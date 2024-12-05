# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Inference-related utility functions for model export and optimizations.

Used for exporting models into various formats such ONNX, torchscript etc. and apply optimizations such as fusion.
"""
from __future__ import annotations

# Standard library imports
import contextlib
import copy
import functools
import logging
import os
import tempfile
import yaml
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

# Third-party imports
import tensorrt as trt
import torch
import torch.nn as nn

# Composer imports
from composer.utils import dist
from composer.utils.checkpoint import download_checkpoint
from composer.utils.iter_helpers import ensure_tuple
from composer.utils.misc import is_model_ddp, is_model_deepspeed, model_eval_mode
from composer.utils.object_store import ObjectStore
from composer.utils.string_enum import StringEnum

# Dataset utils
from datasets.utils import disable_progress_bar

# Type checking imports
if TYPE_CHECKING:
    from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = [
    "export_for_inference",
    "ExportFormat",
    "export_with_logger",
    "quantize_dynamic",
]

Transform = Callable[[nn.Module], nn.Module]

# This is the most common way to use dynamic quantization.
#  Example:
#    from composer.utils import quantize_dynamic
#    export_for_inference(
#        ...
#        transforms = [quantize_dynamic],
#        ...
#    )
#  A user can always redefine it with extra options. This also serves as an example of what to pass to transforms.
quantize_dynamic = functools.partial(
    torch.quantization.quantize_dynamic, qconfig_spec={torch.nn.Linear}
)


def __from_params_file(params_file):
    disable_progress_bar()
    trainer_config = {}
    with open(params_file) as f:
        trainer_config = yaml.safe_load(f)
    return trainer_config


def get_trainer_config():
    params_file = "/mnt/config/parameters.yaml"
    return __from_params_file(params_file)


def export_to_tensorrt(
    onnx_model_save_path, tensorrt_model_save_path, tensorrt_max_batch_size
):
    max_len = get_trainer_config()["maxlen"]
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_builder = trt.Builder(trt_logger)
    # trt_builder.fp16_mode = True
    # trt_builder.max_batch_size = 12
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_network = trt_builder.create_network(explicit_batch)
    trt_parser = trt.OnnxParser(trt_network, trt_logger)
    with open(onnx_model_save_path, "rb") as f:
        assert trt_parser.parse(f.read())

    trt_config = trt_builder.create_builder_config()
    # trt_config.max_workspace_size = 1 * (1 << 30)
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    assert trt_builder.platform_has_fast_fp16
    trt_config.set_flag(trt.BuilderFlag.FP16)
    trt_optim_profile = trt_builder.create_optimization_profile()
    trt_optim_profile.set_shape(
        "input_ids",
        (1, max_len),
        (tensorrt_max_batch_size, max_len),
        (tensorrt_max_batch_size, max_len),
    )
    trt_optim_profile.set_shape(
        "attention_mask",
        (1, max_len),
        (tensorrt_max_batch_size, max_len),
        (tensorrt_max_batch_size, max_len),
    )
    trt_config.add_optimization_profile(trt_optim_profile)
    # trt_engine = trt_builder.build_engine(trt_network, trt_config)
    trt_serialized_engine = trt_builder.build_serialized_network(
        trt_network, trt_config
    )
    with open(f"{tensorrt_model_save_path}", "wb") as f:
        f.write(trt_serialized_engine)

    print(f"completed tensorrt models saving {tensorrt_model_save_path}")
    # endregion


class ExportFormat(StringEnum):
    """Enum class for the supported export formats.

    Attributes:
        torchscript: Export in "torchscript" format.
        onnx:  Export in "onnx" format.
    """

    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


def export_for_inference(
    model: nn.Module,
    save_format: Union[str, ExportFormat],
    save_path: str,
    save_object_store: Optional[ObjectStore] = None,
    sample_input: Optional[Any] = None,
    surgery_algs: Optional[
        Union[
            Callable[[nn.Module], nn.Module], Sequence[Callable[[nn.Module], nn.Module]]
        ]
    ] = None,
    transforms: Optional[Sequence[Transform]] = None,
    load_path: Optional[str] = None,
    load_object_store: Optional[ObjectStore] = None,
    load_strict: bool = False,
    dynamic_axes=None,
) -> None:
    """Export a model for inference.

    Args:
        model (nn.Module): An instance of nn.Module. Please note that model is not modified inplace.
            Instead, export-related transformations are applied to a  copy of the model.
        save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
        save_path: (str): The path for storing the exported model. It can be a path to a file on the local disk,
        a URL, or if ``save_object_store`` is set, the object name
            in a cloud bucket. For example, ``my_run/exported_model``.
        save_object_store (ObjectStore, optional): If the ``save_path`` is in an object name in a cloud bucket
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` which will be used
            to store the exported model. Set this to ``None`` if ``save_path`` is a local filepath.
            (default: ``None``)
        sample_input (Any, optional): Example model inputs used for tracing. This is needed for "onnx" export.
            The ``sample_input`` need not match the batch size you intend to use for inference. However, the model
            should accept the ``sample_input`` as is. (default: ``None``)
        surgery_algs (Union[Callable, Sequence[Callable]], optional): Algorithms that should be applied to the model
            before loading a checkpoint. Each should be callable that takes a model and returns modified model.
            ``surgery_algs`` are applied before ``transforms``. (default: ``None``)
        transforms (Sequence[Transform], optional): transformations (usually optimizations) that should
            be applied to the model. Each Transform should be a callable that takes a model and returns a modified model.
            ``transforms`` are applied after ``surgery_algs``. (default: ``None``)
        load_path (str): The path to an existing checkpoint file.
            It can be a path to a file on the local disk, a URL, or if ``load_object_store`` is set, the object name
            for a checkpoint in a cloud bucket. For example, run_name/checkpoints/ep0-ba4-rank0. (default: ``None``)
        load_object_store (ObjectStore, optional): If the ``load_path`` is in an object name  in a cloud bucket
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` which will be used to retreive the checkpoint.
            Otherwise, if the checkpoint is a local filepath, set to ``None``. (default: ``None``)
        load_strict (bool): Whether the keys (i.e., model parameter names) in the model state dict should
            perfectly match the keys in the model instance. (default: ``False``)

    Returns:
        None
    """
    save_format = ExportFormat(save_format)

    if is_model_deepspeed(model):
        raise ValueError(f"Exporting for deepspeed models is currently not supported.")

    if is_model_ddp(model):
        raise ValueError(
            f"Directly exporting a DistributedDataParallel model is not supported. Export the module instead."
        )

    # Only rank0 exports the model
    if dist.get_global_rank() != 0:
        return

    # make a copy of the model so that we don't modify the original model
    model = copy.deepcopy(model)

    # make a copy of the sample input so that we don't modify the original sample input
    sample_input = copy.deepcopy(sample_input)

    # Apply surgery algorithms in the given order
    for alg in ensure_tuple(surgery_algs):
        model = alg(model)

    if load_path is not None:
        # download checkpoint and load weights only
        log.debug("Loading checkpoint at %s", load_path)
        with tempfile.TemporaryDirectory() as tempdir:
            composer_states_filepath, _, _ = download_checkpoint(
                path=load_path,
                node_checkpoint_folder=tempdir,
                object_store=load_object_store,
                progress_bar=True,
            )
            state_dict = torch.load(composer_states_filepath, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict["state"]["model"], strict=load_strict
            )
            if len(missing_keys) > 0:
                log.warning(
                    f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}"
                )
            if len(unexpected_keys) > 0:
                log.warning(
                    f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}"
                )

    with model_eval_mode(model):
        # Apply transformations (i.e., inference optimizations) in the given order
        for transform in ensure_tuple(transforms):
            model = transform(model)

        is_remote_store = save_object_store is not None
        tempdir_ctx = (
            tempfile.TemporaryDirectory()
            if is_remote_store
            else contextlib.nullcontext(None)
        )
        with tempdir_ctx as tempdir:
            if is_remote_store:
                local_save_path = os.path.join(str(tempdir), "model.export")
            else:
                local_save_path = save_path

            if save_format == ExportFormat.TORCHSCRIPT:
                export_model = None
                try:
                    export_model = torch.jit.script(model)
                except Exception:
                    if sample_input is not None:
                        log.warning(
                            "Scripting with torch.jit.script failed. Trying torch.jit.trace!",
                        )
                        export_model = torch.jit.trace(model, sample_input)
                    else:
                        log.warning(
                            "Scripting with torch.jit.script failed and sample inputs are not provided for tracing "
                            "with torch.jit.trace",
                            exc_info=True,
                        )

                if export_model is not None:
                    torch.jit.save(export_model, local_save_path)
                else:
                    raise RuntimeError(
                        "Scritping and tracing failed! No model is getting exported."
                    )

            if save_format == ExportFormat.ONNX or save_format == ExportFormat.TENSORRT:
                if sample_input is None:
                    raise ValueError(
                        f"sample_input argument is required for onnx export"
                    )
                sample_input = ensure_tuple(sample_input)

                input_names = ["input"]

                # Extract input names from sample_input if it is a dict
                if isinstance(sample_input[0], dict):
                    input_names = list(sample_input[0].keys())

                torch.onnx.export(
                    model,
                    sample_input,
                    os.path.join(local_save_path, "model.onnx"),
                    input_names=input_names,
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                )

                if save_format == ExportFormat.TENSORRT:
                    export_to_tensorrt(
                        os.path.join(local_save_path, "model.onnx"),
                        os.path.join(local_save_path, "model.plan"),
                        tensorrt_max_batch_size=32 if "bulk" in local_save_path else 12,
                    )
                    # delete the onnx model
                    os.remove(os.path.join(local_save_path, "model.onnx"))

            # upload if required.
            if is_remote_store:
                save_object_store.upload_object(save_path, local_save_path)


def export_with_logger(
    model: nn.Module,
    save_format: Union[str, ExportFormat],
    save_path: str,
    logger: Logger,
    save_object_store: Optional[ObjectStore] = None,
    sample_input: Optional[Any] = None,
    transforms: Optional[Sequence[Transform]] = None,
) -> None:
    """Helper method for exporting a model for inference.

    Exports the model to:
    1) save_object_store, if one is provided,
    2) logger.file_artifact(save_path), if (1) does not apply and the logger has a destination that supports file artifact logging,
    3) locally, if (1) and (2) do not apply.

    Args:
        model (nn.Module): An instance of nn.Module. Please note that model is not modified inplace.
            Instead, export-related transformations are applied to a  copy of the model.
        save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
        save_path: (str): The path for storing the exported model. It can be a path to a file on the local disk,
        a URL, or if ``save_object_store`` is set, the object name
            in a cloud bucket. For example, ``my_run/exported_model``.
        logger (Logger): If this logger has a destination that supports file artifacting logging, and save_object_store
            is not provided, this logger is used to export the model.
        save_object_store (ObjectStore, optional): If the ``save_path`` is in an object name in a cloud bucket
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` which will be used
            to store the exported model. Set this to ``None`` if the logger should be used to export the model or
            if ``save_path`` is a local filepath.
            (default: ``None``)
        sample_input (Any, optional): Example model inputs used for tracing. This is needed for "onnx" export.
            The ``sample_input`` need not match the batch size you intend to use for inference. However, the model
            should accept the ``sample_input`` as is. (default: ``None``)
        transforms (Sequence[Transform], optional): transformations (usually optimizations) that should
            be applied to the model. Each Transform should be a callable that takes a model and returns a modified model.
            ``transforms`` are applied after ``surgery_algs``. (default: ``None``)

    Returns:
        None
    """
    from composer.loggers import LogLevel

    if save_object_store == None and logger.has_file_artifact_destination():
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_local_save_path = os.path.join(str(tmpdir), f"model")
            export_for_inference(
                model=model,
                save_format=save_format,
                save_path=temp_local_save_path,
                sample_input=sample_input,
                transforms=transforms,
            )
            logger.file_artifact(
                log_level=LogLevel.FIT,
                artifact_name=save_path,
                file_path=temp_local_save_path,
            )
    else:
        export_for_inference(
            model=model,
            save_format=save_format,
            save_path=save_path,
            save_object_store=save_object_store,
            sample_input=sample_input,
            transforms=transforms,
        )
