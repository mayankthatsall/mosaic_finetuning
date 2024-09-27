import tabulate
import transformers
import boto3
from multiprocessing import cpu_count
import composer
# import composer.functional as cf

# from composer.algorithms import FusedLayerNorm
# from composer.algorithms import GatedLinearUnits
from composer.devices import DeviceGPU
# from composer.utils.object_store import S3ObjectStore
from torchmetrics import Accuracy
from torchmetrics import F1Score
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy
from composer.optim import DecoupledAdamW
import torch
from composer import Trainer
from datasets import load_dataset, Features, Value
import os
import json
from functools import partial
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import numpy as np
import yaml
from composer.utils import dist
from sklearn.metrics import classification_report
from composer import Callback, Event, Logger, State
from datasets.utils import disable_progress_bar
import time
import datetime
# from pysrc.inference_export import export_for_inference
import pandas as pd



TRAINING_COLUMNS = ["input_ids", "attention_mask", "labels"]
# label_column = "taxcode"
label_column = "label"
feature_column = 'input'

def load_data(local_dir: str):
    dfs = {}
    for s in ["train", "test", "validation"]:
        lp = f"{local_dir.rstrip('/')}/{s}"
        if os.path.exists(lp) and len(os.listdir(lp)) > 0:
            # print(f"{lp} exists. loading dataset")
            dfs[s] = f"{local_dir.rstrip('/')}/{s}/*.csv"

    features = Features({feature_column: Value(dtype='string', id=None), label_column: Value(dtype='string', id=None)})
    ds = load_dataset("csv", data_files=dfs, features=features)

    all_labels = set()
    for k, vds in ds.items():
        for l in vds[label_column]:
            all_labels.add(l)

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(list(all_labels))

    print(
        f"\nFound following datasets under {local_dir} \n{json.dumps(dfs)}\n{len(label_encoder.classes_)} labels fit"
    )

    return ds, label_encoder


def tokenize_dataset(tokenizer, max_length, label_encoder, sample):
    src = tokenizer(
        sample[feature_column],
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    labels = sample[label_column]
    tgt = label_encoder.transform(labels)
    encodings = {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": tgt,
    }
    return encodings


def s3_sync(s3_path: str, local_dir: str, pull=True) -> None:
    # import os
    # cmd = 'aws s3 sync s3://source-bucket/ my-dir'
    # os.system(cmd)
    # if we have aws cli installed
    #
    s3_path = s3_path.strip()
    local_dir = local_dir.strip()

    # region aws s3 sync implementation
    def download_dir(client, resource, prefix, start_prefix, local, bucket):
        paginator = client.get_paginator("list_objects")
        for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=prefix):
            if result.get("CommonPrefixes") is not None:
                for subdir in result.get("CommonPrefixes"):
                    download_dir(
                        client,
                        resource,
                        subdir.get("Prefix"),
                        start_prefix,
                        local,
                        bucket,
                    )
            if result.get("Contents") is not None:
                for file in result.get("Contents"):
                    # local + os.sep + key_relative
                    key_relative = file.get("Key").replace(start_prefix, "")
                    local_path = os.path.join(local, key_relative.lstrip("/"))
                    local_dir = "/".join(local_path.split("/")[:-1])

                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir, exist_ok=True)

                    s3_path = file.get("Key")
                    print(f"Downloading {s3_path} -> {local_path}")
                    resource.meta.client.download_file(bucket, s3_path, local_path)

    def pull_s3_prefix(dst_dir, bucket, prefix):
        client = boto3.client("s3")
        resource = boto3.resource("s3")
        download_dir(client, resource, prefix, prefix, dst_dir, bucket)

    def upload_dir_s3(source_dir, bucket, prefix=""):
        client = boto3.client("s3")
        # enumerate local files recursively
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                # construct the full local path
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, source_dir)
                s3_path = os.path.join(prefix, relative_path)
                try:
                    print("Uploading %s..." % s3_path)
                    client.upload_file(local_path, bucket, s3_path)
                except Exception as e:
                    print(f"Failed to upload {local_path} to {s3_path}")

    def push_to_s3(local_dir, bucket, prefix):
        upload_dir_s3(source_dir=local_dir, bucket=bucket, prefix=prefix)

    # endregion

    paths = s3_path.split(":")[1].lstrip("//").split("/")
    bucket = paths[0]
    prefix = "/".join(paths[1:])
    print(f"bucket [{bucket}] prefix [{prefix}] local [{local_dir}]")
    if pull:
        pull_s3_prefix(dst_dir=local_dir, bucket=bucket, prefix=prefix)
    else:
        push_to_s3(local_dir=local_dir, bucket=bucket, prefix=prefix)
    #


def __from_params_file(params_file):
    disable_progress_bar()
    trainer_config = {}
    with open(params_file) as f:
        trainer_config = yaml.safe_load(f)
    return trainer_config


def get_trainer_config():
    params_file = "/mnt/config/parameters.yaml"
    return __from_params_file(params_file)

class BatchLoggerCallback(Callback):
    def __init__(self,batch_size,train_records,global_train_batch_size):
        print("Batch logger initialized")
        self.log_interval = batch_size
        self.st = time.time()
        self.batch_count = 0
        self.train_records = train_records
        self.global_train_batch_size = global_train_batch_size
        self.total = 0
        self.epoch = 0
        self.epoch_st = time.time()
        self.epoch_nd = time.time()
        self.estimated_batched = self.train_records / self.global_train_batch_size
        self.time_per_batch = 0

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.BATCH_START:
            if self.batch_count == 0:
                self.st = time.time()
            self.batch_count += 1
            self.total += 1

        if event == Event.EPOCH_START:
            self.epoch_st = time.time()

        if event == Event.EPOCH_END:
            self.epoch_nd = time.time()
            print(f"{(self.epoch_nd - self.epoch_st):0.2f}s epoch {self.epoch} -  loss {state.loss}")
            self.epoch += 1
            self.epoch_st = time.time()


        if event == Event.BATCH_END:
            if self.batch_count >= self.log_interval:
                self.nd  = time.time()
                print(f"{(self.nd - self.st):0.2f}s / {self.log_interval} batches Done - {self.total} epoch {self.epoch} -  loss {state.loss}")
                self.batch_count = 0

if __name__ == "__main__":
    # we need
    # 1. path to the pre trained model
    # 2. path to the dataset
    # region args
    train_config = get_trainer_config()
    if train_config["dataset"] is None:
        raise Exception("dataset path mandatory")

    # save_onnx = str(train_config.get("save_onnx","true")).strip().lower() in ["true"]

    # region prepare_disk
    w = train_config["workdir"].rstrip("/")
    local_data_dir = f"{w}/data/"
    local_model_dir = f"{w}/model/"
    checkpoint_dir = f"{w}/checkpoint/"
    final_model_dir = f"{w}/trained_model/"
    labels_path = f"{final_model_dir.rstrip('/')}/classes.npy"
    for d in [local_data_dir, local_model_dir, final_model_dir]:
        os.makedirs(d, exist_ok=True)
    # endregion

    # region download_data_models
    #composer.utils.dist.initialize_dist(composer.trainer.devices.DeviceGPU(), timeout=datetime.timedelta(seconds=300))
    # the composer version 11, throws an error if ^^ is used
    # upgrade the composer version to 11 in the yaml if you see
    # TypeError: unsupported type for timedelta seconds component: datetime.timedelta
    composer.utils.dist.initialize_dist(DeviceGPU(), timeout=1000)
    with dist.run_local_rank_zero_first():
        if (
            train_config.get("pretrained", None)
            and len(train_config["pretrained"].strip()) > 0
        ):
            pretrained_model = train_config["pretrained"].strip().rstrip('/')
            pretrained_model = f"{pretrained_model}/"
            s3_sync(s3_path=pretrained_model, local_dir=f"{w}/model/")

        if not train_config.get("skip_ds_download", False):
            s3_sync(s3_path=train_config["dataset"], local_dir=f"{w}/data/")
    # endregion

    # region prepare label encoder
    ds, label_encoder = load_data(local_dir=local_data_dir)
    num_labels = len(label_encoder.classes_)
    np.save(labels_path, label_encoder.classes_)
    # endregion

    # Build a blank model from the config
    model_name_or_location = local_model_dir if train_config.get("pretrained", None) else train_config["model"]
    config = transformers.AutoConfig.from_pretrained(model_name_or_location, num_labels=num_labels)

    print(config.to_json_string())

    print("Loading model from ",model_name_or_location)

    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)

    print("Loading tokenizer from ",model_name_or_location)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_location)
    

    max_len = int(train_config["maxlen"])
    global_train_batch_size = int(train_config.get("train_batch_size",250))
    global_eval_batch_size = int(train_config.get("eval_batch_size",500))

    save_composer_checkpoint = str(train_config.get("save_composer_checkpoint","true")).strip().lower() in ["true"]
    save_pytorch_bin = train_config.get("save_pytorch_bin","true").strip().lower() in ["true"]

    device_train_batch_size = global_train_batch_size // dist.get_world_size()
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()

    #region prepare_datasets
    p_tokenized = partial(tokenize_dataset, tokenizer, max_len, label_encoder)

    vestigial_columns = set()
    for k, d in ds.items():
        for c in d.column_names:
            if c not in TRAINING_COLUMNS:
                vestigial_columns.add(c)

    print(f"Removing {vestigial_columns}")

    tokenized_datasets = ds.map(
        function=p_tokenized,
        batched=True,
        num_proc=cpu_count(),
        remove_columns=list(vestigial_columns),
    )
    for k, d in tokenized_datasets.items():
        d.set_format(type="torch", columns=TRAINING_COLUMNS)

    data_collator = transformers.data.data_collator.default_data_collator
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=device_train_batch_size,
        sampler=dist.get_sampler(train_dataset, drop_last=False, shuffle=True),
        drop_last=False,
        collate_fn=data_collator,
    )
    validation_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=device_eval_batch_size,
        sampler=dist.get_sampler(validation_dataset, drop_last=False, shuffle=False),
        drop_last=False,
        collate_fn=data_collator,
    )
    # No DistributedSampler for the test_dataloader
    # because it is used in a trainer.predict loop that only tracks results from rank0
    # TODO? Make the predict loop PredictionCallback compatible with multi-gpu
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
        sampler=None #dist.get_sampler(validation_dataset, drop_last=False, shuffle=False),
    )

    print(f"Datasets tokenized {tokenized_datasets.shape}")
    #endregion

    metrics = [CrossEntropy(), Accuracy(task='multiclass', num_classes=num_labels), F1Score(task="multiclass", num_classes=num_labels)]
    # Package as a composer model
    composer_model = HuggingFaceModel(model=hf_model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    try:
      composer_model.model_inputs.remove('token_type_ids')
    except Exception:
      print("Unable to remove token_type_ids from composer_model.model_inputs")

    # Apply surgery algorithms using functional API
    # algorithms = train_config.get('algorithms',[])
    # if algorithms is not None and len(algorithms) > 0:
    #     # write out the algorithms and other special layers included
    #     # so that we can re-apply them when we initialize the models.
    #     composer_config_file = os.path.join(final_model_dir, "composer_config.json")
    #     ccfg = { 'algorithms': algorithms }
    #     with open(composer_config_file,'w') as fd:
    #         json.dump(ccfg,fd,indent=2)

    #     if 'GatedLinearUnits' in algorithms:
    #         cf.apply_gated_linear_units(composer_model, optimizers=None)
    #     if 'FusedLayerNorm' in algorithms:
    #         cf.apply_fused_layernorm(composer_model, optimizers=None)
    #     if 'Alibi' in algorithms:
    #         cf.apply_alibi(composer_model,optimizers=None,max_sequence_length=256)

    if train_config.get("print_composer",False):
        print(composer_model)

    # Load the weights
    if train_config.get("load_as_weights", False):
        pretrained_hf_state_dict = torch.load(os.path.join(local_model_dir, "pytorch_model.bin"))
        #missing_keys, unexpected_keys = composer_model.model.load_state_dict(pretrained_hf_state_dict, strict=False)
        missing_keys, unexpected_keys = composer_model.load_state_dict(pretrained_hf_state_dict, strict=False)
        if len(missing_keys) > 0:
            print ("MISSING_KEYS")
            for k in missing_keys:
                print (k)
        else:
            print("No missing keys: OK")

        if len(unexpected_keys) > 0:
            print ("UNEXPECTED_KEYS")
            for k in (unexpected_keys):
                print (k)
        else:
            print("No unexpected keys: OK")

    # if 'Alibi' in algorithms:
    #     cf.apply_alibi(composer_model,optimizers=None,max_sequence_length=256)
        
    adam_lr=float(train_config["optimizer"]["adam"]["lr"])
    adam_beta_st=float(train_config["optimizer"]["adam"]["betas"][0])
    adam_beta_nd=float(train_config["optimizer"]["adam"]["betas"][1])
    adam_eps=float(train_config["optimizer"]["adam"]["eps"])
    adam_weight_decay=float(train_config["optimizer"]["adam"]["weight_decay"])

    optimizer = DecoupledAdamW(
        params=composer_model.parameters(),
        lr=adam_lr,
        betas=(adam_beta_st,adam_beta_nd),
        eps=adam_eps,
        weight_decay=adam_weight_decay
    )

    print(train_config["optimizer"]["adam"])
    print(train_config["scheduler"]["linear_scheduler"])

    ls_alpha_i=float(train_config["scheduler"]["linear_scheduler"]["alpha_i"])
    ls_alpha_f=float(train_config["scheduler"]["linear_scheduler"]["alpha_f"])
    t_max=train_config["scheduler"]["linear_scheduler"]["t_max"]

    linear_lr_decay = composer.optim.scheduler.LinearScheduler(alpha_i=ls_alpha_i, alpha_f=ls_alpha_f, t_max=t_max)

    algorithms = []
    
    print(f"Composer version: {composer.__version__}")

    # from composer.loggers import ObjectStoreLogger
    # 
    # object_store_logger = ObjectStoreLogger(
    #     object_store_cls=S3ObjectStore,
    #     object_store_kwargs={
    #         'bucket':'mosaicml-68c98fa5-0b21-4c7b-b40b-c4482db8832a',
    #     }
    # )
    # loggers=[object_store_logger, TensorboardLogger(flush_interval=10)]
    loggers = []
    # c4_algorithms=[
    #     composer.algorithms.GatedLinearUnits(),
    #     composer.algorithms.Alibi(max_sequence_length=128, train_sequence_length_scaling=1.0),
    #     composer.algorithms.FusedLayerNorm(),
    # ]
    #
    train_records = tokenized_datasets.get('train').shape[0] if 'train' in tokenized_datasets else 0

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        run_name=os.environ.get("RUN_NAME"),
        # console_log_level="batch",
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=train_config.get("max_duration", "1ep"),
        optimizers=optimizer,
        schedulers=[linear_lr_decay],
        # grad_accum=train_config.get("grad_accum","auto"),
        loggers=loggers,
        algorithms=algorithms, 
        device="gpu" if torch.cuda.is_available() else "cpu",
        train_subset_num_batches=int(train_config.get("train_subset_num_batches", -1)),
        eval_subset_num_batches=int(train_config.get("eval_subset_num_batches", -1)),
        precision=train_config.get("precision", "fp32"),
        seed=17,
        save_folder=checkpoint_dir,
        save_weights_only=True,
        save_overwrite=True,
        save_latest_filename="latest",
        save_interval=train_config.get("save_interval", "1ep"),
        callbacks=[BatchLoggerCallback(batch_size=train_config.get("log_every_x_batches",1000),
                                       train_records=train_records,
                                       global_train_batch_size=global_train_batch_size)
                   ],
        progress_bar=train_config.get("progress_bar",True),
        log_to_console=train_config.get("log_to_console",None),
    )

    trn_st = time.time()
    print("Starting training")
    # Start training
    trainer.fit()
    trn_nd = time.time()
    training_time = str(datetime.timedelta(seconds=trn_nd-trn_st))
    print(f"Training and validation done. {training_time}")

    print(trainer.state.eval_metrics)

    # from composer.utils.checkpoint import save_checkpoint
    # final_state_file = os.path.join(final_model_dir,"fs_composer_state.pt")
    # sfn = save_checkpoint(trainer.state,final_state_file,weights_only=True)

    with dist.run_local_rank_zero_first():
        if dist.get_global_rank() == 0:
            hf_model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            print(f"pytorch model saved to {final_model_dir}")

    #region prediction loop
    print("Starting Prediction on test_dataloader")
    trainer.state.model.eval()
    y_true = []
    y_pred = []
    probs = []
    final_res = []


    # probs = fn(logits, **params)
    #
    # # Apply filters
    # assert probs.shape == filters_mask.shape
    # probs = torch.mul(probs, filters_mask)
    #
    # probs, ix = torch.topk(probs, k=len(self.labels), dim=-1)

    from composer.utils import get_device
    with torch.no_grad():
        _device = get_device("gpu" if torch.cuda.is_available() else "cpu")
        for batch in test_dataloader:
            #batch = trainer._device.batch_to_device(batch)
            batch = _device.batch_to_device(batch)
            # batch = trainer.device.batch_to_device(batch)
            y_true.extend(batch['labels'].cpu().numpy())
            predicted = trainer.state.model(batch)
            logits = predicted.logits.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs, ix = torch.topk(probs, k=3, dim=-1)
            pred_probs = ([[a.item() for a in p] for p in probs])
            pred_labels = np.apply_along_axis(
                func1d=label_encoder.inverse_transform, arr=ix, axis=-1
            )
            res = [[(k, l) for k, l in zip(i, j)] for i, j in zip(pred_labels, pred_probs)]
            final_res.extend(res)
            for p in predicted.logits:
                y_pred.append(torch.argmax(p).item())

        target_names = list(label_encoder.classes_)

        y_true = label_encoder.inverse_transform(y_true)
        y_pred = label_encoder.inverse_transform(y_pred)

        def get_invalids(l,c):
            invalids = set()
            for t in c:
                if t not in l:
                    invalids.add(t)
            return invalids

        print(f"Invalid codes in y_true {get_invalids(l=target_names,c=y_true)}")
        print(f"Invalid codes in y_true {get_invalids(l=target_names,c=y_pred)}")
        res_df = pd.DataFrame(ds['test'])
        res_df["y_true"] = y_true
        res_df["y_pred"] = y_pred
        ff = pd.DataFrame(final_res, columns=['res_1', 'res_2', 'res_3'])
        f1 = pd.DataFrame(ff['res_1'].values.tolist(), index=ff.index, columns=['pred_1', 'score_1'])
        f2 = pd.DataFrame(ff['res_2'].values.tolist(), index=ff.index, columns=['pred_2', 'score_2'])
        f3 = pd.DataFrame(ff['res_3'].values.tolist(), index=ff.index, columns=['pred_3', 'score_3'])
        final_res_df = pd.concat([res_df, f1, f2, f3], axis=1)
        result_file = f"{final_model_dir}/predictions.tsv"
        final_res_df.to_csv(result_file, sep='\t', index=False)

        # str_cr = classification_report(
        #     y_true=y_true, y_pred=y_pred, labels=label_encoder.classes_,target_names=target_names,
        #     zero_division=0
        # )

        json_cr = classification_report(
            y_true=y_true, y_pred=y_pred, labels=label_encoder.classes_,output_dict=True,target_names=target_names,
            zero_division=0
        )

        koi = ["precision","recall","f1-score","support"]
        rows = []
        for k in json_cr.keys():
            if k in ["accuracy"]:
                continue
            row = [k]
            for n in koi:
                row.append(json_cr[k][n])
            rows.append(row)
        rows.sort(key=lambda x : x[-1],reverse=True)
        headers = ["label","precision","recall","f1-score","support"]
        import tabulate
        str_cr = tabulate.tabulate(rows,headers=headers)
        print(str_cr)

        cr_file = os.path.join(final_model_dir,"cr.txt")
        json_cr_file = os.path.join(final_model_dir,"cr.json")

        with open(cr_file, "w") as w:
            w.write(str_cr)

        with open(json_cr_file, "w") as w:
            json.dump(json_cr,w,indent=1)
    #endregion


    #region export for inference using ONNX
    # if save_onnx:
    #     with dist.run_local_rank_zero_first():
    #         if dist.get_global_rank() == 0:
    #             onnx_model_save_path = os.path.join(final_model_dir, 'model2.onnx')

    #             # print(trainer.state.batch)
    #             print(trainer.state.batch.keys())
    #             sample_input = trainer.state.batch
    #             # the keys when we tokenize will be
    #             # input_ids, attention_mask
    #             # remove all other keys
    #             if 'labels' in sample_input:
    #                 del sample_input['labels']

    #             # ONNX export requires all values to be moved to the CPU
    #             device = torch.device('cpu')
    #             composer_model.to(device=device)
    #             for key, value in sample_input.items():
    #                 sample_input[key] = value.to(device=device)

    #             export_for_inference(
    #                 model=composer_model,
    #                 save_format="onnx",
    #                 save_path=onnx_model_save_path,
    #                 sample_input=(trainer.state.batch,{}),
    #                 dynamic_axes={'input_ids' :{0 : 'batch_size',1: 'sentence_length'},
    #                               'attention_mask' :{0 : 'batch_size',1: 'sentence_length'},
    #                               'output': {0: 'batch_size'}}
    #             )
    # else:
    #     print("Skipping ONNX model")
    #endregion




    #region move final_model_dir to s3
    with dist.run_local_rank_zero_first():
        if dist.get_global_rank() == 0:
            s3_out_dir = train_config.get("s3_out_dest", None)
            import shutil
            copy_files = ["args.json","input_text_config_"]
            for f in os.listdir(local_data_dir):
                do_copy = False
                for c in copy_files:
                    if f.startswith(c):
                        do_copy = True
                        break
                if do_copy:
                    src_file = os.path.join(local_data_dir,f)
                    dst_file = os.path.join(final_model_dir,f)
                    shutil.copy(src=src_file,dst=dst_file)
                    print(f"Copied {src_file} to {dst_file}")

            if s3_out_dir is not None:
                s3_out_dir = s3_out_dir.rstrip("/")
                from datetime import datetime
                s = datetime.now()
                rnd = f"{s}".split()[1].split(".")[0].replace(":","-")
                dt = f"{s}".split()[0].split(".")[0].replace(":","-")
                ym = trainer.state.run_name
                s3_path = f"{s3_out_dir}/{rnd}_{dt}_{ym}/"
                print("Writing model to ",s3_path)
                s3_sync(s3_path=s3_path, local_dir=final_model_dir, pull=False)
    #endregion


