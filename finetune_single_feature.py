# Standard library imports
import datetime
from datetime import datetime as DateTime
from functools import partial
import json
from multiprocessing import cpu_count
import os
import shutil
import time

# Third-party imports
import boto3
import composer
from composer import Callback, Event, Logger, State, Trainer
from composer.devices import DeviceGPU
from composer.metrics import CrossEntropy
from composer.models.huggingface import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.utils import dist, get_device
from datasets import load_dataset, Features, Value
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tabulate
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
import transformers
from sentence_transformers import SentenceTransformer, util

# Local imports
from pysrc.inference_export import export_for_inference, get_trainer_config

# Columns for training, now including similarity vector
TRAINING_COLUMNS = ["input_ids", "attention_mask", "sims", "labels"]
label_column = "taxcode"


def load_data(local_dir: str):
    """
    Load datasets from local_dir under train/, validation/, test/ as TSVs,
    return a DatasetDict and a fitted LabelEncoder on taxcode.
    """
    dfs = {}
    for split in ["train", "validation", "test"]:
        path = os.path.join(local_dir, split)
        if os.path.exists(path) and os.listdir(path):
            dfs[split] = f"{path}/*.tsv"

    features = Features({
        "sentence1": Value("string"),
        "company_name": Value("string"),
        "sentence2": Value("string"),
        "category": Value("string"),
        "taxcode": Value("string"),
        "source": Value("string"),
    })
    ds = load_dataset("csv", data_files=dfs, delimiter="\t", features=features)

    # Collect all labels and fit encoder
    all_labels = set()
    for split_ds in ds.values():
        all_labels.update(split_ds[label_column])

    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_labels))

    print(f"Found datasets under {local_dir}: {dfs}, {len(label_encoder.classes_)} labels")
    return ds, label_encoder


def tokenize_with_similarity(tokenizer, max_length, label_encoder, tax_embeds, embedder, samples):
    """
    tokenizer + similarity vector computation for a batch of samples.
    """
    texts = samples["sentence1"]
    labels = samples[label_column]
    # Batch tokenize
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Compute similarities batch-wise if embeddings provided
    if tax_embeds is not None and embedder is not None:
        text_embs = embedder.encode(texts, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(text_embs, tax_embeds).cpu().numpy()
    else:
        sim_matrix = np.zeros((len(texts), len(label_encoder.classes_)), dtype=np.float32)

    # Transform labels
    tgt = label_encoder.transform(labels)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "sims": sim_matrix.tolist(),
        "labels": tgt,
    }


def s3_sync(s3_path: str, local_dir: str, pull=True) -> None:
    """
    Sync files between S3 and local directory.
    """
    s3_path = s3_path.strip()
    local_dir = local_dir.strip()

    def download_dir(client, resource, prefix, start_prefix, local, bucket):
        paginator = client.get_paginator("list_objects")
        for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=prefix):
            for sub in result.get("CommonPrefixes", []):
                download_dir(client, resource, sub["Prefix"], start_prefix, local, bucket)
            for obj in result.get("Contents", []):
                key = obj["Key"]
                rel = key.replace(start_prefix, "").lstrip("/")
                out_path = os.path.join(local, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                resource.meta.client.download_file(bucket, key, out_path)

    def pull_prefix(dst, bucket, prefix):
        client = boto3.client("s3")
        resource = boto3.resource("s3")
        download_dir(client, resource, prefix, prefix, dst, bucket)

    def upload_dir(source, bucket, prefix=""):
        client = boto3.client("s3")
        for root, _, files in os.walk(source):
            for f in files:
                local_path = os.path.join(root, f)
                rel = os.path.relpath(local_path, source)
                key = os.path.join(prefix, rel)
                client.upload_file(local_path, bucket, key)

    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if pull:
        pull_prefix(local_dir, bucket, prefix)
    else:
        upload_dir(local_dir, bucket, prefix)


def create_and_save_pbtxt(model_name, save_path, max_seq_len, labels):
    save_path = f"{'/'.join(save_path.split('/')[:-2])}/config.pbtxt"
    with open(save_path, "w") as f:
        f.write(
            f"""name: {model_name}
backend: \"tensorrt\"
max_batch_size: {32 if 'bulk' in save_path else 12}
instance_group [
{{
    count: 1
    kind: KIND_GPU
}}
]
input: [
{{
    name: \"input_ids\",
    data_type: TYPE_INT32,
    dims: [{max_seq_len}]
}},
{{
    name: \"attention_mask\",
    data_type: TYPE_INT32,
    dims: [{max_seq_len}]
}}
]
output: [
{{
    name: \"output\",
    data_type: TYPE_FP32,
    dims: [{labels}]
}}
]"""
        )

class BatchLoggerCallback(Callback):
    def __init__(self, batch_size, train_records, global_train_batch_size):
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
            print(
                f"{(self.epoch_nd - self.epoch_st):0.2f}s epoch {self.epoch} -  loss {state.loss}"
            )
            self.epoch += 1
            self.epoch_st = time.time()
        if event == Event.BATCH_END and self.batch_count >= self.log_interval:
            nd = time.time()
            print(
                f"{(nd - self.st):0.2f}s / {self.log_interval} batches Done - {self.total} epoch {self.epoch} -  loss {state.loss}"
            )
            self.batch_count = 0

if __name__ == "__main__":
    # Load config
    train_config = get_trainer_config()
    if train_config["dataset"] is None:
        raise Exception("dataset path mandatory")

    # New args
    taxcode_file = train_config.get("taxcode_file")
    sim_model = train_config.get("similarity_model", "all-MiniLM-L6-v2")

    # Prepare dirs
    w = train_config["workdir"].rstrip("/")
    local_data_dir = f"{w}/data/"
    local_model_dir = f"{w}/model/"
    checkpoint_dir = f"{w}/checkpoint/"
    final_model_dir = f"{w}/final_model/"
    MODEL_CATEGORY_NAME = str(train_config.get("model_category_name", "")).strip().lower()
    bulk_inference_model_dir = f"{final_model_dir}bulk_inference/models/repository/{MODEL_CATEGORY_NAME}/1/"
    single_inference_model_dir = f"{final_model_dir}single_inference/models/repository/{MODEL_CATEGORY_NAME}/1/"
    labels_path = f"{final_model_dir.rstrip('/')}/classes.npy"
    SAVE_TENSORRT = str(train_config.get("save_tensorrt", "true")).strip().lower() in ["true"]
    for d in [local_data_dir, local_model_dir, final_model_dir, bulk_inference_model_dir, single_inference_model_dir]:
        os.makedirs(d, exist_ok=True)

    # region download pretrained and dataset via S3
    composer.utils.dist.initialize_dist(DeviceGPU(), timeout=1000)
    with dist.run_local_rank_zero_first():
        # If using a pretrained S3 model, download it
        if train_config.get("pretrained") and train_config.get("pretrained").strip():
            pretrained_model = train_config["pretrained"].strip().rstrip("/") + "/"
            s3_sync(s3_path=pretrained_model, local_dir=local_model_dir)
        # Download training/validation/test data
        if not train_config.get("skip_ds_download", False):
            s3_sync(s3_path=train_config.get("dataset"), local_dir=local_data_dir)
    # endregion

    # Download and embed taxcode descriptions if provided
    if taxcode_file:
        s3_sync(s3_path=taxcode_file, local_dir=local_data_dir)
        tax_df = pd.read_csv(os.path.join(local_data_dir, os.path.basename(taxcode_file)))
        tax_descs = tax_df["Combined_Text"].tolist()
        embedder = SentenceTransformer(sim_model)
        tax_embeds = embedder.encode(tax_descs, convert_to_tensor=True)
    else:
        embedder = None
        tax_embeds = None

    # Load data and encoder
    ds, label_encoder = load_data(local_data_dir)
    num_labels = len(label_encoder.classes_)
    np.save(labels_path, label_encoder.classes_)

    # Load model and tokenizer
    model_source = local_model_dir if train_config.get("pretrained") else train_config["model"]
    config = transformers.AutoConfig.from_pretrained(model_source, num_labels=num_labels)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    max_len = int(train_config.get("maxlen", 256))
    global_train_batch_size = int(train_config.get("train_batch_size", 250))
    global_eval_batch_size = int(train_config.get("eval_batch_size", 500))

    # Tokenize and map
    TRAINING_COLUMNS[:] = ["input_ids", "attention_mask", "sims", "labels"]
    map_fn = partial(tokenize_with_similarity, tokenizer, max_len, label_encoder, tax_embeds, embedder)
    vest_cols = set()
    for split_ds in ds.values():
        vest_cols.update(c for c in split_ds.column_names if c not in TRAINING_COLUMNS)
    tokenized = ds.map(function=map_fn, batched=True, num_proc=cpu_count(), remove_columns=list(vest_cols))
    for split_ds in tokenized.values():
        split_ds.set_format(type="torch", columns=TRAINING_COLUMNS)
    data_collator = transformers.data.data_collator.default_data_collator

    train_dataset = tokenized["train"]
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=global_train_batch_size // dist.get_world_size(),
                                  sampler=dist.get_sampler(train_dataset, shuffle=True, drop_last=False),
                                  collate_fn=data_collator)

    validation_dataloader = DataLoader(dataset=tokenized.get("validation",[]),
                                       batch_size=global_eval_batch_size // dist.get_world_size(),
                                       sampler=dist.get_sampler(tokenized.get("validation",[]), shuffle=False, drop_last=False),
                                       collate_fn=data_collator)

    test_dataloader = DataLoader(dataset=tokenized.get("test",[]),
                                 batch_size=global_eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=data_collator)

    print(f"Datasets tokenized and loaded: train={len(train_dataset)}, val={len(tokenized.get('validation',[]))}, test={len(tokenized.get('test',[]))}")

    # Metrics
    metrics = [CrossEntropy(), Accuracy(task="multiclass", num_classes=num_labels), F1Score(task="multiclass", num_classes=num_labels)]

    # Wrap HF model to consume similarity vector
    class HFWithSimilarity(HuggingFaceModel):
        def __init__(self, model, tokenizer, metrics, num_labels):
            super().__init__(model=model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
            hidden = model.config.hidden_size
            self.head = torch.nn.Linear(hidden + num_labels, num_labels)
            self.num_labels = num_labels

        def forward(self, input_ids, attention_mask, sims=None, labels=None):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = out.last_hidden_state[:,0]
            x = torch.cat([cls_emb, sims], dim=-1) if sims is not None else cls_emb
            logits = self.head(x)
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

    composer_model = HFWithSimilarity(hf_model, tokenizer, metrics, num_labels)
    try:
        composer_model.model_inputs.remove("token_type_ids")
    except:
        pass

    # Load pretrained weights if requested
    if train_config.get("load_as_weights", False):
        state_dict = torch.load(os.path.join(local_model_dir, "pytorch_model.bin"))
        missing, unexpected = composer_model.load_state_dict(state_dict, strict=False)
        if missing: print("MISSING KEYS", missing)
        if unexpected: print("UNEXPECTED KEYS", unexpected)

    # Optimizer & scheduler
    adam_cfg = train_config["optimizer"]["adam"]
    optimizer = DecoupledAdamW(params=composer_model.parameters(),
                               lr=float(adam_cfg["lr"]),
                               betas=(float(adam_cfg["betas"][0]), float(adam_cfg["betas"][1])),
                               eps=float(adam_cfg["eps"]),
                               weight_decay=float(adam_cfg["weight_decay"]))

    sched_cfg = train_config["scheduler"]["linear_scheduler"]
    linear_lr_decay = composer.optim.scheduler.LinearScheduler(
        alpha_i=float(sched_cfg["alpha_i"]), alpha_f=float(sched_cfg["alpha_f"]), t_max=sched_cfg["t_max"]
    )

    # Trainer
    trainer = Trainer(
    model=composer_model,
    run_name=os.environ.get("RUN_NAME"),
    train_dataloader=train_dataloader,
    eval_dataloader=validation_dataloader,
    max_duration=train_config.get("max_duration", "1ep"),
    optimizers=optimizer,
    schedulers=[linear_lr_decay],
    callbacks=[
        BatchLoggerCallback(
            batch_size=train_config.get("log_every_x_batches", 1000),
            train_records=len(train_dataset),
            global_train_batch_size=global_train_batch_size,
        )
    ],
    loggers=[],
    algorithms=train_config.get("algorithms", []),
    device="gpu" if torch.cuda.is_available() else "cpu",
    precision=train_config.get("precision", "fp32"),
    seed=17,
    save_folder=checkpoint_dir,
    save_weights_only=True,
    save_overwrite=True,
    save_interval=train_config.get("save_interval", "1ep"),
    progress_bar=train_config.get("progress_bar", False),
    log_to_console=train_config.get("log_to_console", False),
)

# Training
print("Starting training...")
trainer.fit()
print("Training complete.")

# Save final model
with dist.run_local_rank_zero_first():
    if dist.get_global_rank() == 0:
        hf_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

# Prediction loop
print("Running predictions on test set...")
trainer.state.model.eval()
y_true, y_pred, final_res = [], [], []
with torch.no_grad():
    _device = get_device("gpu" if torch.cuda.is_available() else "cpu")
    for batch in test_dataloader:
        batch = _device.batch_to_device(batch)
        y_true.extend(batch["labels"].cpu().numpy())
        out = trainer.state.model(batch)
        logits = out["logits"].cpu()
        probs, ix = torch.topk(torch.nn.functional.softmax(logits, dim=-1), k=3, dim=-1)
        preds = ix.numpy()
        scores = probs.numpy()
        for p, s in zip(preds, scores):
            final_res.append(list(zip(p.tolist(), s.tolist())))
        y_pred.extend(np.argmax(logits.numpy(), axis=1).tolist())

# Build results DataFrame
res_df = pd.DataFrame(ds.get("test", []))
if not res_df.empty:
    res_df["y_true"] = label_encoder.inverse_transform(y_true)
    res_df["y_pred"] = label_encoder.inverse_transform(y_pred)
    ff = pd.DataFrame(final_res, columns=["res_1", "res_2", "res_3"])
    f1 = pd.DataFrame(ff["res_1"].tolist(), columns=["pred_1", "score_1"])
    f2 = pd.DataFrame(ff["res_2"].tolist(), columns=["pred_2", "score_2"])
    f3 = pd.DataFrame(ff["res_3"].tolist(), columns=["pred_3", "score_3"])
    final_df = pd.concat([res_df, f1, f2, f3], axis=1)
    final_df.to_csv(
        os.path.join(final_model_dir, "predictions.tsv"),
        sep="	",
        index=False,
    )

# Classification report
json_cr = classification_report(
    y_true=label_encoder.inverse_transform(y_true),
    y_pred=label_encoder.inverse_transform(y_pred),
    labels=label_encoder.classes_,
    output_dict=True,
    target_names=list(label_encoder.classes_),
    zero_division=0,
)
with open(os.path.join(final_model_dir, "cr.json"), "w") as w:
    json.dump(json_cr, w, indent=2)
print("Results and report saved.")

# ONNX/TensorRT export
if SAVE_TENSORRT:
    with dist.run_local_rank_zero_first():
        if dist.get_global_rank() == 0:
            sample_batch = {k: v.cpu() for k, v in next(iter(test_dataloader)).items() if k != "labels"}
            export_for_inference(
                model=composer_model,
                save_format="tensorrt",
                save_path=bulk_inference_model_dir,
                sample_input=(sample_batch, {}),
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "output": {0: "batch"},
                },
            )
            create_and_save_pbtxt(MODEL_CATEGORY_NAME, bulk_inference_model_dir, max_len, num_labels)
            export_for_inference(
                model=composer_model,
                save_format="tensorrt",
                save_path=single_inference_model_dir,
                sample_input=(sample_batch, {}),
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "output": {0: "batch"},
                },
            )
            create_and_save_pbtxt(MODEL_CATEGORY_NAME, single_inference_model_dir, max_len, num_labels)
else:
    print("Skipping ONNX/TensorRT export")

# Final S3 upload
with dist.run_local_rank_zero_first():
    if dist.get_global_rank() == 0 and train_config.get("s3_out_dest"):
        s3_sync(s3_path=train_config["s3_out_dest"], local_dir=final_model_dir, pull=False)
