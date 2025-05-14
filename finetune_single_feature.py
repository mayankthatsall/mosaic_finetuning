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
from composer import Callback, Event, Logger, State, Trainer, ComposerModel
from composer.devices import DeviceGPU
from composer.metrics import CrossEntropy
from composer.models.huggingface import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.utils import dist, get_device
from datasets import load_dataset, Features, Value
import mlflow
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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from composer.core import DataSpec

# Local imports
from pysrc.inference_export import export_for_inference, get_trainer_config


mlflow.set_tracking_uri("https://hercule-mlflow.gamma.qa.us-west-2.aws.avalara.io/")

TRAINING_COLUMNS = ["input_ids", "attention_mask", "labels", "similarity_scores"]
# label_column = "taxcode"
label_column = "label"
feature_column = "input"


def compute_similarity_scores(input_texts, taxcode_descriptions, model_name='all-MiniLM-L6-v2'):
    """Compute similarity scores between input texts and taxcode descriptions."""
    sbert = SentenceTransformer(model_name)
    # Embed taxcode descriptions once
    tax_embeds = sbert.encode(taxcode_descriptions, convert_to_tensor=True)
    # Embed all input texts
    input_embeds = sbert.encode(input_texts, convert_to_tensor=True)
    # Compute cosine similarity matrix
    sim_matrix = util.pytorch_cos_sim(input_embeds, tax_embeds).cpu().numpy()
    return sim_matrix


def load_data(local_dir: str, taxcode_file: str = None):
    dfs = {}
    for s in ["train", "test", "validation"]:
        lp = f"{local_dir.rstrip('/')}/{s}"
        if os.path.exists(lp) and len(os.listdir(lp)) > 0:
            # print(f"{lp} exists. loading dataset")
            dfs[s] = f"{local_dir.rstrip('/')}/{s}/*.csv"

    features = Features(
        {
            feature_column: Value(dtype="string", id=None),
            label_column: Value(dtype="string", id=None),
        }
    )
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

    # Load taxcode descriptions if provided
    taxcode_descriptions = None
    if taxcode_file and os.path.exists(taxcode_file):
        tax_df = pd.read_csv(taxcode_file)
        taxcode_descriptions = tax_df['Combined_Text'].tolist()
        print(f"Loaded {len(taxcode_descriptions)} taxcode descriptions")

    return ds, label_encoder, taxcode_descriptions


def tokenize_dataset(sample, tokenizer, label_encoder, label_column, maxlen, taxcode_descriptions=None):
    """Tokenize a single example from the dataset."""
    # Tokenize the text
    tokenized = tokenizer(
        sample[feature_column],
        padding="max_length",
        max_length=maxlen,
        truncation=True,
    )
    
    # Handle labels - ensure it's a list/array
    labels = sample[label_column]
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels]
    
    # Transform labels to numeric values
    try:
        tgt = label_encoder.transform(labels)
    except ValueError as e:
        print(f"Warning: Invalid label found: {labels}")
        # Return a default value or skip this example
        return None

    # Compute similarity scores if taxcode descriptions are available
    similarity_scores = None
    if taxcode_descriptions is not None:
        # Get the taxcode description for this example
        taxcode_desc = taxcode_descriptions.get(sample[label_column], "")
        if taxcode_desc:
            # Compute similarity between the text and taxcode description
            similarity_scores = compute_similarity_scores([sample[feature_column]], [taxcode_desc])[0][0][0]

    # Combine all features
    result = {
        **tokenized,
        "labels": tgt,
    }
    
    if similarity_scores is not None:
        result["similarity_scores"] = similarity_scores
        
    return result


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


def create_and_save_pbtxt(model_name, save_path, max_seq_len, labels):
    if model_name is None:
        model_name = "default_model"
    with open(save_path + "config.pbtxt", "w") as f:
        f.write(
            f"""name: "{model_name}"
            backend: "tensorrt"
            max_batch_size: {32 if "bulk" in save_path else 8}
            instance_group [
            {{
                count: 1
                kind: KIND_GPU
            }}
            ]
            input: [
            {{
                name: "input_ids",
                data_type: TYPE_INT32,
                dims: [{max_seq_len}]
            }},
            {{
                name: "attention_mask",
                data_type: TYPE_INT32,
                dims: [{max_seq_len}]
            }}
            ]
            output: [
            {{
                name: "output",
                data_type: TYPE_FP32,
                dims: [{labels}]
            }}
            ]"""
        )


def freeze_layers_except_last_n(model, n_layers):
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    layers_to_train = 0

    # For BERT-like models
    if hasattr(model, "bert"):
        # Unfreeze the last n transformer layers
        for layer in model.bert.encoder.layer[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                layers_to_train += 1

    # For RoBERTa-like models
    elif hasattr(model, "roberta"):
        # Unfreeze the last n transformer layers
        for layer in model.roberta.encoder.layer[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                layers_to_train += 1

    elif hasattr(model, "distilbert"):  # distilbert
        # Unfreeze the last n transformer layers
        for layer in model.distilbert.transformer.layer[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                layers_to_train += 1

    elif hasattr(model, "model"):  # modernbert
        # Unfreeze the last n transformer layers
        for layer in model.model.layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                layers_to_train += 1

    print(f"Last n trainable layers: {layers_to_train}")

    # Always unfreeze the classification head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


# Print trainable parameters to verify
def count_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}"
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

        if event == Event.BATCH_END:
            if self.batch_count >= self.log_interval:
                self.nd = time.time()
                print(
                    f"{(self.nd - self.st):0.2f}s / {self.log_interval} batches Done - {self.total} epoch {self.epoch} -  loss {state.loss}"
                )
                self.batch_count = 0


class DistilBertWithSimilarity(torch.nn.Module):
    def __init__(self, num_labels, num_taxcodes, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = transformers.DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        # classifier accepts [CLS] embedding + similarity vector
        self.classifier = torch.nn.Linear(hidden_size + num_taxcodes, num_labels)

    def forward(self, input_ids, attention_mask, similarity_scores=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0]  # [CLS]
        x = self.dropout(cls_embed)
        
        if similarity_scores is not None:
            x = torch.cat([x, similarity_scores], dim=1)
        
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}


def create_model(train_config, num_labels, num_taxcodes=None):
    model_name = train_config.get("model", "distilbert-base-uncased")
    
    if num_taxcodes is not None:
        # Use custom model with similarity scores
        model = DistilBertWithSimilarity(
            num_labels=num_labels,
            num_taxcodes=num_taxcodes,
            model_name=model_name
        )
    else:
        # Use standard HuggingFace model
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    return model


class DistilBertComposerModel(ComposerModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch)

    def loss(self, outputs, batch):
        return outputs['loss']

    def eval_forward(self, batch, outputs=None):
        if outputs is None:
            outputs = self.forward(batch)
        return outputs

    def get_metrics(self, is_train=False):
        return {}


class CustomDataSpec(DataSpec):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        
    def get_num_samples_in_batch(self, batch):
        # Get the size of the first tensor (input_ids)
        return batch['input_ids'].size(0)
        
    def batch_transforms(self, batch):
        # Convert batch to tensors and move to device
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
            'similarity_scores': batch.get('similarity_scores', None)
        }


if __name__ == "__main__":
    # region prepare config
    train_config = get_trainer_config()
    print(f"train_config: {train_config}")

    # region prepare paths
    local_data_dir = train_config.get("dataset", None)
    if local_data_dir.startswith("s3://"):
        local_data_dir = "/tmp/data"
        s3_sync(train_config.get("dataset"), local_data_dir, pull=True)

    local_model_dir = train_config.get("s3_out_dest", None)
    if local_model_dir.startswith("s3://"):
        local_model_dir = "/tmp/model"
        os.makedirs(local_model_dir, exist_ok=True)

    labels_path = os.path.join(local_model_dir, "classes.npy")
    # endregion

    # region prepare label encoder
    ds, label_encoder, taxcode_descriptions = load_data(
        local_dir=local_data_dir,
        taxcode_file=train_config.get("taxcode_file", None)
    )
    num_labels = len(label_encoder.classes_)
    np.save(labels_path, label_encoder.classes_)
    # endregion

    # region prepare tokenizer
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
        train_config.get("model", "distilbert-base-uncased")
    )
    max_len = train_config.get("maxlen", 256)
    # endregion

    # region prepare_datasets
    vestigial_columns = set()
    for k, vds in ds.items():
        vestigial_columns.update(vds.column_names)
    vestigial_columns = vestigial_columns - set(TRAINING_COLUMNS)

    ds = ds.map(
        lambda x: tokenize_dataset(
            sample=x,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            label_column=label_column,
            maxlen=max_len,
            taxcode_descriptions=taxcode_descriptions
        ),
        batched=False,
        remove_columns=vestigial_columns,
        desc="Tokenizing dataset",
    ).filter(
        lambda x: x is not None,  # Remove any None values
        desc="Filtering invalid examples"
    )
    # endregion

    # region prepare model
    num_taxcodes = len(taxcode_descriptions) if taxcode_descriptions else None
    base_model = create_model(train_config, num_labels, num_taxcodes)
    model = DistilBertComposerModel(base_model)

    # Add optimizer configuration
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=float(train_config['optimizer']['adam']['lr']),
        betas=tuple(train_config['optimizer']['adam']['betas']),
        eps=float(train_config['optimizer']['adam']['eps']),
        weight_decay=train_config['optimizer']['adam']['weight_decay']
    )
    # endregion

    # region prepare trainer
    # Initialize distributed process group
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    # Get local rank from environment variable
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    # Set tokenizer parallelism to false to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_sampler = DistributedSampler(ds["train"])
    val_sampler = DistributedSampler(ds["validation"], shuffle=False)

    train_dataloader = DataLoader(
        ds["train"],
        batch_size=train_config.get("train_batch_size", 128),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        ds["validation"],
        batch_size=train_config.get("eval_batch_size", 128),
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=CustomDataSpec(train_dataloader),
        eval_dataloader=CustomDataSpec(eval_dataloader),
        max_duration=train_config.get("max_duration", "5ep"),
        device=DeviceGPU(),
        optimizers=optimizer,
        callbacks=[
            BatchLoggerCallback(
                batch_size=train_config.get("train_batch_size", 128),
                train_records=len(ds["train"]),
                global_train_batch_size=train_config.get("train_batch_size", 128)
                * train_config.get("grad_accum", 1),
            )
        ],
    )
    # endregion

    # region train
    trainer.fit()
    # endregion

    # region save model
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    # endregion

    # region export for inference
    if train_config.get("save_tensorrt", False):
        export_for_inference(
            model_dir=local_model_dir,
            bulk_s3_out_dest=train_config.get("bulk_s3_out_dest", None),
            single_s3_out_dest=train_config.get("single_s3_out_dest", None),
            max_seq_len=max_len,
            labels=label_encoder.classes_,
        )
    # endregion

    # region upload to s3
    if train_config.get("s3_out_dest", None).startswith("s3://"):
        s3_sync(train_config.get("s3_out_dest"), local_model_dir, pull=False)
    # endregion
