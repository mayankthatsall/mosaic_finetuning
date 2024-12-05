# region imports
import random

import transformers
import boto3
from multiprocessing import cpu_count
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics.collections import MetricCollection
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import torch
from composer import Trainer
import argparse
from datasets import load_dataset
import os
import json
from functools import partial
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import numpy as np
import yaml
from composer.utils import dist
from sklearn.metrics import  classification_report
# endregion

TRAINING_COLUMNS = ["input_ids", "attention_mask", "labels"]


def load_data(local_dir: str):
    dfs = {}
    for s in ["train", "test", "validation"]:
        lp = f"{local_dir.rstrip('/')}/{s}"
        if os.path.exists(lp) and len(os.listdir(lp)) > 0:
            # print(f"{lp} exists. loading dataset")
            dfs[s] = f"{local_dir.rstrip('/')}/{s}/*.csv"

    ds = load_dataset("csv", data_files=dfs, column_names=["input", "label"])

    all_labels = set()
    for k, vds in ds.items():
        for l in vds["label"]:
            all_labels.add(l)

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(list(all_labels))

    print(
        f"\nFound following datasets under {local_dir} \n{json.dumps(dfs)}\n{len(label_encoder.classes_)} labels fit"
    )

    return ds, label_encoder


def tokenize_dataset(tokenizer, max_length, label_encoder, sample):
    src = tokenizer(
        text=sample["input"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    labels = sample["label"]
    tgt = label_encoder.transform(labels)
    encodings = {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": tgt,
    }
    return encodings


if __name__ == '__main__':

    from datasets.utils.logging import set_verbosity_error
    set_verbosity_error()

    ds, label_encoder = load_data(local_dir='/tmp/mosaicml/data')
    num_labels = len(label_encoder.classes_)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = transformers.data.data_collator.default_data_collator

    p_tokenized = partial(tokenize_dataset, tokenizer, 256, label_encoder)
    batch_size = 16
    vestigial_columns = set()
    for k, d in ds.items():
        for c in d.column_names:
            if c not in TRAINING_COLUMNS:
                vestigial_columns.add(c)

    tokenized_datasets = ds.map(
        function=p_tokenized,
        batched=True,
        batch_size=batch_size,
        num_proc=cpu_count(),
        remove_columns=list(vestigial_columns),
    )
    for k,d in tokenized_datasets.items():
        d.set_format(type="torch", columns=TRAINING_COLUMNS)

    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )

    for tb in iter(test_dataloader):
        print(len(tb),tb.keys())

