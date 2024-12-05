import boto3
import os
import yaml
import csv
import time
import json
from typing import Iterator, Dict, List


def __from_params_file(params_file):
    trainer_config = {}
    with open(params_file) as f:
        trainer_config = yaml.safe_load(f)
    return trainer_config


def get_trainer_config():
    params_file = "/mnt/config/parameters.yaml"
    return __from_params_file(params_file)


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


def single_file_download(s3_path: str, local_dir: str) -> str:
    paths = s3_path.split(":")[1].lstrip("//").split("/")
    bucket = paths[0]
    prefix = "/".join(paths[1:])
    client = boto3.client("s3")
    client.download_file(
        Bucket=bucket, Key=prefix, Filename=os.path.join(local_dir, paths[-1])
    )
    return os.path.join(local_dir, paths[-1])


def predict_on_file(fn_predict, dataset_file, prediction_file, batch_size):
    #
    y_true = []
    y_pred = []
    with open(prediction_file, "w") as w:
        writer = csv.writer(w)
        writer.writerow(["y_true", "y_pred"])
        with open(dataset_file, "r") as r:
            reader = csv.reader(r)
            headers = next(reader)
            batch = []
            # batch_size = 250
            st = time.time()
            rc = 0
            for cols in reader:
                txt = cols[-2]
                label = cols[-1]
                rc += 1
                batch.append(txt)
                y_true.append(label)
                if len(batch) >= batch_size:
                    predictions, confidence = fn_predict(sentences=batch)
                    batch.clear()
                    for pb in predictions:
                        y_pred.append(pb[0])
                    rows = [[t, p] for t, p in zip(y_true, y_pred)]
                    writer.writerows(rows)
                    y_true.clear()
                    y_pred.clear()

            if len(batch) > 0:
                predictions, confidence = fn_predict(sentences=batch)
                batch.clear()
                for pb in predictions:
                    y_pred.append(pb[0])
                rows = [[t, p] for t, p in zip(y_true, y_pred)]
                writer.writerows(rows)
                y_true.clear()
                y_pred.clear()

    print(f"Done prediction in {(time.time() - st):.4}s")


class Reader:
    def get_line(self) -> Iterator[Dict[str, str]]:
        raise NotImplementedError


class JSONLReader(Reader):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def get_line(self) -> Iterator[Dict[str, str]]:
        with open(self.file_name) as f:
            for line in f:
                yield json.loads(line)
