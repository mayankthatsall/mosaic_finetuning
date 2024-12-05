# s3://mosaicml-68c98fa5-0b21-4c7b-b40b-c4482db8832a/mosaicml-test-bert-5jj4/checkpoints/ep1-ba147783-rank0

import os
import yaml
import torch
import transformers
import boto3
import time

#
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

def s3_download_file(s3_path: str, local_dir)->None:
    s3_path = s3_path.strip()
    local_dir = local_dir.strip()
    bucket = None
    key = None

    bucket = s3_path.split(":")[1].replace("//","").split('/')[0]
    file = s3_path.split(":")[1].replace("//","").split('/')[-1]
    key = '/'.join(s3_path.split(":")[1].replace("//","").split('/')[1:])

    local_file = os.path.join(local_dir, file)

    print(f"{bucket} {key} {local_file}")

    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, local_file)
    return local_file
#

def main():
    """
    Converts a Composer checkpoint to a HF checkpoint.

    Example Usage: python convert_composer_states_to_hf.py --checkpoint-path <CHECKPOINT_PATH_HERE> --output-dir custom_trained_bert

    Then, you can load the model using the `transformers` package:
        import transformers
        transformers.AutoModelForMaskedLM.from_pretrained("custom_trained_bert/")

    and it should successfully work.
    """
    # load the Composer state
    params_file = "/mnt/config/parameters.yaml"
    with open(params_file) as f:
        app_config = yaml.safe_load(f)

    work_dir = app_config.get("work_dir","/workspace/mosaic")
    out_dir = app_config.get("out_dir","/workspace/hf/model/")
    for d in [work_dir,out_dir]:
        os.makedirs(d,exist_ok=True)


    st = time.time()

    checkpoint_path = s3_download_file(s3_path=app_config["s3_checkpoint"],local_dir=work_dir)

    print(f"Checkpoint downloaded in {time.time() - st}s")
    st = time.time()

    composer_state = torch.load(checkpoint_path, map_location="cpu")
    print(f"Composer state loaded in {time.time() - st}s")
    st = time.time()

    # consume the `module.` prefix created by DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        composer_state['state']['model'], "module.")

    # save the pytorch_model.bin file
    hf_model_path = os.path.join(out_dir, "pytorch_model.bin")
    torch.save(composer_state['state']['model'], hf_model_path)

    print(f"Model saved in {time.time() - st}s")
    st = time.time()

    # load and save the config.json file
    config = transformers.AutoConfig.from_pretrained(app_config["model_name"])
    config_path = os.path.join(out_dir, "config.json")
    config.to_json_file(config_path)
    print(f"Successfully saved {app_config['s3_checkpoint']} to {out_dir}.")
    print(f"Config saved in {time.time() - st}s")
    st = time.time()

    if "tokenizer_base" in app_config and "tokenizer_files" in app_config and len(app_config["tokenizer_files"]) > 0:
        base = app_config["tokenizer_base"].rstrip('/')
        for file in app_config["tokenizer_files"]:
            s3file = f"{base}/{file.lstrip('/')}"
            s3_download_file(s3_path=s3file,local_dir=out_dir)

    s3_sync(s3_path=app_config["s3_lm_path"],local_dir=out_dir,pull=False)
    print(f"HF language model uploaded in {time.time() - st}s")

if __name__ == "__main__":
    main()