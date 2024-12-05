from util import s3_sync, get_trainer_config
import os
from dask import dataframe as dd
import pandas as pd
import tabulate

if __name__ == "__main__":
    config = get_trainer_config()
    # region dir_setup
    dirs = [
        "/workspace/model",
        "/workspace/data",
        "/workspace/lm",
        "/workspace/input/partitions/",
        "/workspace/output/partitions/",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    # endregion

    data = config["dataset"]
    print("Downloading data")
    s3_sync(s3_path=data, local_dir="/workspace/data/")

    local_data = "/workspace/data/" #"/var/data/taxcode/toy2"
    for source in os.listdir(local_data):
        rows = []
        keys = ["count","mean","std",
                "min","25%","50%",
                "75%","max"
                ]
        for part in ["train","test","validation"]:
            testset = os.path.join(local_data,source,part,f"{part}.csv")
            df = pd.read_csv(testset)
            df['token_count'] = df.apply(lambda r: len([t for t in r['input'].split() if len(t.strip()) > 0]),axis=1)
            sts = {}
            for k,v in df['token_count'].describe().items():
                sts[k] = v
            row = [source,part]
            for k in keys:
                row.append(sts[k])
            rows.append(row)
            del df
        header = ["source","part"]
        header.extend(keys)
        print(tabulate.tabulate(rows,headers=header))