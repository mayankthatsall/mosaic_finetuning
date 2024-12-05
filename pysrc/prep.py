import pandas as pd
from sklearn.model_selection import train_test_split

op = "/tmp/toy2"
src = '/var/data/taxcode/taxcode_all.tsv'

df = pd.read_csv(src,sep='\t')
balanced_df=df.groupby('Category',as_index = False,group_keys=False).apply(lambda s: s.sample(1000,replace=True))
balanced_df["label"] = balanced_df.apply (lambda row: row["Category"], axis=1)

train, test = train_test_split(balanced_df, test_size=0.2, random_state=0, stratify=balanced_df[['Category']])
train.to_csv(f'{op}/train/train.csv')
test.to_csv(f'{op}/test/test.csv')
test.to_csv(f'{op}/validation/validation.csv')
