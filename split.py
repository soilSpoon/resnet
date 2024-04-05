import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = "/data/ephemeral/home/resnet/data"

attr = pd.read_csv(f"{ROOT}/Anno/list_attr_celeba.txt", sep="\\s+", skiprows=1)

attr_train, attr_test = train_test_split(attr, test_size=0.2, random_state=42)
print(attr_train.index)

attr_train.to_csv(f"{ROOT}/Anno/train.csv")
attr_test.to_csv(f"{ROOT}/Anno/test.csv")

a = pd.read_csv(f"{ROOT}/Anno/train.csv", index_col=0)

print(attr_train.head())
print(a.head())
