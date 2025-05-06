import pandas as pd
from sklearn.model_selection import train_test_split


def clean_dataset(ds):
    x = ds[ds["text"] != ""]
    x = x[x["label"] != ""]
    x["text"] = x["text"].str.lower()
    return x


def main():
    train_split = pd.read_csv("train.csv", header=None, names=["label", "text"])
    test_split = pd.read_csv("test.csv", header=None, names=["label", "text"])
    train_split, dev_split = train_test_split(
        train_split, test_size=0.1, random_state=23, stratify=train_split["label"])
    train_split = clean_dataset(train_split)
    dev_split = clean_dataset(dev_split)
    test_split = clean_dataset(test_split)
    train_split.to_csv("train_split.csv", index=False)
    dev_split.to_csv("dev_split.csv", index=False)
    test_split.to_csv("test_split.csv", index=False)


if __name__ == '__main__':
    main()
