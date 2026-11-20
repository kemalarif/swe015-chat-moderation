"""Download OffensEval-TR 2020 and build train / val / test splits.

- test  = the official OffensEval 2020 Turkish test set (gold labels), held out for every model
- train / val = stratified 90 / 10 split of the official training set

Usage:  python -m src.data
"""
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_URL, LABEL2ID, PROC_DIR, RAW_DIR, SEED, VAL_FRACTION

BASE = RAW_DIR / "offenseval2020-turkish"
TRAIN_TSV = BASE / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv"
TEST_TSV = BASE / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv"
TEST_LABELS = BASE / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv"


def download():
    if TRAIN_TSV.exists():
        return
    print(f"Downloading {DATA_URL}")
    with urllib.request.urlopen(DATA_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW_DIR)


def _read_tsv(path, n_fields):
    # The README says: no quoting, split each line at tabs manually.
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < n_fields:
                continue
            rows.append(parts[:n_fields])
    return rows


def load_raw():
    download()
    train = pd.DataFrame(_read_tsv(TRAIN_TSV, 3), columns=["id", "text", "label"])
    test = pd.DataFrame(_read_tsv(TEST_TSV, 2), columns=["id", "text"])
    gold = {}
    with open(TEST_LABELS, encoding="utf-8-sig") as f:
        for line in f:
            i, l = line.strip().split(",")
            gold[i] = l
    test["label"] = test["id"].map(gold)
    for df in (train, test):
        df["y"] = df["label"].map(LABEL2ID)
    assert train["y"].notna().all() and test["y"].notna().all()
    return train, test


def build_splits():
    train_full, test = load_raw()
    train, val = train_test_split(
        train_full, test_size=VAL_FRACTION, stratify=train_full["y"], random_state=SEED
    )
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df = df.reset_index(drop=True)
        df.to_csv(PROC_DIR / f"{name}.csv", index=False)
        print(f"{name:5s} n={len(df):6d}  OFF={df['y'].mean():.3f}")


def load_split(name):
    path = PROC_DIR / f"{name}.csv"
    if not path.exists():
        build_splits()
    df = pd.read_csv(path, dtype={"id": str, "text": str})
    df["text"] = df["text"].fillna("")
    return df


if __name__ == "__main__":
    build_splits()
