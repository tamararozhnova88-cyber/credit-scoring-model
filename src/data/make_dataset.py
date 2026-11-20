"""Prepare deterministic stratified split; holdout never used for model selection."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.validation import validate, TARGET


def prepare(source, destination):
    df = pd.read_csv(source)
    validate(df)
    df = df.drop(columns=["ID"], errors="ignore")
    if df.duplicated().any():
        df = df.drop_duplicates().reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    train.to_csv(dest / "train.csv", index=False)
    test.to_csv(dest / "test.csv", index=False)
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)
