import argparse
import pandas as pd
from src.data.validation import validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="data/raw/UCI_Credit_Card.csv")
    args = parser.parse_args()
    validated = validate(pd.read_csv(args.file))
    print(f"Validation passed: {len(validated)} rows")
