"""Data contract using Pandera. Validates raw and prepared datasets."""

import pandas as pd
import pandera.pandas as pa
from pandera import Check

TARGET = "default.payment.next.month"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]


def schema():
    cols = {
        "LIMIT_BAL": pa.Column(float, Check.gt(0), nullable=False, coerce=True),
        "AGE": pa.Column(int, Check.in_range(18, 100), nullable=False, coerce=True),
        "SEX": pa.Column(int, Check.isin([1, 2]), nullable=False, coerce=True),
        "EDUCATION": pa.Column(int, Check.in_range(0, 6), nullable=False, coerce=True),
        "MARRIAGE": pa.Column(int, Check.in_range(0, 3), nullable=False, coerce=True),
        TARGET: pa.Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
    }
    cols.update(
        {p: pa.Column(int, Check.in_range(-2, 9), nullable=False, coerce=True) for p in PAY}
    )
    cols.update({p: pa.Column(float, nullable=False, coerce=True) for p in BILL})
    cols.update({p: pa.Column(float, Check.ge(0), nullable=False, coerce=True) for p in PAID})
    return pa.DataFrameSchema(cols, strict=False)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    return schema().validate(df, lazy=True)
