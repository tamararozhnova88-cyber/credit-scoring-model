"""Deterministic feature engineering, reused at train and inference."""
import pandas as pd
from src.data.validation import PAY, BILL, PAID


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["avg_delay"] = out[PAY].mean(axis=1)
    out["max_delay"] = out[PAY].max(axis=1)
    out["late_months"] = (out[PAY] > 0).sum(axis=1)
    out["avg_bill"] = out[BILL].mean(axis=1)
    out["total_paid"] = out[PAID].sum(axis=1)
    out["payment_to_bill"] = out["total_paid"] / (out[BILL].clip(lower=0).sum(axis=1) + 1)
    out["age_group"] = pd.cut(out["AGE"], bins=[17, 25, 35, 45, 60, 101], labels=["18-25", "26-35", "36-45", "46-60", "61+"]).astype(str)
    return out
