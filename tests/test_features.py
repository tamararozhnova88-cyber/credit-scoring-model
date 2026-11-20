import pandas as pd
from src.features.build_features import add_features


def test_features(raw_row):
    result = add_features(pd.DataFrame([raw_row]))
    assert result.loc[0, "total_paid"] == 600
    assert result.loc[0, "late_months"] == 0
    assert result.loc[0, "age_group"] == "26-35"
