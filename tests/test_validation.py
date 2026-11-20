import pandas as pd
import pandera.errors
import pytest
from src.data.validation import validate


def test_valid_data(raw_row):
    assert len(validate(pd.DataFrame([raw_row]))) == 1


@pytest.mark.parametrize("column,value", [("AGE", 12), ("LIMIT_BAL", -100), ("PAY_0", 50), ("SEX", 5), ("default.payment.next.month", 3), ("PAY_AMT1", -1), ("AGE", None)])
def test_anomalies_fail(raw_row, column, value):
    raw_row[column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(pd.DataFrame([raw_row]))
