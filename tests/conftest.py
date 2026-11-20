import pytest


@pytest.fixture
def raw_row():
    row = {"LIMIT_BAL": 100000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35, "default.payment.next.month": 0}
    row.update({p: 0 for p in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]})
    row.update({f"BILL_AMT{i}": 1000.0 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 100.0 for i in range(1, 7)})
    return row
