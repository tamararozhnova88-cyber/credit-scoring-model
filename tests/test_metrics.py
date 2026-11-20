from src.models.metrics import classification_metrics
from src.monitoring.drift import psi


def test_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert all(m[k] == 1.0 for k in ["roc_auc", "precision", "recall", "f1"])


def test_psi_no_drift():
    assert abs(psi(list(range(100)), list(range(100)))) < 1e-8


def test_psi_shift():
    assert psi(list(range(100)), list(range(100, 200))) > 0.2
