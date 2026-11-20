import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, probability, threshold=0.5):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }
