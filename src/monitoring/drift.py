"""Population Stability Index on fixed reference-score bins."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.data.validation import TARGET


def psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Both populations must be nonempty")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError("Reference scores have no variance")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_hist = np.histogram(current, bins=edges)[0] / len(current)
    ref_hist = np.clip(ref_hist, 1e-6, None)
    cur_hist = np.clip(cur_hist, 1e-6, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def monitor(api_url="http://localhost:8000", sample_size=200):
    reference = pd.read_csv("models/reference_scores.csv")["probability"].to_numpy()
    test = pd.read_csv("data/processed/test.csv").drop(columns=TARGET)
    sample = test.sample(n=min(sample_size, len(test)), random_state=42)
    probabilities = []
    for row in sample.to_dict(orient="records"):
        response = requests.post(f"{api_url.rstrip('/')}/predict", json=row, timeout=10)
        response.raise_for_status()
        probabilities.append(response.json()["default_probability"])
    score = psi(reference, probabilities)
    result = {"psi": score, "sample_size": len(sample), "drift_warning": score >= 0.2}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    monitor(args.api_url, args.sample_size)
