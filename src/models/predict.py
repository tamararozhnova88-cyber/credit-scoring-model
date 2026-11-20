import joblib
import pandas as pd


def predict(model_path, records):
    model = joblib.load(model_path)
    probs = model.predict_proba(pd.DataFrame(records))[:, 1]
    return [{"default_prediction": int(p >= 0.5), "default_probability": float(p)} for p in probs]
