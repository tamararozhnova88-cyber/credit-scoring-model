"""Five MLflow runs + CV hyperparameter search; untouched holdout evaluated once."""
import json
import os
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from src.data.validation import TARGET
from src.models.metrics import classification_metrics
from src.models.pipeline import candidates, create_pipeline


def run_training(train_path="data/processed/train.csv", test_path="data/processed/test.csv", output="models"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
    mlflow.set_experiment("Credit_Default_Prediction")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    experiments = []
    for name, estimator in candidates():
        with mlflow.start_run(run_name=name):
            pipeline = create_pipeline(estimator)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
            mlflow.log_param("algorithm", name)
            mlflow.log_params({"model__" + k: str(v) for k, v in estimator.get_params().items() if k in ["C", "n_estimators", "min_samples_leaf", "class_weight"]})
            mlflow.log_metric("cv_roc_auc_mean", float(scores.mean()))
            mlflow.log_metric("cv_roc_auc_std", float(scores.std()))
            experiments.append((name, float(scores.mean()), estimator))
    best_name, best_cv, best_estimator = max(experiments, key=lambda x: x[1])
    # Hyperparameter tuning is done on TRAIN only, with no test-set peeking.
    if best_name.startswith("logistic"):
        grid = {"classifier__C": [0.05, 0.1, 1.0, 3.0]}
    elif best_name == "random_forest":
        grid = {"classifier__max_depth": [8, 16, None], "classifier__min_samples_leaf": [3, 8]}
    else:
        grid = {"classifier__learning_rate": [0.05, 0.1], "classifier__n_estimators": [80, 120]}
    search = GridSearchCV(create_pipeline(best_estimator), grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    with mlflow.start_run(run_name="best_tuned_final"):
        search.fit(X_train, y_train)
        model = search.best_estimator_
        prob = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, prob)
        mlflow.log_param("selected_experiment", best_name)
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        mlflow.log_metrics({"test_" + k: v for k, v in metrics.items()})
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
        fig.tight_layout()
        fig.savefig("reports/figures/roc_curve.png", dpi=150)
        plt.close(fig)
        mlflow.log_artifact("reports/figures/roc_curve.png")
        mlflow.sklearn.log_model(
    model,
    name="model",
    serialization_format="cloudpickle"
)
        joblib.dump(model, output / "credit_default_model.pkl")
        # A fixed reference distribution for PSI, created from TRAIN only.
        ref = model.predict_proba(X_train)[:, 1]
        pd.DataFrame({"probability": ref}).to_csv(output / "reference_scores.csv", index=False)
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump({**metrics, "cv_roc_auc": float(search.best_score_), "selected_model": best_name}, f, indent=2)
        mlflow.log_artifact("metrics.json")
    print(json.dumps({"model": best_name, "metrics": metrics}, indent=2))
    return model, metrics


if __name__ == "__main__":
    run_training()
