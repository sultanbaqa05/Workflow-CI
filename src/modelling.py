"""
Modelling Script - MLflow Autolog
==================================
Training model dengan MLflow autolog.
Model: Random Forest, Logistic Regression, XGBoost
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore")

# DagsHub Integration
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME", "baqa27")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO", "prediksi-kelulusan-mahasiswa")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "")

if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print(f"[INFO] MLflow tracking: DagsHub ({DAGSHUB_USERNAME}/{DAGSHUB_REPO})")
else:
    # Try localhost first to comply with reviewer guidelines (image1)
    try:
        import urllib.request
        # Check if local MLflow server is active
        urllib.request.urlopen("http://127.0.0.1:5000", timeout=1)
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        print("[INFO] MLflow tracking: local server (http://127.0.0.1:5000/)")
    except Exception:
        mlflow.set_tracking_uri("mlruns")
        print("[INFO] MLflow tracking: local directory (mlruns/)")


def load_processed_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["target"].values
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["target"].values
    le_path = os.path.join(data_dir, "label_encoder.joblib")
    le_target = joblib.load(le_path) if os.path.exists(le_path) else None
    return X_train, X_test, y_train, y_test, le_target


def save_artifacts(y_test, y_pred, model, feature_names, class_names, model_name, artifacts_dir):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    mlflow.log_artifact(cm_path)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    rpt_path = os.path.join(artifacts_dir, f"classification_report_{model_name}.txt")
    with open(rpt_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(rpt_path)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0)
    else:
        return
    idx = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(idx)), imp[idx], color="steelblue")
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    fi_path = os.path.join(artifacts_dir, f"feature_importance_{model_name}.png")
    plt.savefig(fi_path, dpi=150); plt.close()
    mlflow.log_artifact(fi_path)


def train_models_autolog():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, le_target = load_processed_data(data_dir)
    class_names = le_target.classes_ if le_target else [str(i) for i in sorted(np.unique(y_train))]
    feature_names = X_train.columns.tolist()

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    }

    experiment_name = "Prediksi_Kelulusan_Mahasiswa_Autolog"
    if mlflow.active_run() is None:
        mlflow.set_experiment(experiment_name)
    best_model_name = None
    best_f1 = 0
    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}\nTraining: {name} (Autolog)\n{'='*50}")
        if name == "XGBoost":
            mlflow.xgboost.autolog(log_models=True)
        else:
            mlflow.sklearn.autolog(log_models=True)

        with mlflow.start_run(run_name=f"{name}_autolog", nested=True) as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
            save_artifacts(y_test, y_pred, model, feature_names, class_names, name, artifacts_dir)
            results[name] = {"accuracy": round(acc,4), "precision": round(prec,4),
                             "recall": round(rec,4), "f1_score": round(f1,4), "run_id": run.info.run_id}
            if f1 > best_f1:
                best_f1 = f1; best_model_name = name

        mlflow.sklearn.autolog(disable=True)
        mlflow.xgboost.autolog(disable=True)

    metric_info = {"experiment": experiment_name, "best_model": best_model_name,
                   "best_f1_score": best_f1, "results": results}
    with open(os.path.join(artifacts_dir, "metric_info.json"), "w") as f:
        json.dump(metric_info, f, indent=2)

    print(f"\nBEST MODEL: {best_model_name} (F1={best_f1:.4f})")
    return metric_info


if __name__ == "__main__":
    train_models_autolog()
