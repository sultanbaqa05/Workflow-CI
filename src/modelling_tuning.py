"""
Modelling Tuning Script - MLflow Manual Logging
=================================================
Training model dengan hyperparameter tuning + manual MLflow logging.
Menggunakan GridSearchCV/RandomizedSearchCV.
"""

import os
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
from sklearn.model_selection import RandomizedSearchCV
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


def get_param_grids():
    return {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, multi_class="multinomial", max_iter=2000),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "newton-cg"],
                "penalty": ["l2"],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            },
        },
    }


def save_artifacts(y_test, y_pred, model, feature_names, class_names, model_name, artifacts_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Tuned) - {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    
    # Save the model-specific named file for directory submission
    cm_path = os.path.join(artifacts_dir, f"confusion_matrix_tuned_{model_name}.png")
    plt.savefig(cm_path, dpi=150)
    
    # Save the training_confusion_matrix.png for MLflow artifact UI matching (image 2)
    temp_cm_path = os.path.join(artifacts_dir, "training_confusion_matrix.png")
    plt.savefig(temp_cm_path, dpi=150); plt.close()
    mlflow.log_artifact(temp_cm_path)
    mlflow.log_artifact(cm_path)

    report = classification_report(y_test, y_pred, target_names=class_names)
    rpt_path = os.path.join(artifacts_dir, f"classification_report_tuned_{model_name}.txt")
    with open(rpt_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(rpt_path)

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0)
    else:
        return
    idx = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(idx)), imp[idx], color="forestgreen")
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance (Tuned) - {model_name}")
    plt.tight_layout()
    fi_path = os.path.join(artifacts_dir, f"feature_importance_tuned_{model_name}.png")
    plt.savefig(fi_path, dpi=150); plt.close()
    mlflow.log_artifact(fi_path)


def train_models_with_tuning():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, le_target = load_processed_data(data_dir)
    class_names = le_target.classes_ if le_target else [str(i) for i in sorted(np.unique(y_train))]
    feature_names = X_train.columns.tolist()

    param_grids = get_param_grids()
    experiment_name = "Prediksi_Kelulusan_Mahasiswa_Tuning"
    if mlflow.active_run() is None:
        mlflow.set_experiment(experiment_name)

    best_model_name = None
    best_f1 = 0
    results = {}

    for name, config in param_grids.items():
        print(f"\n{'='*50}\nTuning: {name} (Manual Logging)\n{'='*50}")

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=20,
            cv=5,
            scoring="f1_weighted",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # MANUAL LOGGING
        with mlflow.start_run(run_name=f"{name}_tuned", nested=True) as run:
            # Log parameters manually
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("n_iter", 20)
            mlflow.log_param("scoring", "f1_weighted")
            for param_name, param_val in search.best_params_.items():
                mlflow.log_param(f"best_{param_name}", param_val)

            # Log metrics manually
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_weighted", prec)
            mlflow.log_metric("recall_weighted", rec)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric("best_cv_score", search.best_score_)

            # Log model manually
            if name == "XGBoost":
                mlflow.xgboost.log_model(best_estimator, artifact_path="model")
            else:
                mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            # Log tags
            mlflow.set_tag("tuning_method", "RandomizedSearchCV")
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("dataset", "student_graduation")

            # Save and log metric_info.json for matching image 2
            run_metric_info = {
                "model_name": name,
                "accuracy": acc,
                "precision_weighted": prec,
                "recall_weighted": rec,
                "f1_weighted": f1,
                "best_cv_score": search.best_score_,
                "best_params": {k: str(v) for k, v in search.best_params_.items()}
            }
            run_metric_path = os.path.join(artifacts_dir, "metric_info.json")
            with open(run_metric_path, "w") as f:
                json.dump(run_metric_info, f, indent=2)
            mlflow.log_artifact(run_metric_path)

            # Save & log artifacts
            save_artifacts(y_test, y_pred, best_estimator, feature_names,
                         class_names, name, artifacts_dir)

            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV score: {search.best_score_:.4f}")
            print(f"  Test - Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

            results[name] = {
                "accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "f1_score": round(f1, 4),
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "best_cv_score": round(search.best_score_, 4),
                "run_id": run.info.run_id,
            }

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name

    metric_info = {
        "experiment": experiment_name,
        "tuning_method": "RandomizedSearchCV",
        "best_model": best_model_name,
        "best_f1_score": best_f1,
        "results": results,
    }
    with open(os.path.join(artifacts_dir, "metric_info_tuned.json"), "w") as f:
        json.dump(metric_info, f, indent=2)

    print(f"\nBEST TUNED MODEL: {best_model_name} (F1={best_f1:.4f})")
    return metric_info


if __name__ == "__main__":
    train_models_with_tuning()
