"""
Inference Script
=================
Script untuk melakukan prediksi menggunakan model terbaik.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow


def load_model_from_mlflow(run_id: str = None, model_uri: str = None):
    """Load model dari MLflow."""
    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri)
    elif run_id:
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    else:
        raise ValueError("Harus menyediakan run_id atau model_uri")
    return model


def load_preprocessors(processed_dir: str):
    """Load scaler dan label encoder."""
    scaler = joblib.load(os.path.join(processed_dir, "scaler.joblib"))
    le_target = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))
    with open(os.path.join(processed_dir, "feature_info.json"), "r") as f:
        feature_info = json.load(f)
    return scaler, le_target, feature_info


def preprocess_input(data: dict, scaler, feature_info: dict) -> pd.DataFrame:
    """Preprocess input data untuk prediksi."""
    df = pd.DataFrame([data])

    # Feature engineering
    ips_cols = [col for col in df.columns if col.startswith("ips_")]
    if ips_cols:
        df["ipk_rata_rata"] = df[ips_cols].mean(axis=1).round(2)
        if len(ips_cols) >= 2:
            df["ips_trend"] = (df[ips_cols[-1]] - df[ips_cols[0]]).round(2)
            df["ips_std"] = df[ips_cols].std(axis=1).round(4)
    if "jumlah_sks" in df.columns:
        df["sks_ratio"] = (df["jumlah_sks"] / 144).round(4)

    # Encoding
    from sklearn.preprocessing import OrdinalEncoder
    if "penghasilan_ortu" in df.columns:
        ortu_map = {"<2jt": 0, "2-5jt": 1, "5-10jt": 2, ">10jt": 3}
        df["penghasilan_ortu"] = df["penghasilan_ortu"].map(ortu_map).fillna(-1)

    nominal_cols = ["jenis_kelamin", "asal_sekolah", "organisasi", "beasiswa"]
    for col in nominal_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

    # Align columns with training features
    expected_features = feature_info["feature_names"]
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df), columns=expected_features
    )
    return df_scaled


def predict(data: dict, run_id: str = None, model_uri: str = None):
    """Prediksi kelulusan mahasiswa."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")

    scaler, le_target, feature_info = load_preprocessors(processed_dir)
    X = preprocess_input(data, scaler, feature_info)

    model_name = "loaded_model"

    if model_uri:
        model = load_model_from_mlflow(model_uri=model_uri)
    elif run_id:
        model = load_model_from_mlflow(run_id=run_id)
    else:
        # Load from local artifacts
        artifacts_dir = os.path.join(base_dir, "artifacts")
        metric_path = os.path.join(artifacts_dir, "metric_info_tuned.json")
        if not os.path.exists(metric_path):
            metric_path = os.path.join(artifacts_dir, "metric_info.json")
        with open(metric_path, "r") as f:
            metric_info = json.load(f)
        model_name = metric_info.get("best_model", "unknown")
        best_run_id = metric_info["results"][model_name]["run_id"]
        model = load_model_from_mlflow(run_id=best_run_id)

    prediction = model.predict(X)
    predicted_class = le_target.inverse_transform(prediction.astype(int))
    return {
        "prediction": predicted_class[0],
        "prediction_code": int(prediction[0]),
        "model_used": model_name,
    }


if __name__ == "__main__":
    # Contoh data mahasiswa
    sample_data = {
        "ips_1": 3.5, "ips_2": 3.2, "ips_3": 3.8, "ips_4": 3.6,
        "ips_5": 3.4, "ips_6": 3.7, "ips_7": 3.5,
        "jumlah_sks": 140, "umur": 22,
        "jenis_kelamin": "Laki-laki", "asal_sekolah": "SMA",
        "organisasi": "Ya", "beasiswa": "Tidak",
        "jumlah_tanggungan": 2, "penghasilan_ortu": "5-10jt",
    }
    result = predict(sample_data)
    print(f"\nHasil Prediksi: {result}")
