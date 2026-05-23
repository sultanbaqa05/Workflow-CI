"""
Automate Preprocessing Pipeline
================================
Script untuk preprocessing otomatis dataset Prediksi Kelulusan Mahasiswa.
Menjalankan seluruh tahap preprocessing:
1. Load data
2. Handle missing values
3. Feature engineering
4. Encoding
5. Scaling
6. Split data
7. Save processed data

Author: MLOps Submission Dicoding - Membangun Sistem Machine Learning
"""

import os
import sys
import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
import joblib

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari CSV."""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values dengan strategi yang tepat."""
    logger.info("Handling missing values...")
    missing_before = df.isnull().sum().sum()
    logger.info(f"Total missing values sebelum: {missing_before}")

    # Numeric columns: isi dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled with median = {median_val}")

    # Categorical columns: isi dengan mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: filled with mode = {mode_val}")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Total missing values sesudah: {missing_after}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur baru dari fitur yang ada."""
    logger.info("Feature engineering...")

    # IPK rata-rata
    ips_cols = [col for col in df.columns if col.startswith("ips_")]
    if ips_cols:
        df["ipk_rata_rata"] = df[ips_cols].mean(axis=1).round(2)
        logger.info(f"  Created ipk_rata_rata from {ips_cols}")

    # Tren IPS (apakah naik atau turun)
    if len(ips_cols) >= 2:
        df["ips_trend"] = (df[ips_cols[-1]] - df[ips_cols[0]]).round(2)
        logger.info("  Created ips_trend (IPS terakhir - IPS pertama)")

    # IPS standar deviasi (konsistensi)
    if len(ips_cols) >= 2:
        df["ips_std"] = df[ips_cols].std(axis=1).round(4)
        logger.info("  Created ips_std (standar deviasi IPS)")

    # Rasio SKS terhadap target 144
    if "jumlah_sks" in df.columns:
        df["sks_ratio"] = (df["jumlah_sks"] / 144).round(4)
        logger.info("  Created sks_ratio (jumlah_sks / 144)")

    logger.info(f"  Total features after engineering: {df.shape[1]}")
    return df


def encode_features(
    df: pd.DataFrame, target_col: str = "status_kelulusan"
) -> tuple:
    """Encode categorical features."""
    logger.info("Encoding categorical features...")

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Label encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    logger.info(f"  Target classes: {list(le_target.classes_)}")
    logger.info(f"  Target mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Identify categorical columns in X
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"  Categorical columns to encode: {cat_cols}")

    # Ordinal encoding for ordered categories
    if "penghasilan_ortu" in X.columns:
        ortu_order = [["<2jt", "2-5jt", "5-10jt", ">10jt"]]
        oe = OrdinalEncoder(categories=ortu_order, handle_unknown="use_encoded_value", unknown_value=-1)
        X["penghasilan_ortu"] = oe.fit_transform(X[["penghasilan_ortu"]])
        logger.info(f"  Ordinal encoded: penghasilan_ortu -> {ortu_order[0]}")

    # One-hot encode nominal columns
    nominal_cols = [col for col in cat_cols if col != "penghasilan_ortu"]
    if nominal_cols:
        X = pd.get_dummies(X, columns=nominal_cols, drop_first=True, dtype=int)
        logger.info(f"  One-hot encoded: {nominal_cols}")

    logger.info(f"  Features after encoding: {X.shape[1]}")
    return X, y_encoded, le_target


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scaling fitur numerik menggunakan StandardScaler."""
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()

    feature_names = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    logger.info(f"  Scaler fitted on {len(feature_names)} features")
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data menjadi train dan test set."""
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"  Train target distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"  Test target distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train, X_test, y_train, y_test, scaler, le_target, output_dir: str
) -> None:
    """Simpan data yang sudah diproses."""
    logger.info(f"Saving processed data to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save CSVs
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["target"]).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False
    )

    # Save scaler & encoder
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(le_target, os.path.join(output_dir, "label_encoder.joblib"))

    # Save feature names
    feature_info = {
        "feature_names": X_train.columns.tolist(),
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
        "target_classes": le_target.classes_.tolist(),
    }
    with open(os.path.join(output_dir, "feature_info.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    logger.info("  All processed data saved successfully!")
    logger.info(f"  Files: {os.listdir(output_dir)}")


def main():
    """Main preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("AUTOMATE PREPROCESSING PIPELINE")
    logger.info("Dataset: Prediksi Kelulusan Mahasiswa")
    logger.info("=" * 60)

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, "data", "raw", "student_data.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")

    # Step 1: Load data
    df = load_data(raw_data_path)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Feature engineering
    df = feature_engineering(df)

    # Step 4: Encode features
    X, y, le_target = encode_features(df, target_col="status_kelulusan")

    # Step 5: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 6: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 7: Save processed data
    save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_target, processed_dir
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
