"""
Generate synthetic student graduation prediction dataset.
Dataset: Prediksi Kelulusan Mahasiswa

Target classes:
  - Lulus Tepat Waktu
  - Lulus Terlambat
  - Drop Out
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    "ips_1": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_2": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_3": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_4": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_5": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_6": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "ips_7": np.round(np.random.uniform(1.0, 4.0, n_samples), 2),
    "jumlah_sks": np.random.randint(60, 160, n_samples),
    "umur": np.random.randint(18, 30, n_samples),
    "jenis_kelamin": np.random.choice(["Laki-laki", "Perempuan"], n_samples),
    "asal_sekolah": np.random.choice(["SMA", "SMK", "MA"], n_samples),
    "organisasi": np.random.choice(["Ya", "Tidak"], n_samples),
    "beasiswa": np.random.choice(["Ya", "Tidak"], n_samples),
    "jumlah_tanggungan": np.random.randint(0, 6, n_samples),
    "penghasilan_ortu": np.random.choice(
        ["<2jt", "2-5jt", "5-10jt", ">10jt"], n_samples
    ),
}

df = pd.DataFrame(data)

# Add some missing values realistically
missing_indices = np.random.choice(n_samples, size=50, replace=False)
df.loc[missing_indices[:20], "ips_3"] = np.nan
df.loc[missing_indices[20:35], "umur"] = np.nan
df.loc[missing_indices[35:], "penghasilan_ortu"] = np.nan

# Generate target based on composite score
ipk = (
    df["ips_1"].fillna(2.5)
    + df["ips_2"].fillna(2.5)
    + df["ips_3"].fillna(2.5)
    + df["ips_4"].fillna(2.5)
    + df["ips_5"].fillna(2.5)
    + df["ips_6"].fillna(2.5)
    + df["ips_7"].fillna(2.5)
) / 7

# Normalized score with balanced class distribution
score = (
    (ipk - 1.0) / 3.0 * 0.45              # IPK contribution (normalized 1-4 -> 0-1)
    + (df["jumlah_sks"].fillna(110) - 60) / 100 * 0.25  # SKS contribution
    + np.where(df["beasiswa"] == "Ya", 0.10, 0)
    + np.where(df["organisasi"] == "Ya", 0.05, 0)
    - (df["jumlah_tanggungan"] / 10) * 0.05
    + np.random.normal(0, 0.12, n_samples)  # noise
)

conditions = [
    score >= 0.55,
    (score >= 0.30) & (score < 0.55),
    score < 0.30,
]
choices = ["Lulus Tepat Waktu", "Lulus Terlambat", "Drop Out"]
df["status_kelulusan"] = np.select(conditions, choices, default="Lulus Terlambat")

df.to_csv("student_data.csv", index=False)
print(f"Dataset generated: {df.shape}")
print(f"\nDistribusi target:\n{df['status_kelulusan'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSample data:\n{df.head()}")
