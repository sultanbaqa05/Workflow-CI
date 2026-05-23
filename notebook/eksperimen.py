# %% [markdown]
# # Eksperimen: Prediksi Kelulusan Mahasiswa
# 
# Notebook ini berisi eksperimen lengkap untuk prediksi kelulusan mahasiswa:
# 1. Data Loading
# 2. Exploratory Data Analysis (EDA)
# 3. Preprocessing
# 4. Feature Engineering
# 5. Modelling & Evaluasi

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

print("Libraries imported successfully!")

# %% [markdown]
# ## 2. Data Loading

# %%
df = pd.read_csv("../data/raw/student_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 3.1 Distribusi Target

# %%
plt.figure(figsize=(8, 5))
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
df["status_kelulusan"].value_counts().plot(kind="bar", color=colors, edgecolor="black")
plt.title("Distribusi Status Kelulusan", fontsize=14, fontweight="bold")
plt.xlabel("Status Kelulusan")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../artifacts/eda_target_distribution.png", dpi=150)
plt.show()
print(df["status_kelulusan"].value_counts())

# %% [markdown]
# ### 3.2 Missing Values

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
print("Missing Values:")
print(missing_df)

plt.figure(figsize=(8, 4))
if len(missing_df) > 0:
    missing_df["Missing"].plot(kind="barh", color="#e74c3c")
    plt.title("Missing Values per Column", fontsize=14, fontweight="bold")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("../artifacts/eda_missing_values.png", dpi=150)
    plt.show()

# %% [markdown]
# ### 3.3 Distribusi Fitur Numerik

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):
        df[col].hist(bins=20, ax=axes[i], color="#3498db", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribusi Fitur Numerik", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../artifacts/eda_numeric_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3.4 Korelasi Fitur Numerik

# %%
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../artifacts/eda_correlation_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3.5 Distribusi Fitur Kategorikal

# %%
cat_cols = df.select_dtypes(include=["object"]).columns.drop("status_kelulusan").tolist()
fig, axes = plt.subplots(1, len(cat_cols), figsize=(4 * len(cat_cols), 4))
if len(cat_cols) == 1:
    axes = [axes]
for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind="bar", ax=axes[i], color="#9b59b6", edgecolor="black")
    axes[i].set_title(col, fontsize=11)
    axes[i].tick_params(axis="x", rotation=45)
plt.suptitle("Distribusi Fitur Kategorikal", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../artifacts/eda_categorical_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3.6 IPS per Semester vs Kelulusan

# %%
ips_cols = [col for col in df.columns if col.startswith("ips_")]
fig, axes = plt.subplots(1, len(ips_cols), figsize=(3 * len(ips_cols), 4))
for i, col in enumerate(ips_cols):
    df.boxplot(column=col, by="status_kelulusan", ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)
plt.suptitle("IPS vs Status Kelulusan", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../artifacts/eda_ips_vs_target.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Preprocessing

# %% [markdown]
# ### 4.1 Handle Missing Values

# %%
print("Sebelum handling:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Numeric: median, Categorical: mode
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nSesudah handling:")
print(f"Total missing: {df.isnull().sum().sum()}")

# %% [markdown]
# ### 4.2 Feature Engineering

# %%
df["ipk_rata_rata"] = df[ips_cols].mean(axis=1).round(2)
df["ips_trend"] = (df["ips_7"] - df["ips_1"]).round(2)
df["ips_std"] = df[ips_cols].std(axis=1).round(4)
df["sks_ratio"] = (df["jumlah_sks"] / 144).round(4)

print("Fitur baru yang dibuat:")
print("- ipk_rata_rata: rata-rata IPS semester 1-7")
print("- ips_trend: selisih IPS terakhir dan pertama")
print("- ips_std: standar deviasi IPS (konsistensi)")
print("- sks_ratio: rasio SKS terhadap target 144")
print(f"\nTotal fitur sekarang: {df.shape[1]}")

# %% [markdown]
# ### 4.3 Encoding

# %%
# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(df["status_kelulusan"])
X = df.drop(columns=["status_kelulusan"])
print(f"Target classes: {list(le_target.classes_)}")

# Ordinal: penghasilan_ortu
ortu_order = [["<2jt", "2-5jt", "5-10jt", ">10jt"]]
oe = OrdinalEncoder(categories=ortu_order, handle_unknown="use_encoded_value", unknown_value=-1)
X["penghasilan_ortu"] = oe.fit_transform(X[["penghasilan_ortu"]])

# One-hot: nominal columns
nominal_cols = ["jenis_kelamin", "asal_sekolah", "organisasi", "beasiswa"]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True, dtype=int)
print(f"Features after encoding: {X.shape[1]}")
print(X.columns.tolist())

# %% [markdown]
# ### 4.4 Split Data

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ### 4.5 Scaling

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
print("Scaling completed with StandardScaler")

# %% [markdown]
# ## 5. Modelling & Evaluasi

# %%
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# %% [markdown]
# ### 5.1 Perbandingan Model

# %%
results_df = pd.DataFrame(results).T
print(results_df.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
plt.title("Perbandingan Performa Model", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../artifacts/model_comparison.png", dpi=150)
plt.show()

best_model = results_df["F1"].idxmax()
print(f"\nModel Terbaik: {best_model} (F1={results_df.loc[best_model, 'F1']:.4f})")

# %% [markdown]
# ### 5.2 Confusion Matrix Model Terbaik

# %%
best_clf = models[best_model]
y_pred_best = best_clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title(f"Confusion Matrix - {best_model}", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../artifacts/confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Kesimpulan
# 
# - Dataset berisi 1000 data mahasiswa dengan 15 fitur asal + 4 fitur engineering
# - Missing values ditangani dengan median (numerik) dan mode (kategorikal)
# - Feature engineering: IPK rata-rata, IPS trend, IPS std, SKS ratio
# - Model terbaik dipilih berdasarkan F1-Score weighted
# - Semua artefak disimpan untuk tracking MLflow
