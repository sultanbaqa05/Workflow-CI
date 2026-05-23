# Prediksi Kelulusan Mahasiswa - MLOps Pipeline

## 📋 Deskripsi Project

Project end-to-end MLOps untuk memprediksi status kelulusan mahasiswa (Lulus Tepat Waktu / Lulus Terlambat / Drop Out) menggunakan dataset tabular dan Scikit-Learn.

**Submission Dicoding: Membangun Sistem Machine Learning**

## 🏗️ Arsitektur

```
Dataset CSV → Preprocessing → Modelling (MLflow) → DagsHub → CI/CD → Docker → Serving → Monitoring
```

## 📁 Struktur Folder

```
mml/
├── .github/workflows/         # GitHub Actions CI/CD
│   ├── preprocessing.yml      # Workflow preprocessing (Kriteria 1 Advanced)
│   └── ci-cd.yml              # Workflow CI/CD (Kriteria 3 Advanced)
├── data/
│   ├── raw/                   # Dataset mentah
│   │   ├── student_data.csv
│   │   └── generate_dataset.py
│   └── processed/             # Data hasil preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.joblib
│       ├── label_encoder.joblib
│       └── feature_info.json
├── notebook/
│   ├── Eksperimen_Sultan.ipynb # Notebook eksperimen (Kriteria 1)
│   └── eksperimen.py          # Versi .py dari notebook
├── src/
│   ├── automate_preprocessing.py  # Preprocessing otomatis (Kriteria 1 Skilled)
│   ├── modelling.py               # MLflow autolog (Kriteria 2 Basic)
│   ├── modelling_tuning.py        # Manual logging + tuning (Kriteria 2 Skilled/Adv)
│   └── inference.py               # Inference script (Kriteria 4)
├── serving/
│   ├── app.py                 # FastAPI serving + Prometheus metrics
│   └── prometheus_exporter.py # Custom Prometheus exporter
├── monitoring/
│   ├── prometheus.yml         # Konfigurasi Prometheus
│   └── grafana/
│       ├── dashboard.json     # Dashboard Grafana (15 panel)
│       ├── alerting.json      # Alerting rules (3 rules)
│       └── provisioning/
│           ├── datasources/datasource.yml
│           └── dashboards/dashboard.yml
├── MLProject/                 # MLflow Project (Kriteria 3)
│   ├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   └── student_data_preprocessing/
├── artifacts/                 # Artefak model
│   ├── confusion_matrix_*.png
│   ├── classification_report_*.txt
│   ├── feature_importance_*.png
│   ├── metric_info.json
│   └── metric_info_tuned.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── DagsHub.txt
├── create_submission_package.py
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Generate Dataset (jika belum ada)
```bash
python data/raw/generate_dataset.py
```

### 3. Preprocessing
```bash
python src/automate_preprocessing.py
```

### 4. Jalankan MLflow UI (Terminal 1)
```bash
mlflow ui --port 5000
# Buka http://127.0.0.1:5000
```

### 5. Training - Autolog (Terminal 2)
```bash
python src/modelling.py
```

### 6. Training dengan Tuning - Manual Logging
```bash
python src/modelling_tuning.py
```

### 7. Run via MLProject
```bash
mlflow run MLProject --env-manager=local
```

### 8. Model Serving
```bash
python serving/app.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Metrics: http://localhost:8000/metrics
```

### 9. Prometheus Exporter (Terminal terpisah)
```bash
python serving/prometheus_exporter.py
# Metrics: http://localhost:8001
```

### 10. Monitoring Stack (Docker)
```bash
docker-compose up -d
# Grafana: http://localhost:3000 (admin/admin123)
# Prometheus: http://localhost:9090
```

### 11. Test Inference
```bash
python src/inference.py
```

### 12. Buat Submission Package
```bash
python create_submission_package.py
```

## 🔑 GitHub Secrets yang Diperlukan

| Secret | Keterangan |
|--------|-----------|
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_PASSWORD` | Password/Token Docker Hub |
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_REPO` | Nama repository DagsHub |
| `DAGSHUB_TOKEN` | Access token DagsHub |

## 📊 Model yang Digunakan

- **Random Forest Classifier** - Ensemble tree-based model
- **Logistic Regression** - Linear classification model
- **XGBoost Classifier** - Gradient boosting model

Model terbaik dipilih berdasarkan **F1-Score (weighted)**.

## 📈 Monitoring Metrics (15 Metrics)

### FastAPI App Metrics:
1. `model_request_total` - Total request per endpoint
2. `http_requests_total` - Total HTTP requests
3. `model_request_latency_seconds` - Request latency (histogram)
4. `model_prediction_total` - Prediksi per kelas
5. `model_prediction_latency_seconds` - Inference latency (summary)
6. `model_prediction_errors_total` - Error count per type
7. `model_active_requests` - Active concurrent requests
8. `model_input_ipk_distribution` - Distribusi IPK input
9. `model_prediction_confidence` - Confidence distribution
10. `model_load_time_seconds` - Model load time
11. `model_health_check_total` - Health check count
12. `model_total_requests_served` - Total requests served

### Custom Exporter Metrics:
13. `system_cpu_usage_percent` - CPU usage
14. `system_memory_usage_percent` - Memory usage
15. `system_disk_usage_percent` - Disk usage
16. `system_memory_used_mb` - Memory used in MB
17. `model_best_f1_score` - Best model F1 score
18. `model_best_accuracy` - Best model accuracy

## 🚨 Grafana Alerting (3 Rules)

1. **High Prediction Error Rate** - Alert jika error rate > threshold (severity: critical)
2. **High Prediction Latency** - Alert jika p95 latency > 2s (severity: warning)
3. **Model F1 Score Drop** - Alert jika F1 < 0.7 (severity: critical)

## 🔗 Contoh Query Prometheus

```promql
# Total requests per detik
sum(rate(model_request_total[5m]))

# P95 latency
histogram_quantile(0.95, sum(rate(model_request_latency_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(model_prediction_errors_total[5m])) / sum(rate(model_request_total[5m]))

# Prediksi per kelas
sum by (predicted_class) (model_prediction_total)

# CPU usage
system_cpu_usage_percent

# Memory usage
system_memory_usage_percent

# Model F1 Score
model_best_f1_score

# Disk usage
system_disk_usage_percent

# Active requests
model_active_requests

# Prediction confidence p95
histogram_quantile(0.95, sum(rate(model_prediction_confidence_bucket[5m])) by (le))
```

## 📸 Screenshot Checklist untuk Reviewer

### Kriteria 1 (Eksperimen):
- [ ] Repository GitHub `Eksperimen_SML_Sultan` dengan struktur folder benar
- [ ] Notebook `.ipynb` dengan output EDA & preprocessing
- [ ] Workflow preprocessing GitHub Actions minimal 1x berhasil (green check)

### Kriteria 2 (Membangun Model):
- [ ] MLflow Tracking UI di http://127.0.0.1:5000 (screenshoot_dashboard.jpg)
- [ ] MLflow Artifacts: confusion_matrix, classification_report, feature_importance, metric_info.json (screenshoot_artifak.jpg)
- [ ] DagsHub experiment tracking

### Kriteria 3 (Workflow CI):
- [ ] Repository GitHub `Workflow-CI` dengan struktur folder benar
- [ ] MLProject folder lengkap
- [ ] GitHub Actions workflow sukses (green check)
- [ ] Docker Hub image berhasil di-push

### Kriteria 4 (Monitoring):
- [ ] FastAPI serving running (screenshot /docs atau curl)
- [ ] Prometheus targets UP
- [ ] Grafana dashboard dengan 10+ panel (nama: "sultan")
- [ ] 3 alerting rules di Grafana
- [ ] 3 notifikasi alerting

## ⚠️ Potensi Penolakan & Solusi

| Masalah | Solusi |
|---------|--------|
| Dashboard Grafana tanpa username Dicoding | Pastikan title dashboard mengandung username |
| Workflow GH Actions gagal | Test lokal dulu, pastikan semua secrets sudah diset |
| Docker build error | Pastikan mlruns/ dan artifacts/ ada sebelum build |
| MLflow tracking error | Jalankan `mlflow ui --port 5000` dulu sebelum training |
| Missing artifacts | Jalankan modelling.py lalu modelling_tuning.py berurutan |
| Notebook tanpa output | Run All cells di notebook sebelum commit |
| Import error | Pastikan virtual env aktif dan semua package terinstall |
| Repo private | Pastikan kedua repo GitHub visibility **Public** |
