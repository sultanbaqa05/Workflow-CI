# 🏆 PANDUAN LENGKAP MENJALANKAN PIPELINE & PENGAMBILAN BUKTI SCREENSHOT BINTANG 5
## Proyek: Prediksi Kelulusan Mahasiswa - Membangun Sistem Machine Learning (Dicoding)
**Nama Siswa: Sultan**

Panduan ini berisi instruksi lengkap untuk **menjalankan seluruh pipeline** secara runut, melakukan verifikasi fungsionalitas, serta langkah demi langkah mengambil screenshot bukti untuk melengkapi folder submission **`SMSML_Sultan`** agar mendapatkan nilai **Advanced (Bintang 5)**.

---

## 🛠️ PRASYARAT UTAMA (PENTING!)

Agar semua command berjalan lancar tanpa error seperti `ModuleNotFoundError` atau encoding emoji di Windows:
1. **Gunakan Terminal PowerShell** pada direktori root project `D:\dicoding\mml`.
2. **Aktifkan Virtual Environment** terlebih dahulu.
3. **Paksa Mode UTF-8** pada Python di Windows agar tidak terjadi error representasi emoji MLflow.

Setiap kali Anda membuka terminal baru, **jalankan 3 baris perintah ini terlebih dahulu**:
```powershell
cd D:\dicoding\mml
.\venv\Scripts\activate
$env:PYTHONUTF8=1
```

---

## 🚀 KRONOLOGI EKSEKUSI PIPELINE (RUN ALL)

Ikuti urutan eksekusi berikut dari awal hingga akhir untuk memastikan seluruh siklus MLOps Anda terbentuk dengan sempurna:

### Langkah 1: Jalankan Preprocessing Otomatis (Kriteria 1)
Mengubah data mentah menjadi data preprocessed yang siap dilatih:
```powershell
d:\dicoding\mml\venv\Scripts\python.exe src\automate_preprocessing.py
```
*   **Verifikasi**: Harus muncul logs `PREPROCESSING COMPLETED SUCCESSFULLY!` di akhir.
*   **Output**: Dataset preprocessed tersimpan lengkap di `data/processed/`.

### Langkah 2: Jalankan MLflow Tracking UI (Terminal 1)
Aktifkan server pelacak lokal yang akan menampung seluruh metadata dan parameter:
```powershell
d:\dicoding\mml\venv\Scripts\mlflow.exe ui --port 5000
```
*   **Verifikasi**: Biarkan terminal ini terbuka. Buka browser Anda ke http://127.0.0.1:5000 untuk memastikan UI berjalan.

### Langkah 3: Jalankan Baseline Model Training (Terminal 2 - Kriteria 2 Basic)
Melatih model Random Forest, Logistic Regression, dan XGBoost menggunakan MLflow Autolog:
```powershell
d:\dicoding\mml\venv\Scripts\python.exe src\modelling.py
```
*   **Verifikasi**: Harus menampilkan metrics F1-Score untuk ketiga model dan menyimpan model di bawah run name `[ModelName]_autolog`.

### Langkah 4: Jalankan Hyperparameter Tuning & Manual Logging (Terminal 2 - Kriteria 2 Skilled/Advanced)
Melakukan `RandomizedSearchCV` dengan logging manual parameter serta penyimpanan plot visualisasi (confusion matrix, feature importance):
```powershell
d:\dicoding\mml\venv\Scripts\python.exe src\modelling_tuning.py
```
*   **Verifikasi**: Harus menampilkan `BEST TUNED MODEL: LogisticRegression (F1=0.5106)`.
*   **Output**: File plot `.png` dan metadata `.json` tersimpan di folder `artifacts/`.

### Langkah 5: Uji MLProject Run (Kriteria 3)
Memastikan MLflow Project dapat mereproduksi training secara mandiri:
```powershell
d:\dicoding\mml\venv\Scripts\mlflow.exe run MLProject --env-manager=local
```
*   **Verifikasi**: Menjalankan re-training otomatis tanpa ada error conda/local.

### Langkah 6: Jalankan Model Serving & Prometheus Exporter (Kriteria 4)
Menyajikan model terbaik sebagai REST API serta mengekspos metrik sistem/model untuk Prometheus:
```powershell
# Jalankan FastAPI Serving (akan berjalan di port 8000)
d:\dicoding\mml\venv\Scripts\python.exe serving\app.py

# Buka terminal baru, aktifkan venv + UTF8, lalu jalankan Prometheus Exporter (port 8001)
d:\dicoding\mml\venv\Scripts\python.exe serving\prometheus_exporter.py
```

### Langkah 7: Naikkan Container Docker (Prometheus & Grafana)
Mengaktifkan stack pemantauan visual terintegrasi:
```powershell
docker-compose up -d
```
*   **Verifikasi**: Buka docker desktop Anda untuk memastikan container prometheus, grafana, model-serving, dan prometheus-exporter berjalan dengan status hijau.

### Langkah 8: Buat Folder Pengumpulan Akhir
Secara otomatis membangun struktur folder pengumpulan `SMSML_Sultan` sesuai format baku reviewer:
```powershell
d:\dicoding\mml\venv\Scripts\python.exe create_submission_package.py
```

---

## 📸 LANGKAH-LANGKAH PENGAMBILAN SCREENSHOT BUKTI

Semua file screenshot di bawah ini wajib diletakkan ke folder pengumpulan `SMSML_Sultan` menggantikan file placeholder `.jpg` yang ada.

### 1. Bukti Kriteria 2 (Membangun Model)
*   **`screenshoot_dashboard.jpg`**:
    *   Buka browser ke http://127.0.0.1:5000/
    *   Screenshot halaman utama MLflow UI yang menampilkan runs `RandomForest_autolog`, `LogisticRegression_tuned`, dll.
    *   Simpan ke: `SMSML_Sultan/Membangun_model/screenshoot_dashboard.jpg`
*   **`screenshoot_artifak.jpg`**:
    *   Di MLflow UI, klik salah satu run (contoh: `LogisticRegression_tuned`).
    *   Scroll ke bawah hingga bagian **Artifacts**. Pastikan file `training_confusion_matrix.png`, `feature_importance.png`, dan `metric_info_tuned.json` terlihat di panel sebelah kanan.
    *   Ambil screenshot run detail yang memuat folder artifacts tersebut.
    *   Simpan ke: `SMSML_Sultan/Membangun_model/screenshoot_artifak.jpg`

### 2. Bukti Kriteria 4 (FastAPI Model Serving)
*   **`bukti_serving.jpg`**:
    *   Buka http://localhost:8000/docs (Swagger UI).
    *   Klik endpoint POST `/predict` -> **Try it out** -> masukkan data contoh -> klik **Execute**.
    *   Screenshot layar yang memperlihatkan response body JSON sukses (menampilkan `prediction` seperti `"Lulus Tepat Waktu"` dan `confidence` skor).
    *   Simpan ke: `SMSML_Sultan/Monitoring dan Logging/1.bukti_serving/bukti_serving.jpg`

### 3. Bukti Kriteria 4 (Prometheus Targets & Metrics)
*   Buka http://localhost:9090 (Prometheus UI).
*   Klik **Status** -> **Targets**. Pastikan target model-serving dan custom-exporter berstatus **UP** (hijau). Screenshot jika diperlukan.
*   Buka tab Graph, ketikkan query berikut satu per satu, klik **Execute**, lalu screenshot hasilnya (tab **Table** atau **Graph**):
    1.  `model_request_total` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/1.monitoring_model_request_total.jpg`)
    2.  `model_prediction_total` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/2.monitoring_model_prediction_total.jpg`)
    3.  `model_request_latency_seconds_bucket` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/3.monitoring_model_request_latency_seconds.jpg`)
    4.  `model_prediction_errors_total` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/4.monitoring_model_prediction_errors_total.jpg`)
    5.  `model_active_requests` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/5.monitoring_model_active_requests.jpg`)
    6.  `system_cpu_usage_percent` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/6.monitoring_system_cpu_usage_percent.jpg`)
    7.  `system_memory_usage_percent` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/7.monitoring_system_memory_usage_percent.jpg`)
    8.  `model_best_f1_score` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/8.monitoring_model_best_f1_score.jpg`)
    9.  `model_best_accuracy` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/9.monitoring_model_best_accuracy.jpg`)
    10. `model_prediction_confidence_bucket` (Simpan ke: `SMSML_Sultan/Monitoring dan Logging/4.bukti monitoring Prometheus/10.monitoring_model_prediction_confidence.jpg`)

### 4. Bukti Kriteria 4 (Grafana Dashboard - 10+ Panels)
*   Buka http://localhost:3000 (Grafana). Login dengan `admin` / `admin123`.
*   Buka dashboard **"Prediksi Kelulusan Mahasiswa - sultan"** (otomatis termuat berkat konfigurasi provisioning yang kita buat).
*   Screenshot visualisasi dari panel-panel utama berikut untuk membuktikan visualisasi berjalan dengan baik di dashboard berlabel nama Anda:
    1.  `total_prediction_requests` (Simpan ke: `5.bukti monitoring Grafana/1.monitoring_total_prediction_requests.jpg`)
    2.  `prediction_errors` (Simpan ke: `5.bukti monitoring Grafana/2.monitoring_prediction_errors.jpg`)
    3.  `model_f1_score` (Simpan ke: `5.bukti monitoring Grafana/3.monitoring_model_f1_score.jpg`)
    4.  `active_requests` (Simpan ke: `5.bukti monitoring Grafana/4.monitoring_active_requests.jpg`)
    5.  `request_latency` (Simpan ke: `5.bukti monitoring Grafana/5.monitoring_request_latency.jpg`)
    6.  `predictions_by_class` (Simpan ke: `5.bukti monitoring Grafana/6.monitoring_predictions_by_class.jpg`)
    7.  `cpu_usage` (Simpan ke: `5.bukti monitoring Grafana/7.monitoring_cpu_usage.jpg`)
    8.  `memory_usage` (Simpan ke: `5.bukti monitoring Grafana/8.monitoring_memory_usage.jpg`)
    9.  `request_rate` (Simpan ke: `5.bukti monitoring Grafana/9.monitoring_request_rate.jpg`)
    10. `prediction_confidence` (Simpan ke: `5.bukti monitoring Grafana/10.monitoring_prediction_confidence.jpg`)
    11. `model_performance_table` (Simpan ke: `5.bukti monitoring Grafana/11.monitoring_model_performance_table.jpg`)
    12. `disk_usage` (Simpan ke: `5.bukti monitoring Grafana/12.monitoring_disk_usage.jpg`)

### 5. Bukti Kriteria 4 (Alerting & Notifikasi Grafana)
*   Masuk ke menu **Alerting** -> **Alert rules** di Grafana.
*   Buat 3 aturan alerting baru berbasis data Prometheus:
    *   **Rule 1: CPU Alert** -> Warning jika `system_cpu_usage_percent > 85`.
    *   **Rule 2: Error Alert** -> Warning jika `model_prediction_errors_total > 5`.
    *   **Rule 3: Latency Alert** -> Warning jika `model_request_latency_seconds_bucket > 2.0`.
*   Screenshot rules tersebut dan notifikasinya (firing / pending) lalu simpan ke folder `6.bukti alerting Grafana/`:
    *   `1.rules_high_error_rate.jpg` & `2.notifikasi_high_error_rate.jpg`
    *   `3.rules_high_latency.jpg` & `4.notifikasi_high_latency.jpg`
    *   `5.rules_model_f1_drop.jpg` & `6.notifikasi_model_f1_drop.jpg`

---

## 🌟 TIPS PENTING AGAR PASTI BINTANG 5 (ANTI-REJECT)

1.  **Visibilitas Repository GitHub**:
    Pastikan kedua repositori eksternal Anda (**Eksperimen_SML_Sultan** dan **Workflow-CI**) diatur sebagai **PUBLIC** di GitHub. Reviewer akan langsung menolak submission jika link repo tidak bisa dibuka (404).
2.  **Jalankan Cell Notebook sampai Akhir**:
    Pastikan file notebook `.ipynb` yang Anda push ke repo eksperimen sudah di-run-all (menampilkan hasil output cell secara utuh tanpa ada cell yang kosong atau error).
3.  **Tautan yang Benar di File TXT**:
    Buka `Eksperimen_SML_Sultan.txt` dan `Workflow-CI.txt` di root folder submission, ganti placeholder URL dengan link repositori GitHub publik Anda yang sebenarnya.
4.  **Penamaan Dashboard Grafana**:
    Judul dashboard Grafana yang kami setup sudah mengandung nama **"sultan"** (`Prediksi Kelulusan Mahasiswa - sultan`). **Jangan merubah judul dashboard ini** agar reviewer langsung memvalidasi keaslian kepemilikan Anda.
5.  **Gunakan Conda/Local Environment Manager**:
    Ketika menguji `mlflow project`, pastikan parameternya lengkap dan sesuai dengan ketersediaan library agar tidak merusak run tracking online DagsHub.

Setelah semua langkah di atas lengkap dan semua screenshot placeholder diganti dengan gambar asli, bungkus folder `SMSML_Sultan` menjadi format `.zip` menggunakan perintah:
```powershell
Compress-Archive -Path "SMSML_Sultan" -DestinationPath "SMSML_Sultan.zip" -Force
```
Unggah `SMSML_Sultan.zip` ke platform Dicoding dan nikmati nilai **Bintang 5** Anda! 🚀
