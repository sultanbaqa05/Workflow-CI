"""
Create Submission Package - SMSML_Sultan
==========================================
Script untuk membuat struktur folder submission sesuai format reviewer Dicoding.

Struktur output:
SMSML_Sultan/
├── Eksperimen_SML_Sultan.txt
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── student_data_preprocessing/
│   ├── screenshoot_dashboard.jpg
│   ├── screenshoot_artifak.jpg
│   ├── requirements.txt
│   └── DagsHub.txt
├── Workflow-CI.txt
├── Monitoring dan Logging/
│   ├── 1.bukti_serving
│   ├── 2.prometheus.yml
│   ├── 3.prometheus_exporter.py
│   ├── 4.bukti monitoring Prometheus/
│   ├── 5.bukti monitoring Grafana/
│   ├── 6.bukti alerting Grafana/
│   └── 7.Inference.py
"""

import os
import shutil
import sys

# ============================================================
# CONFIG - Sesuaikan dengan data Anda
# ============================================================
STUDENT_NAME = "Sultan"
GITHUB_EKSPERIMEN = "https://github.com/baqa27/Eksperimen_SML_Sultan"
GITHUB_WORKFLOW_CI = "https://github.com/baqa27/Workflow-CI"

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"SMSML_{STUDENT_NAME}")


def create_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
    print(f"  [DIR]  {os.path.relpath(path, BASE_DIR)}")


def copy_file(src, dst):
    """Copy file if source exists."""
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [COPY] {os.path.relpath(dst, BASE_DIR)}")
    else:
        print(f"  [WARN] Source not found: {src}")
        # Create placeholder
        with open(dst, "w", encoding="utf-8") as f:
            f.write(f"# PLACEHOLDER - File asli belum ada: {os.path.basename(src)}\n")
            f.write(f"# Silakan ganti dengan file yang benar\n")
        print(f"  [PLACEHOLDER] {os.path.relpath(dst, BASE_DIR)}")


def copy_dir(src, dst):
    """Copy directory recursively."""
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [DIR COPY] {os.path.relpath(dst, BASE_DIR)}")
    else:
        create_dir(dst)
        print(f"  [WARN] Source dir not found: {src}")


def write_text(path, content):
    """Write text to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [WRITE] {os.path.relpath(path, BASE_DIR)}")


def create_placeholder_image(path, text="PLACEHOLDER"):
    """Create a placeholder text file for screenshot (user must replace)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PLACEHOLDER: {text}\n")
        f.write("Ganti file ini dengan screenshot yang sebenarnya (.jpg/.png)\n")
    print(f"  [PLACEHOLDER] {os.path.relpath(path, BASE_DIR)} - GANTI DENGAN SCREENSHOT!")


def main():
    print("=" * 60)
    print(f"MEMBUAT SUBMISSION PACKAGE: SMSML_{STUDENT_NAME}")
    print("=" * 60)

    # Clean previous output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"[CLEAN] Removed old: {OUTPUT_DIR}")

    create_dir(OUTPUT_DIR)

    # ============================================================
    # 1. Eksperimen_SML_Sultan.txt
    # ============================================================
    print("\n--- Kriteria 1: Eksperimen ---")
    write_text(
        os.path.join(OUTPUT_DIR, f"Eksperimen_SML_{STUDENT_NAME}.txt"),
        f"{GITHUB_EKSPERIMEN}\n"
    )

    # ============================================================
    # 2. Membangun_model/
    # ============================================================
    print("\n--- Kriteria 2: Membangun Model ---")
    model_dir = os.path.join(OUTPUT_DIR, "Membangun_model")
    create_dir(model_dir)

    # modelling.py
    copy_file(
        os.path.join(BASE_DIR, "src", "modelling.py"),
        os.path.join(model_dir, "modelling.py")
    )

    # modelling_tuning.py
    copy_file(
        os.path.join(BASE_DIR, "src", "modelling_tuning.py"),
        os.path.join(model_dir, "modelling_tuning.py")
    )

    # student_data_preprocessing folder
    copy_dir(
        os.path.join(BASE_DIR, "data", "processed"),
        os.path.join(model_dir, "student_data_preprocessing")
    )

    # requirements.txt
    copy_file(
        os.path.join(BASE_DIR, "requirements.txt"),
        os.path.join(model_dir, "requirements.txt")
    )

    # DagsHub.txt
    copy_file(
        os.path.join(BASE_DIR, "DagsHub.txt"),
        os.path.join(model_dir, "DagsHub.txt")
    )

    # Screenshots (check real_screenshots folder first)
    real_screenshots_dir = os.path.join(BASE_DIR, "real_screenshots")
    os.makedirs(real_screenshots_dir, exist_ok=True)
    
    screenshot_dash = os.path.join(model_dir, "screenshoot_dashboard.jpg")
    screenshot_art = os.path.join(model_dir, "screenshoot_artifak.jpg")

    real_dash = os.path.join(real_screenshots_dir, "screenshoot_dashboard.jpg")
    real_art = os.path.join(real_screenshots_dir, "screenshoot_artifak.jpg")

    if os.path.exists(real_dash):
        copy_file(real_dash, screenshot_dash)
    else:
        create_placeholder_image(screenshot_dash, "Screenshot MLflow Dashboard UI")
        
    if os.path.exists(real_art):
        copy_file(real_art, screenshot_art)
    else:
        create_placeholder_image(screenshot_art, "Screenshot MLflow Artifacts")

    # ============================================================
    # 3. Workflow-CI.txt
    # ============================================================
    print("\n--- Kriteria 3: Workflow CI ---")
    write_text(
        os.path.join(OUTPUT_DIR, "Workflow-CI.txt"),
        f"{GITHUB_WORKFLOW_CI}\n"
    )

    # ============================================================
    # 4. Monitoring dan Logging/
    # ============================================================
    print("\n--- Kriteria 4: Monitoring dan Logging ---")
    mon_dir = os.path.join(OUTPUT_DIR, "Monitoring dan Logging")
    create_dir(mon_dir)

    # 1.bukti_serving
    serving_dir = os.path.join(mon_dir, "1.bukti_serving")
    create_dir(serving_dir)
    real_serving = os.path.join(real_screenshots_dir, "bukti_serving.jpg")
    if os.path.exists(real_serving):
        copy_file(real_serving, os.path.join(serving_dir, "bukti_serving.jpg"))
    else:
        create_placeholder_image(
            os.path.join(serving_dir, "bukti_serving.jpg"),
            "Screenshot FastAPI /docs atau curl response"
        )

    # 2.prometheus.yml
    copy_file(
        os.path.join(BASE_DIR, "monitoring", "prometheus.yml"),
        os.path.join(mon_dir, "2.prometheus.yml")
    )

    # 3.prometheus_exporter.py
    copy_file(
        os.path.join(BASE_DIR, "serving", "prometheus_exporter.py"),
        os.path.join(mon_dir, "3.prometheus_exporter.py")
    )

    # 4.bukti monitoring Prometheus (folder)
    prom_mon_dir = os.path.join(mon_dir, "4.bukti monitoring Prometheus")
    create_dir(prom_mon_dir)

    prometheus_metrics = [
        "model_request_total",
        "model_prediction_total",
        "model_request_latency_seconds",
        "model_prediction_errors_total",
        "model_active_requests",
        "system_cpu_usage_percent",
        "system_memory_usage_percent",
        "model_best_f1_score",
        "model_best_accuracy",
        "model_prediction_confidence",
    ]
    for i, metric in enumerate(prometheus_metrics, 1):
        filename = f"{i}.monitoring_{metric}.jpg"
        real_file = os.path.join(real_screenshots_dir, filename)
        dest_file = os.path.join(prom_mon_dir, filename)
        if os.path.exists(real_file):
            copy_file(real_file, dest_file)
        else:
            create_placeholder_image(
                dest_file,
                f"Screenshot Prometheus query: {metric}"
            )

    # 5.bukti monitoring Grafana (folder)
    grafana_mon_dir = os.path.join(mon_dir, "5.bukti monitoring Grafana")
    create_dir(grafana_mon_dir)

    grafana_metrics = [
        "total_prediction_requests",
        "prediction_errors",
        "model_f1_score",
        "active_requests",
        "request_latency",
        "predictions_by_class",
        "cpu_usage",
        "memory_usage",
        "request_rate",
        "prediction_confidence",
        "model_performance_table",
        "disk_usage",
    ]
    for i, metric in enumerate(grafana_metrics, 1):
        filename = f"{i}.monitoring_{metric}.jpg"
        real_file = os.path.join(real_screenshots_dir, filename)
        dest_file = os.path.join(grafana_mon_dir, filename)
        if os.path.exists(real_file):
            copy_file(real_file, dest_file)
        else:
            create_placeholder_image(
                dest_file,
                f"Screenshot Grafana panel: {metric}"
            )

    # 6.bukti alerting Grafana (folder)
    alert_dir = os.path.join(mon_dir, "6.bukti alerting Grafana")
    create_dir(alert_dir)

    alerts = [
        ("1.rules_high_error_rate", "Screenshot rules High Prediction Error Rate"),
        ("2.notifikasi_high_error_rate", "Screenshot notifikasi High Prediction Error Rate"),
        ("3.rules_high_latency", "Screenshot rules High Prediction Latency"),
        ("4.notifikasi_high_latency", "Screenshot notifikasi High Prediction Latency"),
        ("5.rules_model_f1_drop", "Screenshot rules Model F1 Score Drop"),
        ("6.notifikasi_model_f1_drop", "Screenshot notifikasi Model F1 Score Drop"),
    ]
    for filename, desc in alerts:
        img_name = f"{filename}.jpg"
        real_file = os.path.join(real_screenshots_dir, img_name)
        dest_file = os.path.join(alert_dir, img_name)
        if os.path.exists(real_file):
            copy_file(real_file, dest_file)
        else:
            create_placeholder_image(
                dest_file,
                desc
            )

    # 7.Inference.py
    copy_file(
        os.path.join(BASE_DIR, "src", "inference.py"),
        os.path.join(mon_dir, "7.Inference.py")
    )

    # Additional files
    print("\n--- File Tambahan ---")
    # Copy app.py ke monitoring
    copy_file(
        os.path.join(BASE_DIR, "serving", "app.py"),
        os.path.join(mon_dir, "app.py")
    )

    # Copy dashboard.json
    copy_file(
        os.path.join(BASE_DIR, "monitoring", "grafana", "dashboard.json"),
        os.path.join(mon_dir, "dashboard.json")
    )

    # Copy alerting.json
    copy_file(
        os.path.join(BASE_DIR, "monitoring", "grafana", "alerting.json"),
        os.path.join(mon_dir, "alerting.json")
    )

    print("\n" + "=" * 60)
    print("SUBMISSION PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"\n[IMPORTANT] - Anda harus mengganti file PLACEHOLDER dengan:")
    print(f"  1. screenshoot_dashboard.jpg   -> Screenshot MLflow Tracking UI")
    print(f"  2. screenshoot_artifak.jpg     -> Screenshot MLflow Artifacts")
    print(f"  3. 1.bukti_serving/            -> Screenshot FastAPI running")
    print(f"  4. 4.bukti monitoring Prometheus/ -> Screenshot Prometheus queries")
    print(f"  5. 5.bukti monitoring Grafana/    -> Screenshot Grafana panels")
    print(f"  6. 6.bukti alerting Grafana/      -> Screenshot alerting rules + notifikasi")
    print(f"\n[WARNING] - JANGAN LUPA update link di:")
    print(f"  - Eksperimen_SML_{STUDENT_NAME}.txt -> URL repo GitHub kriteria 1")
    print(f"  - Workflow-CI.txt                   -> URL repo GitHub kriteria 3")

    # Print final structure
    print(f"\nFolder Structure:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")


if __name__ == "__main__":
    main()
