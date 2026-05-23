"""
Prometheus Exporter
====================
Custom Prometheus exporter untuk monitoring model ML.
Mengexpose metrics tambahan selain yang ada di FastAPI app.
"""

import os
import json
import time
import psutil
import logging
from prometheus_client import (
    start_http_server, Counter, Gauge, Histogram, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom registry
registry = CollectorRegistry()

# System metrics
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage", registry=registry)
CPU_USAGE_BASIC = Gauge("system_cpu_usage", "CPU usage percentage (basic)", registry=registry)
MEMORY_USAGE = Gauge("system_memory_usage_percent", "Memory usage percentage", registry=registry)
MEMORY_USAGE_BASIC = Gauge("system_ram_usage", "Memory usage percentage (basic)", registry=registry)
MEMORY_USED_MB = Gauge("system_memory_used_mb", "Memory used in MB", registry=registry)
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage percentage", registry=registry)

# Model metrics
MODEL_ACCURACY = Gauge("model_best_accuracy", "Best model accuracy", registry=registry)
MODEL_F1 = Gauge("model_best_f1_score", "Best model F1 score", registry=registry)
MODEL_PRECISION = Gauge("model_best_precision", "Best model precision", registry=registry)
MODEL_RECALL = Gauge("model_best_recall", "Best model recall", registry=registry)

# Data metrics
TRAINING_SAMPLES = Gauge("data_training_samples", "Number of training samples", registry=registry)
TEST_SAMPLES = Gauge("data_test_samples", "Number of test samples", registry=registry)
NUM_FEATURES = Gauge("data_num_features", "Number of features", registry=registry)

MODEL_SERVING_INFO = Info("model_serving_details", "Details about the serving model", registry=registry)


def collect_system_metrics():
    """Collect system resource metrics."""
    cpu = psutil.cpu_percent(interval=1)
    CPU_USAGE.set(cpu)
    CPU_USAGE_BASIC.set(cpu)
    
    mem = psutil.virtual_memory()
    MEMORY_USAGE.set(mem.percent)
    MEMORY_USAGE_BASIC.set(mem.percent)
    
    MEMORY_USED_MB.set(mem.used / (1024 * 1024))
    
    import os
    disk_path = "C:\\" if os.name == "nt" else "/"
    disk = psutil.disk_usage(disk_path)
    DISK_USAGE.set(disk.percent)


def collect_model_metrics():
    """Collect model performance metrics dari artifact files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    processed_dir = os.path.join(base_dir, "data", "processed")

    # Load metric info
    metric_path = os.path.join(artifacts_dir, "metric_info_tuned.json")
    if not os.path.exists(metric_path):
        metric_path = os.path.join(artifacts_dir, "metric_info.json")

    if os.path.exists(metric_path):
        with open(metric_path, "r") as f:
            metric_info = json.load(f)
        best = metric_info["results"][metric_info["best_model"]]
        MODEL_ACCURACY.set(best["accuracy"])
        MODEL_F1.set(best["f1_score"])
        MODEL_PRECISION.set(best["precision"])
        MODEL_RECALL.set(best["recall"])
        MODEL_SERVING_INFO.info({
            "best_model": metric_info["best_model"],
            "experiment": metric_info.get("experiment", "unknown"),
        })

    # Load feature info
    fi_path = os.path.join(processed_dir, "feature_info.json")
    if os.path.exists(fi_path):
        with open(fi_path, "r") as f:
            fi = json.load(f)
        TRAINING_SAMPLES.set(fi.get("n_train_samples", 0))
        TEST_SAMPLES.set(fi.get("n_test_samples", 0))
        NUM_FEATURES.set(fi.get("n_features", 0))


def run_exporter(port: int = 8001, interval: int = 15):
    """Run Prometheus exporter."""
    logger.info(f"Starting Prometheus exporter on port {port}")
    start_http_server(port, registry=registry)

    # Initial model metrics
    collect_model_metrics()

    while True:
        try:
            collect_system_metrics()
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    run_exporter()
