"""
Model Serving - FastAPI
========================
REST API untuk serving model prediksi kelulusan mahasiswa.
Dilengkapi dengan Prometheus metrics exporter.
"""

import os
import sys
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

import mlflow
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Prometheus Metrics (10+ metrics)
# ============================================================
REQUEST_COUNT = Counter(
    "model_request_total", "Total prediction requests", ["method", "endpoint", "status"]
)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests (basic)", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "model_request_latency_seconds", "Request latency in seconds",
    ["method", "endpoint"], buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
PREDICTION_COUNT = Counter(
    "model_prediction_total", "Total predictions by class", ["predicted_class"]
)
PREDICTION_LATENCY = Summary(
    "model_prediction_latency_seconds", "Prediction inference latency"
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total", "Total prediction errors", ["error_type"]
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds", "Time to load model in seconds"
)
ACTIVE_REQUESTS = Gauge(
    "model_active_requests", "Number of active requests"
)
INPUT_FEATURE_VALUES = Histogram(
    "model_input_ipk_distribution", "Distribution of IPK input values",
    buckets=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence", "Prediction confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_INFO = Info("model_serving", "Model serving information")
TOTAL_REQUESTS_SERVED = Counter(
    "model_total_requests_served", "Total requests served since startup"
)
HEALTH_CHECK_COUNT = Counter(
    "model_health_check_total", "Total health check requests"
)

# ============================================================
# Global state
# ============================================================
model = None
scaler = None
le_target = None
feature_info = None
best_model_name = "unknown"


# ============================================================
# Request/Response Models
# ============================================================
class StudentInput(BaseModel):
    ips_1: float = Field(..., ge=0, le=4, description="IPS Semester 1")
    ips_2: float = Field(..., ge=0, le=4, description="IPS Semester 2")
    ips_3: float = Field(..., ge=0, le=4, description="IPS Semester 3")
    ips_4: float = Field(..., ge=0, le=4, description="IPS Semester 4")
    ips_5: float = Field(..., ge=0, le=4, description="IPS Semester 5")
    ips_6: float = Field(..., ge=0, le=4, description="IPS Semester 6")
    ips_7: float = Field(..., ge=0, le=4, description="IPS Semester 7")
    jumlah_sks: int = Field(..., ge=0, le=200, description="Jumlah SKS")
    umur: int = Field(..., ge=15, le=50, description="Umur mahasiswa")
    jenis_kelamin: str = Field(..., description="Laki-laki / Perempuan")
    asal_sekolah: str = Field(..., description="SMA / SMK / MA")
    organisasi: str = Field(..., description="Ya / Tidak")
    beasiswa: str = Field(..., description="Ya / Tidak")
    jumlah_tanggungan: int = Field(..., ge=0, le=15, description="Jumlah tanggungan")
    penghasilan_ortu: str = Field(..., description="<2jt / 2-5jt / 5-10jt / >10jt")


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    confidence: Optional[float] = None
    model_used: str
    inference_time_ms: float


# ============================================================
# Helper Functions
# ============================================================
def load_artifacts():
    """Load model dan preprocessors."""
    global model, scaler, le_target, feature_info, best_model_name

    start_time = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    artifacts_dir = os.path.join(base_dir, "artifacts")

    # Load preprocessors
    scaler = joblib.load(os.path.join(processed_dir, "scaler.joblib"))
    le_target = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))
    with open(os.path.join(processed_dir, "feature_info.json"), "r") as f:
        feature_info = json.load(f)

    # Load best model info
    metric_path = os.path.join(artifacts_dir, "metric_info_tuned.json")
    if not os.path.exists(metric_path):
        metric_path = os.path.join(artifacts_dir, "metric_info.json")

    with open(metric_path, "r") as f:
        metric_info = json.load(f)

    best_model_name = metric_info["best_model"]
    best_run_id = metric_info["results"][best_model_name]["run_id"]
    model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")

    load_time = time.time() - start_time
    MODEL_LOAD_TIME.set(load_time)
    MODEL_INFO.info({
        "model_name": metric_info["best_model"],
        "f1_score": str(metric_info["best_f1_score"]),
        "run_id": best_run_id,
    })
    logger.info(f"Model loaded in {load_time:.2f}s: {metric_info['best_model']}")


def preprocess_input(data: StudentInput) -> pd.DataFrame:
    """Preprocess input data."""
    d = data.model_dump()
    df = pd.DataFrame([d])

    # Feature engineering
    ips_cols = [f"ips_{i}" for i in range(1, 8)]
    df["ipk_rata_rata"] = df[ips_cols].mean(axis=1).round(2)
    df["ips_trend"] = (df["ips_7"] - df["ips_1"]).round(2)
    df["ips_std"] = df[ips_cols].std(axis=1).round(4)
    df["sks_ratio"] = (df["jumlah_sks"] / 144).round(4)

    # Track IPK distribution
    INPUT_FEATURE_VALUES.observe(float(df["ipk_rata_rata"].iloc[0]))

    # Encoding
    ortu_map = {"<2jt": 0, "2-5jt": 1, "5-10jt": 2, ">10jt": 3}
    df["penghasilan_ortu"] = df["penghasilan_ortu"].map(ortu_map).fillna(-1)

    nominal_cols = ["jenis_kelamin", "asal_sekolah", "organisasi", "beasiswa"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

    # Align with training features
    expected = feature_info["feature_names"]
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]

    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=expected)
    return df_scaled


# ============================================================
# FastAPI App
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="Prediksi Kelulusan Mahasiswa API",
    description="API untuk prediksi status kelulusan mahasiswa menggunakan ML",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    TOTAL_REQUESTS_SERVED.inc()
    method = request.method
    path = request.url.path

    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        REQUEST_COUNT.labels(method=method, endpoint=path, status=response.status_code).inc()
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
        return response
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/")
async def root():
    return {"message": "Prediksi Kelulusan Mahasiswa API", "status": "running"}


@app.get("/health")
async def health():
    HEALTH_CHECK_COUNT.inc()
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(student: StudentInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        X = preprocess_input(student)
        prediction = model.predict(X)
        inference_time = (time.time() - start) * 1000

        pred_code = int(prediction[0])
        pred_class = le_target.inverse_transform([pred_code])[0]

        # Track metrics
        PREDICTION_COUNT.labels(predicted_class=pred_class).inc()
        PREDICTION_LATENCY.observe(inference_time / 1000)

        # Try to get confidence
        confidence = None
        try:
            unwrapped = model._model_impl
            if hasattr(unwrapped, "predict_proba"):
                proba = unwrapped.predict_proba(X)
                confidence = float(np.max(proba))
                PREDICTION_CONFIDENCE.observe(confidence)
        except Exception:
            pass

        return PredictionResponse(
            prediction=pred_class,
            prediction_code=pred_code,
            confidence=confidence,
            model_used=best_model_name,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
