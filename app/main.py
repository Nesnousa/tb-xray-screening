"""
TB Screening API — FastAPI entrypoint.

Endpoints
---------
GET  /health            liveness + model status
POST /predict            single-image classification
POST /predict/batch      multi-image classification in one request
POST /predict/gradcam     classification + Grad-CAM heatmap overlay (PNG)

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.gradcam import make_gradcam_overlay
from app.inference import classifier
from app.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    Probabilities,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tb_screening")


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier.load()
    yield


app = FastAPI(
    title="TB Chest X-ray Screening API",
    description="Deep-learning-assisted tuberculosis screening from chest radiographs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __import__("os").path.isdir("frontend"):
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


def _validate_upload(file: UploadFile, raw: bytes) -> None:
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {settings.max_upload_mb} MB limit.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
        model_version=classifier.version if classifier.is_loaded else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not classifier.is_loaded:
        raise HTTPException(503, "Model not loaded. Check MODEL_PATH.")

    raw = await file.read()
    _validate_upload(file, raw)

    try:
        result = classifier.predict(raw)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(400, f"Could not process image: {exc}") from exc

    return PredictionResponse(
        prediction=result.label,
        confidence=result.confidence,
        probabilities=Probabilities(Normal=result.normal_pct, Tuberculosis=result.tb_pct),
        threshold_used=settings.decision_threshold,
        model_version=classifier.version,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
    if not classifier.is_loaded:
        raise HTTPException(503, "Model not loaded. Check MODEL_PATH.")

    items: list[BatchPredictionItem] = []
    for file in files:
        raw = await file.read()
        try:
            _validate_upload(file, raw)
            result = classifier.predict(raw)
            items.append(
                BatchPredictionItem(
                    filename=file.filename,
                    result=PredictionResponse(
                        prediction=result.label,
                        confidence=result.confidence,
                        probabilities=Probabilities(Normal=result.normal_pct, Tuberculosis=result.tb_pct),
                        threshold_used=settings.decision_threshold,
                        model_version=classifier.version,
                    ),
                )
            )
        except HTTPException as exc:
            items.append(BatchPredictionItem(filename=file.filename, error=exc.detail))
        except Exception as exc:  # pragma: no cover - defensive
            items.append(BatchPredictionItem(filename=file.filename, error=str(exc)))

    return BatchPredictionResponse(count=len(items), results=items)


@app.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)) -> Response:
    if not classifier.is_loaded:
        raise HTTPException(503, "Model not loaded. Check MODEL_PATH.")

    raw = await file.read()
    _validate_upload(file, raw)

    try:
        png_bytes = make_gradcam_overlay(classifier.model, raw)
    except Exception as exc:
        logger.exception("Grad-CAM generation failed")
        raise HTTPException(400, f"Could not generate heatmap: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


@app.get("/")
def root() -> dict:
    return {
        "message": "TB Screening API is running.",
        "docs": "/docs",
        "frontend": "/app" if __import__("os").path.isdir("frontend") else None,
    }
