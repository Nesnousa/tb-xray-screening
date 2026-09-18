"""Pydantic response/request models for the API.

Having typed schemas (rather than returning raw dicts) means FastAPI can
validate responses, generate accurate OpenAPI docs, and callers get
autocomplete-friendly, self-documenting JSON shapes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Probabilities(BaseModel):
    Normal: float = Field(..., ge=0, le=100)
    Tuberculosis: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0, le=100)
    probabilities: Probabilities
    threshold_used: float
    model_version: str


class BatchPredictionItem(BaseModel):
    filename: str
    result: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
