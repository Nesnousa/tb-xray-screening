"""
Central configuration for the TB screening API.

Keeping every tunable in one place (instead of scattered constants across
files) makes it trivial to point the service at a different model, image
size, or label set without touching the inference code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "model.h5")
    img_size: tuple[int, int] = (224, 224)
    labels: tuple[str, str] = ("Normal", "Tuberculosis")
    # 0.94, not the naive 0.5 default: per evaluate.py's threshold sweep on the
    # real validation set (model retrained with milder class weighting + extra
    # augmentation, see training/dataset.py), 94% is the highest cutoff that
    # still catches 100% of Tuberculosis cases, with 94.1% Normal recall.
    # Re-run `python evaluate.py` and re-check its "Threshold sweep" table any
    # time the model is retrained - the best value can shift.
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.94"))
    last_conv_layer: str = os.getenv("LAST_CONV_LAYER", "top_conv")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    allowed_content_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/jpg")


settings = Settings()
