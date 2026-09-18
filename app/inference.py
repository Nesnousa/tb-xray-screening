"""
Model loading and inference wrapper.

Isolating this in its own module (instead of inlining everything in
main.py) keeps the FastAPI route handlers thin and makes the inference
logic independently testable and mockable.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger("tb_screening.inference")


@dataclass
class Prediction:
    label: str
    confidence: float
    normal_pct: float
    tb_pct: float


class TBClassifier:
    """Thin wrapper around a Keras model that handles preprocessing,
    inference, and turning raw logits into a friendly result object."""

    def __init__(self, model_path: str = settings.model_path):
        self.model_path = model_path
        self.model = None
        self.version = "untrained"

    def load(self) -> None:
        import os

        if not os.path.exists(self.model_path):
            logger.warning("Model file '%s' not found - predictions disabled.", self.model_path)
            return

        # Deferred import: tensorflow is only needed once we know there's a
        # model file to load, which keeps startup (and `pytest`, which
        # monkeypatches this class entirely) fast when it isn't installed.
        import tensorflow as tf

        logger.info("Loading model from %s ...", self.model_path)
        self.model = tf.keras.models.load_model(self.model_path)
        self.version = os.path.basename(self.model_path)
        logger.info("Model loaded (%s).", self.version)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, raw_bytes: bytes) -> np.ndarray:
        # IMPORTANT: do NOT divide by 255 here. tf.keras.applications.EfficientNetB0
        # bakes its own Rescaling(1/255) + Normalization layers into the very start
        # of the network (confirmed by inspecting backbone.layers[:3]), because it's
        # designed to be fed raw [0,255] pixel values directly. Dividing by 255
        # before handing the image to the model double-rescales it down to a
        # ~[0, 0.0039] range, which crushes the input signal to near-zero and was
        # the root cause of the model predicting the same ~0% TB probability for
        # every single image regardless of content.
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize(settings.img_size)
        arr = np.asarray(img, dtype="float32")
        return np.expand_dims(arr, axis=0)

    def predict(self, raw_bytes: bytes) -> Prediction:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        batch = self.preprocess(raw_bytes)
        preds = self.model.predict(batch, verbose=0)

        if preds.shape[-1] == 1:
            tb_prob = float(preds[0][0])
            normal_prob = 1.0 - tb_prob
        else:
            normal_prob, tb_prob = float(preds[0][0]), float(preds[0][1])

        threshold = settings.decision_threshold
        label = settings.labels[1] if tb_prob >= threshold else settings.labels[0]
        confidence = tb_prob if tb_prob >= threshold else normal_prob

        return Prediction(
            label=label,
            confidence=round(confidence * 100, 2),
            normal_pct=round(normal_prob * 100, 2),
            tb_pct=round(tb_prob * 100, 2),
        )


classifier = TBClassifier()
