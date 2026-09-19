"""
API tests using a fake classifier — no real model weights needed to run
these, which keeps CI fast and makes the test suite runnable by anyone
who clones the repo without a 100+ MB model file.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import Prediction, classifier
from app.main import app


class _FakeModel:
    """Stands in for a loaded Keras model."""

    def predict(self, batch, verbose=0):
        import numpy as np

        return np.array([[0.97]])  # pretend high TB probability (above the 0.94 decision threshold)


@pytest.fixture(autouse=True)
def fake_loaded_model(monkeypatch):
    monkeypatch.setattr(classifier, "model", _FakeModel())
    monkeypatch.setattr(classifier, "version", "test-model")
    yield
    monkeypatch.setattr(classifier, "model", None)


def _fake_png_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


client = TestClient(app)


def test_health_reports_model_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_expected_shape():
    files = {"file": ("xray.png", _fake_png_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "Tuberculosis"
    assert 0 <= body["confidence"] <= 100
    assert set(body["probabilities"]) == {"Normal", "Tuberculosis"}


def test_predict_rejects_non_image():
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_batch_handles_mixed_valid_and_invalid_files():
    files = [
        ("files", ("xray1.png", _fake_png_bytes(), "image/png")),
        ("files", ("notes.txt", b"hello", "text/plain")),
    ]
    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["results"][0]["result"] is not None
    assert body["results"][1]["error"] is not None


def test_predict_without_loaded_model_returns_503(monkeypatch):
    monkeypatch.setattr(classifier, "model", None)
    files = {"file": ("xray.png", _fake_png_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 503
