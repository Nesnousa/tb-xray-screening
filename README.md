# TB Chest X-ray Screening

A deep-learning-assisted tuberculosis screening tool: a FastAPI backend
serving an EfficientNetB0 transfer-learning classifier, paired with a
lightweight vanilla-JS console for uploading a chest radiograph and getting
back a prediction, a confidence gauge, a Grad-CAM attention heatmap, and a
downloadable PDF report.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)

## Why this project

Chest radiographs are one of the most common and cheapest tools for TB
screening, but reading them reliably takes a trained radiologist — a
bottleneck in a lot of the settings where TB is most prevalent. This project
explores how far a transfer-learning image classifier can go as a
*screening aid* (never a replacement) for that first triage step, and — just
as importantly — how to make its decisions inspectable rather than a black
box, via Grad-CAM.

## Features

- **`/predict`** — single-image classification (Normal vs. Tuberculosis) with calibrated confidence
- **`/predict/batch`** — classify a whole folder of X-rays in one request
- **`/predict/gradcam`** — Grad-CAM heatmap showing which regions of the image drove the decision
- **Full training pipeline** (`training/train.py`) — two-stage transfer learning (frozen backbone → fine-tune), class-weighted loss to handle the typical Normal:TB imbalance, augmentation, early stopping
- **Console UI** — drag-and-drop upload, confidence gauge, per-class probability bars, local scan history (browser-only), one-click PDF report export
- **Tests** (`pytest` + FastAPI `TestClient`, model mocked out — no GPU or weights needed to run CI)
- Dockerfile + docker-compose for one-command deployment

## Project structure

```
.
├── app/                  FastAPI service
│   ├── main.py            routes
│   ├── inference.py       model loading + prediction
│   ├── gradcam.py         Grad-CAM heatmap generation
│   ├── schemas.py         pydantic response models
│   └── config.py          central settings
├── training/              model training pipeline (separate from the API)
│   ├── train.py
│   ├── dataset.py
│   └── config.yaml
├── frontend/               static console (HTML/CSS/vanilla JS, no build step)
├── tests/                  pytest suite (model mocked)
├── prepare_dataset.py      splits a raw Kaggle download into data/train + data/val
├── find_bad_images.py      scans a dataset for corrupt/unreadable files before training
├── evaluate.py             confusion matrix + per-class recall/precision + decision-threshold sweep on data/val
└── .github/workflows/      CI
```

## Getting started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate       # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Get a model

Either train your own (see below) or drop a compatible `model.h5` in the
project root. Without a model file the API still runs, but `/predict`
returns `503` until one is present — this is intentional so the service is
inspectable/testable without a multi-hundred-MB weights file in the repo.

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

- `http://localhost:8000/app` — the console UI
- `http://localhost:8000/docs` — interactive Swagger docs
- `http://localhost:8000/health` — server + model status

### 4. Run the tests

```bash
pytest -v
```

### Docker

```bash
docker compose up --build
```

## Training your own model

Expects data laid out as:

```
data/train/Normal/*.png
data/train/Tuberculosis/*.png
data/val/Normal/*.png
data/val/Tuberculosis/*.png
```

(this matches the public [TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) on Kaggle once split into train/val folders.)

```bash
python training/train.py --config training/config.yaml
```

This runs a two-stage transfer-learning process (frozen backbone, then
fine-tuning), applies class weighting to handle the imbalance between Normal
and TB cases, and saves the best checkpoint to `model.h5` plus a training
curve plot.

## Results

Evaluated on the held-out validation split (629 images, `python evaluate.py`):

| Class | Recall | Precision |
|---|---|---|
| Tuberculosis | 100.0% (105/105 caught) | 77.2% |
| Normal | 94.1% (493/524 caught) | 100.0% |

The decision threshold (`decision_threshold` in `app/config.py`) is deliberately
set high (0.94, not the naive 0.5) and chosen empirically with `evaluate.py`'s
threshold sweep: it's the highest cutoff that still keeps Tuberculosis recall
at 100% on the validation set. For a screening tool, missing a real TB case
is far worse than a false alarm on a Normal x-ray, so the threshold is tuned
to that asymmetry rather than plain accuracy.

The class-weighting used during training (see `compute_class_weights` in
`training/dataset.py`) is deliberately capped at a 2.5:1 ratio rather than
the raw ~5:1 imbalance in the dataset - letting it go fully uncapped pushes
Tuberculosis recall up but also makes the model considerably more prone to
flagging Normal x-rays as Tuberculosis than the threshold alone can undo.

One implementation detail worth calling out: `tf.keras.applications.EfficientNetB0`
bakes its own `Rescaling(1/255)` + `Normalization` layers into the first two
layers of the backbone — it expects raw `[0,255]` pixel input. Preprocessing
code that also divides by 255 before handing the image to the model
(a natural first instinct) double-rescales the input into a near-zero range
and silently collapses the model into predicting the same class for every
image. `app/inference.py`, `app/gradcam.py` and `training/dataset.py` all
feed the backbone raw `[0,255]` arrays for this reason.

## API reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict -F "file=@chest_xray.png"
```

```json
{
  "prediction": "Tuberculosis",
  "confidence": 98.3,
  "probabilities": { "Normal": 1.7, "Tuberculosis": 98.3 },
  "threshold_used": 0.94,
  "model_version": "model.h5"
}
```

### `POST /predict/gradcam`

Same input, returns a PNG (the radiograph blended with a Grad-CAM heatmap)
instead of JSON.

### `POST /predict/batch`

Accepts multiple `files`, returns per-file results/errors in one response.

## Disclaimer

This is a research/portfolio project for educational purposes. It is **not**
a certified medical device and must not be used for real clinical
decision-making. Every prediction should be confirmed by a licensed
radiologist.

## License

MIT — see [LICENSE](LICENSE).
