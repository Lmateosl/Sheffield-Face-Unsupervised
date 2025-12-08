# UMIST Face Classifier Demo (FastAPI + Flask UI)

An end-to-end demo that serves an unsupervised/cluster-assisted face classifier (FastAPI) with a styled Flask frontend for quick experimentation. The model pipeline mirrors the training flow on the UMIST dataset: grayscale → resize to 112x92 → scale → PCA → KMeans cluster → concatenate one-hot cluster → neural classifier.

## Project structure
- `api.py` — FastAPI backend with `/predict-image`, `/predict-pixels`, health endpoints; loads pretrained scaler/PCA/KMeans/TF model from `models/`.
- `frontend_app.py` — Flask UI that calls the API, renders predictions, and showcases built-in UMIST sample faces across multiple categories (clean, blurry, blocked, low-light/contrast, noisy, flipped, color-cast).
- `templates/`, `static/` — HTML/CSS/JS for the UI, including inline analysis report viewer (`static/analysis_report_group6.pdf`).
- `models/` — Serialized artifacts (`scaler.joblib`, `pca.joblib`, `kmeans.joblib`, `face_classifier_pca_kmeans.h5`).
- `umist_cropped.mat` — UMIST dataset used for bundled examples and during training.
- `requirements.txt` — Python dependencies (FastAPI, Flask, TensorFlow CPU, scikit-learn, etc.).
- `group_project.py` — Original training/analysis script (data prep, PCA/KMeans, classifier training).

## Quickstart
1) Create & activate a venv (recommended):
```
python -m venv .venv
.venv\Scripts\activate
```

2) Install deps:
```
pip install -r requirements.txt
```

3) Run the API (FastAPI + uvicorn):
```
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

4) Run the UI (Flask) in a second terminal:
```
.venv\Scripts\activate
set API_BASE_URL=http://127.0.0.1:8000
python frontend_app.py
```

5) Open the UI:
```
http://127.0.0.1:5000
```

## UI features
- Upload any face image (PNG/JPG/BMP/GIF/WEBP, ≤6MB) to get prediction.
- Built-in example gallery with 3 samples per person (all 20 UMIST subjects) across categories: Clean, Blurry, Blocked, Low-light/contrast, Noisy, Horizontal flip, Color-cast.
- Confidence summary (entropy-based certainty), top-k bars, and prediction details.
- Pipeline peek badges showing preprocessing/classification steps.
- Inline analysis report viewer (toggle button) for `static/analysis_report_group6.pdf`.

## Notes on predictions
- The model is “closed set” over the 20 UMIST subjects. Any out-of-distribution face will be forced to one of those IDs; interpret results accordingly.
- The backend always converts to grayscale 112x92 to match training; color casts are for UI preview only.

## Troubleshooting
- Missing dependencies: run `pip install -r requirements.txt` inside the venv.
- `No module named 'sklearn'` or similar when starting uvicorn: ensure the venv is activated and deps installed.
- PDF not rendering inline: click the “View analysis report” button; a direct download link is provided as fallback.

## Scripts/entrypoints
- Start API directly via `python api.py` (uses uvicorn main guard) or `python -m uvicorn api:app`.
- Start UI via `python frontend_app.py`.

## License
This repo uses the UMIST face dataset; respect the dataset’s terms if redistributing. Code is provided for educational/demo purposes.***
