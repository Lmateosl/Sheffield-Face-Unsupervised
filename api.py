import os
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from PIL import Image
import tensorflow as tf

# =============================================================
# Configuration
# =============================================================

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Expected image size (matching UMIST preprocessing)
IMG_HEIGHT = 112
IMG_WIDTH = 92
NUM_PIXELS = IMG_HEIGHT * IMG_WIDTH

# =============================================================
# Load models at startup
# =============================================================

try:
    scaler = load(os.path.join(MODELS_DIR, "scaler.joblib"))
    pca = load(os.path.join(MODELS_DIR, "pca.joblib"))
    kmeans = load(os.path.join(MODELS_DIR, "kmeans.joblib"))
    clf_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "face_classifier_pca_kmeans.h5")
    )
except Exception as e:
    # If something goes wrong, raise a clear error at import time
    raise RuntimeError(f"Error loading models from {MODELS_DIR}: {e}")

NUM_CLUSTERS = kmeans.n_clusters
NUM_CLASSES = clf_model.output_shape[-1]

# =============================================================
# FastAPI app
# =============================================================

app = FastAPI(
    title="UMIST Face Classifier API",
    description=(
        "API to classify UMIST face images into one of 20 person IDs.\n\n"
        "Provides basic health endpoints and a /predict-image endpoint "
        "that accepts an uploaded image and returns the predicted class."
    ),
    version="1.0.0",
)

# Optional CORS (useful if later you call this API from a frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Schemas
# =============================================================

class PredictFromPixelsRequest(BaseModel):
    """
    Optional: predict from a raw flattened pixel vector (length = 10304).
    This is useful for debugging and internal tests, not for end-users.
    """
    pixels: List[float]


class PredictResponse(BaseModel):
    predicted_label_0_based: int
    predicted_group_1_based: int
    probabilities: List[float]


# =============================================================
# Helper functions
# =============================================================

def preprocess_image_to_vector(file: UploadFile) -> np.ndarray:
    """
    Read an uploaded image file, convert to grayscale, resize to (112, 92),
    flatten to a (1, 10304) vector of floats in [0, 255].
    """
    try:
        # Read file contents into memory
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Convert to grayscale (L mode) and resize to match training
    img = img.convert("L")
    # PIL uses (width, height)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to numpy array and flatten
    arr = np.array(img, dtype=np.float32)  # shape (112, 92)
    if arr.shape != (IMG_HEIGHT, IMG_WIDTH):
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected image shape after resize: {arr.shape}. "
                   f"Expected ({IMG_HEIGHT}, {IMG_WIDTH}).",
        )

    flat = arr.reshape(1, -1)  # (1, 10304)
    return flat


def build_classifier_features_from_pca(x_pca: np.ndarray, cluster_label: int) -> np.ndarray:
    """
    Given PCA features (1, n_pca) and a K-Means cluster label, build the
    full classifier feature vector by concatenating PCA + one-hot(cluster).
    """
    one_hot = tf.keras.utils.to_categorical(cluster_label, num_classes=NUM_CLUSTERS)
    one_hot = one_hot.reshape(1, -1)
    return np.concatenate([x_pca, one_hot], axis=1)


def predict_from_flat_pixels(flat_pixels: np.ndarray) -> PredictResponse:
    """
    Core prediction pipeline:
    1) scale
    2) PCA transform
    3) K-Means cluster
    4) build classifier features
    5) NN classifier prediction
    """
    if flat_pixels.shape != (1, NUM_PIXELS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected flattened vector of shape (1, {NUM_PIXELS}), "
                   f"got {flat_pixels.shape} instead.",
        )

    # 1) Scale
    x_scaled = scaler.transform(flat_pixels)

    # 2) PCA
    x_pca = pca.transform(x_scaled)

    # 3) K-Means cluster
    cluster_label = int(kmeans.predict(x_pca)[0])

    # 4) Build classifier features
    x_clf = build_classifier_features_from_pca(x_pca, cluster_label)

    # 5) Predict with NN classifier
    probs = clf_model.predict(x_clf)[0]  # (20,)
    predicted_label = int(np.argmax(probs))

    return PredictResponse(
        predicted_label_0_based=predicted_label,
        predicted_group_1_based=predicted_label + 1,
        probabilities=probs.tolist(),
    )


# =============================================================
# Basic endpoints
# =============================================================

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "UMIST Face Classifier API is running. "
                   "Go to /docs for the interactive Swagger UI."
    }


# =============================================================
# Prediction endpoints
# =============================================================

@app.post("/predict-pixels", response_model=PredictResponse)
def predict_from_pixels(request: PredictFromPixelsRequest):
    """
    Optional endpoint:
    Predict from a raw flattened pixel vector.
    The client must send a list of 10304 values (flattened 112x92 image).
    """
    pixels = np.array(request.pixels, dtype=np.float32).reshape(1, -1)
    return predict_from_flat_pixels(pixels)


import io  # Placed here to avoid circular issues in some tools


@app.post("/predict-image", response_model=PredictResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """
    Main endpoint:
    Accepts an uploaded image (JPG/PNG/etc.), converts it to grayscale 112x92,
    and returns the predicted person/group.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    flat = preprocess_image_to_vector(file)
    # Important: reset file pointer for any potential re-use
    file.file.close()

    return predict_from_flat_pixels(flat)
