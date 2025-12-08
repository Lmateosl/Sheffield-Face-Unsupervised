"""
Flask front-end to drive the FastAPI UMIST face classifier.
- Upload an image, call api.py's /predict-image route, and render the result.
- Includes clickable sample faces pulled from the bundled UMIST dataset.
"""
import base64
import io
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import scipy.io
from PIL import Image
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}
BASE_DIR = os.path.dirname(__file__)
UMIST_PATH = os.path.join(BASE_DIR, "umist_cropped.mat")

app = Flask(__name__)
# Needed for flash messages; override with FLASK_SECRET_KEY in production.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-please")
app.config["API_BASE_URL"] = API_BASE_URL
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6MB upload cap


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _ping_api_health() -> Tuple[bool, str]:
    """
    Quick health probe so the UI can show API status.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=4)
        if resp.ok:
            return True, "Online"
        return False, f"Health check failed ({resp.status_code})"
    except requests.RequestException as exc:
        return False, f"Offline: {exc}"


def _rank_probabilities(probabilities: List[float]) -> List[Dict[str, Any]]:
    """
    Turn the probability vector into a ranked list of dicts the template can read.
    """
    ranked = [
        {"group": idx + 1, "prob": float(prob)}
        for idx, prob in enumerate(probabilities)
    ]
    ranked.sort(key=lambda item: item["prob"], reverse=True)
    return ranked


def _load_example_faces(max_people: int = 20, per_person: int = 2) -> List[Dict[str, Any]]:
    """
    Load a handful of sample faces from the bundled UMIST set and cache as PNG/base64.
    This keeps the demo self-contained.
    """
    if not os.path.exists(UMIST_PATH):
        return []

    try:
        mat = scipy.io.loadmat(UMIST_PATH)
        facedat = mat.get("facedat")
    except Exception:
        return []

    if facedat is None:
        return []

    examples: List[Dict[str, Any]] = []
    num_people = min(max_people, facedat.shape[1])
    for person_idx in range(num_people):
        imgs = facedat[0, person_idx]
        take = min(per_person, imgs.shape[2])
        for i in range(take):
            arr = imgs[:, :, i].astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            raw_bytes = buffer.getvalue()
            examples.append(
                {
                    "id": f"p{person_idx+1}_img{i+1}",
                    "person": person_idx + 1,
                    "sample": i + 1,
                    "preview_b64": base64.b64encode(raw_bytes).decode("utf-8"),
                    "bytes": raw_bytes,
                }
            )
    return examples


EXAMPLE_FACES = _load_example_faces()
EXAMPLE_MAP = {ex["id"]: ex for ex in EXAMPLE_FACES}


def _render_index(
    api_status: bool,
    api_status_msg: str,
    result: Dict[str, Any] | None = None,
    preview_data: str | None = None,
    ranked_probs: List[Dict[str, Any]] | None = None,
) -> Any:
    top_probs = (ranked_probs or [])[:3] if ranked_probs else None
    return render_template(
        "index.html",
        api_status=api_status,
        api_status_msg=api_status_msg,
        result=result,
        preview_data=preview_data,
        ranked_probs=ranked_probs,
        top_probs=top_probs,
        examples=EXAMPLE_FACES,
    )


@app.route("/", methods=["GET"])
def index():
    api_status, api_msg = _ping_api_health()
    return _render_index(api_status, api_msg)


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")
    if not uploaded or uploaded.filename == "":
        flash("Please choose an image to upload.", "error")
        return redirect(url_for("index"))

    if not _allowed_file(uploaded.filename):
        flash("Unsupported format. Use PNG, JPG, JPEG, BMP, GIF, or WEBP.", "error")
        return redirect(url_for("index"))

    file_bytes = uploaded.read()
    if not file_bytes:
        flash("Uploaded file is empty.", "error")
        return redirect(url_for("index"))

    preview_data = base64.b64encode(file_bytes).decode("utf-8")
    mimetype = uploaded.mimetype or "image/jpeg"

    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict-image",
            files={"file": (uploaded.filename, io.BytesIO(file_bytes), mimetype)},
            timeout=20,
        )
    except requests.RequestException as exc:
        flash(f"Could not reach the model API: {exc}", "error")
        return redirect(url_for("index"))

    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        flash(f"API error ({resp.status_code}): {detail}", "error")
        return redirect(url_for("index"))

    payload = resp.json()
    probabilities = payload.get("probabilities", [])
    ranked = _rank_probabilities(probabilities)

    return _render_index(
        api_status=True,
        api_status_msg="Online",
        result=payload,
        preview_data=preview_data,
        ranked_probs=ranked,
    )


@app.route("/predict-example", methods=["POST"])
def predict_example():
    example_id = request.form.get("example_id")
    example = EXAMPLE_MAP.get(example_id or "")
    if not example:
        flash("Example not found. Please try another image.", "error")
        return redirect(url_for("index"))

    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict-image",
            files={
                "file": (
                    f"{example_id}.png",
                    io.BytesIO(example["bytes"]),
                    "image/png",
                )
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        flash(f"Could not reach the model API: {exc}", "error")
        return redirect(url_for("index"))

    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        flash(f"API error ({resp.status_code}): {detail}", "error")
        return redirect(url_for("index"))

    payload = resp.json()
    probabilities = payload.get("probabilities", [])
    ranked = _rank_probabilities(probabilities)
    preview_data = example["preview_b64"]

    return _render_index(
        api_status=True,
        api_status_msg="Online",
        result=payload,
        preview_data=preview_data,
        ranked_probs=ranked,
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False,
    )
