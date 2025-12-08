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
from PIL import Image, ImageDraw, ImageFilter
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


def _blur_image(pil_img: Image.Image) -> Image.Image:
    """Gaussian blur helper for generating blurred variants."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=3))


def _make_blocked_transform(seed: int = 2025):
    """
    Create a transform that masks a random rectangular patch on the face.
    Uses a fixed RNG seed so the gallery stays stable across reloads.
    """
    rng = np.random.default_rng(seed)

    def _block(img: Image.Image) -> Image.Image:
        w, h = img.size
        bw = int(w * rng.uniform(0.22, 0.35))
        bh = int(h * rng.uniform(0.22, 0.35))
        x0 = int(rng.uniform(0, w - bw))
        y0 = int(rng.uniform(0, h - bh))
        x1, y1 = x0 + bw, y0 + bh
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], fill=0)
        return img

    return _block


def _load_example_faces(
    max_people: int = 20,
    per_person: int = 3,
    transform=None,
) -> List[Dict[str, Any]]:
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

    rng = np.random.default_rng(1234)
    examples: List[Dict[str, Any]] = []
    num_people = min(max_people, facedat.shape[1])
    for person_idx in range(num_people):
        imgs = facedat[0, person_idx]
        num_imgs = imgs.shape[2]
        take = min(per_person, num_imgs)
        chosen = rng.choice(num_imgs, size=take, replace=False)
        for sample_rank, img_idx in enumerate(sorted(chosen), start=1):
            arr = imgs[:, :, img_idx].astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="L")
            if transform:
                pil_img = transform(pil_img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            raw_bytes = buffer.getvalue()
            examples.append(
                {
                    "id": f"p{person_idx+1}_img{img_idx+1}",
                    "person": person_idx + 1,
                    "sample": sample_rank,
                    "img_index": img_idx + 1,
                    "preview_b64": base64.b64encode(raw_bytes).decode("utf-8"),
                    "bytes": raw_bytes,
                }
            )
    return examples


EXAMPLE_CATEGORIES = [
    {"key": "clean", "label": "Clean faces", "transform": None},
    {"key": "blurry", "label": "Blurry faces", "transform": _blur_image},
    {"key": "blocked", "label": "Blocked faces", "transform": _make_blocked_transform()},
]


def _build_examples_by_category() -> Dict[str, List[Dict[str, Any]]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat in EXAMPLE_CATEGORIES:
        examples = _load_example_faces(transform=cat["transform"])
        by_cat[cat["key"]] = examples
    return by_cat


EXAMPLES_BY_CATEGORY = _build_examples_by_category()
EXAMPLE_MAP = {
    cat_key: {ex["id"]: ex for ex in ex_list}
    for cat_key, ex_list in EXAMPLES_BY_CATEGORY.items()
}


def _render_index(
    api_status: bool,
    api_status_msg: str,
    result: Dict[str, Any] | None = None,
    preview_data: str | None = None,
    ranked_probs: List[Dict[str, Any]] | None = None,
    selected_category: str = "clean",
) -> Any:
    top_probs = (ranked_probs or [])[:3] if ranked_probs else None
    examples = EXAMPLES_BY_CATEGORY.get(selected_category) or []
    return render_template(
        "index.html",
        api_status=api_status,
        api_status_msg=api_status_msg,
        result=result,
        preview_data=preview_data,
        ranked_probs=ranked_probs,
        top_probs=top_probs,
        examples=examples,
        categories=EXAMPLE_CATEGORIES,
        selected_category=selected_category,
    )


@app.route("/", methods=["GET"])
def index():
    api_status, api_msg = _ping_api_health()
    selected_category = request.args.get("category", "clean")
    if selected_category not in EXAMPLES_BY_CATEGORY:
        selected_category = "clean"
    return _render_index(api_status, api_msg, selected_category=selected_category)


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
    selected_category = request.form.get("category", "clean")
    if selected_category not in EXAMPLES_BY_CATEGORY:
        selected_category = "clean"

    example_id = request.form.get("example_id")
    example = EXAMPLE_MAP.get(selected_category, {}).get(example_id or "")
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
        selected_category=selected_category,
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False,
    )
