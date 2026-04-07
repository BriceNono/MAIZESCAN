import os
import io
import json
import base64
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── App init ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ─── Constants ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)
MODEL_PATH    = os.environ.get("MODEL_PATH", "model/vgg16_maize_best.h5")
ALLOWED_EXTS  = {"jpg", "jpeg", "png", "webp"}

CLASS_INFO = {
    "Blight": {
        "label":       "Northern Leaf Blight",
        "scientific":  "Exserohilum turcicum",
        "severity":    "High",
        "spread":      "Airborne spores — spreads very fast",
        "color":       "#c0392b",
        "icon":        "blight",
        "description": "Large, cigar-shaped, grayish-green to tan lesions running parallel to leaf veins. Can cause 30–60% yield loss when infection occurs before tasseling.",
        "actions": [
            "Isolate affected rows immediately to limit airborne spread",
            "Apply systemic fungicide (mancozeb + metalaxyl) every 7 days",
            "Report to your local agricultural extension officer",
            "Consider early harvest on severely infected plots to salvage yield",
            "Plow under crop debris after harvest to break the disease cycle"
        ],
        "prevention": "Plant resistant hybrids (Ht gene). Rotate with non-host crops (soybean, bean). Avoid overhead irrigation."
    },
    "Common_Rust": {
        "label":       "Common Rust",
        "scientific":  "Puccinia sorghi",
        "severity":    "Moderate",
        "spread":      "Wind-borne spores — spreads rapidly",
        "color":       "#e67e22",
        "icon":        "rust",
        "description": "Small, circular to oval, powdery pustules on both leaf surfaces, brick-red to brown in colour. Most damaging when infection occurs before silking.",
        "actions": [
            "Apply triazole fungicide (e.g. propiconazole) within 48 hours of detection",
            "Remove and destroy severely infected leaves to stop spore dispersal",
            "Increase plant spacing to improve airflow and reduce humidity",
            "Scout remaining plots for early signs of infection spread",
            "Monitor daily for next 10 days — rust spreads quickly in warm humid conditions"
        ],
        "prevention": "Use rust-resistant varieties. Apply preventive fungicide at V6–V8 stage in high-risk seasons."
    },
    "Gray_Leaf_Spot": {
        "label":       "Gray Leaf Spot",
        "scientific":  "Cercospora zeae-maydis",
        "severity":    "High",
        "spread":      "Residue-borne — moderate spread",
        "color":       "#7f8c8d",
        "icon":        "spot",
        "description": "Rectangular, tan to gray lesions with distinct parallel margins, confined between leaf veins. Thrives in warm humid conditions with poor air circulation.",
        "actions": [
            "Apply strobilurin + triazole fungicide immediately — delay risks 30–40% yield loss",
            "Avoid overhead irrigation — switch to drip to reduce leaf wetness duration",
            "Remove lower canopy leaves to improve air circulation",
            "Rotate to a non-host crop (sorghum, legumes) next season",
            "Do not leave infected crop residue on field — deep-plow or compost off-site"
        ],
        "prevention": "Crop rotation is the most effective control. Use certified disease-free seed. Maintain optimal plant nutrition."
    },
    "Healthy": {
        "label":       "Healthy",
        "scientific":  "No pathogen detected",
        "severity":    "None",
        "spread":      "Not applicable",
        "color":       "#27ae60",
        "icon":        "healthy",
        "description": "The leaf shows no signs of fungal infection, lesions, or disease stress. The plant appears to be in good health.",
        "actions": [
            "Continue routine crop monitoring every 3–4 days",
            "Maintain optimal soil nitrogen levels (50–70 kg/ha)",
            "Ensure proper field drainage to prevent future disease pressure",
            "Scout neighbouring plots — healthy plants can be infected by nearby outbreaks",
            "Record this observation in your farm field diary"
        ],
        "prevention": "Continue current practices. Monitor weather — high humidity + warm temperatures increase disease risk."
    }
}

# Map model output indices → class keys
# Adjust this order to match your model's training class_indices
CLASS_ORDER = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# ─── Model loading ─────────────────────────────────────────
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        log.warning(
            f"Model file not found at '{MODEL_PATH}'. "
            "Running in DEMO mode — predictions will be simulated."
        )
        return False
    try:
        log.info(f"Loading model from {MODEL_PATH} …")
        model = keras.models.load_model(MODEL_PATH)
        model.predict(np.zeros((1, *IMG_SIZE, 3)))  # warm-up
        log.info("Model loaded and warmed up successfully.")
        return True
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return False

# ─── Image helpers ─────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Load image bytes → normalised (1,224,224,3) float32 array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def image_to_base64(file_bytes: bytes) -> str:
    """Return base64-encoded data URI for preview embedding."""
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def demo_prediction(filename: str) -> np.ndarray:
    """
    Simulate realistic probabilities when no model is loaded.
    Cycles through classes based on filename hash so results are consistent.
    """
    seed = sum(ord(c) for c in filename) % 4
    demos = [
        [0.024, 0.038, 0.012, 0.926],  # Healthy
        [0.041, 0.913, 0.033, 0.013],  # Common Rust
        [0.021, 0.047, 0.901, 0.031],  # Gray Leaf Spot
        [0.882, 0.064, 0.041, 0.013],  # Blight
    ]
    return np.array(demos[seed])

# ─── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "model":      "loaded" if model is not None else "demo_mode",
        "classes":    CLASS_ORDER,
        "img_size":   IMG_SIZE,
        "tf_version": tf.__version__
    })

@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate request ──
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a multipart/form-data request with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename received."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTS)}"}), 400

    try:
        file_bytes = file.read()

        # ── Run inference ──
        img_array = preprocess_image(file_bytes)
        if model is not None:
            probs = model.predict(img_array, verbose=0)[0]
        else:
            probs = demo_prediction(file.filename)

        pred_idx   = int(np.argmax(probs))
        pred_key   = CLASS_ORDER[pred_idx]
        confidence = float(probs[pred_idx])
        info       = CLASS_INFO[pred_key]

        # ── Build response ──
        all_probs = {
            CLASS_INFO[cls]["label"]: round(float(prob), 4)
            for cls, prob in zip(CLASS_ORDER, probs)
        }

        response = {
            "success":     True,
            "demo_mode":   model is None,
            "prediction":  info["label"],
            "class_key":   pred_key,
            "confidence":  round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "scientific":  info["scientific"],
            "severity":    info["severity"],
            "spread":      info["spread"],
            "color":       info["color"],
            "icon":        info["icon"],
            "description": info["description"],
            "actions":     info["actions"],
            "prevention":  info["prevention"],
            "all_probs":   all_probs,
            "image_preview": image_to_base64(file_bytes)
        }

        log.info(
            f"Prediction: {info['label']} ({confidence*100:.1f}%) "
            f"| file={file.filename} | demo={model is None}"
        )
        return jsonify(response)

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/classes")
def get_classes():
    return jsonify({
        key: {
            "label":      info["label"],
            "scientific": info["scientific"],
            "severity":   info["severity"],
            "color":      info["color"]
        }
        for key, info in CLASS_INFO.items()
    })

# ─── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# Call at module import time so gunicorn picks it up
load_model()
