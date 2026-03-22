"""
model.py - Model loading and inference for Breast Cancer Detection
Uses EfficientNetB7 fine-tuned on IDC dataset
"""

import os
import io
import logging
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)

# Image size for EfficientNetB7
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "breast_cancer_model.keras")
DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

_models = {}


def _model_label(model: tf.keras.Model) -> str:
    name = (model.name or "").lower()
    if "efficientnetb0" in name:
        return "EfficientNetB0"
    if "efficientnetb3" in name:
        return "EfficientNetB3"
    if "efficientnetb7" in name:
        return "EfficientNetB7"
    return model.name or "TransferLearningModel"


def load_model():
    """
    Load all trained models (B0, B3, B7) into memory.
    """
    global _models

    if _models:
        return _models

    model_dir = os.path.join(os.path.dirname(__file__), "saved_model")

    model_paths = {
        "B0": os.path.join(model_dir, "model_B0.keras"),
        "B3": os.path.join(model_dir, "model_B3.keras"),
        "B7": os.path.join(model_dir, "model_B7.keras"),
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            logger.info(f"Loading model {name} from {path}")
            _models[name] = tf.keras.models.load_model(path, compile=False)
        else:
            logger.warning(f"Model {name} not found at {path}")

    if not _models:
        logger.error("No trained models found. Train models first.")

    return _models


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes into model input tensor.
    Resizes to 224x224, normalizes to [0,1] range.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def _find_last_conv_layer_name(model: tf.keras.Model) -> str | None:
    """Find the deepest convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # Search nested models like EfficientNet backbone.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for nested in reversed(layer.layers):
                if isinstance(nested, tf.keras.layers.Conv2D):
                    return nested.name
    return None


def _jet_colormap(value: np.ndarray) -> np.ndarray:
    """Apply a jet-like colormap to a [0,1] float array → (H, W, 3) uint8 RGB."""
    v = np.clip(value, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(v - 0.75) * 4.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(v - 0.50) * 4.0, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(v - 0.25) * 4.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _make_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
) -> np.ndarray | None:
    """Compute the raw Grad-CAM heatmap (H, W) in [0, 1]. Returns None on failure."""
    last_conv_layer_name = _find_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        logger.warning("Could not locate a Conv2D layer for Grad-CAM")
        return None

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        logger.warning("Last conv layer '%s' not found in model graph", last_conv_layer_name)
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    input_tensor_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor_tf, training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        logger.warning("Grad-CAM: gradients are None")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) <= 0:
        logger.warning("Grad-CAM: max activation is 0")
        return None

    heatmap = heatmap / max_val
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()
    return heatmap


def _encode_image(img_array: np.ndarray) -> str:
    """Encode a uint8 RGB numpy array to base64 PNG."""
    out_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _make_gradcam_images(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    alpha: float = 0.5,
) -> dict | None:
    """
    Generate Grad-CAM visualisations.
    Returns dict with 'overlay' (base64 PNG of heatmap on image)
    and 'heatmap' (standalone colormapped heatmap), or None.
    """
    heatmap = _make_gradcam_heatmap(model, input_tensor)
    if heatmap is None:
        return None

    base_img = np.clip(input_tensor[0] * 255.0, 0, 255).astype(np.uint8)

    # Build jet-colormapped heatmap image (0-255)
    colormap = (_jet_colormap(heatmap) * 255.0).astype(np.uint8)

    # Overlay: blend original image with colormapped heatmap
    blended = (base_img.astype(np.float32) * (1 - alpha)
               + colormap.astype(np.float32) * alpha)
    overlay = np.clip(blended, 0, 255).astype(np.uint8)

    return {
        "overlay": _encode_image(overlay),
        "heatmap": _encode_image(colormap),
    }


def predict(image_bytes: bytes, include_heatmap: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Run ensemble prediction using EfficientNetB0, B3, and B7.
    """
    models = load_model()

    if not models:
        return {
            "error": "No trained models loaded.",
            "prediction": None,
            "confidence": None,
        }

    input_tensor = preprocess_image(image_bytes)

    probs = {}
    for name, model in models.items():
        pred = model.predict(input_tensor, verbose=0)
        probs[name] = float(pred[0][0])

    # Average probability from all models
    avg_prob = sum(probs.values()) / len(probs)

    idc_positive_prob = avg_prob
    idc_negative_prob = 1 - avg_prob

    is_malignant = avg_prob >= threshold
    label = "Malignant" if is_malignant else "Benign"
    confidence = idc_positive_prob if is_malignant else idc_negative_prob

    # Grad-CAM: prefer B7, then B3, then B0
    gradcam_result = None
    if include_heatmap:
        for variant in ("B7", "B3", "B0"):
            if variant in models:
                gradcam_result = _make_gradcam_images(models[variant], input_tensor)
                if gradcam_result is not None:
                    break

    response = {
        "prediction": label,
        "confidence": round(confidence, 4),
        "idc_positive_prob": round(idc_positive_prob, 4),
        "idc_negative_prob": round(idc_negative_prob, 4),
        "is_malignant": is_malignant,
        "model_predictions": probs,
        "ensemble_method": "average_probability",
        "img_size_used": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "threshold_used": round(float(threshold), 3),
    }

    if gradcam_result is not None:
        response["gradcam_overlay_base64"] = gradcam_result["overlay"]
        response["gradcam_heatmap_base64"] = gradcam_result["heatmap"]

    return response


def get_model_info() -> dict:
    models = load_model()

    if not models:
        return {"status": "not_loaded"}

    info = {}

    for name, model in models.items():
        info[name] = {
            "architecture": model.name,
            "input_shape": list(model.input_shape),
            "total_params": model.count_params(),
        }

    return {
        "status": "loaded",
        "models_loaded": list(models.keys()),
        "model_details": info,
        "dataset": "IDC Histopathology (Kaggle)",
        "classes": ["Benign", "Malignant"]
    }
