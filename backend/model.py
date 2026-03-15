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

_model = None


def _model_label(model: tf.keras.Model) -> str:
    name = (model.name or "").lower()
    if "efficientnetb0" in name:
        return "EfficientNetB0"
    if "efficientnetb3" in name:
        return "EfficientNetB3"
    if "efficientnetb7" in name:
        return "EfficientNetB7"
    return model.name or "TransferLearningModel"


def load_model() -> tf.keras.Model:
    """Load the trained model from disk."""
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully.")
    else:
        logger.warning("No saved model found. Please run train.py first.")
        _model = None
    return _model


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


def _make_gradcam_overlay(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    alpha: float = 0.35,
) -> str | None:
    """Create Grad-CAM heatmap overlay and return base64 PNG string."""
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
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) <= 0:
        return None
    heatmap = heatmap / max_val
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()

    # Compose an RGB red heat overlay using PIL only (keeps dependencies light).
    base_img = np.uint8(np.clip(input_tensor[0] * 255.0, 0, 255))
    overlay = base_img.astype(np.float32)
    overlay[..., 0] = np.clip(
        (1.0 - alpha * heatmap) * overlay[..., 0] + alpha * 255.0 * heatmap,
        0,
        255,
    )
    overlay[..., 1] = np.clip((1.0 - alpha * heatmap) * overlay[..., 1], 0, 255)
    overlay[..., 2] = np.clip((1.0 - alpha * heatmap) * overlay[..., 2], 0, 255)

    out_img = Image.fromarray(overlay.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict(image_bytes: bytes, include_heatmap: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Run inference on the given image bytes.
    Returns a dict with prediction, confidence, and probabilities.
    """
    model = load_model()

    if model is None:
        return {
            "error": "Model not loaded. Please run train.py first.",
            "prediction": None,
            "confidence": None,
        }

    input_tensor = preprocess_image(image_bytes)
    raw_output = model.predict(input_tensor, verbose=0)

    # Model output: sigmoid → probability of IDC Positive (Malignant)
    idc_positive_prob = float(raw_output[0][0])
    idc_negative_prob = 1.0 - idc_positive_prob

    is_malignant = idc_positive_prob >= threshold
    label = "Malignant" if is_malignant else "Benign"
    confidence = idc_positive_prob if is_malignant else idc_negative_prob

    gradcam_overlay = None
    if include_heatmap:
        gradcam_overlay = _make_gradcam_overlay(model, input_tensor)

    response = {
        "prediction": label,
        "confidence": round(confidence, 4),
        "idc_positive_prob": round(idc_positive_prob, 4),
        "idc_negative_prob": round(idc_negative_prob, 4),
        "is_malignant": is_malignant,
        "model_info": f"{_model_label(model)} transfer learning on IDC Histopathology Dataset",
        "img_size_used": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "threshold_used": round(float(threshold), 3),
    }
    if gradcam_overlay is not None:
        response["gradcam_overlay_base64"] = gradcam_overlay
    return response


def get_model_info() -> dict:
    """Return model metadata."""
    model = load_model()
    if model is None:
        return {"status": "not_loaded", "message": "Run train.py to train the model first."}

    return {
        "status": "loaded",
        "architecture": _model_label(model),
        "dataset": "IDC Histopathology (Kaggle)",
        "classes": ["Benign (IDC-)", "Malignant (IDC+)"],
        "input_shape": list(model.input_shape),
        "total_params": model.count_params(),
        "model_path": MODEL_PATH,
    }
