"""Inference and Grad-CAM for the trained EfficientNetV2 experiment."""
from __future__ import annotations

import base64
import io
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
CLASSES = ["bottle", "cup", "pen", "spoon"]


def image_url(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, "PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def orient(image: Image.Image, orientation: int) -> Image.Image:
    operations = {2: Image.Transpose.FLIP_LEFT_RIGHT,
                  3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM,
                  5: Image.Transpose.TRANSPOSE,
                  6: Image.Transpose.ROTATE_270,
                  7: Image.Transpose.TRANSVERSE,
                  8: Image.Transpose.ROTATE_90}
    return image.transpose(operations[orientation]) if orientation in operations else image


class Predictor:
    def __init__(self):
        self.base = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, input_shape=(224, 224, 3))
        self.base.load_weights(ROOT / "weights" / "efficientnetv2-b0_notop.h5")
        self.base.trainable = False
        self.head = tf.keras.models.load_model(
            ROOT / "artifacts" / "no_augmentation.keras", compile=False)
        self.times: list[float] = []

    def predict_bytes(self, raw: bytes) -> dict:
        started = time.perf_counter()
        try:
            original = Image.open(io.BytesIO(raw))
            original.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Please upload a valid JPEG, PNG, or WebP image.") from exc
        if original.width * original.height > 30_000_000:
            raise ValueError("Image is too large (maximum 30 megapixels).")
        orientation = original.getexif().get(274, 1)
        resized = original.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)[None]
        with tf.GradientTape() as tape:
            maps = self.base(array, training=False)
            tape.watch(maps)
            pooled = tf.reduce_mean(maps, axis=(1, 2))
            probabilities = self.head(pooled, training=False)
            index = int(tf.argmax(probabilities[0]))
            score = probabilities[:, index]
        gradients = tape.gradient(score, maps)
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heat = tf.nn.relu(tf.reduce_sum(weights * maps, axis=-1))[0].numpy()
        heat /= max(float(heat.max()), 1e-8)
        heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).resize(
            (224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        # Transparent red/yellow overlay; a coarse class-specific localization.
        rgb = np.asarray(resized, dtype=np.float32).copy()
        alpha = (heat * .62)[..., None]
        color = np.stack([np.full_like(heat, 255), 210 * heat, np.zeros_like(heat)], axis=-1)
        overlay = Image.fromarray(np.uint8(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255)))
        elapsed = (time.perf_counter() - started) * 1000
        self.times.append(elapsed)
        self.times = self.times[-30:]
        scores = probabilities[0].numpy()
        return {
            "class": CLASSES[index],
            "confidence": round(float(scores[index]), 4),
            "scores": {name: round(float(scores[i]), 4) for i, name in enumerate(CLASSES)},
            "image": image_url(orient(resized, orientation)),
            "heatmap": image_url(orient(overlay, orientation)),
            "latency_ms": round(elapsed, 1),
            "average_latency_ms": round(sum(self.times) / len(self.times), 1),
            "device": f"{platform.system()} {platform.machine()} · TensorFlow CPU",
            "explanation": "Grad-CAM highlights image regions that raised the predicted class score. It is not a segmentation mask.",
        }
