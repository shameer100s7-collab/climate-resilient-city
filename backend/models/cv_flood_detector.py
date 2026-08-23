"""
MODEL 2 — CCTV-Based Water Detection
======================================
BASELINE (works right now, real algorithm, no training needed):
  A classical computer-vision water-region detector using color clustering +
  specular reflection/texture analysis. This is a genuine, functional CV
  algorithm — not a placeholder — and gives you a working Model 2 today.

UPGRADE PATH (real deep learning, needs real labeled footage you provide):
  train_segmentation_model.py trains a real U-Net (via segmentation_models_pytorch)
  once you have labeled flood-water frames. I cannot fabricate this training data —
  download a real public dataset (e.g. FloodNet, or Kaggle "Flood Area Segmentation")
  and/or label your own CCTV frames with a tool like CVAT, then run that script.
  Once trained, this file will automatically prefer the deep model over the baseline.
"""
import os
import cv2
import numpy as np

DL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_segmentation.pt")


def _classical_water_detection(frame: np.ndarray) -> dict:
    """
    Real, working baseline:
    1. Convert to HSV, isolate low-saturation/grayish-blue regions typical of
       standing water and wet asphalt.
    2. Detect specular highlights (bright, low-saturation blobs) characteristic
       of reflective water surfaces.
    3. Combine into a water-likelihood mask, then compute % of frame covered.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Water/wet-road tends to be low saturation, mid-to-high value, blue-gray hue
    hue_mask = cv2.inRange(h, 90, 140)
    sat_mask = cv2.inRange(s, 0, 90)
    val_mask = cv2.inRange(v, 60, 220)
    color_mask = cv2.bitwise_and(cv2.bitwise_and(hue_mask, sat_mask), val_mask)

    # Specular reflection: very bright, low-saturation spots
    reflection_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    water_mask = cv2.bitwise_or(color_mask, reflection_mask)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    total_pixels = water_mask.shape[0] * water_mask.shape[1]
    water_pixels = int(np.count_nonzero(water_mask))
    coverage_pct = round((water_pixels / total_pixels) * 100, 2)

    if coverage_pct > 25:
        severity = "SEVERE"
    elif coverage_pct > 10:
        severity = "MODERATE"
    elif coverage_pct > 2:
        severity = "LOW"
    else:
        severity = "NONE"

    return {
        "water_coverage_pct": coverage_pct,
        "severity": severity,
        "detection_mode": "classical_cv_baseline",
    }


def analyze_frame(image_path: str) -> dict:
    frame = cv2.imread(image_path)
    if frame is None:
        return {"ok": False, "error": f"Could not read image at {image_path}"}

    if os.path.exists(DL_MODEL_PATH):
        # Deep-learning path activates automatically once you've trained it
        try:
            from .dl_inference import segment_water  # only imported if model exists
            result = segment_water(frame, DL_MODEL_PATH)
            result["detection_mode"] = "trained_deep_learning"
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": f"DL model failed, check dl_inference.py: {e}"}

    result = _classical_water_detection(frame)
    return {"ok": True, **result}
