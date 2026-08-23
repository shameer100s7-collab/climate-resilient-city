"""
MODEL 1 — Flood/Waterlogging Predictor
=======================================
Two modes, used automatically depending on what data you have:

1. TRAINED MODE (real ML): if you have historical_flood_events.csv (real records),
   run `python train_flood_model.py` to train a real XGBoost classifier and save it.
   This file will then load and use that trained model.

2. FALLBACK MODE (rule-based, works with zero training data): if no trained model
   exists yet, this uses an explainable weighted-scoring formula so your system is
   never non-functional while you're still collecting real historical data.

Nothing here is simulated math dressed up as ML — the fallback is explicitly a
transparent formula, and the trained mode is a real scikit-learn/XGBoost model
trained only when you supply it real data.
"""
import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_model.joblib")


def _rule_based_score(rainfall_mm_next_6h: float, elevation_m: float,
                       drainage_capacity_score: float, historical_flood_frequency: int) -> float:
    """
    Transparent, explainable weighted formula. Weights are a reasonable starting
    point (drainage deficit and rainfall dominate) — recalibrate them once you
    have real historical events to check predictions against.
    """
    rainfall_component = min(rainfall_mm_next_6h / 100.0, 1.0)          # heavier rain -> higher risk
    elevation_component = 1.0 - min(elevation_m / 50.0, 1.0)             # lower elevation -> higher risk
    drainage_component = 1.0 - drainage_capacity_score                  # worse drainage -> higher risk
    history_component = min(historical_flood_frequency / 10.0, 1.0)     # more past floods -> higher risk

    score = (
        0.40 * rainfall_component
        + 0.20 * elevation_component
        + 0.25 * drainage_component
        + 0.15 * history_component
    )
    return round(float(score), 3)


def predict_flood_risk(rainfall_mm_next_6h: float, elevation_m: float,
                        drainage_capacity_score: float, historical_flood_frequency: int) -> dict:
    features = np.array([[rainfall_mm_next_6h, elevation_m,
                           drainage_capacity_score, historical_flood_frequency]])

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        proba = float(model.predict_proba(features)[0][1])
        mode = "trained_ml"
    else:
        proba = _rule_based_score(rainfall_mm_next_6h, elevation_m,
                                   drainage_capacity_score, historical_flood_frequency)
        mode = "rule_based_fallback"

    if proba >= 0.7:
        level = "RED"
    elif proba >= 0.4:
        level = "ORANGE"
    elif proba >= 0.2:
        level = "YELLOW"
    else:
        level = "GREEN"

    return {"risk_score": round(proba, 3), "alert_level": level, "model_mode": mode}
