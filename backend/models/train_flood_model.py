"""
Train the real flood-prediction model on REAL historical data.

You must supply: backend/data/historical_flood_events.csv
Required columns (one row per past event, per zone):
    rainfall_mm_6h, elevation_m, drainage_capacity_score,
    historical_flood_frequency, flooded (0 or 1)

WHERE TO GET REAL DATA FOR THIS (no fabricated/simulated rows):
  - Rainfall: India Meteorological Department (IMD) historical records —
    https://mausam.imd.gov.in  or  https://data.gov.in (search "rainfall")
  - Past flood incidents: your municipal corporation's disaster management
    reports, local news archive search ("<your city> waterlogging <year>"),
    or state disaster management authority reports
  - Elevation: use fetch_real_elevation.py in this repo (real Open-Elevation API)
  - Drainage capacity score: start with your own 0-1 estimate per zone based on
    drain width/maintenance status from municipal records; refine over time

Run:
    python train_flood_model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historical_flood_events.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "flood_model.joblib")

FEATURES = ["rainfall_mm_6h", "elevation_m", "drainage_capacity_score", "historical_flood_frequency"]
LABEL = "flooded"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Create this CSV with real historical event data before training.")
        print("See the docstring at the top of this file for exactly where to source it.")
        return

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        return

    if len(df) < 30:
        print(f"WARNING: Only {len(df)} rows found. XGBoost will train but with very "
              f"little real data the model may not generalize well. Keep collecting "
              f"real events and retrain periodically.")

    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("=== Evaluation on held-out real data ===")
    print(classification_report(y_test, preds))
    if y_test.nunique() > 1:
        print("ROC-AUC:", round(roc_auc_score(y_test, probs), 3))

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
    print("The API will now automatically use this trained model instead of the rule-based fallback.")


if __name__ == "__main__":
    main()
