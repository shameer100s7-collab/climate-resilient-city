"""
rainfall_categorization.py
---------------------------------
Add-on Module 1 of the Climate-Resilient City flood risk pipeline.

Takes daily rainfall records (from IMD / CWC gauge stations, or gridded
IMD data resampled to points) and:
  1. Buckets each day's rainfall into 6 standard intensity categories.
  2. Computes, per station/location, a "rainfall signature" = how often
     that location historically falls into each category.

This signature (not the raw daily value) is what feeds Model 1 — it
captures a location's long-run exposure to different rain severities,
which is what you want for a *risk* model rather than a *forecast* model.

Input CSV format expected (one row per station-day):
    station_id, lat, lon, date, rainfall_mm

Real data sources (Tamil Nadu):
  - IMD gridded rainfall (0.25° x 0.25°): https://www.imdpune.gov.in/
  - CWC station-level gauge data: https://cwc.gov.in/
  - TN Open Data Portal flood/rainfall datasets: https://tn.data.gov.in/
Network access from this environment is restricted, so this script
is built to run on whatever CSV you export from those sources — point
RAINFALL_CSV_PATH at your real file and everything else just works.
"""

import pandas as pd
import numpy as np

# --- 6-category IMD-style rainfall intensity buckets (mm / 24h) ---
RAINFALL_CATEGORIES = [
    ("no_trace",        0.0,   2.4),
    ("light",           2.5,   15.5),
    ("moderate",        15.6,  64.4),
    ("heavy",           64.5,  115.5),
    ("very_heavy",      115.6, 204.4),
    ("extremely_heavy", 204.5, np.inf),
]
CATEGORY_NAMES = [c[0] for c in RAINFALL_CATEGORIES]


def categorize_rainfall(mm: float) -> str:
    """Return the category name for a single day's rainfall in mm."""
    if pd.isna(mm):
        return np.nan
    for name, lo, hi in RAINFALL_CATEGORIES:
        if lo <= mm <= hi:
            return name
    return np.nan


def load_and_categorize(csv_path: str) -> pd.DataFrame:
    """Load raw station-day rainfall and tag each row with a category."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"station_id", "lat", "lon", "date", "rainfall_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")
    df["category"] = df["rainfall_mm"].apply(categorize_rainfall)
    return df


def build_rainfall_signature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse years of daily records into one row per station:
    the frequency (0-1) of each category, plus lat/lon and total
    days observed (useful as a data-quality / confidence weight).
    """
    grouped = df.groupby(["station_id", "lat", "lon"])
    total_days = grouped.size().rename("n_days_observed")

    counts = (
        df.groupby(["station_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_NAMES, fill_value=0)
    )

    freq = counts.div(counts.sum(axis=1), axis=0)
    freq.columns = [f"freq_{c}" for c in freq.columns]

    meta = df.groupby("station_id")[["lat", "lon"]].first()
    signature = meta.join(freq).join(total_days.reset_index(level=["lat", "lon"], drop=True))
    signature = signature.reset_index()
    return signature


def generate_synthetic_chennai_data(n_stations=25, n_years=10, seed=42) -> pd.DataFrame:
    """
    Synthetic stand-in for real IMD/CWC data, ONLY so the pipeline can be
    run and sanity-checked end-to-end before you plug in real records.
    Roughly covers Chennai's bounding box (12.85-13.25 N, 80.10-80.35 E)
    with a monsoon-season bias (Oct-Dec heavier rain), matching TN's
    Northeast Monsoon pattern. Replace with load_and_categorize() on
    real IMD/CWC data for anything beyond pipeline testing.
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(12.85, 13.25, n_stations)
    lons = rng.uniform(80.10, 80.35, n_stations)
    station_ids = [f"CHN_{i:03d}" for i in range(n_stations)]

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=365 * n_years, freq="D"
    )

    rows = []
    for sid, lat, lon in zip(station_ids, lats, lons):
        # station-specific bias: some stations sit in naturally
        # wetter/low-lying catchments (proxy for later GIS coupling)
        station_bias = rng.uniform(0.7, 1.6)
        for d in dates:
            monsoon = d.month in (10, 11, 12)
            base_p_rain = 0.35 if monsoon else 0.12
            if rng.random() < base_p_rain:
                shape = 1.6 if monsoon else 1.1
                mm = rng.gamma(shape, 20.0) * station_bias
                mm = min(mm, 500.0)
            else:
                mm = 0.0
            rows.append((sid, lat, lon, d, round(mm, 1)))

    return pd.DataFrame(rows, columns=["station_id", "lat", "lon", "date", "rainfall_mm"])


if __name__ == "__main__":
    import os

    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Swap this block for real data when available ----
    RAINFALL_CSV_PATH = os.path.join(OUT_DIR, "chennai_rainfall_daily_synthetic.csv")
    if not os.path.exists(RAINFALL_CSV_PATH):
        synth = generate_synthetic_chennai_data()
        synth.to_csv(RAINFALL_CSV_PATH, index=False)
        print(f"[synthetic] wrote {RAINFALL_CSV_PATH} ({len(synth)} rows) — "
              f"replace with real IMD/CWC export for production use")
    # --------------------------------------------------------

    df = load_and_categorize(RAINFALL_CSV_PATH)
    signature = build_rainfall_signature(df)

    out_path = os.path.join(OUT_DIR, "rainfall_signature_by_station.csv")
    signature.to_csv(out_path, index=False)

    print(f"\nBuilt rainfall signatures for {len(signature)} stations -> {out_path}")
    print(signature.head(8).to_string(index=False))
