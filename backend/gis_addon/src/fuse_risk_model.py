"""
fuse_risk_model.py
---------------------------------
Fusion step — combines the three add-on modules into ONE composite
risk score per grid cell. This is what plugs into Model 1's existing
DEM+rainfall+drainage pipeline as the add-on feature set.

Risk(cell) = f(rainfall_signature, flow_accumulation, encroachment_flag,
               drainage_deficit)

Design notes:
- Rainfall signature is per-STATION (sparse points), while flow
  accumulation / encroachment / drainage are per-GRID-CELL (dense
  raster). This script nearest-neighbour-joins the rainfall signature
  onto the raster grid so every cell gets an estimated rainfall
  exposure, then combines everything into one 0-1 risk score.
- The weighting below (see WEIGHTS) is a starting point, not a fitted
  model. Once you have real historical flood event locations as
  ground truth (TN Open Data Portal / district disaster reports /
  HDX flood-extent geodata), replace this hand-weighted formula with
  a trained classifier (XGBoost/random forest) using these same four
  features as inputs — the fusion logic here IS your feature
  engineering step for that model.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Starting weights — replace with learned feature importances once you
# train on real labeled flood events.
WEIGHTS = {
    "rainfall_risk": 0.35,   # frequency of heavy+ rainfall categories nearby
    "flow_accum": 0.30,      # how much upstream runoff converges here
    "encroached": 0.20,      # built over a former waterbody
    "drainage_deficit": 0.15,  # inflow vs drain capacity shortfall
}


def nearest_station_rainfall_risk(grid_df: pd.DataFrame,
                                   rainfall_signature: pd.DataFrame,
                                   grid_extent) -> np.ndarray:
    """
    For each grid cell, find the nearest rainfall station (by simple
    Euclidean distance in row/col space mapped to the station's lat/lon
    rank within the grid extent) and take that station's "heavy+ rain"
    frequency as the cell's rainfall risk input.

    grid_extent: (min_row, max_row, min_col, max_col) used only to map
    station lat/lon into the same row/col space as the synthetic grid
    for this prototype. When you switch to a real DEM with a real CRS,
    replace this with a proper spatial join (e.g. via geopandas
    `sjoin_nearest`, matching on actual lat/lon/CRS) instead of this
    rank-based placeholder mapping.
    """
    heavy_cols = ["freq_heavy", "freq_very_heavy", "freq_extremely_heavy"]
    rainfall_signature = rainfall_signature.copy()
    rainfall_signature["heavy_plus_freq"] = rainfall_signature[heavy_cols].sum(axis=1)

    # Map station lat/lon into row/col space via rank-normalization
    lat_rank = rainfall_signature["lat"].rank(pct=True)
    lon_rank = rainfall_signature["lon"].rank(pct=True)
    min_row, max_row, min_col, max_col = grid_extent
    station_rows = (min_row + lat_rank * (max_row - min_row)).values
    station_cols = (min_col + lon_rank * (max_col - min_col)).values
    station_vals = rainfall_signature["heavy_plus_freq"].values

    grid_rows = grid_df["row"].values[:, None]
    grid_cols = grid_df["col"].values[:, None]
    dists = np.sqrt(
        (grid_rows - station_rows[None, :]) ** 2
        + (grid_cols - station_cols[None, :]) ** 2
    )
    nearest_idx = np.argmin(dists, axis=1)
    return station_vals[nearest_idx]


def normalize(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    if rng == 0:
        return series * 0
    return (series - series.min()) / rng


def fuse():
    flow_df = pd.read_csv(os.path.join(DATA_DIR, "flow_accumulation_grid.csv"))
    vuln_df = pd.read_csv(os.path.join(DATA_DIR, "gis_vulnerability_grid.csv"))
    rainfall_sig = pd.read_csv(os.path.join(DATA_DIR, "rainfall_signature_by_station.csv"))

    grid = flow_df.merge(vuln_df, on=["row", "col"])

    grid_extent = (grid["row"].min(), grid["row"].max(),
                   grid["col"].min(), grid["col"].max())
    grid["rainfall_risk_raw"] = nearest_station_rainfall_risk(grid, rainfall_sig, grid_extent)

    # Normalize each feature to 0-1 so weights are comparable
    grid["rainfall_risk"] = normalize(grid["rainfall_risk_raw"])
    grid["flow_accum_norm"] = normalize(grid["flow_accum"])
    grid["encroached_norm"] = grid["encroached"].astype(float)  # already 0/1
    grid["drainage_deficit_norm"] = normalize(grid["drainage_deficit"])

    grid["composite_risk"] = (
        WEIGHTS["rainfall_risk"] * grid["rainfall_risk"]
        + WEIGHTS["flow_accum"] * grid["flow_accum_norm"]
        + WEIGHTS["encroached"] * grid["encroached_norm"]
        + WEIGHTS["drainage_deficit"] * grid["drainage_deficit_norm"]
    )

    # 5-tier risk classification for map display / alerting thresholds
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    grid["risk_tier"] = pd.cut(grid["composite_risk"], bins=bins, labels=labels, right=False)

    out_path = os.path.join(DATA_DIR, "composite_flood_risk_grid.csv")
    grid.to_csv(out_path, index=False)
    return grid, out_path


if __name__ == "__main__":
    grid, out_path = fuse()
    print(f"Saved composite risk grid -> {out_path}\n")

    print("Risk tier distribution:")
    print(grid["risk_tier"].value_counts().sort_index().to_string())

    print("\nTop 10 highest-risk cells:")
    top = grid.sort_values("composite_risk", ascending=False).head(10)
    cols = ["row", "col", "composite_risk", "risk_tier",
            "rainfall_risk", "flow_accum_norm", "encroached_norm", "drainage_deficit_norm"]
    print(top[cols].to_string(index=False))
