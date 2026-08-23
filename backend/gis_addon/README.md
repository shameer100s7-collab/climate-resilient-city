# Tamil Nadu Flood Risk — Rainfall + GIS Add-on Module (Chennai Prototype)

This is Add-on Module for Model 1 of the Climate-Resilient City early-warning
system: it turns "how much rain fell here" + "what does the terrain and
land-use look like" into a single composite flood-risk score per grid cell.

**Core idea (the Chennai scenario, generalized):** a location doesn't need
heavy rain *itself* to flood — it can be the point where runoff from
higher-elevation areas converges, especially if it used to be a lake/pond/
riverbed that's now built over. This pipeline scores that directly.

## What's in here

```
tn_flood_ews/
  run_all.py                     <- run this to execute the full pipeline
  src/
    rainfall_categorization.py   <- Step 1: 6-category rainfall buckets
    flow_accumulation.py         <- Step 2: DEM -> flow direction -> flow accumulation
    gis_vulnerability.py         <- Step 3: encroachment flag + drainage deficit
    fuse_risk_model.py           <- Step 4: combines 1-3 into composite_risk (0-1)
    visualize.py                 <- Step 5: heatmap PNG
  data/                          <- all outputs land here (gitignored-style scratch)
```

## Quick start (synthetic data — verifies the pipeline works)

```bash
pip install rasterio numpy pandas matplotlib scipy
python3 run_all.py
```

This runs on **clearly-labeled synthetic data** (a fake Chennai basin DEM,
fake rainfall records, a fake "vanished lake") so you can see the whole
pipeline work end-to-end and inspect `data/flood_risk_heatmap.png` before
touching real data.

## Swapping in real data

### 1. Rainfall — `src/rainfall_categorization.py`
Replace the synthetic generator with a real CSV export in this format:
```
station_id, lat, lon, date, rainfall_mm
```
Sources:
- **IMD** gridded rainfall (0.25°): https://www.imdpune.gov.in/
- **CWC** gauge stations: https://cwc.gov.in/
- **TN Open Data Portal**: https://tn.data.gov.in/ (search "rainfall" / "flood")

Set `RAINFALL_CSV_PATH` at the bottom of the script to your real file —
`load_and_categorize()` and `build_rainfall_signature()` need no changes.

### 2. DEM — `src/flow_accumulation.py`
Download a real elevation raster (GeoTIFF) covering your area of interest:
- **SRTM 30m** (global, free): https://earthexplorer.usgs.gov/
- **Cartosat-1 DEM** (India-specific, often better hydrologic detail):
  https://bhuvan.nrsc.gov.in/

Replace `generate_synthetic_chennai_dem()` with:
```python
dem, transform, crs = load_dem("path/to/your_dem.tif")
```
Everything downstream (`fill_depressions`, `compute_flow_direction`,
`compute_flow_accumulation`) works unchanged on a real DEM array.

**Note on scale:** the depression-filling and flow-accumulation code here
is pure NumPy for portability (no C++ build step). It's fine for a city or
a few districts. For a full Tamil Nadu-wide raster at 30m resolution,
either tile it by district and run in batches, or invest time in getting
`richdem`/`pysheds`/GRASS GIS built in your target environment for
production speed — the algorithm logic transfers directly.

### 3. GIS vulnerability — `src/gis_vulnerability.py`
Replace `generate_synthetic_masks()` with two real boolean rasters on the
same grid as your DEM:
- **Historical water mask**: digitized Survey of India toposheets (older
  editions), or Landsat imagery from the 1970s-90s with an NDWI water
  threshold.
- **Current water mask**: recent Sentinel-2 imagery (NDWI) or Bhuvan LULC
  layer: https://bhuvan.nrsc.gov.in/

For `drain_capacity` in `compute_drainage_deficit()`, use TN PWD or your
city corporation's stormwater drain GIS layer if you can obtain it. If not
available, the function already degrades gracefully to a flow-accumulation-
only proxy — you'll still get a usable signal, just less precise.

### 4. Fusion — `src/fuse_risk_model.py`
Once you have **real historical flood event locations** as ground truth
(TN Open Data Portal flood datasets, district disaster reports, or HDX
satellite flood-extent geodata for past events), don't just use the
hand-set `WEIGHTS` dict — train a classifier instead:

```python
from sklearn.ensemble import GradientBoostingClassifier
# features: rainfall_risk, flow_accum_norm, encroached_norm, drainage_deficit_norm
# label: 1 if a real flood was recorded at/near that cell, else 0
```

The four columns this script already produces (`rainfall_risk`,
`flow_accum_norm`, `encroached_norm`, `drainage_deficit_norm`) are exactly
the feature set that classifier needs — this script's job then becomes
feature engineering, and a trained model replaces the weighted sum.

### 5. Statewide scale-up
Same pipeline, run per-district (or in DEM tiles), then mosaic the
`composite_flood_risk_grid.csv` outputs together. The biggest lift at
statewide scale is usually the historical-vs-current water mask
(Step 3) — budget more time for that than for the DEM/rainfall steps,
which are more mechanical.

## Output

`data/composite_flood_risk_grid.csv` — one row per grid cell:
`row, col, composite_risk (0-1), risk_tier (very_low..very_high),
rainfall_risk, flow_accum_norm, encroached_norm, drainage_deficit_norm`

This is what feeds into Model 1's map layer / alerting thresholds.
