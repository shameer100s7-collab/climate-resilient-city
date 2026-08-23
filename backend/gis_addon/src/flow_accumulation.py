"""
flow_accumulation.py
---------------------------------
Add-on Module 2 of the Climate-Resilient City flood risk pipeline.

Computes, from a Digital Elevation Model (DEM):
  1. Depression filling (so water doesn't get "trapped" in DEM noise)
  2. D8 flow direction (which of the 8 neighbouring cells each cell drains to)
  3. Flow accumulation (how many upstream cells drain through each cell)

Flow accumulation is the key output: a cell with high accumulation is a
natural water-convergence point. Combined with the encroachment layer,
this tells you "this specific low point used to be a lake and is now
where every high-elevation cell around it dumps its runoff" — the
Chennai apartment-on-a-former-lake scenario, generalized to any DEM.

Implemented in pure NumPy (no richdem/GDAL-heavy build step) so it runs
anywhere Python + numpy + rasterio run, which matters for your
step-by-step local run requirement.

Real DEM sources for Tamil Nadu:
  - SRTM 30m (global, free): https://earthexplorer.usgs.gov/
  - Cartosat-1 DEM (India, 30m, often better hydrologic detail):
    https://bhuvan.nrsc.gov.in/
  - ISRO Bhuvan also hosts TN-specific elevation tiles.
Point DEM_PATH at a real downloaded GeoTIFF and everything downstream
of load_dem() works unchanged.
"""

import numpy as np

try:
    import rasterio
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False

# D8 neighbour offsets (row, col) and their direction codes (ESRI-style)
D8_OFFSETS = {
    1:  (0, 1),    # E
    2:  (1, 1),    # SE
    4:  (1, 0),    # S
    8:  (1, -1),   # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
}


def load_dem(path: str):
    """Load a DEM GeoTIFF as a numpy array + its transform/CRS."""
    if not HAVE_RASTERIO:
        raise ImportError("rasterio is required to load real DEM files")
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, transform, crs


def fill_depressions(dem: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    Simple iterative priority-fill-lite: raise each interior cell that is
    a local sink (lower than all neighbours) up to its lowest neighbour,
    repeated until stable. This is a simplified stand-in for the
    Wang & Liu / Planchon-Darboux algorithms used in richdem/GRASS —
    adequate for moderate-resolution city/state DEMs; swap in richdem's
    implementation later if you need bit-exact hydrologic correctness
    on very large statewide rasters.
    """
    filled = dem.copy()
    rows, cols = filled.shape
    for _ in range(max_iter):
        changed = False
        interior = filled[1:-1, 1:-1]
        neighbours = np.stack([
            filled[0:-2, 0:-2], filled[0:-2, 1:-1], filled[0:-2, 2:],
            filled[1:-1, 0:-2],                     filled[1:-1, 2:],
            filled[2:,   0:-2], filled[2:,   1:-1], filled[2:,   2:],
        ])
        min_neighbour = np.nanmin(neighbours, axis=0)
        is_sink = interior < min_neighbour
        if np.any(is_sink):
            interior[is_sink] = min_neighbour[is_sink] + 1e-6
            filled[1:-1, 1:-1] = interior
            changed = True
        if not changed:
            break
    return filled


def compute_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    D8 flow direction: each cell points to whichever of its 8 neighbours
    has the steepest downhill slope. Border cells flow off-grid (code 0).
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.int32)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            centre = dem[r, c]
            if np.isnan(centre):
                continue
            best_slope = -np.inf
            best_code = 0
            for code, (dr, dc) in D8_OFFSETS.items():
                nr, nc = r + dr, c + dc
                neighbour = dem[nr, nc]
                if np.isnan(neighbour):
                    continue
                # diagonal neighbours are sqrt(2) further away
                dist = np.sqrt(2) if dr != 0 and dc != 0 else 1.0
                slope = (centre - neighbour) / dist
                if slope > best_slope:
                    best_slope = slope
                    best_code = code
            flow_dir[r, c] = best_code
    return flow_dir


def compute_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Flow accumulation: for every cell, how many upstream cells eventually
    drain into it (each cell contributes 1 unit of "runoff" that follows
    the D8 direction chain until it exits the grid).
    Implemented by processing cells in elevation-independent topological
    order via repeated passes (fine for city/state-scale rasters).
    """
    rows, cols = flow_dir.shape
    accum = np.ones((rows, cols), dtype=np.float64)  # each cell starts with its own unit

    # Build reverse lookup: for each cell, which neighbours flow INTO it
    inflow_from = {(r, c): [] for r in range(rows) for c in range(cols)}
    for r in range(rows):
        for c in range(cols):
            code = flow_dir[r, c]
            if code == 0:
                continue
            dr, dc = D8_OFFSETS[code]
            tr, tc = r + dr, c + dc
            if 0 <= tr < rows and 0 <= tc < cols:
                inflow_from[(tr, tc)].append((r, c))

    # Process cells in order of decreasing elevation isn't available here
    # (flow_dir doesn't carry elevation), so use iterative propagation:
    # repeatedly push accumulation downstream until no more changes.
    indegree = {k: len(v) for k, v in inflow_from.items()}
    ready = [k for k, v in indegree.items() if v == 0]
    processed = set()

    while ready:
        r, c = ready.pop()
        if (r, c) in processed:
            continue
        processed.add((r, c))
        code = flow_dir[r, c]
        if code != 0:
            dr, dc = D8_OFFSETS[code]
            tr, tc = r + dr, c + dc
            if 0 <= tr < rows and 0 <= tc < cols:
                accum[tr, tc] += accum[r, c]
                indegree[(tr, tc)] -= 1
                if indegree[(tr, tc)] == 0:
                    ready.append((tr, tc))

    return accum


def generate_synthetic_chennai_dem(size=60, seed=7):
    """
    Small synthetic DEM standing in for a real SRTM/Cartosat tile, ONLY
    for pipeline testing. Shape: a gentle basin sloping toward one
    corner (simulating drainage toward a low-lying area), with a
    couple of raised ridges (simulating the high-elevation source
    areas whose runoff flows toward the basin) and noise.
    Replace with load_dem() on a real GeoTIFF for production use.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    # base slope: elevation decreases toward bottom-right corner (the "basin")
    base = (size - x) * 0.6 + (size - y) * 0.6
    # two ridges (high elevation source zones)
    ridge1 = 30 * np.exp(-(((x - 12) ** 2 + (y - 45) ** 2) / 80))
    ridge2 = 25 * np.exp(-(((x - 45) ** 2 + (y - 10) ** 2) / 100))
    noise = rng.normal(0, 0.5, size=(size, size))
    dem = base + ridge1 + ridge2 + noise
    return dem


if __name__ == "__main__":
    import os
    import pandas as pd

    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Swap this block for a real DEM (load_dem(DEM_PATH)) ----
    print("[synthetic] using synthetic Chennai-basin DEM — "
          "replace with load_dem() on a real SRTM/Cartosat GeoTIFF for production use")
    dem = generate_synthetic_chennai_dem()
    # ---------------------------------------------------------------

    print("Filling depressions...")
    filled = fill_depressions(dem)

    print("Computing D8 flow direction...")
    flow_dir = compute_flow_direction(filled)

    print("Computing flow accumulation...")
    accum = compute_flow_accumulation(flow_dir)

    # Save as a flat CSV of (row, col, elevation, flow_accum) for the
    # fusion step, and also as .npy for fast reload
    np.save(os.path.join(OUT_DIR, "dem_filled.npy"), filled)
    np.save(os.path.join(OUT_DIR, "flow_accum.npy"), accum)

    rows, cols = accum.shape
    rr, cc = np.meshgrid(range(rows), range(cols), indexing="ij")
    df = pd.DataFrame({
        "row": rr.ravel(),
        "col": cc.ravel(),
        "elevation": filled.ravel(),
        "flow_accum": accum.ravel(),
    })
    out_path = os.path.join(OUT_DIR, "flow_accumulation_grid.csv")
    df.to_csv(out_path, index=False)

    top_sinks = df.sort_values("flow_accum", ascending=False).head(5)
    print(f"\nSaved flow accumulation grid -> {out_path}")
    print("\nTop 5 runoff-convergence cells (highest flow accumulation):")
    print(top_sinks.to_string(index=False))
