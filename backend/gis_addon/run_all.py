"""
run_all.py
---------------------------------
Runs the full Chennai prototype pipeline end-to-end:
  1. Rainfall categorization
  2. DEM flow accumulation
  3. GIS vulnerability (encroachment + drainage deficit)
  4. Fusion into composite risk score
  5. Heatmap visualization

Usage:
    python3 run_all.py

By default this runs entirely on SYNTHETIC data (clearly labeled at
each step) so you can verify the pipeline works before plugging in
real IMD/CWC rainfall and SRTM/Cartosat DEM data. See README.md for
where to get real data and exactly which lines to change.
"""

import subprocess
import sys
import os

STEPS = [
    ("Rainfall categorization", "src/rainfall_categorization.py"),
    ("DEM flow accumulation", "src/flow_accumulation.py"),
    ("GIS vulnerability (encroachment + drainage)", "src/gis_vulnerability.py"),
    ("Fusion into composite risk", "src/fuse_risk_model.py"),
    ("Heatmap visualization", "src/visualize.py"),
]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for label, script in STEPS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        result = subprocess.run([sys.executable, script], cwd=base_dir)
        if result.returncode != 0:
            print(f"\nStep failed: {label}. Stopping.")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Done. Outputs in data/:")
    print("  - composite_flood_risk_grid.csv  (per-cell risk score + tier)")
    print("  - flood_risk_heatmap.png         (visual check)")
    print(f"{'=' * 60}")
