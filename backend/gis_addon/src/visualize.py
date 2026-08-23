"""
visualize.py
---------------------------------
Renders the composite flood risk grid as a heatmap, with the
encroached (former-waterbody) cells outlined, so you can visually
confirm the "vanished lake" effect before wiring this into the map UI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    grid = pd.read_csv(os.path.join(DATA_DIR, "composite_flood_risk_grid.csv"))
    size_row = grid["row"].max() + 1
    size_col = grid["col"].max() + 1

    risk_map = np.full((size_row, size_col), np.nan)
    encroach_map = np.zeros((size_row, size_col), dtype=bool)
    for _, row in grid.iterrows():
        risk_map[int(row["row"]), int(row["col"])] = row["composite_risk"]
        encroach_map[int(row["row"]), int(row["col"])] = bool(row["encroached"])

    cmap = LinearSegmentedColormap.from_list(
        "flood_risk", ["#2b6cb0", "#ecc94b", "#e53e3e"]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(risk_map, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Composite flood risk (0-1)")

    # Outline encroached cells
    ys, xs = np.where(encroach_map)
    ax.scatter(xs, ys, s=6, facecolors="none", edgecolors="black",
               linewidths=0.6, label="Encroached (former waterbody)")

    ax.set_title("Chennai prototype: composite flood risk\n(rainfall + flow accumulation + encroachment + drainage)")
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    out_path = os.path.join(DATA_DIR, "flood_risk_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap -> {out_path}")


if __name__ == "__main__":
    main()
