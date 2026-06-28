"""Side-by-side topology overview of the three evaluation worlds, read off the *baked* full-population
snapshots (research/data/worlds/<name>/facilities.parquet). Each facility is drawn at its real
coordinate, with both marker size and colour encoding its realized usage (the `potential` =
visit count handed to the solvers): rarely-visited facilities stay small and faint, heavily-used
ones grow and brighten, so the effective attractiveness field reads clearly. Visits are heavy-tailed,
so colour uses a log norm. One paper figure that shows, at a glance, how the three worlds differ in
extent (city vs 120 km super-region) and in how usage concentrates.

    uv run --project research python research/scripts/plot_world_topologies.py
    uv run --project research python research/scripts/plot_world_topologies.py --out papers/chainsolvers-journal/figures/world_topologies.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(__file__)
WORLDS_DIR = os.path.join(HERE, "..", "data", "worlds")
DEFAULT_OUT = os.path.join(HERE, "..", "out", "world_topologies.pdf")

# (key, display title). Order = city, real city, super-region.
WORLDS = [
    ("gauss_hannover", "Gauss-Hannover"),
    ("osm_hannover", "OSM-Hannover"),
    ("two_zone", "Two-zone super-region"),
]
CMAP = "YlOrRd"   # heatmap ramp: rare facilities pale (fade to white), popular ones dark red


def _panel(ax, name: str, title: str):
    d = os.path.join(WORLDS_DIR, name)
    fac = pd.read_parquet(os.path.join(d, "facilities.parquet"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    # facilities.parquet is one row per (location, type); collapse to distinct locations and sum
    # visits across the types a location offers -> its effective usage (matches the paper's counts).
    loc = fac.groupby("loc_id", sort=False).agg(x=("x", "first"), y=("y", "first"),
                                                potential=("potential", "sum"))
    x, y = loc["x"].to_numpy() / 1e3, loc["y"].to_numpy() / 1e3   # -> km
    v = loc["potential"].to_numpy(float)
    vis = v > 0

    # Unvisited facilities: the latent topology backdrop (tiny, faint).
    ax.scatter(x[~vis], y[~vis], s=0.4, c="0.88", lw=0, rasterized=True, zorder=1)

    # Visited facilities: size grows with sqrt(visits), colour on a log scale (heavy tail).
    # Draw low -> high so the heavily-used (big, dark) facilities sit on top and read clearly.
    o = np.argsort(v[vis])
    xv, yv, vv = x[vis][o], y[vis][o], v[vis][o]
    vmax = float(vv.max())
    sizes = 1.0 + 32.0 * np.sqrt(vv / vmax)
    sc = ax.scatter(xv, yv, s=sizes, c=vv, cmap=CMAP,
                    norm=LogNorm(vmin=1.0, vmax=vmax), lw=0, alpha=0.9,
                    rasterized=True, zorder=2)

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_title(f"{title}\n{len(loc):,} facilities · {meta.get('n_persons', 0):,} persons",
                 fontsize=9)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("visits (log)", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    avail = [(n, t) for n, t in WORLDS if os.path.exists(os.path.join(WORLDS_DIR, n, "facilities.parquet"))]
    if not avail:
        raise SystemExit("no baked worlds found; run research/scripts/bake_worlds.py first")

    fig, axes = plt.subplots(1, len(avail), figsize=(5.0 * len(avail), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (n, t) in zip(axes, avail):
        _panel(ax, n, t)
    axes[0].set_ylabel("y [km]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
