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

from block_a_style import apply_paper_style, FS_LABEL, FS_TICK

apply_paper_style()  # seaborn whitegrid + canonical font sizes (grid suppressed on the maps below)

HERE = os.path.dirname(__file__)
WORLDS_DIR = os.path.join(HERE, "..", "data", "worlds")
DEFAULT_OUT = os.path.join(HERE, "..", "out", "world_topologies.pdf")

# (key, display title). Order = city, real city, super-region.
WORLDS = [
    ("gauss_hannover", "Gauss-Hannover"),
    ("osm_hannover", "OSM-Hannover"),
    ("two_zone", "Two-zone super-region"),
]
CMAP = "magma"   # perceptually-uniform purple->gold ramp (not the old "blood" YlOrRd)

# Per-world point rendering at FULL alpha, no subsampling: density reads through the overlap of
# many tiny dots, and a flat size ramp keeps high-usage facilities from dominating (their usage
# reads via colour instead). two_zone is the largest/densest world, so it gets the smallest,
# flattest style; the two city worlds get slightly larger dots with a bit more size contrast.
RENDER = {
    "two_zone": dict(base_size=0.10, size_gain=2.0),
    "_default": dict(base_size=0.15, size_gain=5.0),
}


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
    style = RENDER.get(name, RENDER["_default"])

    # Unvisited facilities: the latent topology backdrop (tiny, faint).
    ax.scatter(x[~vis], y[~vis], s=0.4, c="0.88", lw=0, rasterized=True, zorder=1)

    # Visited facilities: size grows mildly with sqrt(visits); colour on a log scale clipped to
    # [p55, p99.5] of the visit counts so the heavy tail doesn't crush the bulk into one shade and
    # potential differences between facilities stay visible. Draw low -> high so popular sit on top.
    o = np.argsort(v[vis])
    xv, yv, vv = x[vis][o], y[vis][o], v[vis][o]
    vmax = float(vv.max())
    lo = max(1.0, float(np.percentile(vv, 55)))
    hi = max(float(np.percentile(vv, 99.5)), lo * 1.01)
    sizes = style["base_size"] + style["size_gain"] * np.sqrt(vv / vmax)
    sc = ax.scatter(xv, yv, s=sizes, c=vv, cmap=CMAP,
                    norm=LogNorm(vmin=lo, vmax=hi), lw=0, alpha=0.9,
                    rasterized=True, zorder=2)

    # Same-size square box for every world: undistorted geography (equal data aspect) on an
    # equal-sided square data window, plus a forced unit box aspect so the three axes match.
    xc, yc = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.03
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.grid(False)   # geographic scatter: no whitegrid streaks across the map
    ax.set_xlabel("x [km]")
    ax.set_title(f"{title}\n{len(loc):,} facilities · {meta.get('n_persons', 0):,} persons")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("visits (log)", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)


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
