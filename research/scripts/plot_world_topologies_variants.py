"""Variant explorer for the world-topology figure (fig 1): same panels as
`plot_world_topologies.py`, but sweeps colormap + colour-normalization so we can pick a
ramp that (a) doesn't read as "blood" (the YlOrRd default) and (b) actually spreads the
heavy-tailed visit counts so potential differences between facilities are visible.

Each variant -> one 3-panel PNG (the three worlds). A contact sheet stacks small previews
of every variant so they can be compared at a glance.

    uv run --project research python research/scripts/plot_world_topologies_variants.py

Outputs to research/out/world_variants/.
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
from matplotlib.colors import LogNorm, PowerNorm

HERE = os.path.dirname(__file__)
WORLDS_DIR = os.path.join(HERE, "..", "data", "worlds")
DEFAULT_OUT = os.path.join(HERE, "..", "out", "world_variants")

WORLDS = [
    ("gauss_hannover", "Gauss-Hannover"),
    ("osm_hannover", "OSM-Hannover"),
    ("two_zone", "Two-zone super-region"),
]


def _norm_factory(kind: str, v: np.ndarray):
    """Build a colour norm for the positive visit-count vector `v`.

    The visits are heavy-tailed, so a plain LogNorm(vmin=1) pins the bulk near the bottom of
    the ramp and almost nothing changes colour. The percentile-clipped and power variants pull
    the visible dynamic range onto the *bulk* of the distribution so differences show."""
    vmax = float(v.max())
    if kind == "log":                       # the current default norm (baseline)
        return LogNorm(vmin=1.0, vmax=vmax)
    if kind == "log_p":                      # log, clipped to [p55, p99.5] -> spreads the bulk
        lo = max(1.0, float(np.percentile(v, 55)))
        hi = float(np.percentile(v, 99.5))
        return LogNorm(vmin=lo, vmax=max(hi, lo * 1.01))
    if kind == "power":                      # gamma<1 stretches the low end without going full log
        return PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
    raise ValueError(kind)


# (id, colormap, norm-kind, human label, faint-backdrop colour)
VARIANTS = [
    ("viridis_log",  "viridis", "log",   "viridis · log",                       "0.88"),
    ("magma_log",    "magma",   "log",   "magma · log",                         "0.85"),
    ("magma_logp",   "magma",   "log_p", "magma · log clipped [p55,p99.5]",     "0.85"),
    ("cividis_log",  "cividis", "log",   "cividis · log (colourblind-safe)",    "0.88"),
    ("plasma_power", "plasma",  "power", "plasma · power(0.4)",                 "0.90"),
    ("mako_log",     "mako",    "log",   "mako · log",                          "0.88"),
    ("rocket_logp",  "rocket",  "log_p", "rocket · log clipped [p55,p99.5]",    "0.90"),
    ("turbo_log",    "turbo",   "log",   "turbo · log (high contrast)",         "0.90"),
]


def _resolve_cmap(name: str):
    """matplotlib has viridis/magma/etc.; mako/rocket are seaborn ramps. Fall back gracefully."""
    try:
        return matplotlib.colormaps[name]
    except KeyError:
        try:
            import seaborn as sns
            return sns.color_palette(name, as_cmap=True)
        except Exception:
            fallback = {"mako": "viridis", "rocket": "magma"}.get(name, "viridis")
            return matplotlib.colormaps[fallback]


def _load(name: str):
    d = os.path.join(WORLDS_DIR, name)
    fac = pd.read_parquet(os.path.join(d, "facilities.parquet"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    loc = fac.groupby("loc_id", sort=False).agg(x=("x", "first"), y=("y", "first"),
                                                potential=("potential", "sum"))
    x, y = loc["x"].to_numpy() / 1e3, loc["y"].to_numpy() / 1e3
    v = loc["potential"].to_numpy(float)
    return x, y, v, len(loc), meta.get("n_persons", 0)


def _panel(ax, data, cmap, norm_kind, backdrop, show_cbar=True, title=None,
           base_size=1.0, size_gain=32.0, alpha=0.9, subsample=1.0, seed=0):
    x, y, v, nloc, npers = data
    # Optional uniform subsample: thins overplotting so DENSE regions still read as denser
    # (clusters otherwise saturate into solid blobs and density structure is lost).
    if subsample < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.random(x.shape[0]) < subsample
        x, y, v = x[keep], y[keep], v[keep]
    vis = v > 0
    ax.scatter(x[~vis], y[~vis], s=0.4, c=backdrop, lw=0, rasterized=True, zorder=1)
    o = np.argsort(v[vis])
    xv, yv, vv = x[vis][o], y[vis][o], v[vis][o]
    vmax = float(vv.max())
    sizes = base_size + size_gain * np.sqrt(vv / vmax)
    sc = ax.scatter(xv, yv, s=sizes, c=vv, cmap=cmap, norm=_norm_factory(norm_kind, vv),
                    lw=0, alpha=alpha, rasterized=True, zorder=2)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(f"{title}\n{nloc:,} facilities · {npers:,} persons", fontsize=9)
    if show_cbar:
        ax.set_xlabel("x [km]")
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("visits", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    avail = [(n, t) for n, t in WORLDS
             if os.path.exists(os.path.join(WORLDS_DIR, n, "facilities.parquet"))]
    if not avail:
        raise SystemExit("no baked worlds found; run research/scripts/bake_worlds.py first")
    os.makedirs(args.out, exist_ok=True)

    loaded = {n: _load(n) for n, _ in avail}

    # Density-friendly rendering: small points, FULL alpha, no subsample. Density reads through
    # overlap of many tiny dots; a flatter size ramp keeps high-usage points from dominating.
    DENS = dict(base_size=0.25, size_gain=7.0, alpha=0.9, subsample=1.0)

    # one full 3-panel figure per colour variant
    for vid, cmap_name, norm_kind, label, backdrop in VARIANTS:
        cmap = _resolve_cmap(cmap_name)
        fig, axes = plt.subplots(1, len(avail), figsize=(5.0 * len(avail), 4.8))
        axes = np.atleast_1d(axes)
        for ax, (n, t) in zip(axes, avail):
            _panel(ax, loaded[n], cmap, norm_kind, backdrop, show_cbar=True, title=t, **DENS)
        axes[0].set_ylabel("y [km]")
        fig.suptitle(label, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(args.out, f"world_{vid}.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print("wrote", out, flush=True)

    # colour contact sheet: one row per colour variant, columns = worlds
    nrows, ncols = len(VARIANTS), len(avail)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for r, (vid, cmap_name, norm_kind, label, backdrop) in enumerate(VARIANTS):
        cmap = _resolve_cmap(cmap_name)
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=8)
        for c, (n, t) in enumerate(avail):
            _panel(axes[r, c], loaded[n], cmap, norm_kind, backdrop,
                   show_cbar=False, title=(t if r == 0 else None), **DENS)
    fig.suptitle("Fig-1 colour-scale variants (rows) × worlds (cols)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    sheet = os.path.join(args.out, "contact_sheet.png")
    fig.savefig(sheet, dpi=args.dpi)
    plt.close(fig)
    print("wrote", sheet, flush=True)

    # render-style sheet on the leading ramp (magma · log-clipped): vary point size / alpha /
    # subsample so density legibility (esp. two_zone) can be picked independently of the ramp.
    RAMP = ("magma", "log_p", "0.85")
    # full alpha, no subsample; sweep point size (base) + size-ramp steepness (gain) small->flat
    STYLES = [
        ("s_a", "base 0.20 · gain 6",  dict(base_size=0.20, size_gain=6.0, alpha=0.9, subsample=1.0)),
        ("s_b", "base 0.15 · gain 5",  dict(base_size=0.15, size_gain=5.0, alpha=0.9, subsample=1.0)),
        ("s_c", "base 0.15 · gain 3",  dict(base_size=0.15, size_gain=3.0, alpha=0.9, subsample=1.0)),
        ("s_d", "base 0.10 · gain 4",  dict(base_size=0.10, size_gain=4.0, alpha=0.9, subsample=1.0)),
        ("s_e", "base 0.10 · gain 2 (near-uniform)", dict(base_size=0.10, size_gain=2.0, alpha=0.9, subsample=1.0)),
    ]
    cmap = _resolve_cmap(RAMP[0])
    fig, axes = plt.subplots(len(STYLES), ncols, figsize=(3.2 * ncols, 3.0 * len(STYLES)), squeeze=False)
    for r, (sid, label, kw) in enumerate(STYLES):
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=8)
        for c, (n, t) in enumerate(avail):
            _panel(axes[r, c], loaded[n], cmap, RAMP[1], RAMP[2],
                   show_cbar=False, title=(t if r == 0 else None), **kw)
    fig.suptitle("magma · log-clipped, full alpha — point size / size-ramp sweep (rows) × worlds (cols)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    rsheet = os.path.join(args.out, "render_styles.png")
    fig.savefig(rsheet, dpi=args.dpi)
    plt.close(fig)
    print("wrote", rsheet, flush=True)


if __name__ == "__main__":
    main()
