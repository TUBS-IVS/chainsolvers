#!/usr/bin/env python
"""Plotting for Block B's alpha sweep (importable + standalone replot from the CSV).

Improvements over the inline plots: distinct color+marker+linestyle per solver (hollow faces so
overlaps stay readable), ALL solvers on ALL panels (effective cost-gap computed for rda/dp_sample
too), symlog x (alpha=0 and the wide tail on one axis), and a 2x2 (work x input) facet per metric.

Effective cost-gap: every solver's `combined_cost` minus the oracle's at the same
(work, input, alpha). Oracle = dp_full where present (anchored), else dp_carla_pot (demoted).
dp_full is the global combined-objective minimum, so gaps are >= 0; rda (ignores potential) and
dp_sample (samples, doesn't minimize) get a real, growing gap -- so they appear on the cost panel.

    python research/scripts/plot_block_b.py                 # replot gauss_hannover
    python research/scripts/plot_block_b.py --world two_zone
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from block_a_style import line  # shared house style (per-solver colour/marker/linestyle, hollow faces)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")

# Block-B draw order; per-solver styling comes from the shared block_a_style module so Block A and
# Block B figures match (do not redefine colours here).
ORDER = ["dp_full", "dp_carla_pot", "dp_carla", "dp_carla_refine", "carla",
         "dp_sample", "dp_sample_capped", "carla_sample", "gravity_independent", "rda", "rda_guided"]

# (column, y-label, symlog?). PANELS = the 3 stacked panels in the per-combo plot;
# FACET_COLS = every metric we facet (guarded by presence in the df, so older CSVs still plot).
# Headline B panels. Distance is evaluated TWO ways: Wasserstein (distribution shape) AND absolute
# per-person deviation (A-style, but ABSOLUTE metres -- NOT gap-to-oracle, since dp_full no longer
# minimises distance once alpha>0). DEMOTED (computed + in the CSV, not plotted): eff_gap (a
# gap-to-oracle, invalid at alpha>0 / no true oracle present -> Block A); pot_fit_tv (facility,
# sparsity-floored) and pot_fit_zone_tv (zone, washes out attractiveness) mislead.
PANELS = [
    ("pot_decile_tv", "potential-mass-decile fit TV (lower=better)", False),
    ("dist_w_m", "distance-fit Wasserstein m (shape)", False),
    ("dist_dev_m", "mean distance deviation m (absolute)", False),
]
FACET_COLS = [
    ("pot_decile_tv", "potential-mass-decile fit TV (headline, lower=better)", False),
    ("pot_spearman", "visits~potential Spearman (higher=better)", False),
    ("dist_w_m", "distance-fit Wasserstein m (shape)", False),
    ("dist_dev_m", "mean distance deviation m (absolute, not gap-to-oracle)", False),
]
WORKS = ["anchored", "demoted"]
INPUTS = ["true", "sampled"]


def add_eff_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Effective combined-cost gap to the oracle (dp_full, else dp_carla_pot) per (work,input,alpha)."""
    out = []
    for _, g in df.groupby(["work", "input", "alpha"], sort=False):
        base = np.nan
        for cand in ("dp_full", "dp_carla_pot"):
            v = g.loc[g["solver"] == cand, "combined_cost"]
            if len(v) and np.isfinite(v.iloc[0]):
                base = float(v.iloc[0]); break
        g = g.copy()
        g["eff_gap"] = g["combined_cost"] - base
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _panel(ax, df, col, ylab, *, symlog=False, legend=True):
    for s in ORDER:
        d = df[(df["solver"] == s) & df[col].notna()].sort_values("alpha")
        if d.empty:
            continue
        line(ax, d["alpha"].to_numpy(), d[col].to_numpy(), s, ms=6)
    # symlog x: alpha=0 (pure-distance) sits on a linear stub, the wide 0.25..50 sweep on a log axis
    # so the low-alpha optimum region is legible instead of crushed against the origin.
    ax.set_xscale("symlog", linthresh=0.25, linscale=0.5)
    if symlog:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    if legend:
        ax.legend(fontsize=7, ncol=2)


def plot_combo(df, alpha_cal, path, title):
    """Three stacked panels (cost gap / pot-fit / dist-fit) for ONE (work,input) combo."""
    import matplotlib.pyplot as plt
    n = len(PANELS)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 4.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, ylab, symlog) in zip(axes, PANELS):
        _panel(ax, df, col, ylab, symlog=symlog)
        ax.axvline(alpha_cal, color="k", ls="--", lw=1, alpha=0.6)
    axes[0].legend(fontsize=7, ncol=2, title=f"calibrated α={alpha_cal:.2f} (dashed)")
    axes[-1].set_xlabel("alpha (potential weight)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def paper_figure(df, alpha_cal, path, *, work="anchored", inp="sampled"):
    """Publication 3-panel alpha sweep for ONE survey-realistic regime (anchors known, distances
    sampled): attractiveness fit (potential-decile TV) / distance shape (Wasserstein) / per-person
    deviation. Saves both .pdf (paper) and .png (preview)."""
    import matplotlib.pyplot as plt
    sub = df[(df["work"] == work) & (df["input"] == inp)]
    panels = [("pot_decile_tv", "attractiveness fit\n(potential-decile TV, lower better)", False),
              ("dist_w_m", "distance shape\n(Wasserstein-1, m)", False),
              ("dist_dev_m", r"per-person deviation" "\n" r"(mean $|\Delta d|$, m)", True)]
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.6), sharex=True)
    for ax, (col, ylab, ylog) in zip(axes, panels):
        for s in ORDER:
            d = sub[(sub["solver"] == s) & sub[col].notna()].sort_values("alpha")
            if d.empty:
                continue
            line(ax, d["alpha"].to_numpy(), d[col].to_numpy(), s, ms=5)
        ax.set_xscale("symlog", linthresh=0.25, linscale=0.5)
        if ylog:
            ax.set_yscale("log")
        ax.axvline(alpha_cal, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6.3, ncol=2, title=f"calibrated α={alpha_cal:.2f} (dashed)")
    axes[-1].set_xlabel("attractiveness weight α")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path[:-4] + ".png", dpi=130)
    plt.close(fig)


def facet_metric(df, col, ylab, alpha_cal, path, title, *, symlog=False):
    """2x2 (work rows x input cols) facet for ONE metric — compare regimes side by side."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(WORKS), len(INPUTS), figsize=(12, 9), sharex=True, sharey=True)
    for r, w in enumerate(WORKS):
        for c, i in enumerate(INPUTS):
            ax = np.atleast_2d(axes)[r][c]
            sub = df[(df["work"] == w) & (df["input"] == i)]
            _panel(ax, sub, col, ylab if c == 0 else "", symlog=symlog, legend=(r == 0 and c == 0))
            ax.axvline(alpha_cal, color="k", ls="--", lw=1, alpha=0.6)
            ax.set_title(f"work={w} / input={i}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_all(df: pd.DataFrame, world: str, out_dir: str, alpha_cal: float) -> None:
    """Emit per-combo stacked plots + per-metric 2x2 facets. df must already have eff_gap."""
    for w in WORKS:
        for i in INPUTS:
            sub = df[(df["work"] == w) & (df["input"] == i)]
            if not sub.empty:
                plot_combo(sub, alpha_cal, os.path.join(out_dir, f"sweep_{world}_{w}_{i}.png"),
                           f"Block B — {world} / work={w} / input={i}")
    for col, ylab, symlog in FACET_COLS:
        if col not in df.columns or not df[col].notna().any():
            continue
        facet_metric(df, col, ylab, alpha_cal, os.path.join(out_dir, f"facet_{world}_{col}.png"),
                     f"Block B — {world} — {ylab}", symlog=symlog)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", default="gauss_hannover")
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args(argv)

    df = pd.read_csv(os.path.join(args.out, f"alpha_sweep_{args.world}.csv"))
    df = add_eff_gap(df)
    df.to_csv(os.path.join(args.out, f"alpha_sweep_{args.world}_eff.csv"), index=False)
    plot_all(df, args.world, args.out, float(df["alpha_cal"].iloc[0]))
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
