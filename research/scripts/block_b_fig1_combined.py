#!/usr/bin/env python
"""Block B Fig 1: combined alpha-sweep, two worlds side by side (osm, two_zone), four stacked
metrics per world. Rows, top to bottom:
    1. attractiveness fit  -- potential-mass-decile TV (truth-anchored headline; lower=better)
    2. visits~potential Pearson r  -- a *correlation* (self-referential, magnitude-aware)
    3. distance shape  -- Wasserstein-1 (m)
    4. per-person deviation  -- mean |delta d| (m, absolute)
r sits directly under TV on purpose: the decile fit bottoms at the calibrated alpha while r (like
every correlation) keeps climbing past it -- correlations reward over-concentration; only the
truth-anchored decile fit peaks at calibration. This folds the old standalone "correlations" figure
into the headline.

Uniform source: every panel is the 50k full-population run, its 7 feasible solvers. We measured
(block_b_1k_vs_50k.py) that decile-TV's near-optimum level is sample-size-sensitive (a residual
sampling-noise floor at survey N), so a clean figure draws ALL panels from one large N rather than
mixing. The four 1k-only solvers are excluded because each is redundant or uninteresting here:
dp_full tracks dp_carla_refine to <=0.007 decile-TV (the refined DP is a faithful oracle stand-in);
plain dp_sample is infeasible at full-population scale (its candidate ball blows up on the large
sparse world -- the cap exists for exactly this) and otherwise tracks dp_sample_capped; rda and
rda_guided are flat in alpha (attractiveness-blind) and carry no sweep story.

    python research/scripts/block_b_fig1_combined.py --p50k <dir>   # default p50k = out/block_b_fullpop
"""
from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from block_a import _load_locations_only
from block_a_style import line, apply_paper_style, WORLD_NAME, legend
from block_b_corr_figs import per_type_counts
from plot_block_b import ORDER

apply_paper_style()
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")   # Overleaf clone
WORLDS = ["osm_hannover", "two_zone"]
WORK, INPUT = "anchored", "sampled"                                             # survey-realistic regime


def load_csv(run_dir, world):
    csv = pd.read_csv(os.path.join(run_dir, f"alpha_sweep_{world}.csv"))
    return csv[(csv.work == WORK) & (csv["input"] == INPUT)]


def pearson_curve(p50k_dir, world):
    """Per-(solver, alpha) visit-weighted Pearson r from the 50k raw_legs; cached next to it."""
    cache = os.path.join(p50k_dir, f"pearson_{world}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    W = _load_locations_only(world)
    ids, _, pots = W.locations_tuple
    raw = pd.read_parquet(os.path.join(p50k_dir, f"raw_legs_{world}.parquet"))
    raw = raw[(raw.work == WORK) & (raw["input"] == INPUT)]
    rows = []
    for s in raw.solver.unique():
        rs = raw[raw.solver == s]
        for a in sorted(rs.alpha.unique()):
            num = den = 0.0
            for t, (_, cnt) in per_type_counts(rs[rs.alpha == a], ids).items():
                pot = np.asarray(pots[t], float)
                if pot.sum() <= 0 or cnt.sum() == 0 or len(pot) < 3:
                    continue
                w = cnt.sum()
                num += pearsonr(cnt, pot).statistic * w
                den += w
            if den > 0:
                rows.append((s, a, num / den))
    out = pd.DataFrame(rows, columns=["solver", "alpha", "pearson"])
    out.to_csv(cache, index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p50k", default=os.path.join(os.path.dirname(__file__), "..", "out", "block_b_fullpop"))
    args = p.parse_args(argv)

    # Uniform source: every panel is the 50k full-population run, its 7 feasible solvers. The four
    # 1k-only solvers (dp_full, dp_sample, rda, rda_guided) are excluded -- each is redundant or
    # uninteresting here (dp_full tracks dp_carla_refine to <=0.007 decile-TV; plain dp_sample is
    # infeasible at full-pop scale and tracks dp_sample_capped; rda/rda_guided are flat in alpha).
    metrics = [
        ("pot_decile_tv", "attractiveness fit\n(potential-decile TV, lower=better)", "csv", False),
        ("pearson", "visits~potential\nPearson r (higher=more concentrated)", "r", False),
        ("dist_w_m", "distance shape\n(Wasserstein-1, m)", "csv", False),
        ("dist_dev_m", r"per-person deviation" "\n" r"(mean $|\Delta d|$, m)", "csv", True),
    ]
    fig, axes = plt.subplots(len(metrics), len(WORLDS), figsize=(11, 12.5), sharex=True)
    for ci, world in enumerate(WORLDS):
        c5 = load_csv(args.p50k, world)
        s50 = [s for s in ORDER if s in set(c5.solver.unique())]       # the 7 full-population solvers
        cal = float(c5["alpha_cal"].iloc[0])
        rcurve = pearson_curve(args.p50k, world)
        for ri, (col, ylab, src, ylog) in enumerate(metrics):
            ax = axes[ri][ci]
            data, valcol = (rcurve, "pearson") if src == "r" else (c5, col)
            for s in s50:
                d = data[(data.solver == s) & data[valcol].notna()].sort_values("alpha")
                if not d.empty:
                    line(ax, d["alpha"].to_numpy(), d[valcol].to_numpy(), s, ms=5)
            ax.set_xscale("symlog", linthresh=0.25, linscale=0.5)
            if ylog:
                ax.set_yscale("log")
            ax.axvline(cal, color="k", ls="--", lw=1, alpha=0.6)
            ax.grid(alpha=0.3, which="both")
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 0:
                ax.set_title(f"{WORLD_NAME.get(world, world)}  (calibrated $\\alpha$={cal:.2f}, dashed)")
        axes[-1][ci].set_xlabel(r"attractiveness weight $\alpha$")
        if ci == 0:
            legend(axes[0][0], ncol=2)

    fig.suptitle("Block B — attractiveness fit, correlation, and distance vs $\\alpha$", y=0.99)
    fig.text(0.5, 0.962, "All four panels: one 50,000-person full-population run (uniform sample size; "
             "dp_carla omitted as identical to DP-circle-R)", ha="center", color="#444444", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "block_B_sweep_combined.pdf"))
    fig.savefig(os.path.join(FIG, "block_B_sweep_combined.png"), dpi=130)
    plt.close(fig)
    print("wrote block_B_sweep_combined.pdf", flush=True)


if __name__ == "__main__":
    main()
