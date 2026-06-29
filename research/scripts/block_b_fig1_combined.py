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

Mixed-source rule (validated by block_b_1k_vs_50k.py): decile-TV and deviation are N-stable, so they
carry EVERY solver; the 50k-overlap solvers use the 50k run, the 1k-only solvers (dp_full, dp_sample,
rda, rda_guided, ...) use the 1k canonical run. Pearson r and Wasserstein are N-sensitive (r is
attenuated at 1k; the sampler's Wasserstein level shifts), so those two panels carry ONLY the
solvers present in the 50k run, sourced from 50k. The 50k solver set is auto-detected, so the figure
upgrades itself (more solvers on r/Wasserstein) as the appends land.

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
from block_a_style import line, apply_paper_style, WORLD_NAME
from block_b_corr_figs import per_type_counts
from plot_block_b import ORDER

apply_paper_style()
import matplotlib.pyplot as plt

B1K = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")           # 1k canonical
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")   # Overleaf clone
WORLDS = ["osm_hannover", "two_zone"]
WORK, INPUT = "anchored", "sampled"                                             # survey-realistic regime
PRELIM = True


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

    fig, axes = plt.subplots(4, len(WORLDS), figsize=(11, 12.5), sharex=True)
    for ci, world in enumerate(WORLDS):
        c1 = load_csv(B1K, world)
        c5 = load_csv(args.p50k, world)
        s50 = set(c5.solver.unique())                                  # solvers that have a 50k run
        cal = float(c5["alpha_cal"].iloc[0])                           # 50k calibration (headline solvers)
        rcurve = pearson_curve(args.p50k, world)

        # row 0: decile-TV (ALL solvers; 50k where available, else 1k)  -- N-stable
        ax = axes[0][ci]
        for s in ORDER:
            src = c5 if s in s50 else c1
            d = src[(src.solver == s) & src["pot_decile_tv"].notna()].sort_values("alpha")
            if not d.empty:
                line(ax, d.alpha.to_numpy(), d["pot_decile_tv"].to_numpy(), s, ms=5)
        if ci == 0:
            ax.set_ylabel("attractiveness fit\n(potential-decile TV, lower=better)")
        ax.set_title(f"{WORLD_NAME.get(world, world)}  (calibrated $\\alpha$={cal:.2f}, dashed)")
        ax.legend(ncol=2) if ci == 0 else None

        # row 1: Pearson r (50k solvers only, from 50k)  -- N-sensitive
        ax = axes[1][ci]
        for s in ORDER:
            if s not in s50:
                continue
            d = rcurve[(rcurve.solver == s) & rcurve["pearson"].notna()].sort_values("alpha")
            if not d.empty:
                line(ax, d.alpha.to_numpy(), d["pearson"].to_numpy(), s, ms=5)
        if ci == 0:
            ax.set_ylabel("visits~potential\nPearson r (higher=more concentrated)")

        # row 2: Wasserstein (50k solvers only, from 50k)  -- N-sensitive for the sampler
        ax = axes[2][ci]
        for s in ORDER:
            if s not in s50:
                continue
            d = c5[(c5.solver == s) & c5["dist_w_m"].notna()].sort_values("alpha")
            if not d.empty:
                line(ax, d.alpha.to_numpy(), d["dist_w_m"].to_numpy(), s, ms=5)
        if ci == 0:
            ax.set_ylabel("distance shape\n(Wasserstein-1, m)")

        # row 3: per-person deviation (ALL solvers; 50k where available, else 1k)  -- N-stable
        ax = axes[3][ci]
        for s in ORDER:
            src = c5 if s in s50 else c1
            d = src[(src.solver == s) & src["dist_dev_m"].notna()].sort_values("alpha")
            if not d.empty:
                line(ax, d.alpha.to_numpy(), d["dist_dev_m"].to_numpy(), s, ms=5)
        ax.set_yscale("log")
        if ci == 0:
            ax.set_ylabel(r"per-person deviation" "\n" r"(mean $|\Delta d|$, m)")
        ax.set_xlabel(r"attractiveness weight $\alpha$")

        for ri in range(4):
            a = axes[ri][ci]
            a.set_xscale("symlog", linthresh=0.25, linscale=0.5)
            a.axvline(cal, color="k", ls="--", lw=1, alpha=0.6)
            a.grid(alpha=0.3, which="both")

    fig.suptitle("Block B — attractiveness fit, correlation, and distance vs $\\alpha$", y=0.985)
    if PRELIM:
        fig.text(0.5, 0.958, "Rows 1 & 4: all solvers (decile-TV & deviation are N-stable). "
                 "Rows 2 & 3 (r, Wasserstein): 50k-run solvers only.",
                 ha="center", color="#444444", fontsize=9)
        fig.text(0.5, 0.5, "PRELIMINARY", fontsize=70, color="gray", alpha=0.08,
                 ha="center", va="center", rotation=30, zorder=0)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(os.path.join(FIG, "block_B_sweep_combined.pdf"))
    fig.savefig(os.path.join(FIG, "block_B_sweep_combined.png"), dpi=130)
    plt.close(fig)
    print("wrote block_B_sweep_combined.pdf", flush=True)


if __name__ == "__main__":
    main()
