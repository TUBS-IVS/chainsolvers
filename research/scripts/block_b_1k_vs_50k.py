#!/usr/bin/env python
"""DIAGNOSTIC (not a paper figure): overlay the 1k canonical run vs the 50k full-population run on
EVERY metric, for the headline regime (work=anchored, input=sampled), restricted to the solvers
present in the 50k run. Purpose: confirm the *stable* metrics (decile-TV, Wasserstein, deviation,
spearman) agree between 1k and 50k so the paper sweep can keep reporting all solvers from 1k on
those panels, and only treat Pearson r (and the facility-TV floor) as N-sensitive.

Solid + open marker = 1k; dashed + x = 50k; colour = solver. PDF + PNG into out/block_b_fullpop/.

    python research/scripts/block_b_1k_vs_50k.py --p50k <dir>   # default p50k = out/block_b_fullpop
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
from block_a_style import COL, apply_paper_style, WORLD_NAME
from block_b_corr_figs import per_type_counts
from plot_block_b import ORDER

apply_paper_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P1K = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")
WORK, INPUT = "anchored", "sampled"
# (column, label, log-y?). pearson is computed from raw_legs; the rest come from the CSV.
METRICS = [
    ("pot_decile_tv", "decile-TV (headline)", False),
    ("pearson", "Pearson r (visits~pot)", False),
    ("pot_spearman", "Spearman rho", False),
    ("dist_w_m", "Wasserstein-1 (m)", True),
    ("dist_dev_m", "mean |dd| (m)", True),
    ("pot_fit_tv", "facility-TV (de-floors w/ N)", False),
    ("recovery_pct", "recovery %", False),
]


def pearson_curve(raw_path, world):
    W = _load_locations_only(world)
    ids, _, pots = W.locations_tuple
    raw = pd.read_parquet(raw_path)
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
    return pd.DataFrame(rows, columns=["solver", "alpha", "pearson"])


def load(run_dir, world):
    csv = pd.read_csv(os.path.join(run_dir, f"alpha_sweep_{world}.csv"))
    csv = csv[(csv.work == WORK) & (csv["input"] == INPUT)]
    r = pearson_curve(os.path.join(run_dir, f"raw_legs_{world}.parquet"), world)
    return csv.merge(r, on=["solver", "alpha"], how="left")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p50k", default=os.path.join(os.path.dirname(__file__), "..", "out", "block_b_fullpop"))
    p.add_argument("--worlds", default="gauss_hannover,osm_hannover,two_zone")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    out = args.out or args.p50k
    worlds = [w for w in args.worlds.split(",") if w]

    fig, axes = plt.subplots(len(METRICS), len(worlds), figsize=(6.0 * len(worlds), 3.0 * len(METRICS)),
                             sharex=True, squeeze=False)
    for ci, world in enumerate(worlds):
        d1 = load(P1K, world)
        d5 = load(args.p50k, world)
        solv = [s for s in ORDER if s in set(d5.solver.unique())]                 # the 50k overlap set
        cal = float(d1["alpha_cal"].iloc[0])
        for ri, (col, ylab, ylog) in enumerate(METRICS):
            ax = axes[ri][ci]
            for s in solv:
                for d, ls, mk in [(d1, "-", "o"), (d5, "--", "x")]:
                    g = d[(d.solver == s) & d[col].notna()].sort_values("alpha")
                    if g.empty:
                        continue
                    ax.plot(g.alpha, g[col], ls=ls, marker=mk, ms=5, lw=1.4,
                            color=COL.get(s, "k"), mfc="none", mew=1.2)
            ax.set_xscale("symlog", linthresh=0.25, linscale=0.5)
            if ylog:
                ax.set_yscale("log")
            ax.axvline(cal, color="k", ls=":", lw=1, alpha=0.5)
            ax.grid(alpha=0.3, which="both")
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 0:
                ax.set_title(f"{WORLD_NAME.get(world, world)}  (cal $\\alpha$={cal:.2f})")
        axes[-1][ci].set_xlabel(r"$\alpha$")
    # two legends: solver colours, and run linestyles. Solver set = union of 50k solvers over worlds.
    seen = set()
    for w in worlds:
        seen |= set(pd.read_csv(os.path.join(args.p50k, f"alpha_sweep_{w}.csv")).solver.unique())
    all_solv = [s for s in ORDER if s in seen]
    solver_handles = [Line2D([0], [0], color=COL.get(s, "k"), lw=2, marker="o", mfc="none", label=s)
                      for s in all_solv]
    run_handles = [Line2D([0], [0], color="0.3", ls="-", marker="o", mfc="none", label="1k canonical"),
                   Line2D([0], [0], color="0.3", ls="--", marker="x", label="50k full-pop")]
    axes[0][0].legend(handles=solver_handles + run_handles, ncol=2, fontsize=8, loc="best")
    fig.suptitle("1k vs 50k on all metrics (headline regime: anchored / sampled; 50k-overlap solvers) "
                 "— DIAGNOSTIC, not a paper figure", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    base = os.path.join(out, "compare_1k_vs_50k")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)
    print(f"wrote {base}.pdf", flush=True)


if __name__ == "__main__":
    main()
