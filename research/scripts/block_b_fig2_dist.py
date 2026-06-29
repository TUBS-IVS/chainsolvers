#!/usr/bin/env python
"""Block B Fig 2: free-leg distance distributions, FOUR panels across -- gauss_hannover (2) then
two_zone (2). For each world:
  left  -- at the calibrated alpha: true (DGP) vs argmin (dp_carla, fed the sampled distances ->
           reproduces them), the generative sampler (dp_sample, makes its own), and the distance-
           blind gravity floor. Shows what the Wasserstein numbers mean and why the argmin "hugs"
           truth (it is fed the distances it is scored against).
  right -- the generative sampler alone as alpha grows (0 / calibrated / large): over-weighting
           attractiveness pulls trips onto the few nearest-attractive facilities and shifts the whole
           distribution short of truth -- the visible cause of the rising-Wasserstein panel.

Sourced from the 1k canonical run (out/block_b) -- the only run carrying plain dp_carla + dp_sample;
distribution shape is N-robust, so 1k is fine for an illustration.

    python research/scripts/block_b_fig2_dist.py
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from block_a_style import COL, apply_paper_style, WORLD_NAME

apply_paper_style()
import matplotlib.pyplot as plt

B = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")            # 1k canonical
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")  # Overleaf clone
WORLDS = ["gauss_hannover", "two_zone"]


def draw_world(axL, axR, world):
    r = pd.read_parquet(os.path.join(B, f"raw_legs_{world}.parquet"))
    r = r[r.work == "anchored"].copy()
    samp = r[r["input"] == "sampled"]
    cal = float(pd.read_csv(os.path.join(B, f"alpha_sweep_{world}.csv"))["alpha_cal"].iloc[0])
    grid = np.sort(samp["alpha"].unique())
    a_cal = float(grid[np.argmin(np.abs(grid - cal))])             # snapshot at on-calibration grid point
    a_hi = float(grid[np.argmin(np.abs(grid - 20.0))])             # "large alpha" grid point

    true_d = r[(r["input"] == "true") & (r.solver == "dp_carla")]["observed_dist_m"].to_numpy(float)
    true_d = true_d[np.isfinite(true_d)]
    hi = np.percentile(true_d, 97)                                 # common x-range per world (scales differ)
    xs = np.linspace(0, hi, 400)

    def kde(d):
        d = d[np.isfinite(d) & (d <= hi)]
        return gaussian_kde(d)(xs)

    def ach(s, a):
        return samp[(samp.solver == s) & (samp.alpha == a)]["achieved_dist_m"].to_numpy(float)

    wn = WORLD_NAME.get(world, world)
    # left: solver snapshot at calibrated alpha
    axL.fill_between(xs, kde(true_d), color="0.6", alpha=0.5, label="true (DGP)")
    for s, lab in [("dp_carla", "argmin (fed distances)"), ("dp_sample", "generative sampler"),
                   ("gravity_independent", "gravity floor")]:
        axL.plot(xs, kde(ach(s, a_cal)), color=COL.get(s, "k"), lw=1.8, label=lab)
    axL.set_title(f"{wn} — at calibrated $\\alpha\\approx${cal:.2f}")
    axL.set_xlabel("free-leg distance (m)")
    axL.legend()
    axL.grid(alpha=0.3)

    # right: generative sampler as alpha grows
    axR.fill_between(xs, kde(true_d), color="0.6", alpha=0.5, label="true (DGP)")
    for a_, ls, c in [(0.0, ":", "#4575b4"), (a_cal, "-", COL["dp_sample"]), (a_hi, "--", "#d73027")]:
        axR.plot(xs, kde(ach("dp_sample", a_)), lw=1.9, ls=ls, color=c, label=f"dp_sample, $\\alpha$={a_:g}")
    axR.set_title(f"{wn} — sampler vs $\\alpha$")
    axR.set_xlabel("free-leg distance (m)")
    axR.legend()
    axR.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    draw_world(axes[0], axes[1], WORLDS[0])
    draw_world(axes[2], axes[3], WORLDS[1])
    axes[0].set_ylabel("density")
    fig.suptitle("Free-leg distance distributions: argmin reproduces fed distances; the generative "
                 "sampler is governed by $\\alpha$", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "block_B_distances.pdf"))
    fig.savefig(os.path.join(FIG, "block_B_distances.png"), dpi=130)
    plt.close(fig)
    print("wrote block_B_distances.pdf (4 panels: gauss + two_zone)", flush=True)


if __name__ == "__main__":
    main()
