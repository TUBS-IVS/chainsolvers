#!/usr/bin/env python
"""Illustrative distance distributions behind the Block B Wasserstein panel.

Left  : at the calibrated alpha, the realized free-leg distance distribution for the true DGP, an
        argmin solver (fed the sampled distances -> reproduces them), the generative sampler
        (produces its own), and the distance-blind gravity floor. Shows what the Wasserstein numbers
        mean and why the argmin "hugs" truth -- it is fed the distances it is scored against.
Right : the generative sampler alone as alpha grows (0 / calibrated / large). Over-weighting
        attractiveness pulls trips onto the few nearest-attractive facilities, shifting the whole
        distribution short of truth -- the visible cause of the rising-Wasserstein middle panel.

    python research/scripts/plot_block_b_dist.py            # gauss_hannover, anchored/sampled
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from block_a_style import COL, apply_paper_style

apply_paper_style()  # seaborn whitegrid + canonical font sizes

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")


def main(argv=None):
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", default="gauss_hannover")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--alpha-cal", type=float, default=None,
                   help="snapshot alpha; default = sweep grid point nearest the world's calibrated alpha")
    args = p.parse_args(argv)

    r = pd.read_parquet(os.path.join(args.out, f"raw_legs_{args.world}.parquet"))
    r = r[(r.work == "anchored")].copy()
    samp = r[r["input"] == "sampled"]

    # snapshot at the grid point nearest the world's calibrated alpha (so the panel is on-calibration)
    cal = float(pd.read_csv(os.path.join(args.out, f"alpha_sweep_{args.world}.csv"))["alpha_cal"].iloc[0])
    grid = np.sort(samp["alpha"].unique())
    if args.alpha_cal is None:
        args.alpha_cal = float(grid[np.argmin(np.abs(grid - cal))])

    # true reference = distances fed at input=true (the argmin reproduces them exactly there)
    true_d = r[(r["input"] == "true") & (r.solver == "dp_carla")]["observed_dist_m"].to_numpy(float)
    true_d = true_d[np.isfinite(true_d)]
    hi = np.percentile(true_d, 97)                       # common x-range for legibility
    xs = np.linspace(0, hi, 400)

    def kde(d):
        d = d[np.isfinite(d) & (d <= hi)]                # drop the >p97 tail (range, not clip pile-up)
        return gaussian_kde(d)(xs)

    def ach(s, a):
        return samp[(samp.solver == s) & (samp.alpha == a)]["achieved_dist_m"].to_numpy(float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True)

    # -- left: solver snapshot at calibrated alpha --
    axL.fill_between(xs, kde(true_d), color="0.6", alpha=0.5, label="true (DGP)")
    a = args.alpha_cal
    for s, lab in [("dp_carla", "argmin (fed distances)"), ("dp_sample", "generative sampler"),
                   ("gravity_independent", "gravity floor")]:
        axL.plot(xs, kde(ach(s, a)), color=COL.get(s, "k"), lw=1.8, label=lab)
    axL.set_title(f"distance distribution near calibrated α≈{cal:.2f} (grid α={a:g})")
    axL.set_xlabel("free-leg distance (m)"); axL.set_ylabel("density")
    axL.legend(); axL.grid(alpha=0.3)

    # -- right: generative sampler as alpha grows --
    axR.fill_between(xs, kde(true_d), color="0.6", alpha=0.5, label="true (DGP)")
    for a_, ls, c in [(0.0, ":", "#4575b4"), (args.alpha_cal, "-", COL["dp_sample"]),
                      (20.0, "--", "#d73027")]:
        axR.plot(xs, kde(ach("dp_sample", a_)), lw=1.9, ls=ls, color=c, label=f"dp_sample, α={a_:g}")
    axR.set_title("generative sampler vs α (over-weighting shifts it short)")
    axR.set_xlabel("free-leg distance (m)")
    axR.legend(); axR.grid(alpha=0.3)

    fig.tight_layout()
    base = os.path.join(args.out, f"dist_illustration_{args.world}")
    fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
