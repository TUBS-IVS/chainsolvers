#!/usr/bin/env python
"""Extra Block B paper figures from the stable 1k canonical run (research/out/block_b):
completes the three-world headline trio (adds two_zone 3-panel) and the two_zone regime
facets for the non-decile metrics (Wasserstein, Spearman, deviation). Writes PDF (+PNG) into
the Overleaf clone paper/figures/. Run anytime; reads only the static 1k CSVs."""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import plot_block_b as P
from block_a_style import WORLD_NAME

B = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")              # 1k canonical (stable)
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")     # Overleaf clone


def load(w):
    df = pd.read_csv(os.path.join(B, f"alpha_sweep_{w}.csv"))
    return P.add_eff_gap(df), float(df["alpha_cal"].iloc[0])


# three-panel sweeps for the non-gauss worlds, via the SAME paper_figure as gauss (fig:bsweep) so
# all three sweep figures are identical in layout (one legend on the top panel, matching y-labels,
# "World / work / input" suptitle) -- not plot_combo, which repeats the legend on every panel.
for w_key, out_name in [("two_zone", "block_B_sweep3_two_zone"), ("osm_hannover", "block_B_sweep3_osm")]:
    dfw, calw = load(w_key)
    P.paper_figure(dfw, calw, os.path.join(FIG, f"{out_name}.pdf"), world=w_key)
    print(f"wrote {out_name}.pdf")

# two_zone regime facets (decile is the headline; the other three are the supporting metrics)
df, cal = load("two_zone")
for col, ylab, name in [
    ("pot_decile_tv", "potential-mass-decile fit TV (lower=better)", "decile"),
    ("dist_w_m", "distance-fit Wasserstein m (shape)", "distw"),
    ("pot_spearman", "visits~potential Spearman (higher=better)", "spearman"),
    ("dist_dev_m", "mean distance deviation m (absolute)", "distdev"),
]:
    P.facet_metric(df, col, ylab, cal, os.path.join(FIG, f"block_B_facet_{name}_two_zone.png"),
                   f"{WORLD_NAME['two_zone']} — {ylab}")
    print(f"wrote block_B_facet_{name}_two_zone.pdf")
print("done")
