#!/usr/bin/env python
"""Block B paper facet figure (fig:bfacet) from the stable 1k survey-size canonical run
(research/out/block_b): the two_zone potential-mass-decile-TV facet across the four
work x input regimes. Writes PDF (+PNG) into the Overleaf clone paper/figures/. Reads only the
static 1k CSV.

This is the survey-size regime-invariance illustration; the headline sweep (decile-TV, Pearson r,
Wasserstein, deviation) comes from the 50k full-population run via block_b_fig1_combined.py."""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import plot_block_b as P
from block_a_style import WORLD_NAME

B = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")              # 1k canonical (stable)
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")     # Overleaf clone

df = pd.read_csv(os.path.join(B, "alpha_sweep_two_zone.csv"))
cal = float(df["alpha_cal"].iloc[0])
df = P.add_eff_gap(df)
P.facet_metric(df, "pot_decile_tv", "potential-mass-decile fit TV (lower=better)", cal,
               os.path.join(FIG, "block_B_facet_decile_two_zone.png"),
               f"{WORLD_NAME['two_zone']} — attractiveness fit (decile TV) across regimes")
print("wrote block_B_facet_decile_two_zone.pdf")
