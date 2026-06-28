#!/usr/bin/env python
"""Block C figures (both channels) -- post-hoc replots of prognosis_<scenario>_<world>.csv.

Per scenario:
  C1  headline      -- forecast outcome vs shock level: true vs the model ladder.
  C2  weight family -- true vs the argmin (dp_full) pot_weight sweep + the calibrated generative.
                       attractiveness: no constant weight matches the SHAPE; accessibility: argmin is
                       FLAT at every weight (the shortening isn't in its distance inputs).
  C3  A/B/C tie     -- one axis, three families on the same runs: gap-to-oracle (A: argmin ~0),
                       fit (B: pot_decile_tv), forecast (C: outcome). Optimization-perfect,
                       fit-acceptable, forecast-wrong.

Channels: attractiveness -> y = d district share (pp);  accessibility -> y = median secondary dist (m).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from block_a_style import COL, line

GEN_REP = "dp_sample"
ATTRACT = "attractiveness"


def _argmin_rep(df):
    """The exact-argmin reference present in this run: dp_full (attractiveness) else dp_carla_pot
    (accessibility drops dp_full as infeasible -> dp_carla_pot is the near-oracle)."""
    return "dp_full" if (df["solver"] == "dp_full").any() else "dp_carla_pot"


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path + ".png", dpi=130); fig.savefig(path + ".pdf")


def _ylab(scenario):
    return ("$\\Delta$ district share of secondary stops (pp)" if scenario == ATTRACT
            else "median secondary-trip distance (m)")


def _yval(sub, scenario):
    """Headline y per row, sorted by level."""
    s = sub.sort_values("level")
    y = (s["outcome_delta"] * 100.0) if scenario == ATTRACT else s["outcome_main"]
    return s["level"].to_numpy(), y.to_numpy()


def _sel(df, solver, condition):
    return df[(df["solver"] == solver) & (df["condition"] == condition)]


def _cal_weight(df):
    acal = float(df["alpha_cal"].dropna().iloc[0]) if df["alpha_cal"].notna().any() else 1.0
    arg = df[(df["solver"] == _argmin_rep(df)) & df["condition"].astype(str).str.startswith("w")]
    if not len(arg):
        return None
    return float(arg["alpha"].iloc[(arg["alpha"] - acal).abs().argsort()].iloc[0])


def plot_headline(df, scenario, world, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lx, ly = _yval(_sel(df, "true", "true"), scenario)
    ax.plot(lx, ly, color="black", lw=2.4, marker="o", mfc="black", ms=6, label="true (DGP)", zorder=5)

    for cond, (ls, mk) in {"informed": ("-", "o"), "blind": ("--", "s"), "no_attr": (":", "^")}.items():
        s = _sel(df, GEN_REP, cond)
        if len(s):
            x, y = _yval(s, scenario)
            ax.plot(x, y, color=COL.get(GEN_REP, "#bcbd22"), ls=ls, marker=mk, ms=5, mfc="none",
                    lw=1.6, label=f"{GEN_REP}: {cond}")
    cs = _sel(df, "carla_sample", "informed")
    if len(cs):
        x, y = _yval(cs, scenario); line(ax, x, y, "carla_sample", label="carla_sample: informed")
    w = _cal_weight(df)
    rep = _argmin_rep(df)
    if w is not None:
        s = df[(df["solver"] == rep) & (df["alpha"] == w)]
        if len(s):
            x, y = _yval(s, scenario)
            ax.plot(x, y, color=COL.get(rep, "#8c564b"), ls="-.", marker="*", ms=7, mfc="none",
                    lw=1.6, label=f"{rep} argmin (w={w:g})")
    ax.set(xlabel="shock level", ylabel=_ylab(scenario), title=f"Forecasting ({scenario}) -- {world}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, framealpha=0.9)
    _save(fig, os.path.join(out_dir, f"C1_headline_{scenario}_{world}")); plt.close(fig)


def plot_weight_family(df, scenario, world, out_dir):
    import matplotlib, matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    lx, ly = _yval(_sel(df, "true", "true"), scenario)
    ax.plot(lx, ly, color="black", lw=2.6, marker="o", mfc="black", ms=6, label="true (DGP)", zorder=5)
    rep = _argmin_rep(df)
    arg = df[(df["solver"] == rep) & df["condition"].astype(str).str.startswith("w")]
    ws = sorted(arg["alpha"].unique())
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(ws), 2))
    for i, wv in enumerate(ws):
        x, y = _yval(arg[arg["alpha"] == wv], scenario)
        ax.plot(x, y, color=cmap(i), lw=1.4, marker=".", ms=8, label=f"argmin w={wv:g}")
    g = _sel(df, GEN_REP, "informed")
    if len(g):
        x, y = _yval(g, scenario)
        ax.plot(x, y, color="crimson", lw=2.2, ls="--", marker="D", ms=5, mfc="none",
                label=f"{GEN_REP} informed (calibrated)", zorder=4)
    msg = ("no constant weight tracks the shape" if scenario == ATTRACT
           else "argmin flat at every weight -- the future isn't in its inputs")
    ax.set(xlabel="shock level", ylabel=_ylab(scenario), title=f"{msg} -- {world}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, os.path.join(out_dir, f"C2_weightfamily_{scenario}_{world}")); plt.close(fig)


def plot_abc_tie(df, scenario, world, out_dir):
    """A/B/C on one axis (same runs): gap-to-oracle (A), fit pot_decile_tv (B), forecast outcome (C)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    w = _cal_weight(df)
    rep = _argmin_rep(df)

    # A: gap-to-oracle = combined_cost(solver) - combined_cost(rep) at the same (level, weight)
    if w is not None:
        oracle = df[(df["solver"] == rep) & (df["alpha"] == w)].set_index("level")["combined_cost"]
        for s in ["carla", "dp_carla", "dp_carla_pot", "dp_carla_refine"]:
            sub = df[(df["solver"] == s) & (df["alpha"] == w)].sort_values("level")
            if len(sub) and s != rep:
                gap = sub["combined_cost"].to_numpy() - oracle.reindex(sub["level"]).to_numpy()
                line(ax[0], sub["level"], gap, s, label=s)
        ax[0].axhline(0, color="k", lw=0.8, ls=":")
    ax[0].set(xlabel="shock level", ylabel=f"objective gap vs {rep}", title="A: optimization (argmin ~0)")

    # B: fit (pot_decile_tv) -- generative informed vs argmin@cal
    for s, cond, key in [(GEN_REP, "informed", None), (rep, None, w)]:
        sub = (_sel(df, s, cond) if cond else df[(df["solver"] == s) & (df["alpha"] == key)]).sort_values("level")
        if len(sub):
            line(ax[1], sub["level"], sub["pot_decile_tv"], s,
                 label=f"{s}{':'+cond if cond else f' w={key:g}'}")
    ax[1].set(xlabel="shock level", ylabel="pot_decile_tv (lower=better fit)", title="B: fit (reproduce)")

    # C: forecast outcome vs true
    lx, ly = _yval(_sel(df, "true", "true"), scenario)
    ax[2].plot(lx, ly, color="black", lw=2.2, marker="o", mfc="black", ms=5, label="true")
    for s, cond, key in [(GEN_REP, "informed", None), (rep, None, w)]:
        sub = _sel(df, s, cond) if cond else df[(df["solver"] == s) & (df["alpha"] == key)]
        if len(sub):
            x, y = _yval(sub, scenario)
            line(ax[2], x, y, s, label=f"{s}{':'+cond if cond else f' w={key:g}'}")
    ax[2].set(xlabel="shock level", ylabel=_ylab(scenario), title="C: forecast (predict shift)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, f"C3_abc_tie_{scenario}_{world}")); plt.close(fig)


def plot_true_curves(dfs, scenario, out_dir):
    """Decision figure: TRUE accessibility outcome vs level, per world (+ two_zone rural subpop).
    Shows whether the lambda/accessibility signal is strong+monotone (supply-constrained worlds) or
    weak/non-monotone (dense gauss). dfs: {world: true_curves df}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xl = "attractiveness multiplier $b$" if scenario == ATTRACT else "mixed-use intensity $m$"
    for w, d in dfs.items():
        s = d[d["scenario"] == scenario].sort_values("level")
        if not len(s):
            continue
        ax[0].plot(s["level"], s["median_dist_m"], marker="o", lw=1.6, label=w)
        ax[1].plot(s["level"], s["frac_local"] * 100, marker="o", lw=1.6, label=w)
        if "median_dist_rural_m" in s and s["median_dist_rural_m"].notna().any():
            ax[0].plot(s["level"], s["median_dist_rural_m"], marker="s", ls="--", lw=1.4, label=f"{w} (rural)")
            ax[1].plot(s["level"], s["frac_local_rural"] * 100, marker="s", ls="--", lw=1.4, label=f"{w} (rural)")
    ax[0].set(xlabel=xl, ylabel="TRUE median secondary distance (m)", title=f"TRUE outcome -- {scenario}")
    ax[1].set(xlabel=xl, ylabel="% trips local (<1.2 km)", title="local-trip share")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, f"DECISION_truecurves_{scenario}"))
    plt.close(fig)
    print(f"  [{scenario}] wrote DECISION true-curve figure -> {out_dir}", flush=True)


def plot_all(df, scenario, world, out_dir):
    if isinstance(df, str):
        df = pd.read_csv(df)
    plot_headline(df, scenario, world, out_dir)
    plot_weight_family(df, scenario, world, out_dir)
    plot_abc_tie(df, scenario, world, out_dir)
    print(f"  [{world}/{scenario}] wrote C1/C2/C3 figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="gauss_hannover")
    ap.add_argument("--scenario", default="attractiveness")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "out", "block_c"))
    a = ap.parse_args()
    plot_all(pd.read_csv(os.path.join(a.out, f"prognosis_{a.scenario}_{a.world}.csv")),
             a.scenario, a.world, a.out)
