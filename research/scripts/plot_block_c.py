#!/usr/bin/env python
"""Block C figures (forecasting) -- post-hoc replots of prognosis_<scenario>_<world>.csv, with
bootstrap CI bands read from raw_legs_<scenario>_<world>.parquet.

Per scenario:
  C1  headline      -- forecast outcome vs shock level: true vs the model ladder, with cluster-
                       bootstrap 95% CI bands on the share (attractiveness).
  C2  weight family -- true vs the argmin (dp_full) pot_weight sweep + the calibrated generative:
                       no constant weight reproduces the elasticity shape.
  C3  A/B/C tie     -- one axis, three families on the same runs: gap-to-oracle (A), fit (B),
                       forecast (C). Optimization-perfect, fit-acceptable, forecast-wrong.

Channel: attractiveness -> y = d district share (pp). (Accessibility is reported as a negative
result; no model figure.)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from block_a_style import COL, line, apply_paper_style, WORLD_NAME, LABEL, legend

apply_paper_style()  # seaborn whitegrid + canonical font sizes

GEN_REP = "dp_sample"
ATTRACT = "attractiveness"
_TAG = ""   # anchor-regime tag (fixed/full), set by plot_all so figure names don't clobber


def _name(stem, scenario, world):
    return f"{stem}_{scenario}" + (f"_{_TAG}" if _TAG else "") + f"_{world}"


def _argmin_rep(df):
    return "dp_full" if (df["solver"] == "dp_full").any() else "dp_carla_pot"


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path + ".png", dpi=140); fig.savefig(path + ".pdf")


def _ylab(scenario):
    return ("$\\Delta$ district share of secondary stops (pp)" if scenario == ATTRACT
            else "median secondary-trip distance (m)")


def _yval(sub, scenario):
    s = sub.sort_values("level")
    y = (s["outcome_delta"] * 100.0) if scenario == ATTRACT else s["outcome_main"]
    return s["level"].to_numpy(), y.to_numpy()


def _sel(df, solver, condition):
    return df[(df["solver"] == solver) & (df["condition"] == condition)]


def _argw(df, solver, w):
    """Argmin rows at weight w, selected by the exact condition string (so the truedist control --
    same solver, same alpha=1 -- is never confused with the w1 sweep point)."""
    return _sel(df, solver, f"w{w:g}")


def _cal_weight(df):
    acal = float(df["alpha_cal"].dropna().iloc[0]) if df["alpha_cal"].notna().any() else 1.0
    arg = df[(df["solver"] == _argmin_rep(df)) & df["condition"].astype(str).str.startswith("w")]
    if not len(arg):
        return None
    return float(arg["alpha"].iloc[(arg["alpha"] - acal).abs().argsort()].iloc[0])


# --------------------------------------------------------------------------- #
# cluster (per-person) bootstrap CI on the district share, from the raw legs
# --------------------------------------------------------------------------- #

def _load_raw(out_dir, scenario, world):
    p = os.path.join(out_dir, f"raw_legs_{scenario}" + (f"_{_TAG}" if _TAG else "") + f"_{world}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def _band(raw, solver, condition, alpha, base, levels, nboot=300, seed=0):
    """Per-level 95% CI of (share - base)*100, cluster-bootstrapped over persons. Returns
    {level: (lo_pp, hi_pp)} or {} if raw missing."""
    if raw is None:
        return {}
    rng = np.random.default_rng(seed)
    sub = raw[(raw["solver"] == solver) & (raw["condition"] == condition)]
    if alpha is not None:
        sub = sub[np.isclose(sub["alpha"], alpha)]
    out = {}
    for lv in levels:
        s = sub[np.isclose(sub["level"], lv)]
        if not len(s):
            continue
        g = s.groupby("unique_person_id")["in_district"].agg(["sum", "count"])
        si, ni = g["sum"].to_numpy(float), g["count"].to_numpy(float)
        P = len(si)
        if P < 2:
            continue
        draws = rng.integers(0, P, size=(nboot, P))
        shares = si[draws].sum(1) / ni[draws].sum(1)
        lo, hi = np.percentile(shares, [2.5, 97.5])
        out[lv] = ((lo - base) * 100.0, (hi - base) * 100.0)
    return out


def _draw_band(ax, levels, band, color):
    xs = [lv for lv in levels if lv in band]
    if len(xs) < 2:
        return
    lo = [band[x][0] for x in xs]; hi = [band[x][1] for x in xs]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=1)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def plot_headline(df, scenario, world, out_dir, raw=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    base = float(df["base_main"].dropna().iloc[0])
    levels = sorted(df["level"].unique())
    rep, w = _argmin_rep(df), _cal_weight(df)
    do_band = scenario == ATTRACT and raw is not None

    t = _sel(df, "true", "true").sort_values("level")
    ax.plot(*_yval(t, scenario), color="black", lw=2.6, marker="o", mfc="black", ms=6,
            label="true (DGP)", zorder=6)

    # generative protagonist + flat floor
    plot_set = [(GEN_REP, "informed", "-", "o", f"{LABEL.get(GEN_REP, GEN_REP)}: informed"),
                (GEN_REP, "no_attr", ":", "^", f"{LABEL.get(GEN_REP, GEN_REP)}: no attr.\\ ($\\alpha{{=}}0$)"),
                ("carla_sample", "informed", "-", "d", f"{LABEL.get('carla_sample', 'carla_sample')}: informed")]
    for solver, cond, ls, mk, lab in plot_set:
        s = _sel(df, solver, cond)
        if len(s):
            x, y = _yval(s, scenario)
            c = COL.get(solver, "#888")
            ax.plot(x, y, color=c, ls=ls, marker=mk, ms=5, mfc="none", lw=1.7, label=lab, zorder=4)
            if do_band and cond == "informed" and solver == GEN_REP:
                _draw_band(ax, levels, _band(raw, solver, cond, df[(df.solver == solver) &
                           (df.condition == cond)]["alpha"].iloc[0], base, levels), c)

    if w is not None:                                   # argmin at the calibrated weight
        s = _argw(df, rep, w)
        if len(s):
            x, y = _yval(s, scenario); c = COL.get(rep, "#8c564b")
            ax.plot(x, y, color=c, ls="-.", marker="*", ms=8, mfc="none", lw=1.7,
                    label=f"{LABEL.get(rep, rep)} argmin (w={w:g})", zorder=4)
            if do_band:
                _draw_band(ax, levels, _band(raw, rep, f"w{w:g}", w, base, levels), c)
    tc = _sel(df, rep, "truedist")                      # argmin fed TRUE distances (control)
    if len(tc):
        x, y = _yval(tc, scenario)
        ax.plot(x, y, color="#444", ls="--", marker="x", ms=6, lw=1.4,
                label=f"{LABEL.get(rep, rep)} argmin, true dist.\\ (control)", zorder=3)

    ax.set(xlabel="attractiveness multiplier $b$" if scenario == ATTRACT else "mixed-use $m$",
           ylabel=_ylab(scenario),
           title=f"Attractiveness-shift forecast — {WORLD_NAME.get(world, world)}" if scenario == ATTRACT
                 else f"Mixed-use forecast — {WORLD_NAME.get(world, world)}")
    ax.grid(alpha=0.3); legend(ax, framealpha=0.92, loc="best")
    _save(fig, os.path.join(out_dir, _name("C1_headline", scenario, world))); plt.close(fig)


def plot_weight_family(df, scenario, world, out_dir):
    import matplotlib, matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    rep = _argmin_rep(df)
    t = _sel(df, "true", "true").sort_values("level")
    ax.plot(*_yval(t, scenario), color="black", lw=2.8, marker="o", mfc="black", ms=6,
            label="true (DGP)", zorder=6)
    arg = df[(df["solver"] == rep) & df["condition"].astype(str).str.startswith("w")]
    ws = sorted(arg["alpha"].unique())
    cmap = matplotlib.colormaps["viridis"].resampled(max(len(ws), 2))
    for i, wv in enumerate(ws):
        x, y = _yval(arg[np.isclose(arg["alpha"], wv)], scenario)
        ax.plot(x, y, color=cmap(i), lw=1.5, marker=".", ms=9, label=f"argmin w={wv:g}", zorder=3)
    g = _sel(df, GEN_REP, "informed")
    if len(g):
        ax.plot(*_yval(g, scenario), color="crimson", lw=2.4, ls="--", marker="D", ms=5,
                mfc="none", label=f"{LABEL.get(GEN_REP, GEN_REP)} informed (calibrated)", zorder=5)
    ax.set(xlabel="attractiveness multiplier $b$", ylabel=_ylab(scenario),
           title=f"No constant argmin weight reproduces the shape --- {WORLD_NAME.get(world, world)}")
    ax.grid(alpha=0.3); legend(ax, ncol=2, framealpha=0.92, loc="best")
    _save(fig, os.path.join(out_dir, _name("C2_weightfamily", scenario, world))); plt.close(fig)


def plot_abc_tie(df, scenario, world, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    rep, w = _argmin_rep(df), _cal_weight(df)
    if w is not None:
        oracle = _argw(df, rep, w).set_index("level")["combined_cost"]
        for s in ["carla", "dp_carla", "dp_carla_pot", "dp_carla_refine"]:
            sub = _argw(df, s, w).sort_values("level")
            if len(sub) and s != rep:
                gap = sub["combined_cost"].to_numpy() - oracle.reindex(sub["level"]).to_numpy()
                line(ax[0], sub["level"], gap, s, label=LABEL.get(s, s))
        ax[0].axhline(0, color="k", lw=0.8, ls=":")
    ax[0].set(xlabel="$b$", ylabel=f"objective gap vs {rep}", title="A: optimization (argmin $\\approx$0)")
    for s, cond in [(GEN_REP, "informed"), (rep, f"w{w:g}" if w is not None else None)]:
        sub = _sel(df, s, cond).sort_values("level") if cond else df.iloc[:0]
        if len(sub):
            line(ax[1], sub["level"], sub["pot_decile_tv"], s, label=f"{LABEL.get(s, s)}:{cond}")
    ax[1].set(xlabel="$b$", ylabel="pot\\_decile\\_tv (lower=better)", title="B: fit (reproduce today)")
    t = _sel(df, "true", "true")
    ax[2].plot(*_yval(t, scenario), color="black", lw=2.2, marker="o", mfc="black", ms=5, label="true")
    for s, cond in [(GEN_REP, "informed"), (rep, f"w{w:g}" if w is not None else None)]:
        sub = _sel(df, s, cond) if cond else df.iloc[:0]
        if len(sub):
            line(ax[2], *_yval(sub, scenario), s, label=f"{LABEL.get(s, s)}:{cond}")
    ax[2].set(xlabel="$b$", ylabel=_ylab(scenario), title="C: forecast (predict shift)")
    for a in ax:
        a.grid(alpha=0.3); legend(a)
    _save(fig, os.path.join(out_dir, _name("C3_abc_tie", scenario, world))); plt.close(fig)


def plot_true_curves(dfs, scenario, out_dir):
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
    ax[0].set(xlabel=xl, ylabel="TRUE median secondary distance (m)", title=f"TRUE outcome --- {scenario}")
    ax[1].set(xlabel=xl, ylabel="% trips local (<1.2 km)", title="local-trip share")
    for a in ax:
        a.grid(alpha=0.3); legend(a)
    _save(fig, os.path.join(out_dir, f"DECISION_truecurves_{scenario}")); plt.close(fig)


def plot_distance(df, scenario, world, out_dir):
    """C4 -- distance response to the shock. Left: each model's median achieved trip distance vs b,
    against the TRUE median (does the distance distribution shorten like truth?). Right: Wasserstein
    of each model's distances vs the TRUE CF distances (lower = better; argmin fed stale distances
    fails to track the shortening). The distance-side companion to the share forecast."""
    import matplotlib.pyplot as plt
    rep, w = _argmin_rep(df), _cal_weight(df)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    t = _sel(df, "true", "true").sort_values("level")
    if "true_median_dist_m" in t and t["true_median_dist_m"].notna().any():
        ax[0].plot(t["level"], t["true_median_dist_m"], color="black", lw=2.4, marker="o",
                   mfc="black", ms=6, label="true (DGP)", zorder=6)
    rows = [(GEN_REP, "informed"), (GEN_REP, "no_attr"), ("carla_sample", "informed")]
    if w is not None:
        rows += [(rep, f"w{w:g}"), (rep, "truedist")]
    for s, cond in rows:
        sub = _sel(df, s, cond).sort_values("level")
        if not len(sub):
            continue
        lab = f"{s}:{cond}"
        if "median_achieved_m" in sub:
            ax[0].plot(sub["level"], sub["median_achieved_m"], marker="o", ms=4, lw=1.5,
                       color=COL.get(s, "#444"), label=lab)
        ax[1].plot(sub["level"], sub["dist_w_m"], marker="o", ms=4, lw=1.5,
                   color=COL.get(s, "#444"), label=lab)
    ax[0].set(xlabel="attractiveness multiplier $b$", ylabel="median achieved trip distance (m)",
              title=f"Distance response --- {WORLD_NAME.get(world, world)}")
    ax[1].set(xlabel="attractiveness multiplier $b$", ylabel="distance Wasserstein vs TRUE (m)",
              title="distance fit to the counterfactual")
    for a in ax:
        a.grid(alpha=0.3); legend(a, fontsize=8)
    _save(fig, os.path.join(out_dir, _name("C4_distance", scenario, world))); plt.close(fig)


def plot_all(df, scenario, world, out_dir, tag=""):
    global _TAG
    _TAG = tag
    if isinstance(df, str):
        df = pd.read_csv(df)
    raw = _load_raw(out_dir, scenario, world)
    plot_headline(df, scenario, world, out_dir, raw=raw)
    plot_weight_family(df, scenario, world, out_dir)
    plot_abc_tie(df, scenario, world, out_dir)
    plot_distance(df, scenario, world, out_dir)
    print(f"  [{world}/{scenario}/{tag or 'na'}] wrote C1–C4 figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="gauss_hannover")
    ap.add_argument("--scenario", default="attractiveness")
    ap.add_argument("--tag", default="fixed", help="anchor regime tag (fixed|full)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "out", "block_c"))
    a = ap.parse_args()
    stem = f"prognosis_{a.scenario}" + (f"_{a.tag}" if a.tag else "") + f"_{a.world}.csv"
    plot_all(pd.read_csv(os.path.join(a.out, stem)), a.scenario, a.world, a.out, tag=a.tag)
