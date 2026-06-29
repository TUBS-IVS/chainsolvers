#!/usr/bin/env python
"""Block B: correlation-metric figures from the 50k full-pop run.

(A) block_B_correlations.pdf -- visits~potential association vs alpha for the two samplers on both
    smaller worlds: Pearson r, Spearman, and 1-decile-TV on one axis. Shows every *correlation*
    (rank or linear) peaks to the RIGHT of the decile (calibration) optimum -- they reward
    over-concentration; only decile, anchored to the true distribution, peaks at the calibrated alpha.

(B) block_B_scatter.pdf -- per-facility realized visits vs potential for osm/carla_sample (the
    biggest r-opt vs decile-opt gap: 1.5 vs 5.0), at the decile-opt and the r-opt alpha, with the
    proportional-truth line. At decile-opt the cloud hugs proportionality; at r-opt it over-concentrates
    (steeper, high-leverage) -- higher r, worse mass fit.

    python research/scripts/block_b_corr_figs.py
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from block_a import _load_locations_only
from block_a_style import apply_paper_style, WORLD_NAME

apply_paper_style()
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "block_b_fullpop")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")  # Overleaf clone
OUTB = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")  # scatter is out-only (not a paper fig)
SAMPLERS = ["carla_sample", "dp_sample_capped"]
LABEL = {"carla_sample": "CARLA-sample", "dp_sample_capped": "DP-sample"}


def per_type_counts(raw, ids):
    out = {}
    for t in ids:
        idl = [str(x) for x in ids[t]]
        id2i = {fid: i for i, fid in enumerate(idl)}
        cnt = np.zeros(len(idl), dtype=float)
        vc = raw[raw.to_act_type == t].chosen_loc_id.astype(str).value_counts()
        for fid, c in vc.items():
            i = id2i.get(fid)
            if i is not None:
                cnt[i] += c
        out[t] = (np.asarray(ids[t]), cnt)
    return out


def corr_curves(world):
    """Per-alpha (pearson, spearman) visit-weighted over types, for each sampler. Decile from CSV."""
    W = _load_locations_only(world)
    ids, _, pots = W.locations_tuple
    raw = pd.read_parquet(os.path.join(OUT, f"raw_legs_{world}.parquet"))
    raw = raw[(raw.work == "anchored") & (raw["input"] == "sampled")]
    csv = pd.read_csv(os.path.join(OUT, f"alpha_sweep_{world}.csv"))
    csv = csv[(csv.work == "anchored") & (csv["input"] == "sampled")]
    res = {}
    for s in SAMPLERS:
        r = raw[raw.solver == s]
        rows = []
        for a in sorted(r.alpha.unique()):
            rs, sps, wts = [], [], []
            for t, (_, cnt) in per_type_counts(r[r.alpha == a], ids).items():
                pot = np.asarray(pots[t], float)
                if pot.sum() <= 0 or cnt.sum() == 0 or len(pot) < 3:
                    continue
                w = cnt.sum()
                rs.append(pearsonr(cnt, pot).statistic * w)
                sps.append(spearmanr(cnt, pot).correlation * w)
                wts.append(w)
            tot = sum(wts)
            rows.append((a, sum(rs) / tot, sum(sps) / tot))
        d = pd.DataFrame(rows, columns=["alpha", "pearson", "spearman"])
        dec = csv[csv.solver == s].set_index("alpha")["pot_decile_tv"]
        d["one_minus_decile"] = d.alpha.map(lambda a: 1 - dec.get(a, np.nan))
        res[s] = d
    return res


def fig_correlations(worlds=("gauss_hannover", "osm_hannover")):
    data = {w: corr_curves(w) for w in worlds}
    fig, axes = plt.subplots(len(worlds), len(SAMPLERS), figsize=(11, 8), sharex=True)
    for ri, w in enumerate(worlds):
        for ci, s in enumerate(SAMPLERS):
            ax = axes[ri][ci]
            d = data[w][s]
            ax.plot(d.alpha, d.one_minus_decile, ":d", color="k", ms=5, lw=1.8, label="1 − decile-TV (calibration)")
            ax.plot(d.alpha, d.pearson, "-o", color="#1f77b4", ms=5, lw=1.6, label="Pearson r")
            ax.plot(d.alpha, d.spearman, "-^", color="#2ca02c", ms=5, lw=1.6, label="Spearman ρ")
            # mark the two optima
            a_dec = d.alpha[d.one_minus_decile.idxmax()]
            a_r = d.alpha[d.pearson.idxmax()]
            ax.axvline(a_dec, color="k", ls="--", lw=1, alpha=0.5)
            ax.axvline(a_r, color="#1f77b4", ls="--", lw=1, alpha=0.5)
            ax.set_xscale("symlog", linthresh=0.25, linscale=0.5)
            ax.set_title(f"{WORLD_NAME.get(w, w)} / {LABEL[s]}  "
                         f"(decile-opt α={a_dec:g}, r-opt α={a_r:g})", fontsize=9)
            if ri == len(worlds) - 1:
                ax.set_xlabel("attractiveness weight α")
            if ci == 0:
                ax.set_ylabel("association (higher = better)")
            if ri == 0 and ci == 0:
                ax.legend(loc="lower center")
    fig.suptitle("visits~potential association vs α: every correlation peaks right of the decile optimum")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "block_B_correlations.pdf"))
    fig.savefig(os.path.join(FIG, "block_B_correlations.png"), dpi=130)
    plt.close(fig)
    print("wrote block_B_correlations.pdf")


def fig_scatter(world="osm_hannover", solver="carla_sample", a_dec=1.5, a_r=5.0):
    W = _load_locations_only(world)
    ids, _, pots = W.locations_tuple
    raw = pd.read_parquet(os.path.join(OUT, f"raw_legs_{world}.parquet"))
    raw = raw[(raw.work == "anchored") & (raw["input"] == "sampled") & (raw.solver == solver)]
    csv = pd.read_csv(os.path.join(OUT, f"alpha_sweep_{world}.csv"))
    csv = csv[(csv.work == "anchored") & (csv["input"] == "sampled") & (csv.solver == solver)].set_index("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, a, tag in [(axes[0], a_dec, "decile-optimal"), (axes[1], a_r, "r-optimal")]:
        pv, cv = [], []
        for t, (_, cnt) in per_type_counts(raw[raw.alpha == a], ids).items():
            pot = np.asarray(pots[t], float)
            pv.append(pot); cv.append(cnt)
        pot = np.concatenate(pv); cnt = np.concatenate(cv)
        m = pot > 0
        pot, cnt = pot[m], cnt[m]
        # proportional-truth line: visits ∝ potential, scaled to equal total mass
        scale = cnt.sum() / pot.sum()
        ax.scatter(pot, cnt + 0.5, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
        xs = np.array([pot.min(), pot.max()])
        ax.plot(xs, scale * xs + 0.5, "k--", lw=1.5, label="proportional to potential (truth)")
        ax.set_xscale("log"); ax.set_yscale("log")
        r = pearsonr(cnt, pot).statistic
        dtv = float(csv.loc[a, "pot_decile_tv"])
        ax.set_title(f"α={a:g} ({tag})\nPearson r={r:.2f},  decile-TV={dtv:.3f}", fontsize=9)
        ax.set_xlabel("facility potential (= true visit count)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("realized visits (+0.5, log)")
    fig.suptitle(f"{world.replace('_hannover','')} / {LABEL[solver]}: per-facility visits vs potential — "
                 f"r-opt over-concentrates vs the decile-opt")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTB, "block_B_scatter.pdf"))  # out/block_b, not a paper figure
    fig.savefig(os.path.join(OUTB, "block_B_scatter.png"), dpi=130)
    plt.close(fig)
    print("wrote out/block_b/block_B_scatter.pdf")


if __name__ == "__main__":
    fig_correlations()
    fig_scatter()
