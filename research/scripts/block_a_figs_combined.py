"""Combined Block A figures (no re-solving; reads the same CSVs as block_a_figs.py).

Two paper figures that each merge what used to be separate floats:
 - A2_frontier_combined.pdf : the quality--runtime frontier as a 2x3 grid -- top row
   log--log with every method (the RDA visible), bottom row the placement family alone
   on linear axes (RDA dropped so the bottom-left cluster resolves). One column per world.
 - A8_density_length_combined.pdf : density x chain-length for ALL THREE worlds in one
   figure (gap + runtime rows per world, one column per chain length). OSM-Hannover is
   now shown alongside Gauss-Hannover rather than omitted.

Kept in its own module so importing it does NOT trigger block_a_figs.py's module-level
panel() side effects (which regenerate the whole single-figure set)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from block_a_style import COL, line as _line, apply_paper_style, WORLD_NAME, legend  # noqa: F401

apply_paper_style()
TITLE = WORLD_NAME
WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
B = "research/out/block_a"

PLACEMENT = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]  # dp_full = oracle (0)
BASELINES = ["rda", "carla_sample", "dp_sample"]  # off-scale refs (log y). dp_sample = the calibrated fitted-tail model (== Blocks B/C), surfaced under its canonical name via _consolidate_sample
REGIMES = ["true", "dist_noise=0.15", "dist_sampled", "anchor_disturb=1000m", "anchor_remove"]
RLAB = ["true", "noise", "sampled", "anchor\njitter", "anchor\nremove"]


def _consolidate_sample(raw):
    """Report ONE generative dp_sample: the calibrated fitted-tail model (stored as `dp_sample_tuned`
    by the solve harness; the same calibrated model used unnamed in Blocks B/C). Drops the stock-default
    untuned `dp_sample` ablation rows. Falls back to untuned if the tuned rows are absent."""
    if (raw.solver == "dp_sample_tuned").any():
        raw = raw[raw.solver != "dp_sample"].copy()
        raw.loc[raw.solver == "dp_sample_tuned", "solver"] = "dp_sample"
    return raw


def _gap(raw, s, reg):
    """Mean and SE of (solver - oracle) per-person gap in regime reg (mirrors block_a_figs._gap)."""
    orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    if d.empty:
        return (np.nan, np.nan)
    g = (d - orc).to_numpy(float); ok = ~np.isnan(g)
    return (np.nanmean(g), np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan)


def a1_combined():
    """Gap-to-oracle across the difficulty axes, all three worlds (columns). Big-on-top,
    small-on-bottom (matching the frontier figure): the off-scale baselines on log axes
    (top row) and the placement family on linear axes (bottom row, the headline). Merges
    the former two separate floats into one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, w in enumerate(WORLDS):
        raw = _consolidate_sample(pd.read_csv(f"{B}/{w}/1_gap_raw.csv"))
        top, bot = axes[0, j], axes[1, j]
        for s in BASELINES:                              # top: off-scale baselines, log y
            if not (raw.solver == s).any():
                continue
            m = [_gap(raw, s, r)[0] for r in REGIMES]
            _line(top, range(len(REGIMES)), m, s)
        top.set_yscale("log")
        bot.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
        for s in PLACEMENT:                              # bottom: placement family, linear
            m = [_gap(raw, s, r)[0] for r in REGIMES]
            e = [_gap(raw, s, r)[1] for r in REGIMES]
            _line(bot, range(len(REGIMES)), m, s, yerr=e)
        top.set_title(TITLE[w])
        for ax in (top, bot):
            ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(RLAB); ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("metres above oracle / person (log)\nbaselines")
    axes[1, 0].set_ylabel("metres above oracle / person\nplacement family")
    legend(axes[0, -1]); legend(axes[1, -1], ncol=2)
    fig.tight_layout(); fig.savefig(f"{B}/A1_gap_combined.pdf"); plt.close(fig)
    print("wrote A1_gap_combined.pdf")


def _frontier(ax, w, exclude=(), logscale=True):
    """One frontier panel for world w (mirrors block_a_figs.a2)."""
    raw = pd.read_csv(f"{B}/{w}/2_frontier_raw.csv")
    meta = pd.read_csv(f"{B}/{w}/2_frontier_meta.csv")
    canon = (raw.groupby(["solver", "knob", "val"], as_index=False)["dev_m"].mean()
             .merge(meta, on=["solver", "knob", "val"]))
    robp = f"{B}/{w}/2_frontier_robust_meta.csv"
    rob = pd.read_csv(robp) if os.path.exists(robp) else pd.DataFrame(
        columns=["run", "solver", "val", "runtime_s", "mean_dev_m"])
    rob_solvers = set(rob["solver"].unique())
    for s in canon["solver"].drop_duplicates():
        if s in exclude:
            continue
        if s in rob_solvers:
            m = (rob[rob.solver == s].groupby("val", as_index=False)
                 .agg(runtime_s=("runtime_s", "mean"), dev_m=("mean_dev_m", "mean")).sort_values("val"))
            _line(ax, m["runtime_s"], m["dev_m"], s)
        else:
            d = canon[canon.solver == s].sort_values("runtime_s")
            _line(ax, d["runtime_s"], d["dev_m"], s)
    if logscale:
        ax.set_xscale("log"); ax.set_yscale("log")


def a2_combined():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, w in enumerate(WORLDS):
        _frontier(axes[0, j], w, logscale=True)                 # top: all methods, log--log
        _frontier(axes[1, j], w, exclude={"rda"}, logscale=False)  # bottom: placement family, linear
        axes[0, j].set_title(TITLE[w])
        axes[0, j].set_xlabel("runtime [s] (log)")
        axes[1, j].set_xlabel("runtime [s]")
        for i in (0, 1):
            axes[i, j].grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("mean deviation [m] (log)\nall methods")
    axes[1, 0].set_ylabel("mean deviation [m]\nplacement family (linear)")
    legend(axes[0, -1]); legend(axes[1, -1])
    fig.tight_layout(); fig.savefig(f"{B}/A2_frontier_combined.pdf"); plt.close(fig)
    print("wrote A2_frontier_combined.pdf")


def a8_combined():
    lengths = [2, 6, 10]
    sol_rt = ["carla", "carla_sample", "rda", "dp_rings", "dp_carla",
              "dp_rings_refine", "dp_carla_refine", "dp_full"]
    sol_gap = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]  # rda off-scale, dp_full=0
    nrow, ncol = 2 * len(WORLDS), len(lengths)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow), sharex=True)
    for wi, w in enumerate(WORLDS):
        df = pd.read_csv(f"{B}/{w}/8_density_length.csv")
        gr, rr = 2 * wi, 2 * wi + 1                              # gap row, runtime row for this world
        for j, n in enumerate(lengths):
            sub = df[df.n_free == n]
            for s in sol_gap:
                d = sub[sub.solver == s].sort_values("N_per_type")
                if not d.empty:
                    _line(axes[gr, j], d.N_per_type, d.gap_m, s, ms=4)
            for s in sol_rt:
                d = sub[sub.solver == s].sort_values("N_per_type")
                if not d.empty:
                    _line(axes[rr, j], d.N_per_type, d.runtime_ms, s, ms=4)
            axes[gr, j].set_xscale("log"); axes[gr, j].grid(alpha=0.3, which="both")
            axes[rr, j].set_xscale("log"); axes[rr, j].set_yscale("log"); axes[rr, j].grid(alpha=0.3, which="both")
            if wi == 0:
                axes[0, j].set_title(f"chain length n={n}")
            axes[nrow - 1, j].set_xlabel("facilities per type N (log)")
        axes[gr, 0].set_ylabel(f"{TITLE[w]}\ngap above oracle [m]")
        axes[rr, 0].set_ylabel("ms / person (log)")
    legend(axes[0, -1]); legend(axes[1, -1])
    fig.tight_layout(); fig.savefig(f"{B}/A8_density_length_combined.pdf"); plt.close(fig)
    print("wrote A8_density_length_combined.pdf")


if __name__ == "__main__":
    a1_combined()
    a2_combined()
    a8_combined()
    print("DONE")
