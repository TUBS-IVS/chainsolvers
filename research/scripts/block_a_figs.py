"""Paper-ready Block A figure set, rebuilt entirely from the saved CSVs (no re-solving). Vector PDF,
no figure titles (LaTeX caption carries them). A fixed per-solver colour map keeps each solver the
same colour in every panel/figure. dp_full is dropped from the gap plots (it is the oracle, gap 0)
except A5 where its runtime is the point."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
# TITLE (display names) comes from the shared WORLD_NAME map, set right after the import below.
PLACEMENT = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]  # dp_full = oracle (0), omitted
BASELINES = ["rda", "carla_sample", "dp_sample"]  # non-argmin refs (RDA + generative): carla_sample (constrained, near-feasible) vs dp_sample (joint, off-scale). dp_sample here is the CALIBRATED fitted-tail model -- the same generative model reported in Blocks B/C -- surfaced under its canonical name (see _consolidate_sample); rda_guided broken -> not plotted
REGIMES = ["true", "dist_noise=0.15", "dist_sampled", "anchor_disturb=1000m", "anchor_remove"]
RLAB = ["true", "noise", "sampled", "anchor\njitter", "anchor\nremove"]
B = "research/out/block_a"

# House style (per-solver colour+marker+linestyle, hollow faces) lives in block_a_style so Block A,
# Block B, the robust companions, and any future figure share it. Import; do not redefine.
from block_a_style import STYLE, COL, line as _line, apply_paper_style, WORLD_NAME, legend  # noqa: E402,F401

apply_paper_style()  # seaborn whitegrid + canonical font sizes for the whole figure set
TITLE = WORLD_NAME   # shared display names (consistent across every figure)


def _consolidate_sample(raw):
    """Report ONE generative dp_sample everywhere: the calibrated fitted-tail model (MLE body scale +
    heavy-tail mixture), which the solve harness stores under `dp_sample_tuned`. This is the same
    calibrated generative model used (unnamed) in Blocks B/C; the stock-default untuned `dp_sample`
    rows were only ever an ablation and are dropped here. Falls back gracefully (keeps untuned) if the
    tuned rows are absent."""
    if (raw.solver == "dp_sample_tuned").any():
        raw = raw[raw.solver != "dp_sample"].copy()
        raw.loc[raw.solver == "dp_sample_tuned", "solver"] = "dp_sample"
    return raw


def _gap(raw, s, reg):
    orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    if d.empty:
        return (np.nan, np.nan)
    g = (d - orc).to_numpy(float); ok = ~np.isnan(g)
    return (np.nanmean(g), np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan)


def panel(fn, fname, figsize=(13, 4.2)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, w in zip(axes, WORLDS):
        fn(ax, w)
        ax.set_title(TITLE[w]); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{B}/{fname}"); plt.close(fig)
    print("wrote", fname)


def a1(ax, w):
    raw = pd.read_csv(f"{B}/{w}/1_gap_raw.csv")
    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
    for s in PLACEMENT:
        m = [_gap(raw, s, r)[0] for r in REGIMES]; e = [_gap(raw, s, r)[1] for r in REGIMES]
        _line(ax, range(len(REGIMES)), m, s, yerr=e)
    ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(RLAB)
    if w == WORLDS[0]:
        ax.set_ylabel("metres above oracle / person")
    if w == WORLDS[-1]:
        legend(ax, ncol=2)


def a1b(ax, w):
    raw = _consolidate_sample(pd.read_csv(f"{B}/{w}/1_gap_raw.csv"))
    for s in BASELINES:
        if not (raw.solver == s).any():
            continue
        m = [_gap(raw, s, r)[0] for r in REGIMES]
        _line(ax, range(len(REGIMES)), m, s)
    ax.set_yscale("log"); ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(RLAB)
    if w == WORLDS[0]:
        ax.set_ylabel("metres above oracle / person (log)")
        legend(ax)


def a2(ax, w, exclude=(), logscale=True):
    """Quality--runtime frontier. The placement family uses the mean over the REPEATED robustness runs
    (2_frontier_robust_meta.csv) for runtime (run-to-run spread is small -- reported in the text, not
    drawn). rda (and any solver absent from the robust file) is drawn from the single canonical sweep
    as-is. Deviation is deterministic, so the mean line's y is exact."""
    import os
    raw = pd.read_csv(f"{B}/{w}/2_frontier_raw.csv")
    meta = pd.read_csv(f"{B}/{w}/2_frontier_meta.csv")
    canon = raw.groupby(["solver", "knob", "val"], as_index=False)["dev_m"].mean().merge(meta, on=["solver", "knob", "val"])
    robp = f"{B}/{w}/2_frontier_robust_meta.csv"
    rob = pd.read_csv(robp) if os.path.exists(robp) else pd.DataFrame(columns=["run", "solver", "val", "runtime_s", "mean_dev_m"])
    rob_solvers = set(rob["solver"].unique())
    for s in canon["solver"].drop_duplicates():
        if s in exclude:
            continue
        if s in rob_solvers:                                # robust: mean-over-runs line
            m = rob[rob.solver == s].groupby("val", as_index=False).agg(
                runtime_s=("runtime_s", "mean"), dev_m=("mean_dev_m", "mean")).sort_values("val")
            _line(ax, m["runtime_s"], m["dev_m"], s)
        else:                                               # canonical (rda) as-is
            d = canon[canon.solver == s].sort_values("runtime_s")
            _line(ax, d["runtime_s"], d["dev_m"], s)
    if logscale:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("runtime [s] (log)" if logscale else "runtime [s]")
    if w == WORLDS[0]:
        ax.set_ylabel("mean deviation [m] (log)" if logscale else "mean deviation [m]")
    if w == WORLDS[-1]:
        legend(ax)


def a2_norda(ax, w):  # additional zoomed frontier: rda dropped so the placement family's axes expand
    a2(ax, w, exclude={"rda"})


def a2_norda_lin(ax, w):  # additional: rda-excluded frontier on linear axes
    a2(ax, w, exclude={"rda"}, logscale=False)


def a3(ax, w):
    df = pd.read_csv(f"{B}/{w}/3_scaling.csv")
    for s in df["solver"].drop_duplicates():
        d = df[df.solver == s].sort_values("n_free")
        _line(ax, d["n_free"], d["ms_per_person"], s)
    ax.set_yscale("log"); ax.set_xlabel("chain length (free nodes)")
    if w == WORLDS[0]:
        ax.set_ylabel("ms / person (log)")
    if w == WORLDS[-1]:
        legend(ax)


def a5(ax, w):
    df = pd.read_csv(f"{B}/{w}/5_nwall.csv")
    for s in df["solver"].drop_duplicates():
        d = df[df.solver == s].sort_values("N_per_type")
        _line(ax, d["N_per_type"], d["ms_per_person"], s)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("facilities per type N (log)")
    if w == WORLDS[0]:
        ax.set_ylabel("ms / person (log)")
    if w == WORLDS[-1]:
        legend(ax, ncol=2)


def a6(ax, w):
    df = pd.read_csv(f"{B}/{w}/6_recall.csv").sort_values("min_candidates")
    _line(ax, df["min_candidates"], df["gap_m"], "dp_carla", yerr=df["se"], capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("min_candidates (log)")
    if w == WORLDS[0]:
        ax.set_ylabel("dp_carla gap above oracle [m]")


def a7():  # DEPRECATED: A7 (density-trade) is superseded by A8 (density x length); not called/plotted.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, w in enumerate(WORLDS):
        df = pd.read_csv(f"{B}/{w}/7_density.csv")
        for s in ["carla", "dp_carla", "dp_full"]:
            d = df[df.solver == s].sort_values("N_per_type")
            _line(axes[0, j], d.N_per_type, d.gap_m, s)
            _line(axes[1, j], d.N_per_type, d.runtime_ms, s)
        axes[0, j].set_title(TITLE[w])
        for i in (0, 1):
            axes[i, j].set_xscale("log"); axes[i, j].grid(alpha=0.3, which="both")
        axes[1, j].set_yscale("log"); axes[1, j].set_xlabel("facilities per type N (log)")
    axes[0, 0].set_ylabel("gap above oracle [m]"); axes[1, 0].set_ylabel("ms / person (log)")
    legend(axes[0, 2])
    fig.tight_layout(); fig.savefig(f"{B}/A7_density_trade.pdf"); plt.close(fig); print("wrote A7_density_trade.pdf")


def a8(w):
    df = pd.read_csv(f"{B}/{w}/8_density_length.csv")
    lengths = sorted(df.n_free.unique())
    sol_rt = ["carla", "carla_sample", "rda", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_full"]
    sol_gap = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]  # rda off-scale, dp_full=0
    fig, axes = plt.subplots(2, len(lengths), figsize=(4 * len(lengths), 7), sharex=True)
    for j, n in enumerate(lengths):
        sub = df[df.n_free == n]
        for s in sol_gap:
            d = sub[sub.solver == s].sort_values("N_per_type")
            if not d.empty:
                _line(axes[0, j], d.N_per_type, d.gap_m, s, ms=4)
        for s in sol_rt:
            d = sub[sub.solver == s].sort_values("N_per_type")
            if not d.empty:
                _line(axes[1, j], d.N_per_type, d.runtime_ms, s, ms=4)
        axes[0, j].set_title(f"chain length n={n}")
        for i in (0, 1):
            axes[i, j].set_xscale("log"); axes[i, j].grid(alpha=0.3, which="both")
        axes[1, j].set_yscale("log"); axes[1, j].set_xlabel("facilities per type N (log)")
    axes[0, 0].set_ylabel("gap above oracle [m]\n(placement family)"); axes[1, 0].set_ylabel("ms / person (log)")
    legend(axes[0, -1]); legend(axes[1, -1])
    fig.tight_layout(); fig.savefig(f"{B}/A8_density_length_{w}.pdf"); plt.close(fig)
    print(f"wrote A8_density_length_{w}.pdf")


def a4():
    LAD = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]  # dp_full = oracle (0), omitted
    fig, ax = plt.subplots(figsize=(9, 4.8)); width = 0.8 / len(WORLDS); x = np.arange(len(LAD))
    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
    hatch = {"gauss_hannover": None, "osm_hannover": "//", "two_zone": ".."}
    for wi, w in enumerate(WORLDS):
        raw = pd.read_csv(f"{B}/{w}/4_generation_raw.csv")
        piv = raw.pivot_table(index="unique_person_id", columns="solver", values="dev_m"); orc = piv["dp_full"]
        m, e = [], []
        for s in LAD:
            g = (piv[s] - orc).to_numpy(float); ok = ~np.isnan(g)
            m.append(np.nanmean(g)); e.append(np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan)
        ax.bar(x + wi * width - 0.4 + width / 2, m, width, yerr=e, capsize=2, label=TITLE[w],
               color=[COL[s] for s in LAD], hatch=hatch[w], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(LAD, rotation=20, ha="right"); ax.set_ylabel("metres above oracle")
    # legend: worlds are hatches (bars are solver-coloured)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.7", hatch=hatch[w], edgecolor="white", label=TITLE[w]) for w in WORLDS])
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(f"{B}/A4_generation_cross.pdf"); plt.close(fig); print("wrote A4_generation_cross.pdf")


panel(a1, "A1_gap_difficulty.pdf")
panel(a1b, "A1b_baselines.pdf")
panel(a2, "A2_frontier.pdf")
panel(a2_norda, "A2b_frontier_norda.pdf")  # additional: same frontier, rda excluded (placement family zoom)
panel(a2_norda_lin, "A2c_frontier_norda_lin.pdf")  # additional: rda excluded, linear axes
panel(a3, "A3_scaling.pdf")
a4()
# a7() is DEPRECATED -- A7 density-trade superseded by A8 (density x length). Not generated.
for _w in WORLDS:
    a8(_w)
panel(a5, "A5_nwall.pdf")
# a6 (recall sweep) DEPRECATED -- removed from the paper (dead flat line; result kept as prose). Not generated.
print("DONE")
