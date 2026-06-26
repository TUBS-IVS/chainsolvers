"""Robust companions to A1 (gap x difficulty), to show the noise/sampled means are heavy-tail /
single-outlier driven (SE approxeq mean is the tell). Two figures, same layout as A1:

  A1_median.pdf            -- per-person gap MEDIAN (+- bootstrap 95% CI) instead of mean.
  A1_mean_excl_outlier.pdf -- MEAN, but the single worst-gap person per world dropped (same person
                              dropped for every solver/regime, so it stays a fair comparison).

Also prints, per world, who was dropped and the per-(solver,regime) mean before/after, so the
outlier's reach across solvers is explicit. Reads research/out/block_a/<world>/1_gap_raw.csv only.

    python research/scripts/block_a_figs_robust.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
TITLE = {"gauss_hannover": "Gauss-Hannover", "osm_hannover": "OSM-Hannover", "two_zone": "Two-zone"}
PLACEMENT = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]
REGIMES = ["true", "dist_noise=0.15", "dist_sampled", "anchor_disturb=1000m", "anchor_remove"]
RLAB = ["true", "noise", "sampled", "anchor\njitter", "anchor\nremove"]
COL = {"carla": "#1f77b4", "dp_rings": "#ff7f0e", "dp_carla": "#2ca02c",
       "dp_rings_refine": "#d62728", "dp_carla_refine": "#9467bd"}
B = "research/out/block_a"
RNG = np.random.default_rng(0)


def gaps(raw, s, reg):
    orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    return (d - orc).reindex(orc.index)  # per-person gap (NaN where unplaced)


def boot_med_ci(x, n=1000):
    x = x[~np.isnan(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    meds = np.median(RNG.choice(x, size=(n, x.size), replace=True), axis=1)
    return (np.percentile(meds, 2.5), np.percentile(meds, 97.5))


def worst_person(raw):
    """Person with the single largest placement-solver gap over all regimes (the spike to drop)."""
    best_pid, best_val = None, -np.inf
    for s in PLACEMENT:
        for reg in REGIMES:
            g = gaps(raw, s, reg)
            if g.notna().any():
                pid = g.idxmax()
                if g[pid] > best_val:
                    best_val, best_pid = g[pid], pid
    return best_pid, best_val


def main():
    fig_m, ax_m = plt.subplots(1, 3, figsize=(13, 4.2))
    fig_e, ax_e = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, w in enumerate(WORLDS):
        raw = pd.read_csv(f"{B}/{w}/1_gap_raw.csv")
        pid, val = worst_person(raw)
        print(f"\n== {w} ==  worst person dropped: {pid} (max placement gap {val:.0f} m)")
        print(f"   {'solver':16s} " + "  ".join(f"{r:>16s}" for r in RLAB))
        for s in PLACEMENT:
            mean_all, mean_excl, med, ci = [], [], [], []
            for reg in REGIMES:
                g = gaps(raw, s, reg)
                mean_all.append(np.nanmean(g.to_numpy(float)))
                mean_excl.append(np.nanmean(g.drop(index=[pid], errors="ignore").to_numpy(float)))
                gv = g.to_numpy(float); med.append(np.nanmedian(gv)); ci.append(boot_med_ci(gv))
            lo = [m - c[0] for m, c in zip(med, ci)]; hi = [c[1] - m for m, c in zip(med, ci)]
            ax_m[j].errorbar(range(5), med, yerr=[lo, hi], marker="o", ms=4, capsize=2, color=COL[s], label=s)
            ax_e[j].plot(range(5), mean_excl, marker="o", ms=4, color=COL[s], label=s)
            chg = "  ".join(f"{a:6.1f}->{b:5.1f}" for a, b in zip(mean_all, mean_excl))
            print(f"   {s:16s} mean all->excl: {chg}")
        for ax in (ax_m[j], ax_e[j]):
            ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
            ax.set_title(TITLE[w]); ax.grid(alpha=0.3); ax.set_xticks(range(5)); ax.set_xticklabels(RLAB, fontsize=8)
        ax_m[j].annotate(f"drop {pid}", (0.5, 0.95), xycoords="axes fraction", fontsize=6, ha="center", va="top")
    ax_m[0].set_ylabel("MEDIAN metres above oracle / person"); ax_e[0].set_ylabel("MEAN metres above oracle\n(worst person dropped)")
    ax_m[-1].legend(fontsize=7, ncol=2); ax_e[-1].legend(fontsize=7, ncol=2)
    fig_m.tight_layout(); fig_m.savefig(f"{B}/A1_median.pdf"); plt.close(fig_m)
    fig_e.tight_layout(); fig_e.savefig(f"{B}/A1_mean_excl_outlier.pdf"); plt.close(fig_e)
    print(f"\nwrote {B}/A1_median.pdf and {B}/A1_mean_excl_outlier.pdf")


if __name__ == "__main__":
    main()
