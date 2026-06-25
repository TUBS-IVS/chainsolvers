"""Block A.4 cross-world: the search/generation decomposition for all worlds on one axis, from the
saved per-person raw (4_generation_raw.csv). Shows search collapses to ~0 everywhere while the
generation gap (pruned DP -> dp_full) binds only on the dispersed two_zone world."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
LADDER = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_full"]
BASE = "research/out/block_a"


def world_gaps(w):
    raw = pd.read_csv(f"{BASE}/{w}/4_generation_raw.csv")
    piv = raw.pivot_table(index="unique_person_id", columns="solver", values="dev_m")
    orc = piv["dp_full"]
    out = {}
    for s in LADDER:
        if s not in piv:
            continue
        g = (piv[s] - orc).to_numpy(float); ok = ~np.isnan(g)
        out[s] = (float(np.nanmean(g)),
                  float(np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan)
    return out


data = {w: world_gaps(w) for w in WORLDS}
fig, ax = plt.subplots(figsize=(9, 4.8))
width = 0.8 / len(WORLDS)
x = np.arange(len(LADDER))
for wi, w in enumerate(WORLDS):
    means = [data[w].get(s, (np.nan, np.nan))[0] for s in LADDER]
    ses = [data[w].get(s, (np.nan, np.nan))[1] for s in LADDER]
    ax.bar(x + wi * width - 0.4 + width / 2, means, width, yerr=ses, capsize=2, label=w)
ax.set_xticks(x); ax.set_xticklabels(LADDER, rotation=20, ha="right")
ax.set_ylabel("metres above oracle")
ax.set_title("Block A.4 (cross-world) — search solved everywhere; generation binds only on two_zone")
ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(f"{BASE}/4_generation_cross.png", dpi=130); plt.close(fig)
print("wrote", f"{BASE}/4_generation_cross.png")
for w in WORLDS:
    print(f"{w:16s}", {s: round(data[w][s][0], 2) for s in data[w]})
