"""anchor_disturb on its AFFECTED subpopulation. The all-persons mean dilutes the regime because
only persons with a free secondary bounded by work feel the jittered anchor. Affected := the oracle
(dp_full) deviation changes between `true` and `anchor_disturb` (so the work-bounded geometry moved).
Report each solver's gap-to-oracle on that subset, true vs disturbed, from the saved per-person raw."""
import numpy as np
import pandas as pd

WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
SOLVERS = ["carla", "dp_carla_refine", "rda"]
BASE = "research/out/block_a"
DIST = "anchor_disturb=1000m"


def _mean_se(x):
    x = x[~np.isnan(x)]
    return (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)


rows = []
for w in WORLDS:
    raw = pd.read_csv(f"{BASE}/{w}/1_gap_raw.csv")
    dev = {s: raw[raw.solver == s].pivot_table(index="unique_person_id", columns="regime", values="dev_m")
           for s in set(SOLVERS) | {"dp_full"}}
    orc = dev["dp_full"]
    affected = (orc[DIST] - orc["true"]).abs() > 0.5
    n, na = len(orc), int(affected.sum())
    for s in SOLVERS:
        gt = (dev[s]["true"] - orc["true"])
        gd = (dev[s][DIST] - orc[DIST])
        # affected subset only
        at = gt[affected].to_numpy(float); ad = gd[affected].to_numpy(float)
        mt, st = _mean_se(at); md, sd = _mean_se(ad)
        rows.append({"world": w, "affected%": round(100 * na / n, 1), "n_aff": na, "solver": s,
                     "gap_true": round(mt, 2), "gap_disturb": round(md, 2),
                     "degradation": round(md - mt, 2), "se_disturb": round(sd, 2)})

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/anchor_subpop.csv", index=False)
print(df.to_string(index=False))
print(f"\nwrote {BASE}/anchor_subpop.csv")
