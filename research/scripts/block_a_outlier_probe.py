"""Trace the perturbed-regime outlier persons behind the A1 noise/sampled means: how many outliers
per world (top-N gap table), and for the worst one(s) the chain geometry + a per-solver re-solve
showing where the placement blows up. Explains the gauss-vs-osm noise gap and the two_zone-sampled
spike. Reads result CSVs + reconstructs regimes deterministically (seed 0).

    python research/scripts/block_a_outlier_probe.py
"""
import importlib.util

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers import run  # noqa: E402
from chainsolvers_eval import survey as S  # noqa: E402

PLACEMENT = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]
PROBES = [("osm_hannover", "dist_noise=0.15"), ("gauss_hannover", "dist_noise=0.15"),
          ("two_zone", "dist_sampled")]
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def topN(raw, reg, n=8):
    orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    cols = {}
    for s in PLACEMENT:
        d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
        cols[s] = (d - orc)
    G = pd.DataFrame(cols)
    G["MAX"] = G.max(axis=1)
    return G.sort_values("MAX", ascending=False).head(n).round(0)


def resolve_person(world, pl, gt, pid):
    legs = pl[pl.unique_person_id == pid].copy()
    g = gt[gt.unique_leg_id.isin(legs.unique_leg_id)]
    scored = ba._scored_legs(legs, g)
    print(f"\n   --- chain {pid}: {len(legs)} legs ---")
    show = legs[["unique_leg_id", "to_act_type", "mode", "distance_meters", "from_x", "from_y", "to_x", "to_y"]].copy()
    show["free_to"] = show.unique_leg_id.isin(set(g.loc[g.to_is_free, "unique_leg_id"]))
    show["scored"] = show.unique_leg_id.isin(scored)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    # 2-leg / single-intermediate geometry: for each free node bounded by two known endpoints
    for s in ["dp_full", "dp_rings", "dp_carla"]:
        rdf, _, _ = ba._solve_timed(world, legs, s, 0)
        per = ba.per_person_dev(rdf, legs, g, scored=scored)
        sub = rdf[rdf.unique_leg_id.isin(scored)]
        ach = np.hypot(sub.to_x - sub.from_x, sub.to_y - sub.from_y)
        dev = (sub.distance_meters - ach).abs()
        print(f"   {s:9s} total_dev={float(per.get(pid, np.nan)):.0f} m | per-scored-leg |obs-ach|: "
              + " ".join(f"{d:.0f}" for d in dev))


def main():
    for w, reg in PROBES:
        world = ba.load(w); samp = S.sample_persons(world, 1000, seed=0); rng = np.random.default_rng(0)
        regs = {l: (pl, gt) for l, pl, gt in ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=1000.0)}
        pl, gt = regs[reg]
        raw = pd.read_csv(f"research/out/block_a/{w}/1_gap_raw.csv")
        print(f"\n{'='*90}\n{w}  regime={reg}\n{'='*90}")
        tn = topN(raw, reg)
        print("top gap persons (m above oracle):")
        print(tn.to_string())
        # how concentrated: share of the total mean from the #1 person
        orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
        for s in ["carla", "dp_carla", "dp_rings"]:
            d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
            g = (d - orc).dropna()
            top = g.idxmax()
            print(f"   {s}: mean={g.mean():.2f}  median={g.median():.2f}  #1 person {top}={g[top]:.0f}m contributes {g[top]/len(g):.2f}m ({100*g[top]/g.sum():.0f}% of the summed gap)")
        # trace the single worst person
        worst = tn.index[0]
        resolve_person(world, pl, gt, worst)


if __name__ == "__main__":
    main()
