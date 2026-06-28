"""Append ONLY dp_rings + dp_rings_refine to A2 (frontier), re-solving nothing else. Reconstructs the
EXACT dist_sampled regime result_frontier uses (seed 0, 1000 persons, feasible resample) and merges the
two new solvers into the existing per-world 2_frontier_raw.csv + 2_frontier_meta.csv -- the clean points
for carla/dp_carla/dp_carla_refine/rda (measured 2026-06-26) are kept UNTOUCHED. Gated by a self-check:
an existing frontier point (dp_carla @ min_candidates=25) must reproduce bit-identically from the
reconstructed regime, proving the append is consistent. NOTE: the two new solvers' runtime is measured
under current load -> a tendency relative to the 06-26 absolutes; deviation is deterministic.
Idempotent (drops any prior dp_rings/dp_rings_refine rows first).

    python research/scripts/add_rings_frontier.py            # dry: verify self-check + report
    python research/scripts/add_rings_frontier.py --write
"""
import argparse
import importlib.util
import time

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers import run                       # noqa: E402
from chainsolvers.locations import Locations       # noqa: E402
from chainsolvers_eval import survey as S          # noqa: E402

SEED, NPERS = 0, 1000
NEW = ["dp_rings", "dp_rings_refine"]
CHECK = ("dp_carla", "min_candidates", 25)          # existing frontier point used as the consistency gate


def _regime(world):
    """Reconstruct result_frontier's dist_sampled regime EXACTLY (same seed/sequence)."""
    w = S.sample_persons(world, NPERS, seed=SEED)
    rng = np.random.default_rng(SEED)
    samples = S.per_mode_distance_samples(world.plans_df, world.ground_truth)
    pl = S.resample_distances(w.plans_df, w.ground_truth, samples, rng, feasible=True,
                              locations=Locations(*world.locations_tuple))
    return pl, w.ground_truth, ba._scored_legs(pl, w.ground_truth)


def _solve_point(world, pl, gt, scored, solver, knob, val):
    cls, params = ba._frontier_spec(solver, knob, val)
    ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=SEED, parameters=params)
    t0 = time.perf_counter(); rdf, _, _ = run.solve(ctx=ctx, plans_df=pl); dt = time.perf_counter() - t0
    return ba.per_person_dev(rdf, pl, gt, scored), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for w in args.worlds:
        world = ba.load(w)
        pl, gt, scored = _regime(world)
        rawp = f"research/out/block_a/{w}/2_frontier_raw.csv"
        metap = f"research/out/block_a/{w}/2_frontier_meta.csv"
        raw = pd.read_csv(rawp); meta = pd.read_csv(metap)

        # self-check: existing dp_carla@25 must reproduce bit-identically from the reconstructed regime
        cs, ck, cv = CHECK
        chk, _ = _solve_point(world, pl, gt, scored, cs, ck, cv)
        old = raw[(raw.solver == cs) & (raw.knob == ck) & (raw.val == cv)].set_index("unique_person_id")["dev_m"]
        idx = sorted(set(chk.index) & set(old.index))
        a, b = chk.reindex(idx).to_numpy(float), old.reindex(idx).to_numpy(float)
        fin = ~(np.isnan(a) | np.isnan(b)); md = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        ok = md <= 1e-6 and len(idx) == len(old)
        print(f"[{w}] self-check {cs}@{cv}: {'PASS' if ok else 'FAIL'} "
              f"(max|diff|={md:.2g}m, n={len(idx)}/{len(old)})", flush=True)
        if not ok:
            print(f"[{w}] -> skipped (self-check failed)"); continue

        rows, metarows = [], []
        for s in NEW:
            knob, vals = ba._KNOBS[s]
            for v in vals:
                dev, dt = _solve_point(world, pl, gt, scored, s, knob, v)
                rows.append(pd.DataFrame({"solver": s, "knob": knob, "val": v,
                                          "unique_person_id": np.asarray(dev.index),
                                          "dev_m": dev.to_numpy(float)}))
                metarows.append({"solver": s, "knob": knob, "val": v, "runtime_s": dt})
                print(f"   {s:16s} {knob}={v:<4} dev_mean={float(dev.mean()):.1f}m  rt={dt:.2f}s", flush=True)
        if not args.write:
            print(f"[{w}] dry run, not written"); continue
        raw = raw[~raw.solver.isin(NEW)]
        pd.concat([raw, *rows], ignore_index=True).to_csv(rawp, index=False)
        meta = meta[~meta.solver.isin(NEW)]
        pd.concat([meta, pd.DataFrame(metarows)], ignore_index=True).to_csv(metap, index=False)
        print(f"[{w}] appended {NEW} -> {rawp}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
