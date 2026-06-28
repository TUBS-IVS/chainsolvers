"""A2 frontier ROBUSTNESS runs: re-measure the PLACEMENT-FAMILY frontier solvers (all _KNOBS solvers
EXCEPT rda -- rda is the slow pole and its canonical sweep is reused as-is) several times in one
session, to characterise run-to-run RUNTIME variance (the deviation axis is deterministic -- identical
every run, which is asserted as an internal check). Reconstructs the EXACT seed-0 dist_sampled regime
result_frontier uses (self-check gated on dp_carla@25), so these runs are directly comparable to the
canonical 2_frontier_*.csv -- but they are written to a SEPARATE, append-only file
(2_frontier_robust_meta.csv) and NEVER touch the canonical frontier data. Each invocation appends new
run indices (continues numbering), so repeated calls accumulate samples.

    python research/scripts/a2_robust.py --runs 3
    python research/scripts/a2_robust.py --runs 2 --worlds gauss_hannover   # add more later
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers import run                       # noqa: E402
from chainsolvers.locations import Locations       # noqa: E402
from chainsolvers_eval import survey as S          # noqa: E402

SEED, NPERS = 0, 1000
CHECK = ("dp_carla", "min_candidates", 25)          # existing frontier point used as the consistency gate
SKIP = {"rda"}                                       # rda is the slow pole; reuse its canonical sweep


def _regime(world):
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
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for w in args.worlds:
        world = ba.load(w)
        pl, gt, scored = _regime(world)
        # self-check vs canonical: existing dp_carla@25 deviation must reproduce bit-identically
        cs, ck, cv = CHECK
        chk, _ = _solve_point(world, pl, gt, scored, cs, ck, cv)
        canon = pd.read_csv(f"research/out/block_a/{w}/2_frontier_raw.csv")
        old = canon[(canon.solver == cs) & (canon.knob == ck) & (canon.val == cv)].set_index("unique_person_id")["dev_m"]
        idx = sorted(set(chk.index) & set(old.index))
        a, b = chk.reindex(idx).to_numpy(float), old.reindex(idx).to_numpy(float)
        fin = ~(np.isnan(a) | np.isnan(b)); md = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        ok = md <= 1e-6 and len(idx) == len(old)
        print(f"[{w}] self-check {cs}@{cv}: {'PASS' if ok else 'FAIL'} (max|diff|={md:.2g}m)", flush=True)
        if not ok:
            print(f"[{w}] -> skipped (self-check failed)"); continue

        outp = f"research/out/block_a/{w}/2_frontier_robust_meta.csv"
        prev = pd.read_csv(outp) if os.path.exists(outp) else pd.DataFrame()
        run0 = int(prev["run"].max()) + 1 if len(prev) else 0
        rows = []
        for r in range(run0, run0 + args.runs):
            t_run = time.perf_counter()
            for s, (knob, vals) in ba._KNOBS.items():
                if s in SKIP:
                    continue
                for v in vals:
                    dev, dt = _solve_point(world, pl, gt, scored, s, knob, v)
                    rows.append({"run": r, "solver": s, "knob": knob, "val": v,
                                 "runtime_s": dt, "mean_dev_m": float(dev.mean())})
            print(f"[{w}] run {r} done ({time.perf_counter()-t_run:.0f}s)", flush=True)
        out = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(outp, index=False)
        print(f"[{w}] wrote runs {run0}..{run0+args.runs-1} -> {outp} (total {int(out['run'].max())+1} runs)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
