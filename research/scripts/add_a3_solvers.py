"""Append dp_rings, dp_rings_refine, and dp_full to A3 (chain-length scaling) WITHOUT re-solving the
existing 4 solvers. dp_rings/dp_rings_refine use the same per-length sample as the canonical run
(k = 60/30/12); dp_full (the O(n N^2) oracle) uses a SMALLER sample and is stopped early the moment a
point blows past the time cap or errors out (OOM on a long chain x full catalog) -- where it walls is
the informative part, to be kept or dropped after inspection. Idempotent (drops any prior rows for
these three solvers first). Rewrites research/out/block_a/<world>/3_scaling.csv (gitignored).

    python research/scripts/add_a3_solvers.py
    python research/scripts/add_a3_solvers.py --worlds gauss_hannover
"""
import argparse
import importlib.util
import time

import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers import run  # noqa: E402

SEED, B = 0, "research/out/block_a"
NEW = ["dp_rings", "dp_rings_refine", "dp_full"]
LENGTHS = (1, 2, 3, 4, 6, 8, 10, 12, 14)
CAP_S = 45.0                                              # same cap as result_scaling: blew up -> stop
k_scaling = lambda n: 60 if n <= 4 else 30 if n <= 8 else 12          # canonical (dp_rings/refine match)
k_full = lambda n: min(k_scaling(n), 8)                               # dp_full: smaller, expensive oracle


def _time(world, solver, n, k):
    pl, _ = ba.synth_chain_plans(world, n, k, SEED)
    cls, params = ba._spec(solver)
    ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=SEED, parameters=params)
    t0 = time.perf_counter(); run.solve(ctx=ctx, plans_df=pl); return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    args = ap.parse_args()
    for w in args.worlds:
        world = ba.load(w)
        p = f"{B}/{w}/3_scaling.csv"; df = pd.read_csv(p)
        rows = []
        for solver in NEW:
            kf = k_full if solver == "dp_full" else k_scaling
            for n in LENGTHS:
                k = kf(n)
                try:
                    dt = _time(world, solver, n, k)
                except Exception as e:                    # OOM / error on long chain x full catalog
                    print(f"[{w}] {solver} n={n}: FAILED ({type(e).__name__}) -> stop", flush=True); break
                rows.append({"solver": solver, "n_free": n, "k": k, "dt_s": dt, "ms_per_person": 1000 * dt / k})
                print(f"[{w}] {solver:16s} n={n:2d} k={k:2d}  {1000*dt/k:8.1f} ms/p  (dt {dt:.1f}s)", flush=True)
                if dt > CAP_S:                            # blew past the cap -> stop pushing to longer n
                    print(f"[{w}] {solver} exceeded {CAP_S}s at n={n} -> stop", flush=True); break
        df = df[~df.solver.isin(NEW)]
        pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
        print(f"[{w}] appended {len(rows)} rows for {NEW} -> {p}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
