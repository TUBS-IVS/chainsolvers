"""TENDENCY run: append ONLY carla_sample's runtime to A3 (scaling), A5 (N-wall), A8 (density x length),
re-solving nothing else. A8 gap is computed against the STORED per-person dp_full from
8_density_length_raw.csv (no oracle re-solve). Cells/lengths/levels are read from the existing CSVs so
they match exactly; k follows the result functions' formulas. NOTE: measured under concurrent load ->
absolute timings are a tendency, not a clean baseline; re-run all-solvers-one-pass on a quiet machine
for paper-grade absolutes. Idempotent (drops prior carla_sample rows first).

    python research/scripts/add_carla_sample_runtime.py
"""
import importlib.util, time
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers import run  # noqa: E402
from chainsolvers_eval import survey as S  # noqa: E402

SEED = 0; SOL = "carla_sample"; B = "research/out/block_a"
k_scaling = lambda n: 60 if n <= 4 else 30 if n <= 8 else 12          # result_scaling
k_dl = lambda n: max(20, 60 if n <= 2 else 40 if n <= 6 else 0)        # result_density_length _k


def _time(loc, pl):
    ctx = run.setup(locations_tuple=loc, solver=SOL, rng_seed=SEED)
    t0 = time.perf_counter(); rdf, _, _ = run.solve(ctx=ctx, plans_df=pl)
    return rdf, time.perf_counter() - t0


for w in ["gauss_hannover", "osm_hannover", "two_zone"]:
    world = ba.load(w); full = world.locations_tuple
    # ---- A3 scaling (full locations, vary length) ----
    p = f"{B}/{w}/3_scaling.csv"; df = pd.read_csv(p)
    rows = []
    for n in sorted(df.n_free.unique()):
        pl, _ = ba.synth_chain_plans(world, int(n), k_scaling(int(n)), SEED); npers = pl.unique_person_id.nunique()
        try:
            _, rt = _time(full, pl); rows.append({"n_free": int(n), "solver": SOL, "ms_per_person": 1000 * rt / npers})
        except Exception as e:
            print(f"[{w}] R3 n={n}: {SOL} failed ({e!r})", flush=True)
    df = df[df.solver != SOL]; pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
    print(f"[{w}] R3: appended {len(rows)} carla_sample rows", flush=True)

    # ---- A5 N-wall (subsampled locations, 60 persons) ----
    p = f"{B}/{w}/5_nwall.csv"; df = pd.read_csv(p)
    samp = S.sample_persons(world, 60, seed=SEED); pl = samp.plans_df; npers = pl.unique_person_id.nunique()
    rows = []
    for N in sorted(df.N_per_type.unique()):
        loc = ba._subsample_locations(full, int(N), SEED)
        _, rt = _time(loc, pl); rows.append({"N_per_type": int(N), "solver": SOL, "ms_per_person": 1000 * rt / npers})
    df = df[df.solver != SOL]; pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
    print(f"[{w}] R5: appended {len(rows)} carla_sample rows", flush=True)

    # ---- A8 density x length (synth chains, subsampled loc; gap from stored oracle raw) ----
    p = f"{B}/{w}/8_density_length.csv"; rp = f"{B}/{w}/8_density_length_raw.csv"
    df = pd.read_csv(p); raw = pd.read_csv(rp)
    cells = df[["n_free", "N_per_type"]].drop_duplicates().itertuples(index=False)
    agg, rawrows = [], []
    for n, N in cells:
        n, N = int(n), int(N)
        pl, gt = ba.synth_chain_plans(world, n, k_dl(n), SEED); scored = ba._scored_legs(pl, gt)
        loc = ba._subsample_locations(full, N, SEED); npers = pl.unique_person_id.nunique()
        rdf, rt = _time(loc, pl); dev = ba.per_person_dev(rdf, pl, gt, scored)
        orc = raw[(raw.n_free == n) & (raw.N_per_type == N) & (raw.solver == "dp_full")].set_index("unique_person_id")["dev_m"]
        g = (dev - orc).reindex(orc.index).to_numpy(float)
        agg.append({"n_free": n, "N_per_type": N, "solver": SOL, "runtime_ms": 1000 * rt / npers,
                    "gap_m": float(np.nanmean(g)) if np.isfinite(g).any() else float("nan")})
        rawrows += [{"n_free": n, "N_per_type": N, "solver": SOL, "unique_person_id": pid, "dev_m": float(v)}
                    for pid, v in dev.items()]
    df = df[df.solver != SOL]; pd.concat([df, pd.DataFrame(agg)], ignore_index=True).to_csv(p, index=False)
    raw = raw[raw.solver != SOL]; pd.concat([raw, pd.DataFrame(rawrows)], ignore_index=True).to_csv(rp, index=False)
    print(f"[{w}] R8: appended {len(agg)} carla_sample cells", flush=True)
print("DONE")
