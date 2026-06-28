"""Append carla_sample (CARLA greedy-ancestral generative sampler) to Block A result-1, as an
off-scale reference in A1b alongside dp_sample. Unlike dp_sample (unconstrained joint MNL sampler,
~km deviation), carla_sample samples among CARLA's geometric candidates and stays near-feasible
(~tens of m) -- a distinct generative point worth showing. Solves only carla_sample on the replicated
CRN regimes (seed 0) and appends to 1_gap_raw.csv (+ meta); gated by a self-check that an existing
unaffected solver still reproduces, so the append is provably consistent.

    python research/scripts/add_carla_sample.py            # dry: verify + report
    python research/scripts/add_carla_sample.py --write
"""
import argparse, importlib.util
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers_eval import survey as S  # noqa: E402

SEED, SIGMA, SOLVER = 0, 1000.0, "carla_sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for w in args.worlds:
        world = ba.load(w); samp = S.sample_persons(world, args.persons, seed=SEED)
        rng = np.random.default_rng(SEED)
        regimes = ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=SIGMA)
        rawp = f"research/out/block_a/{w}/1_gap_raw.csv"; metap = f"research/out/block_a/{w}/1_gap_raw.csv".replace("1_gap_raw", "1_gap_meta")
        raw = pd.read_csv(rawp)
        # self-check: dp_rings@true (unaffected) must reproduce -> regime/CRN replication is exact
        plt, gtt = regimes[0][1], regimes[0][2]; sc = ba._scored_legs(plt, gtt)
        chk = ba.per_person_dev(ba._solve_timed(world, plt, "dp_rings", SEED)[0], plt, gtt, scored=sc)
        old = raw[(raw.solver == "dp_rings") & (raw.regime == "true")].set_index("unique_person_id")["dev_m"]
        idx = sorted(set(chk.index) & set(old.index))
        a, b = chk.reindex(idx).to_numpy(float), old.reindex(idx).to_numpy(float)
        fin = ~(np.isnan(a) | np.isnan(b)); md = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        ok = md <= 1e-6 and len(idx) == len(old)
        print(f"[{w}] self-check dp_rings@true: {'PASS' if ok else 'FAIL'} (max|diff|={md:.2g}m)")
        if not ok:
            print(f"[{w}] -> skipped (self-check failed)"); continue
        rows, meta = [], []
        for label, pl, gt in regimes:
            scored = ba._scored_legs(pl, gt)
            rdf, rt, valid = ba._solve_timed(world, pl, SOLVER, SEED)
            dev = ba.per_person_dev(rdf, pl, gt, scored)
            rows.append(pd.DataFrame({"solver": SOLVER, "regime": label,
                                      "unique_person_id": dev.index, "dev_m": dev.to_numpy(float)}))
            meta.append({"solver": SOLVER, "regime": label, "runtime_s": rt, "valid": valid})
            print(f"   {label:22s} carla_sample raw_dev_mean={np.nanmean(dev.to_numpy(float)):.0f} m "
                  f"(gap vs oracle computed downstream from the CSV)", flush=True)
        if not args.write:
            print(f"[{w}] dry run, not written"); continue
        raw = raw[raw.solver != SOLVER]
        pd.concat([raw, *rows], ignore_index=True).to_csv(rawp, index=False)
        m = pd.read_csv(metap); m = m[m.solver != SOLVER]
        pd.concat([m, pd.DataFrame(meta)], ignore_index=True).to_csv(metap, index=False)
        print(f"[{w}] appended carla_sample -> {rawp}")
    print("DONE")


if __name__ == "__main__":
    main()
