"""Add `dp_sample_tuned` to Block A result-1 CSVs WITHOUT re-running the oracle/DP family, with a
world-unchanged self-check.

The gap metric pairs `dp_sample_tuned.dev - dp_full.dev` per person, so the appended rows must be
computed on the SAME world + SAME CRN regimes that produced the rows already in the CSV. To prove
that, we re-solve the UNTUNED `dp_sample` (already present in the CSV) on the replicated regimes and
assert its per-person dev matches the CSV bit-for-bit. A match guarantees:
  (a) the rng/regime replication (seed 0, sigma 1000) is exact, AND
  (b) the baked world is byte-identical to the one the other solver rows were produced on
      (a changed world would shift dp_sample's draws and trip the assert).
Only on PASS are the freshly-solved tuned rows trustworthy and appended.

    python research/scripts/add_dp_sample_tuned.py                       # dry: verify + report only
    python research/scripts/add_dp_sample_tuned.py --write               # append tuned where verified
    python research/scripts/add_dp_sample_tuned.py --worlds two_zone --write
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers_eval import survey as S  # noqa: E402

SEED, SIGMA = 0, 1000.0


def _cached_calib(world, name):
    """dp_sample_tuned calibration is a deterministic one-off per world (MLE over fixed data, no
    per-run randomness), so cache it to JSON and reuse. NOTE: keyed by world NAME -- if a world is
    re-baked, delete research/out/block_a/<name>/dp_sample_tuned_calib.json to force a refit."""
    p = f"research/out/block_a/{name}/dp_sample_tuned_calib.json"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    c = ba._calibrate_dp_sample(world)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
    return c


def _solve_world(world, name, persons):
    """Replicate the CRN regimes and solve untuned (verification) + tuned (the new rows)."""
    samp = S.sample_persons(world, persons, seed=SEED)
    rng = np.random.default_rng(SEED)                  # same stream as result_gap_difficulty
    regimes = ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=SIGMA)
    tuned = _cached_calib(world, name)
    untuned, tuned_raw, tuned_meta = [], [], []
    for label, pl, gt in regimes:
        scored = ba._scored_legs(pl, gt)
        r0, rt0, _ = ba._solve_timed(world, pl, "dp_sample", SEED)          # verification re-solve
        d0 = ba.per_person_dev(r0, pl, gt, scored)
        rT, rtT, vT = ba._solve_timed(world, pl, "dp_sample_tuned", SEED, tuned)
        dT = ba.per_person_dev(rT, pl, gt, scored)
        print(f"    regime {label:22s} untuned={rt0:5.0f}s tuned={rtT:5.0f}s", flush=True)
        untuned.append(pd.DataFrame({"regime": label, "unique_person_id": d0.index,
                                     "dev_m": d0.to_numpy(float)}))
        tuned_raw.append(pd.DataFrame({"solver": "dp_sample_tuned", "regime": label,
                                       "unique_person_id": dT.index, "dev_m": dT.to_numpy(float)}))
        tuned_meta.append({"solver": "dp_sample_tuned", "regime": label, "runtime_s": rtT, "valid": vT})
    return tuned, pd.concat(untuned, ignore_index=True), tuned_raw, tuned_meta


def _check(existing: pd.DataFrame, resolved: pd.DataFrame):
    """Exact nan-aware match of re-solved untuned dp_sample vs the CSV's existing dp_sample rows."""
    if existing.empty:
        return False, "no existing dp_sample rows in CSV to verify against"
    m = existing.merge(resolved, on=["regime", "unique_person_id"], suffixes=("_csv", "_new"))
    if not (len(m) == len(existing) == len(resolved)):
        return False, f"row-set mismatch: csv={len(existing)} new={len(resolved)} merged={len(m)}"
    a, b = m.dev_m_csv.to_numpy(float), m.dev_m_new.to_numpy(float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not (nan_a == nan_b).all():
        return False, f"NaN pattern differs ({int((nan_a != nan_b).sum())} persons)"
    fin = ~nan_a
    maxd = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    return (maxd <= 1e-6), f"max|diff|={maxd:.3g}m over {int(fin.sum())} placed persons ({int(nan_a.sum())} NaN)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--write", action="store_true", help="append tuned rows where verification PASSES")
    args = ap.parse_args()

    for w in args.worlds:
        world = ba.load(w)
        rawp = f"research/out/block_a/{w}/1_gap_raw.csv"
        metap = f"research/out/block_a/{w}/1_gap_meta.csv"
        raw = pd.read_csv(rawp)
        existing = raw[raw.solver == "dp_sample"][["regime", "unique_person_id", "dev_m"]]

        tuned, untuned_now, tuned_raw, tuned_meta = _solve_world(world, w, args.persons)
        ok, msg = _check(existing, untuned_now)
        tmean = pd.concat(tuned_raw).groupby("regime")["dev_m"].mean()
        print(f"[{w}] calib={ {k: round(v, 3) if isinstance(v, float) else v for k, v in tuned.items()} }")
        print(f"[{w}] world-unchanged self-check: {'PASS' if ok else 'FAIL'}  ({msg})")
        print(f"[{w}] tuned mean dev/regime: " +
              "  ".join(f"{r}={tmean[r]:.0f}m" for r in tmean.index))
        if not ok:
            print(f"[{w}] -> NOT writing (verification failed; world or pipeline differs)\n")
            continue
        if not args.write:
            print(f"[{w}] -> dry run, not written\n")
            continue
        raw = raw[raw.solver != "dp_sample_tuned"]                         # idempotent
        pd.concat([raw, *tuned_raw], ignore_index=True).to_csv(rawp, index=False)
        meta = pd.read_csv(metap); meta = meta[meta.solver != "dp_sample_tuned"]
        pd.concat([meta, pd.DataFrame(tuned_meta)], ignore_index=True).to_csv(metap, index=False)
        print(f"[{w}] appended dp_sample_tuned -> {rawp}\n")
    print("DONE")


if __name__ == "__main__":
    main()
