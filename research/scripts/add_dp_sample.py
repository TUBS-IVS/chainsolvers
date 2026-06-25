"""Backfill the generative dp_sample into Block A result-1 (gap x difficulty) WITHOUT rerunning the
other solvers. Reproduces each world's exact 1000-person sample + 5 regimes (seed 0, sigma 1000 ---
identical rng stream to result_gap_difficulty), runs only dp_sample, and appends its per-person
dev to 1_gap_raw.csv (+ runtime to 1_gap_meta.csv). The gap vs dp_full is then recomputable from
the raw, since the dp_full rows are already there. Idempotent (drops any prior dp_sample rows first).

    python research/scripts/add_dp_sample.py                          # all worlds, n=1000, writes
    python research/scripts/add_dp_sample.py --worlds gauss_hannover --persons 15 --dry   # smoke
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers_eval import survey as S  # noqa: E402

SEED, SIGMA = 0, 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--dry", action="store_true", help="run + report, do not touch the CSVs")
    args = ap.parse_args()

    for w in args.worlds:
        world = ba.load(w)
        samp = S.sample_persons(world, args.persons, seed=SEED)
        rng = np.random.default_rng(SEED)              # same stream as result_gap_difficulty
        regimes = ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=SIGMA)
        raw_rows, meta_rows = [], []
        for label, pl, gt in regimes:
            scored = ba._scored_legs(pl, gt)
            rdf, rt, valid = ba._solve_timed(world, pl, "dp_sample", SEED)
            dev = ba.per_person_dev(rdf, pl, gt, scored)
            raw_rows.append(pd.DataFrame({"solver": "dp_sample", "regime": label,
                                          "unique_person_id": dev.index, "dev_m": dev.to_numpy(float)}))
            meta_rows.append({"solver": "dp_sample", "regime": label, "runtime_s": rt, "valid": valid})
            v = dev.to_numpy(float)
            print(f"[{w}] {label:22s} dp_sample dev mean={np.nanmean(v):8.1f}m  n={np.isfinite(v).sum():4d}  ({rt:.0f}s)", flush=True)
        if args.dry:
            print(f"[{w}] --dry: not written\n", flush=True); continue
        rawp = f"research/out/block_a/{w}/1_gap_raw.csv"
        metap = f"research/out/block_a/{w}/1_gap_meta.csv"
        raw = pd.read_csv(rawp); raw = raw[raw.solver != "dp_sample"]
        pd.concat([raw, *raw_rows], ignore_index=True).to_csv(rawp, index=False)
        meta = pd.read_csv(metap); meta = meta[meta.solver != "dp_sample"]
        pd.concat([meta, pd.DataFrame(meta_rows)], ignore_index=True).to_csv(metap, index=False)
        print(f"[{w}] appended dp_sample -> {rawp}\n", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
