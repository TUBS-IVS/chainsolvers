"""Targeted partial rerun of Block A result-1 after the circle-intersection + resampler fixes.

Only re-solves the (solver, regime) cells those fixes can change, merging into 1_gap_raw.csv; the
rest is bit-identical by CRN. A self-check (re-solve UNAFFECTED cells, assert they match the existing
CSV) runs first -- if it fails, the world is skipped (CSV untouched). This avoids re-solving the
expensive dp_full oracle + vendored RDA on the true/noise/anchor_remove regimes.

Affected cells (why):
  dist_sampled          ALL solvers  -- input changed: facility-aware resampler
  anchor_disturb=1000m  ALL solvers  -- input changed: resampler's RNG draws cascade into it
  dist_noise=0.15       carla/dp_carla/dp_carla_refine -- circle-intersection fix; input unchanged
  anchor_remove         carla/dp_carla/dp_carla_refine -- circle-intersection fix; input unchanged
  true                  none         -- input unchanged + fix doesn't bite (dp_carla==dp_rings)

R2 frontier is rerun separately (block_a.py --results 2). R3-R8 are unaffected (true/synthetic).

    python research/scripts/block_a_rerun_affected.py --write
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
CARLA_FAM = ["carla", "dp_carla", "dp_carla_refine"]
ALL_ORDER = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine",
             "dp_full", "dp_sample", "dp_sample_tuned", "rda"]
SELFCHECK = [("dp_rings", "true"), ("dp_full", "dist_noise=0.15")]  # unaffected -> must match CSV


def _dev(world, pl, gt, solver, scored, extra=None):
    rdf, rt, valid = ba._solve_timed(world, pl, solver, SEED, extra)
    return ba.per_person_dev(rdf, pl, gt, scored=scored), rt, valid


def _match(new: pd.Series, old: pd.Series) -> tuple[bool, str]:
    if set(new.index) != set(old.index):
        return False, f"person-set differs ({len(new)} vs {len(old)})"
    a = new.reindex(sorted(new.index)).to_numpy(float)
    b = old.reindex(sorted(new.index)).to_numpy(float)
    na, nb = np.isnan(a), np.isnan(b)
    if not (na == nb).all():
        return False, "NaN pattern differs"
    fin = ~na
    md = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    return md <= 1e-6, f"max|diff|={md:.3g}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["gauss_hannover", "osm_hannover", "two_zone"])
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for w in args.worlds:
        world = ba.load(w); samp = S.sample_persons(world, args.persons, seed=SEED)
        rng = np.random.default_rng(SEED)
        regimes = {l: (pl, gt) for l, pl, gt in
                   ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=SIGMA)}
        rawp = f"research/out/block_a/{w}/1_gap_raw.csv"; metap = f"research/out/block_a/{w}/1_gap_meta.csv"
        raw = pd.read_csv(rawp)
        present = [s for s in ALL_ORDER if (raw.solver == s).any()]
        calibp = f"research/out/block_a/{w}/dp_sample_tuned_calib.json"
        calib = json.load(open(calibp, encoding="utf-8")) if os.path.exists(calibp) else None
        print(f"\n== {w} ==  solvers present: {present}")

        # --- CRN self-check on UNAFFECTED cells (proves regime reconstruction + merge are valid) ---
        ok_all = True
        for s, reg in SELFCHECK:
            if s not in present:
                continue
            pl, gt = regimes[reg]; scored = ba._scored_legs(pl, gt)
            new, _, _ = _dev(world, pl, gt, s, scored)
            old = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
            ok, msg = _match(new, old)
            print(f"   self-check {s}@{reg}: {'PASS' if ok else 'FAIL'} ({msg})")
            ok_all &= ok
        if not ok_all:
            print(f"   -> self-check FAILED; skipping {w} (CSV untouched)"); continue

        # --- re-solve affected cells ---
        affected = {"dist_sampled": present, "anchor_disturb=1000m": present,
                    "dist_noise=0.15": [s for s in CARLA_FAM if s in present],
                    "anchor_remove": [s for s in CARLA_FAM if s in present]}
        new_raw, new_meta, keys = [], [], set()
        for reg, sols in affected.items():
            pl, gt = regimes[reg]; scored = ba._scored_legs(pl, gt)
            for s in sols:
                extra = calib if s == "dp_sample_tuned" else None
                dev, rt, valid = _dev(world, pl, gt, s, scored, extra)
                new_raw.append(pd.DataFrame({"solver": s, "regime": reg,
                                             "unique_person_id": dev.index, "dev_m": dev.to_numpy(float)}))
                new_meta.append({"solver": s, "regime": reg, "runtime_s": rt, "valid": valid})
                keys.add((s, reg))
                print(f"   re-solved {s:16s}@{reg:20s} raw_dev_mean={np.nanmean(dev.to_numpy(float)):.1f}m "
                      f"(NOT gap; gap=raw-oracle, computed downstream) ({rt:.0f}s)", flush=True)
        if not args.write:
            print(f"   -> dry run, not written"); continue

        # --- merge: drop replaced cells, append fresh ---
        raw["_k"] = list(zip(raw.solver, raw.regime))
        raw = raw[~raw["_k"].isin(keys)].drop(columns="_k")
        pd.concat([raw, *new_raw], ignore_index=True).to_csv(rawp, index=False)
        meta = pd.read_csv(metap); meta["_k"] = list(zip(meta.solver, meta.regime))
        meta = meta[~meta["_k"].isin(keys)].drop(columns="_k")
        pd.concat([meta, pd.DataFrame(new_meta)], ignore_index=True).to_csv(metap, index=False)
        print(f"   -> merged {len(keys)} cells into {rawp}")
    print("\nDONE")


if __name__ == "__main__":
    main()
