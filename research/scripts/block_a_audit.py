"""Canonical Block A number audit -- the single reproducible source of truth for every quantitative
Block A claim in the paper. Recomputes all headline numbers FROM the persisted result CSVs (no
re-solving): gap-to-oracle x difficulty, generation/search decomposition, proven-optimality %, the
recall geometry-limit floor, chain-length scaling, the N-wall, the frontier ranges, and the
anchor-quality subpopulation degradation. Writes a human-readable audit to
research/out/block_a/AUDIT.txt and the raw per-person anchor-subpopulation deltas to
research/out/block_a/<world>/anchor_subpop_raw.csv.

The ONLY thing not stored in the CSVs is which persons are in the work-bounded subpopulation that the
anchor-disturb axis affects; that set is reconstructed *deterministically* (seed 0, same rng stream as
result_gap_difficulty -> identical 1000-person sample + regimes), then the per-person deviations are
read from 1_gap_raw.csv. So this whole audit is reproducible from the committed CSVs + this script.

    python research/scripts/block_a_audit.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("block_a", "research/scripts/block_a.py")
ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
from chainsolvers_eval import survey as S  # noqa: E402

WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
B = "research/out/block_a"                 # working result CSVs (gitignored, regenerable)
AUDIT_DIR = "research/block_a_audit"       # TRACKED frozen snapshot (canonical paper numbers + key raw)
SEED, SIGMA = 0, 1000.0
REGIMES = ["true", "dist_noise=0.15", "dist_sampled", "anchor_disturb=1000m", "anchor_remove"]
PLACEMENT = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine"]
OUT = []


def p(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.append(line)


def gap_series(raw, s, reg):
    orc = raw[(raw.solver == "dp_full") & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    d = raw[(raw.solver == s) & (raw.regime == reg)].set_index("unique_person_id")["dev_m"]
    if d.empty:
        return pd.Series(dtype=float)
    return (d - orc).reindex(orc.index)


def main():
    for w in WORLDS:
        raw = pd.read_csv(f"{B}/{w}/1_gap_raw.csv")
        sols = set(raw.solver.unique())
        p(f"\n{'='*78}\n{w}\n{'='*78}")

        p("\n[R1] gap-to-oracle x difficulty (mean m above dp_full, +- SE):")
        for reg in REGIMES:
            cells = []
            for s in PLACEMENT + ["rda", "carla_sample", "dp_sample", "dp_sample_tuned"]:
                if s not in sols:
                    continue
                g = gap_series(raw, s, reg).to_numpy(float); ok = ~np.isnan(g)
                if ok.sum() == 0:
                    continue
                se = np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else float("nan")
                cells.append(f"{s}={np.nanmean(g):.2f}+-{se:.2f}")
            p(f"   {reg:20s} " + "  ".join(cells))

        # proven optimality on true regime
        p("\n[R1] proven optimality on `true` (% persons with gap < 0.5 m):")
        for s in ["carla", "dp_carla", "dp_carla_refine"]:
            if s in sols:
                g = gap_series(raw, s, "true").to_numpy(float); g = g[~np.isnan(g)]
                p(f"   {s:18s} {100*np.mean(g < 0.5):.1f}%  (mean gap {g.mean():.2f} m)")

        # generation/search decomposition (true)
        p("\n[R4] generation/search decomposition (true regime, mean m above oracle):")
        for s in PLACEMENT:
            if s in sols:
                p(f"   {s:18s} {np.nanmean(gap_series(raw, s, 'true')):.2f}")

        # recall floor
        f6 = f"{B}/{w}/6_recall.csv"
        if os.path.exists(f6):
            r6 = pd.read_csv(f6)
            p(f"\n[R6] recall floor (dp_carla gap vs min_candidates): "
              f"flat at {r6.gap_m.mean():.2f} m over K={list(r6.min_candidates)} (spread {r6.gap_m.max()-r6.gap_m.min():.3f})")

        # scaling
        f3 = f"{B}/{w}/3_scaling.csv"
        if os.path.exists(f3):
            s3 = pd.read_csv(f3)
            all_n = sorted(s3.n_free.unique())
            carla = s3[s3.solver == "carla"].set_index("n_free")["ms_per_person"]
            dpc = s3[s3.solver == "dp_carla"].set_index("n_free")["ms_per_person"]
            # CARLA "fails" where it has no finite runtime (row absent OR NaN) beyond some length
            cdone = [n for n in all_n if n in carla.index and pd.notna(carla[n])]
            cfail = [n for n in all_n if n > max(cdone) or (n in carla.index and pd.isna(carla[n]))]
            p(f"\n[R3] scaling: carla n=8 -> {carla.get(8, float('nan')):.0f} ms; carla max completed n={max(cdone)}, "
              f"fails (no return) at n>= {min(cfail) if cfail else 'none'}; dp_carla runs to n={max(dpc.index)} ({dpc.get(max(dpc.index)):.0f} ms)")

        # n-wall
        f5 = f"{B}/{w}/5_nwall.csv"
        if os.path.exists(f5):
            s5 = pd.read_csv(f5)
            dpf = s5[s5.solver == "dp_full"].set_index("N_per_type")["ms_per_person"]
            rda = s5[s5.solver == "rda"].set_index("N_per_type")["ms_per_person"]
            p(f"\n[R5] N-wall: dp_full {dpf.index.min()}->{dpf.index.max()} = {dpf.iloc[0]:.0f}->{dpf.iloc[-1]:.0f} ms; "
              f"rda flat ~{rda.mean():.0f} ms")

        # density x length (A8, paper figure)
        f8 = f"{B}/{w}/8_density_length.csv"
        if os.path.exists(f8):
            s8 = pd.read_csv(f8)
            p("\n[R8] density x length (gap-to-oracle [m], mean over N; runtime climb across the N sweep):")
            for n in sorted(s8.n_free.unique()):
                sub = s8[s8.n_free == n]
                gaps = {sv: sub[sub.solver == sv].gap_m.mean() for sv in
                        ["carla", "dp_carla", "dp_carla_refine", "dp_rings_refine"] if (sub.solver == sv).any()}
                dpc = sub[sub.solver == "dp_carla"].sort_values("N_per_type")["runtime_ms"]
                dpf8 = sub[sub.solver == "dp_full"].sort_values("N_per_type")["runtime_ms"]
                climb = f"dp_carla rt {dpc.iloc[0]:.0f}->{dpc.iloc[-1]:.0f}ms ({dpc.iloc[-1]/max(dpc.iloc[0],1e-9):.0f}x)" if len(dpc) else ""
                wall = f"dp_full rt {dpf8.iloc[0]:.0f}->{dpf8.iloc[-1]:.0f}ms" if len(dpf8) else ""
                p(f"   n={n:2d}  " + "  ".join(f"{k}={v:.1f}" for k, v in gaps.items()) + f"   {climb}  {wall}")

        # frontier
        f2 = f"{B}/{w}/2_frontier_raw.csv"
        if os.path.exists(f2):
            fr = pd.read_csv(f2).merge(pd.read_csv(f"{B}/{w}/2_frontier_meta.csv"), on=["solver", "knob", "val"])
            agg = fr.groupby(["solver", "val"], as_index=False).agg(dev=("dev_m", "mean"), rt=("runtime_s", "first"))
            for s in ["carla", "dp_carla", "rda"]:
                d = agg[agg.solver == s]
                if not d.empty:
                    p(f"[R2] frontier {s:10s} runtime[{d.rt.min():.2f}-{d.rt.max():.2f}]s dev[{d.dev.min():.0f}-{d.dev.max():.0f}]m")

        # anchor-quality subpopulation (reconstruct affected set deterministically; read dev from CSV)
        world = ba.load(w); samp = S.sample_persons(world, 1000, seed=SEED); rng = np.random.default_rng(SEED)
        regs = {l: (pl, gt) for l, pl, gt in ba.make_regimes(samp.plans_df, samp.ground_truth, world, rng, anchor_sigma=SIGMA)}
        pl_t, gt = regs["true"]; pl_d, _ = regs["anchor_disturb=1000m"]; scored = ba._scored_legs(pl_t, gt)
        kt = pl_t.set_index("unique_leg_id")[["from_x", "from_y", "to_x", "to_y"]]
        kd = pl_d.set_index("unique_leg_id")[["from_x", "from_y", "to_x", "to_y"]]
        comm = kt.index.intersection(kd.index)
        chg = [l for l in comm[((kt.loc[comm] - kd.loc[comm]).abs() > 1e-6).any(axis=1)] if l in scored]
        aff = sorted(set(pl_t.set_index("unique_leg_id")["unique_person_id"].loc[chg].unique()))
        n_placed = raw[raw.regime == "true"].unique_person_id.nunique()
        p(f"\n[R1-anchor] work-bounded affected subpop: {len(aff)} persons ({100*len(aff)/n_placed:.1f}% of placed)")
        rows = []
        for s in ["dp_full", "carla", "rda"]:
            if s not in sols:
                continue
            dt = raw[(raw.solver == s) & (raw.regime == "true")].set_index("unique_person_id")["dev_m"]
            dd = raw[(raw.solver == s) & (raw.regime == "anchor_disturb=1000m")].set_index("unique_person_id")["dev_m"]
            ot = raw[(raw.solver == "dp_full") & (raw.regime == "true")].set_index("unique_person_id")["dev_m"]
            od = raw[(raw.solver == "dp_full") & (raw.regime == "anchor_disturb=1000m")].set_index("unique_person_id")["dev_m"]
            idx = [x for x in aff if x in dt.index and x in dd.index]
            gap_t = (dt.reindex(idx) - ot.reindex(idx)); gap_d = (dd.reindex(idx) - od.reindex(idx))
            p(f"   {s:8s} gap-above-oracle: true={gap_t.mean():.1f} -> disturb={gap_d.mean():.1f}  DEGRADATION=+{gap_d.mean()-gap_t.mean():.0f} m"
              f"   (raw dev: true {dt.reindex(idx).mean():.0f} -> disturb {dd.reindex(idx).mean():.0f})")
            for x in idx:
                rows.append({"solver": s, "person": x, "dev_true": dt.get(x), "dev_disturb": dd.get(x),
                             "gap_true": gap_t.get(x), "gap_disturb": gap_d.get(x)})
        os.makedirs(AUDIT_DIR, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{AUDIT_DIR}/{w}_anchor_subpop_raw.csv", index=False)
        OUT.append(f"   -> raw per-person anchor-subpop deltas: {AUDIT_DIR}/{w}_anchor_subpop_raw.csv")
        # snapshot the cached calibration into the tracked audit dir (frozen record)
        calib = f"{B}/{w}/dp_sample_tuned_calib.json"
        if os.path.exists(calib):
            import shutil
            shutil.copy(calib, f"{AUDIT_DIR}/{w}_dp_sample_tuned_calib.json")

    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(f"{AUDIT_DIR}/AUDIT.txt", "w", encoding="utf-8") as f:
        f.write("Block A canonical numbers -- regenerate with: python research/scripts/block_a_audit.py\n")
        f.write("(reads research/out/block_a/<world>/*.csv; affected subpop reconstructed at seed 0)\n")
        f.write("\n".join(OUT) + "\n")
    print(f"\nwrote {AUDIT_DIR}/AUDIT.txt")


if __name__ == "__main__":
    main()
