#!/usr/bin/env python
"""Benchmark CARLA vs the exact DP / MILP solvers.

Runs out of the box on synthetic data, or on your own CSVs:

    # synthetic population + scaling sweep
    python scripts/benchmark_solvers.py

    # real data (expects chainsolvers' default column schemas)
    python scripts/benchmark_solvers.py \
        --locations-csv facilities.csv --plans-csv plans.csv \
        --solvers carla dp_carla dp_carla_refine

Locations CSV uses LocationColumns defaults (id, activities, x, y, potentials, name);
`activities`/`potentials` may be ';'-separated for multi-type facilities.
Plans CSV uses PlanColumns defaults (unique_person_id, unique_leg_id, to_act_type,
distance_meters, from_x, from_y, to_x, to_y, ...).

Reported per solver:
  - mean / median total |Δd| (sum of per-leg |observed − model| over a person)
  - wall-clock runtime
  - pairwise win / tie / loss vs the first solver (typically carla)

MILP is an exact oracle and does not scale; keep it to small instances.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from chainsolvers import run, Scorer
from chainsolvers import io
from chainsolvers_eval.synth import (
    generate_world, build_topology, single_chain_plans, topology_locations_tuple,
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

_DP_FAMILY = {"dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_carla_pot", "milp"}  # dp_full ignores min_candidates (uses all)


def _params_for(solver: str, args) -> dict | None:
    """min_candidates only applies to the DP-family solvers (CARLA's config rejects it)."""
    if solver in _DP_FAMILY and args.min_candidates:
        return {"min_candidates": int(args.min_candidates)}
    return None


def per_person_total_dev(rdf: pd.DataFrame) -> pd.Series:
    actual = np.hypot(rdf["to_x"] - rdf["from_x"], rdf["to_y"] - rdf["from_y"])
    dev = (rdf["distance_meters"] - actual).abs()
    return dev.groupby(rdf["unique_person_id"]).sum()


def _alpha_beta(mode: str, pot_weight: float, dist_dev_weight: float):
    """Objective weights the solver actually optimizes, per scoring mode (mirrors
    dp._alpha_beta_from_scorer): higher objective = better."""
    if mode == "potential":
        return pot_weight, 0.0
    if mode == "combined":
        return pot_weight, dist_dev_weight
    return 0.0, dist_dev_weight  # geometric


def per_person_objective(rdf: pd.DataFrame, free_ids: set, alpha: float, beta: float) -> pd.Series:
    """Per-person value of the optimized objective  alpha*sum(P_free) - beta*sum|dd|.
    This is the quantity dp_full maximizes exactly, so %gap to it measures how well a
    solver optimizes its *stated* objective (the right solver-quality metric — recovery
    is incidental)."""
    actual = np.hypot(rdf["to_x"] - rdf["from_x"], rdf["to_y"] - rdf["from_y"])
    dev = (rdf["distance_meters"] - actual).abs()
    obj = -beta * dev.groupby(rdf["unique_person_id"]).sum()
    if alpha and "to_act_potential" in rdf and free_ids:
        fr = rdf[rdf["unique_leg_id"].isin(free_ids)]
        pot = alpha * fr["to_act_potential"].groupby(fr["unique_person_id"]).sum()
        obj = obj.add(pot, fill_value=0.0)
    return obj


def recovery_metrics(rdf: pd.DataFrame, gt: pd.DataFrame):
    """Against ground truth: % of free (placed) activities assigned to the true
    facility, and mean placement error (metres) for those activities."""
    free = gt[gt["to_is_free"]].merge(
        rdf[["unique_leg_id", "to_act_identifier", "to_x", "to_y"]], on="unique_leg_id", how="inner"
    )
    if free.empty:
        return float("nan"), float("nan")
    exact = 100.0 * float((free["to_act_identifier"] == free["true_to_identifier"]).mean())
    err = float(np.hypot(free["to_x"] - free["true_to_x"], free["to_y"] - free["true_to_y"]).mean())
    return exact, err


def realism_metrics(rdf, free_ids, obs_free_dist):
    """Distance-distribution fidelity + chosen attractiveness over free (placed) legs.
    Returns (Wasserstein vs observed (clipped@p95), model median, model p90, mean potential)."""
    r = rdf[rdf.unique_leg_id.isin(free_ids)]
    model_d = np.hypot(r.to_x - r.from_x, r.to_y - r.from_y).to_numpy()
    clip = float(np.quantile(obs_free_dist, 0.95))
    w = wasserstein_distance(np.clip(model_d, 0, clip), np.clip(obs_free_dist, 0, clip))
    pot = r["to_act_potential"].mean() if "to_act_potential" in r else float("nan")
    return w, float(np.median(model_d)), float(np.quantile(model_d, 0.9)), float(pot)


def calibrate_decay(plans_df):
    """Exponential-decay scale per mode = mean observed distance (MLE)."""
    if "mode" not in plans_df:
        return {}, float(plans_df["distance_meters"].mean())
    scales = plans_df.groupby("mode")["distance_meters"].mean().to_dict()
    return {str(k): float(v) for k, v in scales.items()}, float(plans_df["distance_meters"].mean())


def run_solver(solver, locations_tuple, plans_df, mode, seed, params=None,
               pot_weight=1.0, dist_dev_weight=1.0):
    scorer = Scorer(mode=mode, pot_weight=pot_weight, dist_dev_weight=dist_dev_weight)
    ctx = run.setup(locations_tuple=locations_tuple, solver=solver, rng_seed=seed,
                    scorer=scorer, parameters=params or None)
    t0 = time.perf_counter()
    rdf, _, valid = run.solve(ctx=ctx, plans_df=plans_df)
    dt = time.perf_counter() - t0
    return rdf, valid, dt


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_real(locations_csv: str, plans_csv: str):
    ldf = pd.read_csv(locations_csv)
    ids, coords, pots = io.build_locations_payload_from_df(ldf)
    plans_df = pd.read_csv(plans_csv)
    return (ids, coords, pots), plans_df


def population_benchmark(args, rng):
    gt = None
    if args.locations_csv and args.plans_csv:
        loc, plans_df = load_real(args.locations_csv, args.plans_csv)
        print(f"Loaded real data: {plans_df['unique_person_id'].nunique()} persons, "
              f"{len(plans_df)} legs.")
    else:
        world = generate_world(n_locations=args.facilities, n_persons=args.persons,
                               gravity_scale=args.gravity_scale, distance_noise=args.noise, rng=rng)
        loc, plans_df, gt = world.locations_tuple, world.plans_df, world.ground_truth
        print(f"Synthetic world: {world.meta['n_locations']} multi-type locations, "
              f"{world.meta['n_persons']} persons, {world.meta['n_legs']} legs "
              f"({world.meta['n_free_legs']} to place); anchors={world.anchor_types}; "
              f"distance_noise={args.noise}.")

    print(f"\nObjective mode: {args.mode}\n")

    decay_scales, default_scale = calibrate_decay(plans_df)
    free_ids = set(gt.loc[gt.to_is_free, "unique_leg_id"]) if gt is not None else set()
    obs_free_dist = (plans_df.set_index("unique_leg_id").loc[list(free_ids), "distance_meters"].to_numpy()
                     if free_ids else None)

    # Run every requested solver, keeping the result frame for recovery / realism scoring.
    results: Dict[str, Tuple[bool, pd.Series, float, pd.DataFrame]] = {}
    for solver in args.solvers:
        params = _params_for(solver, args) or {}
        if solver == "dp_sample":
            params = {**params, "decay_scales": decay_scales, "default_scale": default_scale}
        try:
            rdf, valid, dt = run_solver(solver, loc, plans_df, args.mode, args.seed,
                                        params or None, args.pot_weight, args.dist_dev_weight)
            results[solver] = (valid, per_person_total_dev(rdf), dt, rdf)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{solver:10s} ERROR: {e}")

    alpha, beta = _alpha_beta(args.mode, args.pot_weight, args.dist_dev_weight)
    objs = {s: per_person_objective(r[3], free_ids, alpha, beta) for s, r in results.items()}

    # Optional TRUE-GLOBAL oracle (dp_full): exact DP over the full candidate set ->
    # exact maximum of the objective, in ANY mode (no pruning, so the potential term is
    # admissible too). %gap is therefore on the objective the solver optimizes.
    opt = None
    if args.oracle:
        if "dp_full" in results:
            opt = objs["dp_full"]
        else:
            try:
                rdf_o, _, _ = run_solver("dp_full", loc, plans_df, args.mode, args.seed,
                                         None, args.pot_weight, args.dist_dev_weight)
                opt = per_person_objective(rdf_o, free_ids, alpha, beta)
            except Exception as e:  # noqa: BLE001
                print(f"Global oracle (dp_full) failed: {e}")

    gap_hdr = f" {'opt%':>6s} {'%gap':>7s}" if opt is not None else ""
    rec_hdr = f" {'recov%':>7s} {'err(m)':>9s}" if gt is not None else ""
    baseline = args.solvers[0]
    header = (f"{'solver':11s} {'valid':6s} {'mean|dd|':>10s} {'time(s)':>9s} "
              f"{'w/t/l vs ' + baseline:>18s}{gap_hdr}{rec_hdr}")
    print(header)
    print("-" * len(header))

    base_dev = results[baseline][1] if baseline in results else None
    for solver in args.solvers:
        if solver not in results:
            continue
        valid, dev, dt, rdf = results[solver]
        if base_dev is None or solver == baseline:
            wtl = ""
        else:
            common = dev.index.intersection(base_dev.index)
            a, b = dev.loc[common], base_dev.loc[common]
            wtl = f"{int((a < b - 1e-9).sum())}/{int((np.abs(a - b) <= 1e-9).sum())}/{int((a > b + 1e-9).sum())}"
        gap_cells = ""
        if opt is not None:
            # objective: higher is better, oracle is the max -> solver_obj <= opt_obj.
            so = objs[solver]
            common = so.index.intersection(opt.index)
            a, o = so.loc[common], opt.loc[common]
            opt_frac = 100.0 * float((a >= o - 1e-6).mean())
            denom = float(np.abs(o).sum())
            pct = (float((o - a).sum()) / denom * 100.0) if denom > 1e-9 else float("nan")
            pct_str = f"{pct:6.2f}%" if pct == pct else "     —"  # — when objective ~ 0
            gap_cells = f" {opt_frac:5.1f}% {pct_str}"
        rec_cells = ""
        if gt is not None:
            rec, err = recovery_metrics(rdf, gt)
            rec_cells = f" {rec:6.1f}% {err:9.1f}"
        print(f"{solver:11s} {str(valid):6s} {dev.mean():10.4f} {dt:9.3f} "
              f"{wtl:>18s}{gap_cells}{rec_cells}")
    if opt is not None:
        print(f"\n  opt% = persons reaching the global optimum of the objective (dp_full, mode={args.mode});")
        print("  %gap = (obj_opt - obj_solver) / |obj_opt|  on alpha*P - beta*dev  (>= 0).")
    if gt is not None:
        print("  recov%/err = DIAGNOSTIC only (true-facility id-match) — NOT the objective; "
              "low under noise is expected, not solver failure.")

    # --- realism table (distance-distribution fidelity + chosen attractiveness) ----
    if obs_free_dist is not None and len(obs_free_dist):
        print(f"\n  REALISM (free legs): observed dist median={np.median(obs_free_dist):.0f} "
              f"p90={np.quantile(obs_free_dist, 0.9):.0f}")
        rhdr = f"{'solver':11s} {'distW':>9s} {'modelMed':>9s} {'modelP90':>9s} {'meanAttr':>9s}"
        print(rhdr)
        print("-" * len(rhdr))
        for solver in args.solvers:
            if solver not in results:
                continue
            w, med, p90, attr = realism_metrics(results[solver][3], free_ids, obs_free_dist)
            print(f"{solver:11s} {w:9.1f} {med:9.0f} {p90:9.0f} {attr:9.3f}")
        print("  distW = Wasserstein-1 to observed free-leg distances (clipped @ p95); lower = more realistic.")
        print("  meanAttr = mean potential of chosen facilities (higher = more attractive).")


def scaling_benchmark(args, rng):
    """Runtime vs chain length on the SAME gravity world as the population benchmark:
    one topology, single chains of increasing length placed by the gravity rule."""
    print("\n=== scaling: single chain, increasing length (gravity world) ===")
    topo = build_topology(n_locations=args.facilities, rng=rng)
    loc = topology_locations_tuple(topo, topo.sizes)
    print(f"{'legs':>5s} " + " ".join(f"{s:>14s}" for s in args.solvers))
    print(f"{'':>5s} " + " ".join(f"{'dev / ms':>14s}" for _ in args.solvers))
    for n_legs in args.scaling_legs:
        df, _ = single_chain_plans(topo, n_legs, gravity_scale=args.gravity_scale,
                                   distance_noise=args.noise, rng=rng)
        cells = []
        for solver in args.solvers:
            try:
                rdf, valid, dt = run_solver(solver, loc, df, args.mode, args.seed, _params_for(solver, args))
                dev = float(per_person_total_dev(rdf).iloc[0])
                cells.append(f"{dev:6.2f}/{dt*1000:6.0f}")
            except Exception:  # noqa: BLE001
                cells.append("    --/--")
        print(f"{n_legs:>5d} " + " ".join(f"{c:>14s}" for c in cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solvers", nargs="+", default=["carla", "dp_rings", "dp_carla", "dp_carla_refine"],
                   help="solvers to compare (first is the win/tie/loss baseline)")
    p.add_argument("--oracle", action="store_true",
                   help="also run the exact MILP oracle and report opt%% / %%gap (small instances only)")
    p.add_argument("--mode", default="geometric", choices=["geometric", "potential", "combined"],
                   help="scoring objective")
    p.add_argument("--persons", type=int, default=300)
    p.add_argument("--facilities", type=int, default=1000,
                   help="total multi-type locations in the synthetic world")
    p.add_argument("--gravity-scale", type=float, default=4000.0,
                   help="distance decay (m) for chain generation in the synthetic world")
    p.add_argument("--noise", type=float, default=0.0,
                   help="relative Gaussian noise on observed leg distances (e.g. 0.15)")
    p.add_argument("--pot-weight", type=float, default=1.0, help="alpha for potential/combined mode")
    p.add_argument("--dist-dev-weight", type=float, default=1.0, help="beta for distance deviation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-candidates", type=int, default=None,
                   help="candidate pool size for DP-family solvers (small values expose recall gaps)")
    p.add_argument("--locations-csv", default=None)
    p.add_argument("--plans-csv", default=None)
    p.add_argument("--scaling-legs", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    p.add_argument("--no-scaling", action="store_true")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    population_benchmark(args, rng)
    if not args.no_scaling and not (args.locations_csv and args.plans_csv):
        scaling_benchmark(args, rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
