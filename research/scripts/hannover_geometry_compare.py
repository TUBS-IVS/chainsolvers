#!/usr/bin/env python
"""Run the paper's solver-quality test on BOTH Hannover worlds and check the conclusions
hold in the same direction.

Two worlds, same MiD-2023 chains / decay / mode-split — they differ ONLY in geometry &
attractiveness:

  * GAUSS  = `synth.city_world("hannover")` — clustered Gaussian blobs, density-calibrated
             to the measured OSM counts (the cheap, controllable synthetic world).
  * REAL   = `osm.hannover_osm_world()`     — real OSM facility coordinates + tag-derived
             attractiveness weights (the strongest decoupling of geometry from any solver).

ONE fixed Gauss instance (fixed seed) is used — not averaged over seeds. The real world's
geometry is fixed by the cached snapshot; its chains use the same fixed seed.

For every solver in the registry (minus the `carla_plus` stub) it reports the headline
metrics from `benchmark_solvers.py`:

  * %gap   — (obj_oracle - obj_solver)/|obj_oracle| on the optimized objective (oracle =
             dp_full, the true-global DP). #1 metric: how well the solver optimizes its
             stated objective. Lower = better, >=0.
  * opt%   — fraction of persons reaching the global optimum.
  * recov% / err — DIAGNOSTIC: true-facility id-match and placement error vs ground truth.
  * distW / med  — realism: Wasserstein-1 to the observed free-leg distances + model median.
  * time   — wall clock.

The question is NOT "are the numbers identical" (geometry differs) but "do the same
*directions* hold": same solver ranking by %gap, same combined-vs-geometric recovery lift
under noise, same realism ordering.

    uv run python research/scripts/hannover_geometry_compare.py --persons 250 --noise 0.1 \
        --modes geometric combined --pot-weight 400
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from chainsolvers import Scorer, run
from chainsolvers_eval.synth import city_world
from chainsolvers_eval.osm import DEFAULT_HANNOVER_POIS, hannover_osm_world
from chainsolvers_eval.worlds import load_world
from chainsolvers_eval.survey import sample_persons

# reuse the paper's metric definitions verbatim so this matches benchmark_solvers.py
from benchmark_solvers import (_alpha_beta, calibrate_decay, per_person_objective,
                               per_person_total_dev, realism_metrics, recovery_metrics)

ALL_SOLVERS = ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine",
               "dp_carla_pot", "milp", "dp_sample", "dp_full"]
_DP_FAMILY = {"dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_carla_pot", "milp"}


def add_noise(plans_df: pd.DataFrame, gt: pd.DataFrame, noise: float, rng) -> pd.DataFrame:
    """Relative Gaussian noise on free-leg observed distances (the ambiguity that makes
    usage-based potentials matter)."""
    if noise <= 0:
        return plans_df
    free = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    out = plans_df.copy()
    d = out["distance_meters"].to_numpy(float).copy()
    ids = out["unique_leg_id"].to_numpy()
    mask = np.array([i in free for i in ids])
    d[mask] = np.maximum(1.0, d[mask] * (1.0 + rng.normal(0, noise, size=mask.sum())))
    out["distance_meters"] = d
    return out


def run_world(name: str, world, *, solvers, mode, noise, pot_weight, dist_dev_weight,
              min_candidates: Optional[int], seed: int):
    loc, plans_df, gt = world.locations_tuple, world.plans_df, world.ground_truth
    plans_df = add_noise(plans_df, gt, noise, np.random.default_rng(seed + 99))

    decay_scales, default_scale = calibrate_decay(plans_df)
    free_ids = set(gt.loc[gt.to_is_free, "unique_leg_id"])
    obs_free = (plans_df.set_index("unique_leg_id").loc[list(free_ids), "distance_meters"].to_numpy()
                if free_ids else None)
    alpha, beta = _alpha_beta(mode, pot_weight, dist_dev_weight)

    print(f"\n### {name}  | mode={mode} noise={noise} pot_weight={pot_weight} "
          f"| {world.meta['n_locations']} fac, box {world.meta['box']/1000:.1f} km, "
          f"{world.meta['density_per_km2']:.0f}/km^2 | {world.meta['n_persons']} persons, "
          f"{world.meta['n_free_legs']} free legs")

    results: Dict[str, tuple] = {}
    for s in solvers:
        params = {}
        if s in _DP_FAMILY and min_candidates:
            params["min_candidates"] = int(min_candidates)
        if s == "dp_sample":
            params = {**params, "decay_scales": decay_scales, "default_scale": default_scale}
        try:
            scorer = Scorer(mode=mode, pot_weight=pot_weight, dist_dev_weight=dist_dev_weight)
            ctx = run.setup(locations_tuple=loc, solver=s, rng_seed=seed, scorer=scorer,
                            parameters=params or None)
            t0 = time.perf_counter()
            rdf, _, valid = run.solve(ctx=ctx, plans_df=plans_df)
            dt = time.perf_counter() - t0
            results[s] = (valid, rdf, dt)
        except Exception as e:  # noqa: BLE001
            print(f"  {s:16s} ERROR: {type(e).__name__}: {str(e)[:70]}")

    objs = {s: per_person_objective(r[1], free_ids, alpha, beta) for s, r in results.items()}
    opt = objs.get("dp_full")

    rows = []
    for s in solvers:
        if s not in results:
            continue
        valid, rdf, dt = results[s]
        opt_frac = gap = float("nan")
        if opt is not None:
            so = objs[s]
            common = so.index.intersection(opt.index)
            a, o = so.loc[common], opt.loc[common]
            opt_frac = 100.0 * float((a >= o - 1e-6).mean())
            denom = float(np.abs(o).sum())
            gap = (float((o - a).sum()) / denom * 100.0) if denom > 1e-9 else float("nan")
        rec, err = recovery_metrics(rdf, gt)
        distW = med = float("nan")
        if obs_free is not None and len(obs_free):
            distW, med, _p90, _attr = realism_metrics(rdf, free_ids, obs_free)
        rows.append((s, opt_frac, gap, rec, err, distW, med, dt))

    hdr = (f"  {'solver':16s} {'opt%':>6s} {'%gap':>7s} {'recov%':>7s} {'err(m)':>8s} "
           f"{'distW':>8s} {'med':>7s} {'time':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s, of, gap, rec, err, dw, med, dt in rows:
        print(f"  {s:16s} {of:5.1f}% {gap:6.2f}% {rec:6.1f}% {err:8.1f} {dw:8.1f} {med:7.0f} {dt:6.2f}s")
    return {s: dict(opt=of, gap=gap, recov=rec, err=err, distW=dw, med=med, time=dt)
            for s, of, gap, rec, err, dw, med, dt in rows}


def directional_summary(by_mode_world: dict, solvers):
    """Compare gauss vs real: same solver ranking by %gap, same combined-vs-geometric
    recovery lift."""
    print("\n" + "=" * 78)
    print("DIRECTIONAL COMPARISON  (do gauss & real agree in direction?)")
    print("=" * 78)
    for mode, worlds in by_mode_world.items():
        if "GAUSS" not in worlds or "REAL" not in worlds:
            continue
        g, r = worlds["GAUSS"], worlds["REAL"]
        common = [s for s in solvers if s in g and s in r and s != "dp_full" and s != "dp_sample"]
        # rank argmin solvers by %gap (ascending = better)
        gr = sorted(common, key=lambda s: g[s]["gap"])
        rr = sorted(common, key=lambda s: r[s]["gap"])
        # Tie-aware: solvers with gap < TOL are "exact" — their relative order is arbitrary, so
        # compare the EXACT SET and the non-exact tail order instead of the brittle full order.
        TOL = 0.05
        gex = {s for s in common if g[s]["gap"] < TOL}
        rex = {s for s in common if r[s]["gap"] < TOL}
        gtail = [s for s in gr if s not in gex]
        rtail = [s for s in rr if s not in rex]
        agree = (gex == rex) and (gtail == rtail)
        from scipy.stats import spearmanr
        rho = spearmanr([g[s]["gap"] for s in common], [r[s]["gap"] for s in common]).correlation
        gstr = " < ".join(f"{s}({g[s]['gap']:.1f})" for s in gr)
        rstr = " < ".join(f"{s}({r[s]['gap']:.1f})" for s in rr)
        print(f"\n[{mode}] %gap ranking, best->worst (gap% in parens):")
        print(f"   GAUSS: {gstr}")
        print(f"   REAL : {rstr}")
        print(f"   exact set (gap<{TOL}%): GAUSS={sorted(gex)} REAL={sorted(rex)}")
        print(f"   same exact-set + tail order: {agree} | Spearman rho(gap) = {rho:.3f}")

    if "geometric" in by_mode_world and "combined" in by_mode_world:
        REC_TOL = 3.0  # recovery on ~300 free legs is noisy; |delta| < 3pp = "no real change"
        cls = lambda d: "+" if d > REC_TOL else ("-" if d < -REC_TOL else "~")
        print("\nCombined-vs-geometric RECOVERY change under noise (the potential-aware claim):")
        print(f"   (recovery is a noisy DIAGNOSTIC; |delta|<{REC_TOL:.0f}pp treated as '~' no-change)")
        print(f"   {'solver':16s} {'GAUSS geo->comb':>18s} {'REAL geo->comb':>18s} {'agree':>6s}")
        for s in solvers:
            try:
                gg = by_mode_world["geometric"]["GAUSS"][s]["recov"]
                gc = by_mode_world["combined"]["GAUSS"][s]["recov"]
                rg = by_mode_world["geometric"]["REAL"][s]["recov"]
                rc = by_mode_world["combined"]["REAL"][s]["recov"]
            except KeyError:
                continue
            gcls, rcls = cls(gc - gg), cls(rc - rg)
            print(f"   {s:16s} {gg:6.1f}->{gc:5.1f} ({gcls}){'':2s} "
                  f"{rg:6.1f}->{rc:5.1f} ({rcls}){'':2s} {str(gcls == rcls):>6s}")


def main(argv=None):
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--persons", type=int, default=250)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--modes", nargs="+", default=["geometric", "combined"],
                   choices=["geometric", "potential", "combined"])
    p.add_argument("--pot-weight", type=float, default=400.0)
    p.add_argument("--dist-dev-weight", type=float, default=1.0)
    p.add_argument("--min-candidates", type=int, default=None)
    p.add_argument("--solvers", nargs="+", default=ALL_SOLVERS)
    p.add_argument("--seed", type=int, default=0, help="the ONE fixed gauss/chain seed")
    p.add_argument("--baked", action="store_true",
                   help="load the fixed BAKED worlds from --baked-root and subsample --persons "
                        "(keeps the full-population dense potential field)")
    p.add_argument("--baked-root", default=os.path.join(os.path.dirname(__file__), "..", "data", "worlds"))
    args = p.parse_args(argv)

    if args.baked:
        gdir = os.path.join(args.baked_root, "gauss_hannover")
        odir = os.path.join(args.baked_root, "osm_hannover")
        if not (os.path.isdir(gdir) and os.path.isdir(odir)):
            print(f"Baked worlds missing under {args.baked_root}\n"
                  "Bake them first: uv run --project research python research/scripts/bake_worlds.py",
                  file=sys.stderr)
            return 2
        print(f"Loading BAKED worlds and subsampling {args.persons} persons (seed {args.seed}) ...")
        gauss = sample_persons(load_world(gdir), args.persons, seed=args.seed)
        real = sample_persons(load_world(odir), args.persons, seed=args.seed)
        worlds = {"GAUSS": gauss, "REAL": real}
    else:
        if not os.path.exists(DEFAULT_HANNOVER_POIS):
            print(f"Real OSM snapshot missing: {DEFAULT_HANNOVER_POIS}\n"
                  "Fetch it first: uv run python research/scripts/fetch_osm_topology.py", file=sys.stderr)
            return 2
        print("Building the two fixed Hannover worlds on the fly (one fixed seed) ...")
        gauss = city_world("hannover", n_persons=args.persons, rng=np.random.default_rng(args.seed))
        real = hannover_osm_world(args.persons, rng=np.random.default_rng(args.seed))
        worlds = {"GAUSS": gauss, "REAL": real}

    by_mode_world: dict = {m: {} for m in args.modes}
    for mode in args.modes:
        for wname, world in worlds.items():
            by_mode_world[mode][wname] = run_world(
                wname, world, solvers=args.solvers, mode=mode, noise=args.noise,
                pot_weight=args.pot_weight, dist_dev_weight=args.dist_dev_weight,
                min_candidates=args.min_candidates, seed=args.seed)

    directional_summary(by_mode_world, args.solvers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
