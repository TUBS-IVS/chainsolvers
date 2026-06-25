#!/usr/bin/env python
"""Run the paper's PROGNOSIS (forecasting-elasticity) test on BOTH Hannover geometries and
check the conclusion holds in the same direction.

Same shock, same DGP, same solvers — only the geometry/attractiveness differs:

  * GAUSS = `build_topology` Hannover preset (clustered Gaussian blobs).
  * REAL  = `osm.load_poi_csv` (real OSM coordinates + tag-derived attractiveness weights).

For each geometry: calibrate the structural MNL (alpha, scale) by MLE on a baseline, boost a
corner district's attractiveness x`boost`, regenerate the TRUE counterfactual chains, then ask
every solver to predict the counterfactual placements from baseline-resampled distances (which
carry NO knowledge of the shock). The only route to elasticity is the attractiveness term.
Reports predicted vs true district share, so we can see whether GAUSS reproduces REAL's
direction: structural/attractiveness-aware solvers track the shift; the geometric floor does not.

    uv run python research/scripts/prognosis_geometry_compare.py --persons 600 --boost 6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from chainsolvers import Scorer, run
from chainsolvers_eval.calibration import fit_location_choice
from chainsolvers_eval.osm import DEFAULT_HANNOVER_POIS, load_poi_csv
from chainsolvers_eval.synth import (MID_MODE_DECAY_M, MID_PAIR_DECAY_M, MID_TO_DECAY_M,
                                     MID_TOUR_MODE_SPLIT, MID_URBAN_TEMPLATES, build_topology,
                                     generate_chains, topology_locations_tuple)
from chainsolvers_eval import survey as sv

TRANSFORM = "log"
ARGMIN = {"carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_carla_pot",
          "dp_full", "milp"}
ALL_SOLVERS = ["carla", "dp_carla_refine", "dp_carla_pot", "dp_full", "milp", "dp_sample"]
# MiD chain/decay bundle shared by both geometries (so only geometry differs).
CHAIN_KW = dict(templates=MID_URBAN_TEMPLATES, pair_decay=MID_PAIR_DECAY_M, to_decay=MID_TO_DECAY_M,
                mode_decay=MID_MODE_DECAY_M, mode_split=MID_TOUR_MODE_SPLIT, assign_modes=True)


def _share(ids, district_ids):
    return float(pd.Series(ids).isin(district_ids).mean())


def run_geometry(name, topo, *, persons, boost, attr_meters, solvers, seed):
    box = topo.box
    rng = np.random.default_rng(seed)

    # Baseline -> calibrate (alpha, scale).
    pb, gtb, _ = generate_chains(topo, persons, sizes=topo.sizes, gravity_scale=4000.0, rng=rng,
                                 **CHAIN_KW)
    plans_b, gt_b = pd.DataFrame(pb), pd.DataFrame(gtb)
    alpha_hat, scale_hat = fit_location_choice(topo, plans_b, gt_b, transform=TRANSFORM)

    # Shock a corner district; regenerate TRUE counterfactual.
    district = (topo.coords[:, 0] < 0.4 * box) & (topo.coords[:, 1] < 0.4 * box)
    district_ids = set(topo.loc_ids[district])
    sizes_cf = topo.sizes.copy(); sizes_cf[district] *= boost
    pcf, gtcf, _ = generate_chains(topo, persons, sizes=sizes_cf, gravity_scale=4000.0,
                                   rng=np.random.default_rng(seed + 1), **CHAIN_KW)
    plans_cf, gt_cf = pd.DataFrame(pcf), pd.DataFrame(gtcf)
    free_cf = set(gt_cf.loc[gt_cf.to_is_free, "unique_leg_id"])
    free_b = set(gt_b.loc[gt_b.to_is_free, "unique_leg_id"])
    true_base = _share(gt_b[gt_b.unique_leg_id.isin(free_b)].true_to_identifier, district_ids)
    true_cf = _share(gt_cf[gt_cf.unique_leg_id.isin(free_cf)].true_to_identifier, district_ids)

    # Forecast input: counterfactual chains, distances resampled from the BASELINE distribution
    # (no knowledge of the shock). argmin solvers match these; dp_sample generates its own.
    base_samples = sv.per_mode_distance_samples(plans_b, gt_b)
    forecast_plans = sv.resample_distances(plans_cf, gt_cf, base_samples,
                                           np.random.default_rng(seed + 2))
    loc_argmin = topology_locations_tuple(topo, np.log(np.maximum(sizes_cf, 1e-9)))
    loc_gen = topology_locations_tuple(topo, sizes_cf)
    pot_w = float(attr_meters)

    print(f"\n### {name}  | {len(topo.loc_ids)} fac, box {box/1000:.1f} km | district "
          f"{int(district.sum())} fac boosted x{boost} | persons {persons}")
    print(f"  calibrated alpha={alpha_hat:.2f} scale={scale_hat:.0f} (true 1.0/4000) | "
          f"TRUE district share {true_base*100:.1f}% -> {true_cf*100:.1f}% "
          f"(shift {(true_cf-true_base)*100:+.1f} pp)")
    hdr = f"  {'solver':16s} {'regime':11s} {'predCF%':>8s} {'errVsTrue':>10s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    out = {}

    def score(label, regime, rdf):
        share = _share(rdf[rdf.unique_leg_id.isin(free_cf)].to_act_identifier, district_ids)
        print(f"  {label:16s} {regime:11s} {share*100:7.1f}% {abs(share-true_cf)*100:9.1f}pp")
        out[label] = dict(regime=regime, predCF=share * 100, err=abs(share - true_cf) * 100)

    # zero-elasticity floor: argmin, geometric (no attractiveness).
    ctx = run.setup(locations_tuple=loc_gen, solver="dp_carla_refine", rng_seed=7,
                    scorer=Scorer(mode="geometric"))
    rdf, _, _ = run.solve(ctx=ctx, plans_df=forecast_plans)
    score("geom(floor)", "geom", rdf)

    for s in solvers:
        try:
            if s == "dp_sample":
                ctx = run.setup(locations_tuple=loc_gen, solver="dp_sample", rng_seed=7,
                                scorer=Scorer(mode="combined", pot_weight=alpha_hat),
                                parameters={"default_scale": float(scale_hat), "attr_transform": TRANSFORM})
                rdf, _, _ = run.solve(ctx=ctx, plans_df=plans_cf)
                score(s, "generative", rdf)
            elif s in ARGMIN:
                ctx = run.setup(locations_tuple=loc_argmin, solver=s, rng_seed=7,
                                scorer=Scorer(mode="combined", pot_weight=pot_w, dist_dev_weight=1.0))
                rdf, _, _ = run.solve(ctx=ctx, plans_df=forecast_plans)
                score(s, "combined", rdf)
        except Exception as e:  # noqa: BLE001
            print(f"  {s:16s} ERROR: {type(e).__name__}: {str(e)[:60]}")

    out["_true_cf"] = true_cf * 100
    out["_true_base"] = true_base * 100
    return out


def main(argv=None):
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--persons", type=int, default=600)
    p.add_argument("--boost", type=float, default=6.0)
    p.add_argument("--attr-meters", type=float, default=800.0)
    p.add_argument("--gauss-locations", type=int, default=8000,
                   help="gauss facility count (kept modest so dp_full stays fast)")
    p.add_argument("--solvers", nargs="+", default=ALL_SOLVERS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not os.path.exists(DEFAULT_HANNOVER_POIS):
        print(f"Real OSM snapshot missing: {DEFAULT_HANNOVER_POIS}", file=sys.stderr)
        return 2

    # ONE fixed gauss topology (Hannover preset, fixed seed) + the fixed real OSM topology.
    from chainsolvers_eval.synth import CITY_PRESETS
    pv = CITY_PRESETS["hannover"]["type_prevalence"]
    gauss_topo = build_topology(n_locations=args.gauss_locations, density_per_km2=93.0,
                                n_clusters=10, type_prevalence=pv,
                                rng=np.random.default_rng(args.seed))
    real_topo = load_poi_csv(DEFAULT_HANNOVER_POIS)

    g = run_geometry("GAUSS", gauss_topo, persons=args.persons, boost=args.boost,
                     attr_meters=args.attr_meters, solvers=args.solvers, seed=args.seed)
    r = run_geometry("REAL", real_topo, persons=args.persons, boost=args.boost,
                     attr_meters=args.attr_meters, solvers=args.solvers, seed=args.seed)

    print("\n" + "=" * 70)
    print("DIRECTIONAL COMPARISON  (does each solver show elasticity in BOTH?)")
    print("=" * 70)
    print(f"  {'solver':16s} {'GAUSS predCF':>13s} {'REAL predCF':>12s} {'both>floor':>11s}")
    gfloor, rfloor = g["geom(floor)"]["predCF"], r["geom(floor)"]["predCF"]
    print(f"  {'geom(floor)':16s} {gfloor:12.1f}% {rfloor:11.1f}%  (baseline)")
    print(f"  {'TRUE cf share':16s} {g['_true_cf']:12.1f}% {r['_true_cf']:11.1f}%")
    for s in args.solvers:
        if s in g and s in r:
            ge = g[s]["predCF"] > gfloor + 0.5
            re = r[s]["predCF"] > rfloor + 0.5
            print(f"  {s:16s} {g[s]['predCF']:12.1f}% {r[s]['predCF']:11.1f}% "
                  f"{str(ge and re):>11s}")
    print("\n  Elasticity = predCF lifted above the geometric floor toward TRUE cf share.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
