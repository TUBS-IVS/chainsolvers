#!/usr/bin/env python
"""Prognosis across ALL solvers — does each have forecasting elasticity?

Forecasting regime: the counterfactual has NO observed per-leg distances (it's the
future). So every solver is given the counterfactual chains' anchors + activity types,
and the NEW (shocked) structural attractiveness, but must source distances itself:

  - argmin solvers (carla, carla_dp, dp_refine, carla_dp_refine, dp_full): need a target
    distance to match -> fed BASELINE-distribution-resampled distances (which do NOT
    encode the shock), run in `combined` mode. Their only route to elasticity is the
    attractiveness term tipping the choice among distance-compatible candidates.
  - dp_sample (generative): ignores distances, generates them from the calibrated decay +
    attractiveness. Native forecasting tool.

Consistency: argmin potentials are passed as log(size) so `pot_weight*pot` matches the
log-form calibrated utility; dp_sample applies the log internally (attr_transform='log').
The argmin deviation-vs-attractiveness trade-off is unitful (metres vs log-attractiveness)
and therefore ad hoc -- `--attr-meters` sets how many metres of deviation one unit of
log-attractiveness is worth; sweep it to see how weight-sensitive the (weak) elasticity is.

    python scripts/prognosis_solvers.py --persons 800 --boost 6 \
        --solvers carla carla_dp_refine dp_full dp_sample --attr-meters 800
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from chainsolvers import run, Scorer
from chainsolvers_eval.synth import build_topology, generate_chains, topology_locations_tuple
from chainsolvers_eval.calibration import fit_location_choice
from chainsolvers_eval import survey as sv

TRANSFORM = "log"  # match the DGP gravity form (utility propto log(size)); strictly-positive sizes
ARGMIN = {"carla", "dp", "carla_dp", "dp_refine", "carla_dp_refine", "dp_full", "milp"}


def _share(ids, district_ids):
    return float(pd.Series(ids).isin(district_ids).mean())


def _free_dist(rdf, free_ids):
    r = rdf[rdf.unique_leg_id.isin(free_ids)]
    return np.hypot(r.to_x - r.from_x, r.to_y - r.from_y).to_numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", type=int, default=1200)
    p.add_argument("--persons", type=int, default=800)
    p.add_argument("--gravity-scale", type=float, default=4000.0)
    p.add_argument("--boost", type=float, default=6.0)
    p.add_argument("--attr-meters", type=float, default=800.0,
                   help="argmin combined mode: metres of deviation one log-attractiveness unit is worth")
    p.add_argument("--solvers", nargs="+",
                   default=["carla", "carla_dp_refine", "dp_full", "dp_sample"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    topo = build_topology(n_locations=args.locations, rng=rng)
    box = topo.box

    # Baseline world -> calibrate (alpha, scale) by MLE.
    pb, gtb, _ = generate_chains(topo, args.persons, sizes=topo.sizes, assign_modes=False,
                                 gravity_scale=args.gravity_scale, rng=rng)
    plans_b, gt_b = pd.DataFrame(pb), pd.DataFrame(gtb)
    alpha_hat, scale_hat = fit_location_choice(topo, plans_b, gt_b, transform=TRANSFORM)

    # Shock a corner district's attractiveness; regenerate the TRUE counterfactual.
    district = (topo.coords[:, 0] < 0.4 * box) & (topo.coords[:, 1] < 0.4 * box)
    district_ids = set(topo.loc_ids[district])
    sizes_cf = topo.sizes.copy()
    sizes_cf[district] *= args.boost
    pcf, gtcf, _ = generate_chains(topo, args.persons, sizes=sizes_cf, assign_modes=False,
                                   gravity_scale=args.gravity_scale, rng=np.random.default_rng(args.seed + 1))
    plans_cf, gt_cf = pd.DataFrame(pcf), pd.DataFrame(gtcf)
    free_cf = set(gt_cf.loc[gt_cf.to_is_free, "unique_leg_id"])
    true_cf = _share(gt_cf[gt_cf.unique_leg_id.isin(free_cf)].true_to_identifier, district_ids)
    true_base = _share(gt_b[gt_b.unique_leg_id.isin(set(gt_b.loc[gt_b.to_is_free, "unique_leg_id"]))]
                       .true_to_identifier, district_ids)
    obs_cf = plans_cf.set_index("unique_leg_id").loc[list(free_cf), "distance_meters"].to_numpy()
    clip = float(np.quantile(obs_cf, 0.95))

    # Forecast input: counterfactual chains, free-leg distances RESAMPLED from the baseline
    # distribution (so they carry no knowledge of the shock). argmin matches these.
    base_samples = sv.per_mode_distance_samples(plans_b, gt_b)
    forecast_plans = sv.resample_distances(plans_cf, gt_cf, base_samples, np.random.default_rng(args.seed + 2))

    # Attractiveness payloads: log(size) for argmin (linear pot term), raw for dp_sample (logs internally).
    loc_argmin = topology_locations_tuple(topo, np.log(np.maximum(sizes_cf, 1e-9)))
    loc_gen = topology_locations_tuple(topo, sizes_cf)
    pot_w = float(args.attr_meters)  # dist_dev_weight=1 -> 1 log-attr unit == attr_meters of deviation

    print(f"District = corner quadrant: {int(district.sum())}/{len(topo.loc_ids)} boosted x{args.boost}.")
    print(f"Calibrated (MLE, {TRANSFORM!r}): alpha={alpha_hat:.2f}, scale={scale_hat:.0f} m "
          f"(true 1.0/{args.gravity_scale:.0f})")
    print(f"TRUE district share: baseline={true_base*100:.1f}%  counterfactual={true_cf*100:.1f}%  "
          f"(true shift {(true_cf-true_base)*100:+.1f} pp)\n")
    print(f"  argmin combined weight: 1 log-attr unit == {pot_w:.0f} m deviation\n")

    hdr = f"{'solver':17s} {'regime':10s} {'predCF%':>8s} {'errVsTrue':>10s} {'distW':>8s}"
    print(hdr); print("-" * len(hdr))

    def score(rdf, label, regime):
        share = _share(rdf[rdf.unique_leg_id.isin(free_cf)].to_act_identifier, district_ids)
        w = wasserstein_distance(np.clip(_free_dist(rdf, free_cf), 0, clip), np.clip(obs_cf, 0, clip))
        print(f"{label:17s} {regime:10s} {share*100:7.1f}% {abs(share-true_cf)*100:9.1f}pp {w:8.0f}")

    # zero-elasticity floor: an argmin in geometric mode (no attractiveness at all).
    ctx = run.setup(locations_tuple=loc_gen, solver="carla_dp_refine", rng_seed=7,
                    scorer=Scorer(mode="geometric"))
    rdf, _, _ = run.solve(ctx=ctx, plans_df=forecast_plans)
    score(rdf, "carla_dp_refine", "geom(floor)")

    for solver in args.solvers:
        try:
            if solver == "dp_sample":
                ctx = run.setup(locations_tuple=loc_gen, solver="dp_sample", rng_seed=7,
                                scorer=Scorer(mode="combined", pot_weight=alpha_hat),
                                parameters={"default_scale": float(scale_hat), "attr_transform": TRANSFORM})
                rdf, _, _ = run.solve(ctx=ctx, plans_df=plans_cf)
                score(rdf, solver, "generative")
            elif solver in ARGMIN:
                ctx = run.setup(locations_tuple=loc_argmin, solver=solver, rng_seed=7,
                                scorer=Scorer(mode="combined", pot_weight=pot_w, dist_dev_weight=1.0))
                rdf, _, _ = run.solve(ctx=ctx, plans_df=forecast_plans)
                score(rdf, solver, "combined")
            else:
                print(f"{solver:17s} (skipped — unknown)")
        except Exception as e:  # noqa: BLE001
            print(f"{solver:17s} ERROR: {e}")

    print(f"\n  Target = true counterfactual share {true_cf*100:.1f}%. Lower errVsTrue = better forecast.")
    print("  geom(floor) has no attractiveness -> ~no elasticity; generative samples from the utility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
