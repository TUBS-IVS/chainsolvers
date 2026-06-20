#!/usr/bin/env python
"""Prognosis (forecasting) experiment on the synthetic world.

Realism is not just reproducing today's distribution — it is predicting under a
*changed* world. Because the synthetic world's data-generating process is known, we
can run a ground-truthed counterfactual that real data cannot:

    baseline topology  --(gravity x attractiveness)-->  baseline chains
          |  shock: boost a district's attractiveness (x factor)
          v
    same topology, shocked sizes  -->  TRUE counterfactual chains

A model is then asked to predict the counterfactual placements (given the new activity
chains + the structural attractiveness it is allowed to see). We measure whether it
reproduces the true shift of trips into the boosted district — i.e. whether it has the
right elasticity. Models:

  - structural-informed : sampled-utility (attractiveness + decay), TOLD the new sizes.
  - structural-blind    : same model, but given the OLD sizes (didn't know about the boom).
  - non-structural      : sampled-decay only (alpha=0) — ignores attractiveness.

Expectation: only the structural-informed model tracks the true counterfactual; the
others stay near baseline (no/weak elasticity) — the distinction between *reproducing*
and *forecasting*.

    python scripts/prognosis.py --persons 800 --boost 6
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


def _true_share(gt, free_ids, district_ids):
    g = gt[gt.unique_leg_id.isin(free_ids)]
    return float(g.true_to_identifier.isin(district_ids).mean())


def _pred_share(rdf, free_ids, district_ids):
    r = rdf[rdf.unique_leg_id.isin(free_ids)]
    return float(r.to_act_identifier.isin(district_ids).mean())


def _free_dist(df, free_ids):
    r = df[df.unique_leg_id.isin(free_ids)]
    if "to_x" in r:
        return np.hypot(r.to_x - r.from_x, r.to_y - r.from_y).to_numpy()
    return r["distance_meters"].to_numpy()


TRANSFORM = "log"  # match the DGP's gravity form (utility ∝ log(size)); strictly-positive sizes


def predict(topo, plans_df, model_values, alpha, scale, seed):
    loc = topology_locations_tuple(topo, model_values)
    ctx = run.setup(locations_tuple=loc, solver="dp_sample", rng_seed=seed,
                    scorer=Scorer(mode="combined", pot_weight=alpha),
                    parameters={"default_scale": float(scale), "attr_transform": TRANSFORM})
    rdf, _, _ = run.solve(ctx=ctx, plans_df=plans_df)
    return rdf


def true_counterfactual(topo, n_persons, district, boost, gravity_scale, seed):
    """Regenerate the TRUE chains under an attractiveness shock; return (plans_cf, gt_cf, sizes_cf)."""
    sizes_cf = topo.sizes.copy()
    sizes_cf[district] *= boost
    rng = np.random.default_rng(seed)
    pcf, gtcf, _ = generate_chains(topo, n_persons, sizes=sizes_cf, assign_modes=False,
                                   gravity_scale=gravity_scale, rng=rng)
    return pd.DataFrame(pcf), pd.DataFrame(gtcf), sizes_cf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", type=int, default=1200)
    p.add_argument("--persons", type=int, default=800)
    p.add_argument("--gravity-scale", type=float, default=4000.0)
    p.add_argument("--boost", type=float, default=6.0, help="attractiveness multiplier for the district")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--elasticity-sweep", action="store_true",
                   help="sweep the boost factor and report predicted vs true Δ(district share)")
    p.add_argument("--boosts", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    topo = build_topology(n_locations=args.locations, rng=rng)
    box = topo.box

    # Baseline chains (modes off -> single decay scale; mode left out by request).
    pb, gtb, _ = generate_chains(topo, args.persons, sizes=topo.sizes, assign_modes=False,
                                 gravity_scale=args.gravity_scale, rng=rng)
    plans_b, gt_b = pd.DataFrame(pb), pd.DataFrame(gtb)

    district = (topo.coords[:, 0] < 0.4 * box) & (topo.coords[:, 1] < 0.4 * box)
    district_ids = set(topo.loc_ids[district])
    free_b = set(gt_b.loc[gt_b.to_is_free, "unique_leg_id"])
    true_base = _true_share(gt_b, free_b, district_ids)

    # Calibrate the structural model (alpha, decay scale) on the BASELINE by max-likelihood.
    alpha_hat, scale = fit_location_choice(topo, plans_b, gt_b, transform=TRANSFORM)
    print(f"District = corner quadrant: {int(district.sum())}/{len(topo.loc_ids)} locations.")
    print(f"Calibrated on baseline (MLE, transform={TRANSFORM!r}): alpha={alpha_hat:.2f}, "
          f"scale={scale:.0f} m   (true DGP: alpha=1.0, scale={args.gravity_scale:.0f})")
    print(f"Baseline district share of secondary trips = {true_base*100:.1f}%\n")

    def scenario(boost):
        plans_cf, gt_cf, sizes_cf = true_counterfactual(
            topo, args.persons, district, boost, args.gravity_scale, seed=args.seed + 1)
        free_cf = set(gt_cf.loc[gt_cf.to_is_free, "unique_leg_id"])
        true_cf = _true_share(gt_cf, free_cf, district_ids)
        rdf_s = predict(topo, plans_cf, sizes_cf, alpha_hat, scale, seed=7)
        rdf_n = predict(topo, plans_cf, topo.sizes, 0.0, scale, seed=7)
        return dict(plans_cf=plans_cf, free_cf=free_cf, true_cf=true_cf, sizes_cf=sizes_cf,
                    struct=_pred_share(rdf_s, free_cf, district_ids),
                    nonstruct=_pred_share(rdf_n, free_cf, district_ids),
                    rdf_s=rdf_s, rdf_n=rdf_n)

    if args.elasticity_sweep:
        print("Elasticity sweep -- change in district share vs baseline (percentage points):\n")
        hdr = f"{'boost':>6s} {'true_d':>8s} {'struct_d':>9s} {'nonstruct_d':>12s}"
        print(hdr); print("-" * len(hdr))
        for b in args.boosts:
            s = scenario(b)
            print(f"{b:6.1f} {(s['true_cf']-true_base)*100:7.1f} {(s['struct']-true_base)*100:8.1f} "
                  f"{(s['nonstruct']-true_base)*100:11.1f}")
        print("\n  A structural model tracks the true elasticity curve; the non-structural model is flat.")
        return 0

    # Single-scenario detail
    s = scenario(args.boost)
    obs_cf_dist = s["plans_cf"].set_index("unique_leg_id").loc[list(s["free_cf"]), "distance_meters"].to_numpy()
    clip = float(np.quantile(obs_cf_dist, 0.95))
    rdf_b = predict(topo, s["plans_cf"], topo.sizes, alpha_hat, scale, seed=7)  # structural-blind
    print(f"Boost x{args.boost}:  TRUE district share  baseline={true_base*100:.1f}%  "
          f"counterfactual={s['true_cf']*100:.1f}%  (true shift {(s['true_cf']-true_base)*100:+.1f} pp)\n")
    rows = [("structural-informed", s["rdf_s"]), ("structural-blind", rdf_b), ("non-structural", s["rdf_n"])]
    hdr = f"{'model':22s} {'predCF%':>8s} {'errVsTrueCF':>12s} {'distW(cf)':>10s}"
    print(hdr); print("-" * len(hdr))
    for name, rdf in rows:
        share = _pred_share(rdf, s["free_cf"], district_ids)
        w = wasserstein_distance(np.clip(_free_dist(rdf, s["free_cf"]), 0, clip), np.clip(obs_cf_dist, 0, clip))
        print(f"{name:22s} {share*100:7.1f}% {abs(share-s['true_cf'])*100:11.1f}pp {w:10.0f}")
    print("\n  Only a structural model TOLD the new attractiveness tracks the true shift -> prognosis ability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
