#!/usr/bin/env python
"""Survey-realistic experiment: super-region ground truth -> random survey -> two input
modes -> all solvers -> validate against ground truth.

  Track 1 (direct chains): run argmin solvers on the SURVEYED persons' observed distances;
           score mean deviation + %gap to the true optimum (dp_full) + recovery.
  Track 3 (distance distribution): derive a per-mode distance distribution from the survey,
           apply it to the STUDY-REGION synthetic population (no observed distances), and
           compare the produced distance distribution + attractiveness + recovery against
           the study region's ground truth -- status-quo distance-resampler vs the
           generative structural sampler.

    python scripts/survey_experiment.py --locations 1500 --persons 2000 --survey-frac 0.1
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


def _free_ids(gt):
    return set(gt.loc[gt["to_is_free"], "unique_leg_id"])


def _recovery(rdf, gt, free_ids):
    f = gt[gt.unique_leg_id.isin(free_ids)].merge(
        rdf[["unique_leg_id", "to_act_identifier", "to_x", "to_y"]], on="unique_leg_id")
    rec = 100.0 * float((f.to_act_identifier == f.true_to_identifier).mean())
    err = float(np.hypot(f.to_x - f.true_to_x, f.to_y - f.true_to_y).mean())
    return rec, err


def _free_dist(rdf, free_ids):
    r = rdf[rdf.unique_leg_id.isin(free_ids)]
    return np.hypot(r.to_x - r.from_x, r.to_y - r.from_y).to_numpy()


def _total_dev(rdf):
    a = np.hypot(rdf.to_x - rdf.from_x, rdf.to_y - rdf.from_y)
    return float((rdf.distance_meters - a).abs().sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", type=int, default=1500)
    p.add_argument("--persons", type=int, default=2000)
    p.add_argument("--survey-frac", type=float, default=0.1)
    p.add_argument("--study-frac-side", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    topo = build_topology(n_locations=args.locations, rng=rng)
    pr, gr, _ = generate_chains(topo, args.persons, sizes=topo.sizes, assign_modes=True, rng=rng)
    plans, gt = pd.DataFrame(pr), pd.DataFrame(gr)
    loc_struct = topology_locations_tuple(topo, topo.sizes)  # structural attractiveness everywhere

    # Study sub-region + its synthetic population.
    lo, hi = sv.study_window(topo, args.study_frac_side)
    study_pids = sv.persons_in_window(plans, lo, hi)
    study_plans, study_gt = sv._subset(plans, gt, study_pids)
    study_free = _free_ids(study_gt)

    # Survey: random sample of the WHOLE super-region.
    survey_plans, survey_gt = sv.draw_survey(plans, gt, args.survey_frac, np.random.default_rng(args.seed + 1))
    survey_free = _free_ids(survey_gt)
    print(f"Super-region: {args.locations} locations, {args.persons} persons. "
          f"Study window persons: {len(study_pids)}. Survey persons: {survey_plans['unique_person_id'].nunique()}.\n")

    # ---- Track 1: direct chains (surveyed persons, observed distances) -------------
    print("Track 1 -- direct chains on surveyed persons (deviation vs true optimum):")
    opt = None
    rows = []
    for solver in ["dp_full", "carla", "carla_dp_refine"]:
        ctx = run.setup(locations_tuple=loc_struct, solver=solver, rng_seed=7)
        rdf, _, _ = run.solve(ctx=ctx, plans_df=survey_plans)
        dev = _total_dev(rdf) / survey_plans["unique_person_id"].nunique()
        rec, err = _recovery(rdf, survey_gt, survey_free)
        if solver == "dp_full":
            opt = dev
        gap = (dev - opt) / opt * 100 if opt else float("nan")
        rows.append((solver, dev, gap, rec, err))
    print(f"  {'solver':16s} {'mean|dd|':>9s} {'%gap':>7s} {'recov%':>7s} {'err(m)':>8s}")
    for s, dev, gap, rec, err in rows:
        print(f"  {s:16s} {dev:9.1f} {gap:6.1f}% {rec:6.1f}% {err:8.1f}")

    # ---- Track 3: survey distance distribution applied to the study population -----
    samples = sv.per_mode_distance_samples(survey_plans, survey_gt)
    decay = {str(m): float(np.mean(v)) for m, v in samples.items() if m is not None and len(v)}
    default_scale = float(np.mean(samples[None]))
    alpha_hat, _ = fit_location_choice(topo, survey_plans, survey_gt, transform="log", max_persons=300)

    # observed (true) study-region free-leg distance distribution = the target
    obs = study_plans.set_index("unique_leg_id").loc[list(study_free), "distance_meters"].to_numpy()
    clip = float(np.quantile(obs, 0.95))
    true_attr = study_gt[study_gt.to_is_free].true_to_identifier  # for reference only

    print("\nTrack 3 -- survey distance distribution applied to study population:")
    print(f"  (calibrated: alpha={alpha_hat:.2f}, per-mode decay={ {k:round(v) for k,v in decay.items()} })")
    print(f"  observed study free-leg dist: median={np.median(obs):.0f} p90={np.quantile(obs,0.9):.0f}")
    print(f"  {'pipeline':28s} {'distW':>8s} {'recov%':>7s} {'meanAttr':>9s}")

    # status-quo: resample distances from the survey distribution, then argmin-place
    resampled = sv.resample_distances(study_plans, study_gt, samples, np.random.default_rng(args.seed + 2))
    ctx = run.setup(locations_tuple=loc_struct, solver="carla_dp_refine", rng_seed=7)
    rdf_sq, _, _ = run.solve(ctx=ctx, plans_df=resampled)
    w = wasserstein_distance(np.clip(_free_dist(rdf_sq, study_free), 0, clip), np.clip(obs, 0, clip))
    rec, _ = _recovery(rdf_sq, study_gt, study_free)
    attr = rdf_sq[rdf_sq.unique_leg_id.isin(study_free)]["to_act_potential"].mean()
    print(f"  {'status-quo (resample+argmin)':28s} {w:8.0f} {rec:6.1f}% {attr:9.3f}")

    # generative: structural sampler calibrated on the survey
    ctx = run.setup(locations_tuple=loc_struct, solver="dp_sample", rng_seed=7,
                    scorer=Scorer(mode="combined", pot_weight=alpha_hat),
                    parameters={"decay_scales": decay, "default_scale": default_scale, "attr_transform": "log"})
    rdf_g, _, _ = run.solve(ctx=ctx, plans_df=study_plans)
    w = wasserstein_distance(np.clip(_free_dist(rdf_g, study_free), 0, clip), np.clip(obs, 0, clip))
    rec, _ = _recovery(rdf_g, study_gt, study_free)
    attr = rdf_g[rdf_g.unique_leg_id.isin(study_free)]["to_act_potential"].mean()
    print(f"  {'generative (dp_sample)':28s} {w:8.0f} {rec:6.1f}% {attr:9.3f}")

    print("\n  distW = Wasserstein to the study region's true free-leg distance distribution (lower=better).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
