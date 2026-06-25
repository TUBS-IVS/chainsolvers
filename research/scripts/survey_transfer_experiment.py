"""Country -> national survey -> study-region transfer experiment.

Generate a super-region ground truth, draw a *national* survey, carve a centred **study region**,
and probe solvers on that region. Ground truth (the resident's true chain + anchors) is KEPT
throughout, so recovery and gap-to-oracle stay measurable everywhere; realistic difficulty is
parameterized as two controlled degradations of the resident's true inputs (no primary-activity
placer needed):

  * Axis 1 — distance quality (true anchors kept): true -> noisy eps -> survey-distribution
    distances (`resample_distances`). The status-quo argmin input; the paired recovery-collapse /
    surviving-distance-fit is the crisp "distance-fit != correctness" demonstration.
  * Axis 2 — anchor quality (true distances kept): work jittered (`disturb_anchor`, imperfect
    commuting model -> robustness curve) -> work demoted to a free node (`demote_anchor`, no
    commuting model -> the harder near-single-anchor instance, with GT retained).

Per regime it reports %gap-to-oracle + recovery (+ distance-fit on the distance axis). A
`--frac-sweep` checks whether the result survives realistically tiny survey samples.
(`transplant_survey_chains` — whole-donor-chain HTS donation — is kept as a footnote primitive in
`survey.py`; it forfeits GT and collapses into Axis 1 on two-anchor spans, so it is not run here.)

    uv run python research/scripts/survey_transfer_experiment.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from chainsolvers import Scorer, run
from chainsolvers_eval import survey as S
from chainsolvers_eval.metrics import distribution_fit, free_leg_distances
from chainsolvers_eval.synth import (MID_MODE_DECAY_M, MID_PAIR_DECAY_M, MID_TO_DECAY_M,
                                     MID_TOUR_MODE_SPLIT, MID_URBAN_TEMPLATES, generate_world,
                                     two_zone_world)

# decay_inflate tuned on the HOMOGENEOUS super-region (density 30/km^2, ~20 km box) so realized
# per-mode leg medians match MiD (walk 997/980, bike 2443/2450, car 5519/5910, pt 5240/5400 — the
# big box lifts car/pt closer to target than the single-city presets). Deliberately applied
# UNCHANGED to the inhomogeneous world: there the dense core realizes much shorter trips (car
# ~3050), which is the realistic local mismatch that drives the transfer penalty — not retuned away.
INFLATE = {"walk": 0.60, "bike": 1.02, "car": 2.1, "pt": 2.0}


def _deviation(result_df, gt):
    """Total |observed - achieved| over free legs (the separable placement objective)."""
    free = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    df = result_df[result_df["unique_leg_id"].isin(free)]
    ach = np.hypot(df["to_x"] - df["from_x"], df["to_y"] - df["from_y"])
    return float((df["distance_meters"] - ach).abs().sum())


def _recovery(result_df, gt):
    free = gt[gt["to_is_free"]].merge(result_df[["unique_leg_id", "to_act_identifier"]], on="unique_leg_id")
    return float((free["to_act_identifier"] == free["true_to_identifier"]).mean())


def _gap(sol_dev, orc_dev):
    """%gap of a solver's deviation above the oracle's on the SAME instance (no GT needed)."""
    return (sol_dev - orc_dev) / orc_dev if orc_dev > 1e-9 else 0.0


def _solve(world, plans, solver, seed=0, pot_weight=0.0):
    # Distance alone barely identifies a facility once trips are long and candidate sets large;
    # attractiveness (potentials = usage) disambiguates, so default to the combined scorer.
    scorer = Scorer(mode="combined", pot_weight=pot_weight) if pot_weight > 0 else None
    ctx = run.setup(locations_tuple=world.locations_tuple, solver=solver, rng_seed=seed, scorer=scorer)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=plans)
    return rdf, valid


def run_world(world, *, frac, noise, solver, oracle, seed, label, pot_weight, study_frac=0.4,
              anchor_noise=500.0):
    """GT is kept throughout; difficulty is parameterized on two independent axes (distance
    quality, anchor quality) as controlled degradations of the resident's true inputs. Recovery
    and gap-to-oracle stay live everywhere; no primary-activity placer is needed."""
    rng = np.random.default_rng(seed)
    plans, gt = world.plans_df, world.ground_truth
    lo, hi = S.study_window(world.topology, frac_side=study_frac)
    region_pids = set(S.persons_in_window(plans, lo, hi))
    reg_plans = plans[plans["unique_person_id"].isin(region_pids)].copy()
    reg_gt = gt[gt["unique_leg_id"].isin(set(reg_plans["unique_leg_id"]))].copy()

    survey_plans, survey_gt = S.draw_survey(plans, gt, frac=frac, rng=rng)
    survey_samples = S.per_mode_distance_samples(survey_plans, survey_gt)
    survey_ref = free_leg_distances_from_plans(survey_plans, survey_gt)

    dens = world.meta.get("density_per_km2")
    dens_s = f"density {dens:.0f}/km^2" if dens else (
        f"core/ring {world.meta.get('core_density')}/{world.meta.get('ring_density')}/km^2")
    print(f"\n=== {label} ===")
    print(f"  super-region: {world.meta['n_locations']} fac, box {world.topology.box/1000:.0f} km, "
          f"{dens_s} | region persons {len(region_pids)} | "
          f"survey {len(survey_plans['unique_person_id'].unique())} persons (frac {frac})")

    def report(tag, pl, g, *, dist_noise=0.0, show_fit=False, note=""):
        pl2 = _add_noise(pl, g, dist_noise, rng) if dist_noise > 0 else pl
        orc, _ = _solve(world, pl2, oracle, seed, pot_weight)
        sol, _ = _solve(world, pl2, solver, seed, pot_weight)
        gap = _gap(_deviation(sol, g), _deviation(orc, g))
        rec = _recovery(sol, g)
        fit = f"  dist-fit KS {distribution_fit(free_leg_distances(sol, g), survey_ref)['ks']:.3f}" if show_fit else ""
        print(f"  {tag:24s} %gap {gap*100:5.1f}  recovery {rec:.3f}{fit}   {note}")

    # Baseline: resident's own chain + own true distances + true anchors. Identifiability.
    report("A truth", reg_plans, reg_gt, show_fit=True)
    # Axis 1 — distance quality (true anchors kept): true -> noisy eps -> survey-distribution.
    print("  -- distance axis (true anchors) --")
    report("B1 +noise eps", reg_plans, reg_gt, dist_noise=noise, show_fit=True)
    report("B2 survey-dist", S.resample_distances(reg_plans, reg_gt, survey_samples, rng), reg_gt,
           show_fit=True, note="(recovery collapses; dist-fit survives)")
    # Axis 2 — anchor quality (true distances kept): work jittered -> work removed (free node).
    print("  -- anchor axis (true distances) --")
    report("C work +noise", S.disturb_anchor(reg_plans, anchor="work", noise_m=anchor_noise, rng=rng),
           reg_gt, note=f"(work eps sigma={anchor_noise:.0f}m; robustness)")
    dem_pl, dem_gt = S.demote_anchor(reg_plans, reg_gt, anchor="work")
    report("D work removed", dem_pl, dem_gt, note="(work demoted to free node; harder instance)")


def free_leg_distances_from_plans(plans, gt):
    free = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    return plans.loc[plans["unique_leg_id"].isin(free), "distance_meters"].to_numpy(float)


def _add_noise(plans, gt, noise, rng):
    free = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    out = plans.copy()
    d = out["distance_meters"].to_numpy(float).copy()
    ids = out["unique_leg_id"].to_numpy()
    for i in range(len(out)):
        if ids[i] in free:
            d[i] = max(1.0, d[i] * (1.0 + float(rng.normal(0, noise))))
    out["distance_meters"] = d
    return out


def frac_sweep(world, *, fracs, solver, seed, pot_weight, study_frac=0.4):
    rng = np.random.default_rng(seed)
    plans, gt = world.plans_df, world.ground_truth
    lo, hi = S.study_window(world.topology, frac_side=study_frac)
    reg_pids = set(S.persons_in_window(plans, lo, hi))
    reg_plans = plans[plans["unique_person_id"].isin(reg_pids)].copy()
    reg_gt = gt[gt["unique_leg_id"].isin(set(reg_plans["unique_leg_id"]))].copy()
    print("\n=== survey-fraction sweep (distance axis, GT kept: recovery vs dist-fit vs sample size) ===")
    for f in fracs:
        sp, sg = S.draw_survey(plans, gt, frac=f, rng=rng)
        samples = S.per_mode_distance_samples(sp, sg)
        ref = free_leg_distances_from_plans(sp, sg)
        pl = S.resample_distances(reg_plans, reg_gt, samples, rng)
        sol, _ = _solve(world, pl, solver, seed, pot_weight)
        fit = distribution_fit(free_leg_distances(sol, reg_gt), ref)
        print(f"  frac {f:<6} survey persons {len(sp['unique_person_id'].unique()):5d}  "
              f"recovery {_recovery(sol, reg_gt):.3f}  "
              f"dist-fit KS {fit['ks']:.3f} p50err {fit.get('q50_abs_err_m', float('nan')):.0f}m")


def build_homogeneous(*, n_locations, noise, seed):
    """Uniform control: one MiD-urban regime everywhere at a national-average density."""
    return generate_world(
        n_locations=n_locations, n_persons=int(n_locations * 0.4), density_per_km2=30.0,
        distance_noise=noise, templates=MID_URBAN_TEMPLATES, pair_decay=MID_PAIR_DECAY_M,
        to_decay=MID_TO_DECAY_M, mode_decay=MID_MODE_DECAY_M, mode_split=MID_TOUR_MODE_SPLIT,
        decay_inflate=INFLATE, rng=np.random.default_rng(seed))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-locations", type=int, default=12000, help="homogeneous-control facility count")
    p.add_argument("--persons", type=int, default=6000, help="two-zone population")
    p.add_argument("--box", type=float, default=45000.0, help="two-zone super-region side (m)")
    p.add_argument("--frac", type=float, default=0.05, help="national survey sampling fraction")
    p.add_argument("--noise", type=float, default=0.1, help="small distance noise on free legs")
    p.add_argument("--anchor-noise", type=float, default=500.0,
                   help="work-anchor jitter sigma (m) for the anchor-quality axis")
    p.add_argument("--solver", default="dp_carla_refine")
    p.add_argument("--oracle", default="dp_full")
    p.add_argument("--pot-weight", type=float, default=80.0,
                   help="attractiveness weight in the combined scorer (0 = distance-only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac-sweep", action="store_true")
    args = p.parse_args(argv)

    # 1) Homogeneous control: study window is the central 40 %.
    homo = build_homogeneous(n_locations=args.n_locations, noise=args.noise, seed=args.seed)
    run_world(homo, frac=args.frac, noise=args.noise, solver=args.solver, oracle=args.oracle,
              seed=args.seed, label="HOMOGENEOUS control", pot_weight=args.pot_weight,
              anchor_noise=args.anchor_noise)

    # 2) Two-zone: dense urban core (MiD-urban) in a sparse rural ring (MiD-rural). The study
    #    region is the urban core (core_frac_side), so the national survey is a true urban+rural
    #    mixture and applying it to the core is a genuine transfer mismatch.
    core_frac = 0.33
    tz = two_zone_world(n_persons=args.persons, box=args.box, core_frac_side=core_frac,
                        distance_noise=args.noise, rng=np.random.default_rng(args.seed))
    run_world(tz, frac=args.frac, noise=args.noise, solver=args.solver, oracle=args.oracle,
              seed=args.seed, label="TWO-ZONE (urban core in rural ring)",
              pot_weight=args.pot_weight, study_frac=core_frac, anchor_noise=args.anchor_noise)
    if args.frac_sweep:
        frac_sweep(tz, fracs=[0.05, 0.02, 0.005], solver=args.solver, seed=args.seed,
                   pot_weight=args.pot_weight, study_frac=core_frac)


if __name__ == "__main__":
    main()
