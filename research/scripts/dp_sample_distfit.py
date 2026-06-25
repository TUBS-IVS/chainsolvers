"""Block B generative arm: does dp_sample reproduce the *wanted* free-leg distance distribution?

The generative MNL (dp_sample) samples distances rather than collapsing onto the argmin, so it can
in principle reproduce an observed distance distribution (the hEART-style generative target) while
still plugging into our optimality + prognosis machinery. Out of the box it can't: it undershoots the
median (uncalibrated decay scale) and truncates the long tail (local-only candidate ball). This script
demonstrates the two-part fix on a ground-truthed synthetic world:

    1. calibrated decay scale (MLE via calibration.fit_location_choice)  -> fixes the median
    2. local/global candidate mixing (global_mix_k > 0, new DpSample param) -> fixes the tail

It runs several configurations against the world's own observed free-leg distance distribution (the
"wanted" target) and reports median, q90 and the KS statistic for each:

    default            default_scale=3000, single scale    (status quo -- undershoots + truncates)
    calibrated         MLE scale,          single scale     (median fixed, tail still short)
    calibrated+mix     MLE scale,          global_mix_k>0   (NO tail improvement -- see note)
    calibrated+tail    MLE scale,          heavy-tail kernel (median + tail)

Note: candidate mixing (`global_mix_k`) does NOT fix the tail here. The tail miss is structural,
not candidate-limited: the joint forward-backward conditions each intermediate on *both* anchors,
so far candidates carry ~0 MNL mass however many you add. A scale sweep shows the tail is
*kernel*-governed (raise the scale and q90 climbs, but the median overshoots first), so the fix is a
heavy-tailed decay *mixture* (`tail_weight` mass on a `tail_scale_factor * scale` kernel) -- the
generative counterpart of the synthetic DGP's own long-range draws.

    python research/scripts/dp_sample_distfit.py
    python research/scripts/dp_sample_distfit.py --persons 1500 --tail-weight 0.15 --tail-factor 3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers.scoring_selection import Scorer
from chainsolvers_eval.synth import generate_world
from chainsolvers_eval.calibration import (fit_location_choice, fit_location_choice_mixture,
                                           fit_mode_kernels, fit_mode_powerlaw)

# Cap the fitted long-kernel reach: a near-flat fitted tail_scale (≈uniform global pull) would
# otherwise blow up the candidate ball. Beyond ~this factor the tail is effectively global, which
# `global_mix_k` covers more cheaply.
TAIL_FACTOR_CAP = 10.0


def _free_leg_distances(rdf, gt) -> np.ndarray:
    """Realised free-leg straight-line distances (||from - to||) from a result frame."""
    free = gt[gt.to_is_free].merge(
        rdf[["unique_leg_id", "to_x", "to_y", "from_x", "from_y"]], on="unique_leg_id")
    return np.hypot(free.to_x - free.from_x, free.to_y - free.from_y).to_numpy(float)


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dep, sup|F_a - F_b|)."""
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _run(world, params, *, alpha, seed) -> np.ndarray:
    ctx = run.setup(
        locations_tuple=world.locations_tuple, solver="dp_sample", rng_seed=seed,
        scorer=Scorer(mode="combined", pot_weight=alpha), parameters=params,
    )
    rdf, _, valid = run.solve(ctx=ctx, plans_df=world.plans_df)
    if not valid:
        raise RuntimeError("dp_sample produced an invalid result frame")
    return _free_leg_distances(rdf, world.ground_truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facilities", type=int, default=1000)
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--gravity-scale", type=float, default=4000.0)
    ap.add_argument("--mix-k", type=int, default=800, help="global candidates added per pool")
    ap.add_argument("--tail-weight", type=float, default=0.20, help="mixture mass on the long kernel")
    ap.add_argument("--tail-factor", type=float, default=3.5, help="long kernel = factor * scale")
    ap.add_argument("--transform", default="log1p", choices=["log1p", "log"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", default=None,
                    help="baked world name under research/data/worlds (e.g. two_zone); "
                         "default = freshly generated gauss-style world")
    ap.add_argument("--sample", type=int, default=1500,
                    help="persons to subsample from a baked world (it is too large to solve whole)")
    ap.add_argument("--cal-persons", type=int, default=400,
                    help="persons used for the (single + mixture) MLE calibration; lower it on large "
                         "catalogs where the per-choice likelihood is expensive")
    args = ap.parse_args()

    if args.world:
        import os
        from chainsolvers_eval import survey as sv
        from chainsolvers_eval.worlds import load_world
        wdir = os.path.join(os.path.dirname(__file__), "..", "data", "worlds", args.world)
        world = sv.sample_persons(load_world(wdir), args.sample, seed=args.seed)
        src = f"baked '{args.world}' (sampled {args.sample} persons)"
    else:
        world = generate_world(n_locations=args.facilities, n_persons=args.persons,
                               gravity_scale=args.gravity_scale, rng=np.random.default_rng(args.seed))
        src = f"{args.facilities} facilities, {args.persons} persons, gravity_scale={args.gravity_scale:.0f}"
    gt = world.ground_truth

    # "Wanted" target = the world's own observed free-leg distances.
    free_ids = gt.loc[gt.to_is_free, "unique_leg_id"]
    wanted = (world.plans_df.set_index("unique_leg_id")
              .loc[list(free_ids), "distance_meters"].to_numpy(float))

    # Calibrate the structural decay scale (MLE). alpha drives the potential term.
    alpha_hat, scale_hat = fit_location_choice(
        world.topology, world.plans_df, gt, transform=args.transform, max_persons=args.cal_persons)
    # Structural mixture MLE: (scale, tail_weight, tail_scale) for the heavy-tail kernel. These are
    # estimated from the choice likelihood, NOT tuned to the marginal -> the "fitted" config below is
    # not hand-tuned. IMPORTANT: the mixture's *body* scale is single-anchor-specific (the choice MLE
    # conditions on the previous node only) and does NOT transfer to the both-anchor sampler -- using
    # it collapses the median. Only the *tail shape* (weight + tail/body ratio) transfers; pair it
    # with the single-component body scale (which recovers the true generating scale). tail_factor is
    # capped (see TAIL_FACTOR_CAP).
    alpha_mix, scale_mix, w_mix, ts_mix = fit_location_choice_mixture(
        world.topology, world.plans_df, gt, transform=args.transform, max_persons=args.cal_persons)
    f_mix = float(min(ts_mix / scale_mix, TAIL_FACTOR_CAP))
    # Per-mode structural kernels (the multi-mode fix): body scale + tail shape per mode.
    _, decay_pm, tw_pm, tf_pm, _pooled = fit_mode_kernels(
        world.topology, world.plans_df, gt, transform=args.transform, max_persons=args.cal_persons)
    tf_pm = {m: min(f, TAIL_FACTOR_CAP) for m, f in tf_pm.items()}
    # Per-mode power-law kernel: a single fitted exponent k per mode sets tail thickness.
    _, decay_pl, shape_pl, pooled_pl = fit_mode_powerlaw(
        world.topology, world.plans_df, gt, transform=args.transform, max_persons=args.cal_persons)

    base = {"attr_transform": args.transform}
    print(f"world: {src}")
    print(f"MLE single  : alpha={alpha_hat:.2f}, scale={scale_hat:.0f}")
    print(f"MLE mixture : alpha={alpha_mix:.2f}, body_scale={scale_mix:.0f} (single-anchor; not used "
          f"as body), tail_weight={w_mix:.3f}, tail_factor={ts_mix / scale_mix:.1f} (capped {f_mix:.1f})")
    if decay_pm:
        print("per-mode exp: " + "  ".join(
            f"{m}(s={decay_pm[m]:.0f},w={tw_pm[m]:.2f},f={tf_pm[m]:.1f})" for m in sorted(decay_pm)))
    print(f"powerlaw    : pooled(s={pooled_pl[0]:.0f},k={pooled_pl[1]:.2f})" + (
        "  " + "  ".join(f"{m}(s={decay_pl[m]:.0f},k={shape_pl[m]:.2f})" for m in sorted(decay_pl))
        if decay_pl else ""))
    print()

    configs = [
        ("default",        {**base, "default_scale": 3000.0}),
        ("calibrated",     {**base, "default_scale": scale_hat}),
        ("calibrated+mix", {**base, "default_scale": scale_hat, "global_mix_k": args.mix_k}),
        ("hand-tail",      {**base, "default_scale": scale_hat, "tail_weight": args.tail_weight,
                            "tail_scale_factor": args.tail_factor}),
        ("fitted-tail",    {**base, "default_scale": scale_hat, "global_mix_k": args.mix_k,
                            "tail_weight": w_mix, "tail_scale_factor": f_mix}),
        # per-mode body scales + per-mode tail shapes; pooled single-fit scale as the fallback.
        ("fitted-permode", {**base, "default_scale": scale_hat, "global_mix_k": args.mix_k,
                            "decay_scales": decay_pm, "tail_weights": tw_pm,
                            "tail_scale_factors": tf_pm, "tail_weight": w_mix,
                            "tail_scale_factor": f_mix}),
        # per-mode power-law kernel: fitted scale + fitted exponent k per mode (its own scale transfers).
        ("fitted-powerlaw", {**base, "dist_kernel": "powerlaw", "global_mix_k": args.mix_k,
                             "default_scale": pooled_pl[0], "dist_shape": pooled_pl[1],
                             "decay_scales": decay_pl, "dist_shapes": shape_pl}),
    ]
    print(f"{'target (wanted)':16s}  median={np.median(wanted):7.0f}m  "
          f"q90={np.quantile(wanted, 0.90):8.0f}m")
    print(f"{'-' * 78}")
    for name, params in configs:
        d = _run(world, params, alpha=alpha_hat, seed=args.seed + 1)
        print(f"{name:16s}  median={np.median(d):7.0f}m  q90={np.quantile(d, 0.90):8.0f}m  "
              f"(d_med={np.median(d) - np.median(wanted):+7.0f}  "
              f"d_q90={np.quantile(d, 0.90) - np.quantile(wanted, 0.90):+8.0f})  "
              f"KS={_ks(d, wanted):.3f}")


if __name__ == "__main__":
    main()
