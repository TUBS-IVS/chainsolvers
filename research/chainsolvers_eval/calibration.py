"""Maximum-likelihood calibration of the structural location-choice model.

Forecasting needs *estimated* behavioural parameters, not hand-set ones. We fit the
multinomial-logit utility

    U(candidate) = alpha * log(1 + attractiveness) - distance_from_previous / scale

to observed secondary-location choices, by maximizing the conditional log-likelihood
over the full candidate set per activity type (so no sampling-of-alternatives
correction is needed). `alpha` is the attractiveness sensitivity; `scale` is the
distance-decay length. These are the structural parameters assumed stable under
scenario change — i.e. what gives the model prognosis ability.

The estimator conditions each choice on the *previous* true location, matching a
sequential gravity data-generating process; on synthetic data fit by this estimator,
the recovered (alpha, scale) should track the true (1.0, gravity_scale).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from chainsolvers.solvers.dp import attr_value  # core helper (kept in the library)

# One observed free choice: (log-attractiveness of candidates, distance-from-previous, chosen idx).
Choice = Tuple[np.ndarray, np.ndarray, int]


def _collect_free_choices(
    topo,
    plans_df,
    gt,
    *,
    attractiveness: Optional[np.ndarray] = None,
    transform: str = "log1p",
    max_persons: Optional[int] = 400,
    with_mode: bool = False,
) -> List[tuple]:
    """Each observed free (secondary) choice as (log-attractiveness of candidates,
    distance-from-previous-true-location, chosen local idx), conditioning every choice on the
    *previous* true location to match a sequential gravity DGP. Shared by all estimators.
    With ``with_mode=True`` each tuple gains a trailing ``mode`` field (the leg's travel mode,
    or ``None``) for per-mode grouping."""
    attractiveness = topo.sizes if attractiveness is None else np.asarray(attractiveness, dtype=float)
    log_attr = attr_value(attractiveness, transform)
    id_to_idx = {lid: i for i, lid in enumerate(topo.loc_ids)}
    type_local = {t: {g: i for i, g in enumerate(topo.type_locs[t])} for t in topo.types}

    g = gt.set_index("unique_leg_id")
    tx, ty, tid, free = g["true_to_x"], g["true_to_y"], g["true_to_identifier"], g["to_is_free"]

    pids = plans_df["unique_person_id"].unique()
    if max_persons:
        pids = pids[:max_persons]
    sub = plans_df[plans_df["unique_person_id"].isin(pids)]
    has_mode = "mode" in sub.columns

    choices: List[tuple] = []
    for _, grp in sub.groupby("unique_person_id", sort=False):
        grp = grp.reset_index(drop=True)
        node = [np.array([grp["from_x"][0], grp["from_y"][0]], dtype=float)]  # home start
        for k in range(len(grp)):
            lid = grp["unique_leg_id"][k]
            node.append(np.array([tx[lid], ty[lid]], dtype=float))  # node after leg k = leg k's true 'to'
        for k in range(len(grp)):
            lid = grp["unique_leg_id"][k]
            if not bool(free[lid]):
                continue
            t = grp["to_act_type"][k]
            prev = node[k]                                  # location before this activity
            cand = topo.type_locs[t]
            cc = topo.coords[cand]
            dprev = np.hypot(cc[:, 0] - prev[0], cc[:, 1] - prev[1])
            chosen_local = type_local[t][id_to_idx[tid[lid]]]
            rec = (log_attr[cand], dprev, chosen_local)
            if with_mode:
                rec = rec + (grp["mode"][k] if has_mode else None,)
            choices.append(rec)

    if not choices:
        raise ValueError("No free (secondary) choices found to calibrate on.")
    return choices


def _fit_single(choices: List[Choice]) -> Tuple[float, float]:
    """MLE (alpha, scale) of the single-component conditional MNL over `choices`."""
    def neg_ll(params):
        alpha, log_scale = params
        scale = float(np.exp(log_scale))
        tot = 0.0
        for ls, dprev, cl in choices:
            u = alpha * ls - dprev / scale
            tot -= (u[cl] - logsumexp(u))
        return tot

    init_scale = float(np.median([dprev[cl] for _, dprev, cl in choices]) + 1.0)
    res = minimize(neg_ll, x0=[1.0, np.log(init_scale)], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 2000})
    return float(res.x[0]), float(np.exp(res.x[1]))


def _fit_mixture(choices: List[Choice]) -> Tuple[float, float, float, float]:
    """MLE (alpha, scale, tail_weight, tail_scale) of the two-component (body+tail) MNL."""
    def neg_ll(params):
        # Unconstrained params -> (alpha, scale>0, w in (0,1), tail_scale = scale*(1+gap), gap>0).
        alpha, log_scale, logit_w, log_gap = params
        scale = float(np.exp(log_scale))
        w = 1.0 / (1.0 + np.exp(-logit_w))
        tail_scale = scale * (1.0 + float(np.exp(log_gap)))
        tot = 0.0
        for ls, dprev, cl in choices:
            kern = (1.0 - w) * np.exp(-dprev / scale) + w * np.exp(-dprev / tail_scale)
            u = alpha * ls + np.log(kern)              # log K(d); kern > 0 always
            tot -= (u[cl] - logsumexp(u))
        return tot

    init_scale = float(np.median([dprev[cl] for _, dprev, cl in choices]) + 1.0)
    # init: alpha=1, w≈0.1, tail_scale≈3*scale. Warm-started near the single-component optimum.
    x0 = [1.0, np.log(init_scale), np.log(0.1 / 0.9), np.log(2.0)]
    res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 4000})
    alpha, log_scale, logit_w, log_gap = res.x
    scale = float(np.exp(log_scale))
    w = float(1.0 / (1.0 + np.exp(-logit_w)))
    tail_scale = float(scale * (1.0 + np.exp(log_gap)))
    return alpha, scale, w, tail_scale


def _fit_powerlaw(choices: List[Choice], scale: float) -> Tuple[float, float]:
    """MLE (alpha, shape k) of the conditional MNL under a power-law distance kernel
    ``K(d) = (1 + d/scale)^(-k)`` -> ``log K = -k·log1p(d/scale)``, with ``scale`` **fixed** to the
    (well-identified) exponential MLE. Fitting ``(scale, k)`` jointly is ill-posed — they trade off
    along a ``k/scale`` ridge and diverge — so we pin ``scale`` and fit only the exponent ``k`` (a
    stable 1-D tail-thickness knob: small k = heavier polynomial tail)."""
    s = max(float(scale), 1e-6)
    log1p_d = [np.log1p(dprev / s) for _, dprev, _ in choices]

    def neg_ll(params):
        alpha, log_k = params
        k = float(np.exp(log_k))
        tot = 0.0
        for (ls, _dprev, cl), l1p in zip(choices, log1p_d):
            u = alpha * ls - k * l1p                       # = alpha*ls + log K(d)
            tot -= (u[cl] - logsumexp(u))
        return tot

    res = minimize(neg_ll, x0=[1.0, np.log(1.0)], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 2000})
    return float(res.x[0]), float(np.exp(res.x[1]))


def fit_location_choice(
    topo,
    plans_df,
    gt,
    *,
    attractiveness: Optional[np.ndarray] = None,
    transform: str = "log1p",
    max_persons: Optional[int] = 400,
) -> Tuple[float, float]:
    """Return (alpha, scale) maximizing the conditional MNL likelihood of the observed
    free (secondary) location choices in (plans_df, gt) on topology `topo`. `transform`
    selects the attractiveness form (see :func:`attr_value`) and must match the form the
    forecasting solver uses."""
    return _fit_single(_collect_free_choices(
        topo, plans_df, gt, attractiveness=attractiveness, transform=transform,
        max_persons=max_persons))


def fit_location_choice_mixture(
    topo,
    plans_df,
    gt,
    *,
    attractiveness: Optional[np.ndarray] = None,
    transform: str = "log1p",
    max_persons: Optional[int] = 400,
) -> Tuple[float, float, float, float]:
    """Return (alpha, scale, tail_weight, tail_scale) maximizing the conditional MNL likelihood
    under a **two-component distance kernel** ``K(d) = (1-w)·exp(-d/scale) + w·exp(-d/tail_scale)``
    (``tail_scale > scale``). The utility is ``alpha·log_attr + log K(d)`` — the exact choice-model
    counterpart of :func:`chainsolvers.solvers.dp.sample_chain`'s mixture edge kernel, so the fitted
    ``(scale, tail_weight, tail_scale)`` feed ``dp_sample`` directly.

    This is a *structural* estimator: it maximizes choice likelihood, it never fits the marginal
    distance distribution. Crucially it does not manufacture a tail — on data generated by a single
    decay (no long-range component) the MLE drives ``tail_weight → 0`` (or ``tail_scale → scale``),
    recovering :func:`fit_location_choice`; it reports ``w > 0`` only when the choices actually carry
    a heavier-than-exponential tail. Identifiability of ``w``/``tail_scale`` individually degrades as
    ``w → 0`` (the tail component vanishes); use the implied kernel, not ``w`` alone, when ``w`` is tiny.
    """
    return _fit_mixture(_collect_free_choices(
        topo, plans_df, gt, attractiveness=attractiveness, transform=transform,
        max_persons=max_persons))


def fit_mode_kernels(
    topo,
    plans_df,
    gt,
    *,
    attractiveness: Optional[np.ndarray] = None,
    transform: str = "log1p",
    max_persons: Optional[int] = 400,
    min_choices: int = 40,
) -> Tuple[float, Dict, Dict, Dict, Tuple[float, float, float]]:
    """**Per-mode** structural calibration for ``dp_sample`` on a mode-heterogeneous world. A single
    pooled tail kernel overshoots multi-mode super-regions: car carries a long tail, walk/bike do not,
    and a pooled fit smears car's tail across every mode. This fits each mode group separately.

    Returns ``(alpha, decay_scales, tail_weights, tail_scale_factors, pooled)`` where the three dicts
    are keyed by mode (string) and feed ``dp_sample`` directly (``parameters={"decay_scales": ...,
    "tail_weights": ..., "tail_scale_factors": ..., "default_scale"/"tail_weight"/"tail_scale_factor"
    from `pooled`}``). Per the validated recipe each mode's **body scale is the single-component MLE**
    (which transfers to the both-anchor sampler) while only the **tail shape** (weight, tail/body
    ratio) comes from the mixture MLE. ``alpha`` and ``pooled = (scale, tail_weight, tail_factor)``
    are the pooled fits, used as the fallback for modes with fewer than ``min_choices`` observations
    (and for worlds with no mode column). ``pooled``'s ``tail_factor`` is ``tail_scale / mixture_body``.
    """
    choices = _collect_free_choices(topo, plans_df, gt, attractiveness=attractiveness,
                                    transform=transform, max_persons=max_persons, with_mode=True)
    plain = [(ls, dprev, cl) for ls, dprev, cl, _ in choices]
    alpha_p, scale_p = _fit_single(plain)
    _, smix_p, w_p, ts_p = _fit_mixture(plain)
    pooled = (scale_p, w_p, ts_p / smix_p)

    groups: Dict[object, List[Choice]] = defaultdict(list)
    for ls, dprev, cl, m in choices:
        groups[m].append((ls, dprev, cl))

    decay_scales: Dict[str, float] = {}
    tail_weights: Dict[str, float] = {}
    tail_scale_factors: Dict[str, float] = {}
    # Skip per-mode fitting entirely for a single unlabelled group (no mode column) -> use pooled.
    if not (len(groups) == 1 and next(iter(groups)) is None):
        for m, ch in groups.items():
            if m is None or len(ch) < min_choices:
                continue  # too few observations -> fall back to the pooled defaults
            _, s_single = _fit_single(ch)
            _, s_mix, w_m, ts_m = _fit_mixture(ch)
            key = str(m)
            decay_scales[key] = s_single
            tail_weights[key] = w_m
            tail_scale_factors[key] = ts_m / s_mix
    return alpha_p, decay_scales, tail_weights, tail_scale_factors, pooled


def fit_mode_powerlaw(
    topo,
    plans_df,
    gt,
    *,
    attractiveness: Optional[np.ndarray] = None,
    transform: str = "log1p",
    max_persons: Optional[int] = 400,
    min_choices: int = 40,
) -> Tuple[float, Dict, Dict, Tuple[float, float]]:
    """**Per-mode** power-law calibration for ``dp_sample`` (``dist_kernel="powerlaw"``). Returns
    ``(alpha, decay_scales, dist_shapes, pooled)`` where ``decay_scales``/``dist_shapes`` are mode->
    value dicts (scale and exponent ``k``) and ``pooled = (scale, k)`` is the fallback for thin/absent
    modes. Each mode's ``scale`` is the single-component exponential MLE (well-identified, transfers to
    the both-anchor sampler); only the exponent ``k`` is fitted on top of it (small ``k`` = heavier
    polynomial tail), so it adapts across worlds (light for gauss, heavy for two_zone)."""
    choices = _collect_free_choices(topo, plans_df, gt, attractiveness=attractiveness,
                                    transform=transform, max_persons=max_persons, with_mode=True)
    plain = [(ls, dprev, cl) for ls, dprev, cl, _ in choices]
    alpha_p, scale_p = _fit_single(plain)
    _, k_p = _fit_powerlaw(plain, scale_p)

    groups: Dict[object, List[Choice]] = defaultdict(list)
    for ls, dprev, cl, m in choices:
        groups[m].append((ls, dprev, cl))

    decay_scales: Dict[str, float] = {}
    dist_shapes: Dict[str, float] = {}
    if not (len(groups) == 1 and next(iter(groups)) is None):
        for m, ch in groups.items():
            if m is None or len(ch) < min_choices:
                continue
            _, s_m = _fit_single(ch)
            _, k_m = _fit_powerlaw(ch, s_m)
            decay_scales[str(m)] = s_m
            dist_shapes[str(m)] = k_m
    return alpha_p, decay_scales, dist_shapes, (scale_p, k_p)
