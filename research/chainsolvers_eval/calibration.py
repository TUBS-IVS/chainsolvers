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

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from chainsolvers.solvers.dp import attr_value  # core helper (kept in the library)


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

    # Each choice: (log-attractiveness of candidates, distance-from-previous, chosen local idx)
    choices: List[Tuple[np.ndarray, np.ndarray, int]] = []
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
            choices.append((log_attr[cand], dprev, chosen_local))

    if not choices:
        raise ValueError("No free (secondary) choices found to calibrate on.")

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
    alpha, scale = float(res.x[0]), float(np.exp(res.x[1]))
    return alpha, scale
