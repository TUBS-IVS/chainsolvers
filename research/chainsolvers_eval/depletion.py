"""Population-level potential depletion — an eval-rig wrapper, NOT a solver feature.

Every solver places each person independently on a *static* attractiveness field. That lets
a few highly-attractive facilities be chosen by everyone, so the realized visit distribution
over-concentrates relative to the potential field. ``solve_with_depletion`` adds a capacity /
competition layer *around* any solver: it drives the per-person loop and, after each placement,
decrements the chosen facility's potential so later persons see a "fresher" (reduced) field.
With ``deplete == 1.0`` on usage-derived potentials (visit counts) this is exactly sampling
*without replacement*, which makes the aggregate choices track the potential marginal.

This lives in the eval package on purpose: it couples placements (the per-person separable
exactness no longer implies optimality of the joint allocation, and the result becomes
person-order dependent), so it is a benchmarking mechanism, not part of the MIT solvers. The
solvers are untouched; this only mutates a *working copy* of ``ctx.locations.potentials`` between
persons and restores the original afterwards, so the passed ``ctx`` is unchanged on return.

    from chainsolvers_eval.depletion import solve_with_depletion, visit_potential_fit
    rdf = solve_with_depletion(ctx, plans_df, deplete=1.0, rng=np.random.default_rng(0))
    fit = visit_potential_fit(ctx, rdf, plans_df)   # per-type TV distance to the potential field
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers.types import PlanColumns


def _id_index(locations) -> Dict[Any, Dict[Any, int]]:
    """{activity_type: {facility_id: row index in the potentials/coords arrays}}."""
    return {t: {fid: i for i, fid in enumerate(locations.identifiers[t])}
            for t in locations.identifiers}


def _free_leg_types(sub: pd.DataFrame, cols: PlanColumns) -> Dict[Any, Any]:
    """leg_id -> to_act_type for the legs the solver must place (unknown destination)."""
    free = sub[sub[cols.to_x].isna()]
    return dict(zip(free[cols.unique_leg_id], free[cols.to_act_type]))


def solve_with_depletion(
    ctx,
    plans_df: pd.DataFrame,
    *,
    deplete: float = 1.0,
    floor: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    **solve_kwargs: Any,
) -> pd.DataFrame:
    """Solve every person through ``run.solve``, depleting each chosen facility's potential by
    ``deplete`` (floored at ``floor``) so subsequent persons see the reduced field.

    Solver-agnostic: works for any registered solver or solver class. Returns the concatenated
    ``result_df``. The order of persons is randomised (depletion is order-dependent); pass a
    seeded ``rng`` for reproducibility. ``ctx.locations.potentials`` is restored on return.
    """
    if rng is None:
        rng = np.random.default_rng()
    cols = PlanColumns()
    locs = ctx.locations

    orig_pots = locs.potentials
    work_pots = {t: np.array(a, dtype=float) for t, a in orig_pots.items()}
    id_index = _id_index(locs)

    persons = plans_df[cols.person_id].drop_duplicates().to_numpy()
    persons = persons[rng.permutation(len(persons))]

    results = []
    locs.potentials = work_pots  # solvers read potentials fresh from here every call
    try:
        for pid in persons:
            sub = plans_df[plans_df[cols.person_id] == pid]
            # dp_carla_pot caches its top-K-by-potential on the solver instance; clear it so
            # the augmentation set reflects the depleted field (no-op for other solvers).
            cache = getattr(ctx.solver, "_pot_top_cache", None)
            if isinstance(cache, dict):
                cache.clear()

            rdf, _, _ = run.solve(ctx=ctx, plans_df=sub, **solve_kwargs)
            results.append(rdf)

            if deplete:
                leg_type = _free_leg_types(sub, cols)
                placed = rdf[rdf[cols.unique_leg_id].isin(leg_type)]
                for lid, fid in zip(placed[cols.unique_leg_id], placed[cols.to_act_identifier]):
                    t = leg_type.get(lid)
                    idx = id_index.get(t, {}).get(fid)
                    if idx is not None:
                        work_pots[t][idx] = max(floor, float(work_pots[t][idx]) - deplete)
    finally:
        locs.potentials = orig_pots  # restore — ctx is pristine after the call

    return pd.concat(results, ignore_index=True)


def visit_potential_fit(
    ctx,
    result_df: pd.DataFrame,
    plans_df: Optional[pd.DataFrame] = None,
) -> Dict[Any, float]:
    """Total-variation distance between the realized visit distribution and the (true) potential
    field, per activity type. 0 == the solver's aggregate choices perfectly reproduce the
    attractiveness marginal; higher == more over-/under-concentration. Lower is better.

    If ``plans_df`` is given, only the free (solver-placed) legs are counted (recommended);
    otherwise every placed leg with an identifier of a placeable type is counted.
    """
    cols = PlanColumns()
    locs = ctx.locations
    id_index = _id_index(locs)

    placed = result_df.dropna(subset=[cols.to_act_identifier])
    if plans_df is not None:
        free_legs = set()
        for _, sub in plans_df.groupby(cols.person_id):
            free_legs |= set(_free_leg_types(sub, cols))
        placed = placed[placed[cols.unique_leg_id].isin(free_legs)]

    out: Dict[Any, float] = {}
    for t in locs.identifiers:
        pot = np.asarray(locs.potentials[t], dtype=float)
        if pot.sum() <= 0:
            continue
        counts = np.zeros(len(pot))
        sub = placed[placed[cols.to_act_type] == t] if cols.to_act_type in placed else placed
        for fid in sub[cols.to_act_identifier]:
            i = id_index[t].get(fid)
            if i is not None:
                counts[i] += 1
        if counts.sum() == 0:
            continue
        p = pot / pot.sum()
        q = counts / counts.sum()
        out[t] = 0.5 * float(np.abs(p - q).sum())  # total-variation distance
    return out
