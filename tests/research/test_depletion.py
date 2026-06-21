"""Tests for the eval-rig potential-depletion wrapper (chainsolvers_eval.depletion).

Key guarantee: it is a pure add-on. Not calling it changes nothing; calling it leaves the
context pristine; and with deplete=0 it reproduces a plain solve exactly for a deterministic
solver. A behavioural check confirms depletion actually spreads load toward the potential field.
"""

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers_eval.synth import generate_world
from chainsolvers_eval.depletion import solve_with_depletion, visit_potential_fit


def _world(seed=0, n_locations=400, n_persons=150):
    return generate_world(n_locations=n_locations, n_persons=n_persons,
                          rng=np.random.default_rng(seed))


def _sorted(rdf):
    return rdf.sort_values("unique_leg_id").reset_index(drop=True)


def test_deplete_zero_matches_plain_solve():
    # With a deterministic solver, deplete=0 must reproduce plain run.solve exactly
    # (persons are independent at deplete=0; only row order differs -> compare sorted).
    w = _world()
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_carla", rng_seed=1)
    plain, _, _ = run.solve(ctx=ctx, plans_df=w.plans_df)
    depl = solve_with_depletion(ctx, w.plans_df, deplete=0.0, rng=np.random.default_rng(7))

    a = _sorted(plain); b = _sorted(depl)
    assert len(a) == len(b)
    for col in ["to_x", "to_y", "to_act_identifier"]:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        assert ((x == y) | (pd.isna(x) & pd.isna(y))).all(), col


def test_ctx_potentials_restored_after_run():
    # The wrapper mutates only a working copy; ctx.locations.potentials is unchanged after.
    w = _world()
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=1)
    before = {t: ctx.locations.potentials[t].copy() for t in ctx.locations.potentials}
    same_obj = ctx.locations.potentials

    solve_with_depletion(ctx, w.plans_df, deplete=5.0, rng=np.random.default_rng(3))

    assert ctx.locations.potentials is same_obj  # same dict object restored
    for t, arr in before.items():
        assert np.array_equal(ctx.locations.potentials[t], arr), t


def test_depletion_changes_placements():
    # For a potential-sensitive solver, depletion must actually alter the assignment
    # (otherwise the mechanism is inert).
    w = _world(seed=2)
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=1)
    off = solve_with_depletion(ctx, w.plans_df, deplete=0.0, rng=np.random.default_rng(4))
    on = solve_with_depletion(ctx, w.plans_df, deplete=1.0, rng=np.random.default_rng(4))
    a, b = _sorted(off)["to_act_identifier"].to_numpy(), _sorted(on)["to_act_identifier"].to_numpy()
    assert not ((a == b) | (pd.isna(a) & pd.isna(b))).all()


def test_depletion_improves_potential_fit():
    # Sampling without replacement against the potential field spreads load, so the realized
    # visit distribution tracks the potential marginal at least as well as i.i.d. placement.
    w = _world(seed=5, n_locations=300, n_persons=400)
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=1)
    off = solve_with_depletion(ctx, w.plans_df, deplete=0.0, rng=np.random.default_rng(9))
    on = solve_with_depletion(ctx, w.plans_df, deplete=1.0, rng=np.random.default_rng(9))

    tv_off = np.mean(list(visit_potential_fit(ctx, off, w.plans_df).values()))
    tv_on = np.mean(list(visit_potential_fit(ctx, on, w.plans_df).values()))
    assert tv_on <= tv_off + 1e-9  # never worse; typically strictly better
