"""Test the in-process CallableSolver hook (stand-in for an external solver)."""

import numpy as np

from chainsolvers import run
from chainsolvers_eval.synth import generate_world
from chainsolvers_eval.external import CallableSolver


def test_callable_solver_runs_and_is_valid():
    w = generate_world(n_locations=300, n_persons=60, rng=np.random.default_rng(0))

    def place_fn(S, E, distances, act_types, locations, rng):
        # trivial stand-in: snap the S->E midpoint to the nearest facility of each type
        mid = 0.5 * (S + E)
        chosen = []
        for t in act_types:
            ids, coords, pots = locations.query_closest(t, mid, k=1)
            chosen.append((ids[0], np.asarray(coords[0], dtype=float), float(pots[0])))
        return chosen

    ctx = run.setup(locations_tuple=w.locations_tuple, solver=CallableSolver,
                    rng_seed=1, parameters={"place_fn": place_fn})
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    assert len(rdf) == len(w.plans_df)


def test_callable_solver_requires_place_fn():
    w = generate_world(n_locations=200, n_persons=20, rng=np.random.default_rng(0))
    ctx = run.setup(locations_tuple=w.locations_tuple, solver=CallableSolver, rng_seed=1)
    try:
        run.solve(ctx=ctx, plans_df=w.plans_df)
        raise AssertionError("expected ValueError when place_fn is missing")
    except ValueError:
        pass
