"""Tests for the eval baseline solvers (Hörl RDA + floors) and the solver=<class> path."""

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers_eval.synth import generate_world
from chainsolvers_eval.baselines import (
    RelaxationDiscretization, RelaxationDiscretizationGuided, Nearest, ZoneSample, GravityIndependent,
)


def _world(seed=0, n_locations=300, n_persons=70):
    return generate_world(n_locations=n_locations, n_persons=n_persons, rng=np.random.default_rng(seed))

# keep the assignment loop short in tests (default is 20)
_FAST = {"assignment_iterations": 2}


def _total_dev(rdf):
    a = np.hypot(rdf.to_x - rdf.from_x, rdf.to_y - rdf.from_y)
    return float((rdf.distance_meters - a).abs().sum())


def test_baselines_run_via_solver_class():
    w = _world()
    for cls in (RelaxationDiscretization, Nearest, ZoneSample, GravityIndependent):
        ctx = run.setup(locations_tuple=w.locations_tuple, solver=cls, rng_seed=1, parameters=_FAST)
        rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
        assert valid, f"{cls.__name__} produced invalid output"
        assert len(rdf) == len(w.plans_df)


def test_relaxation_matches_distances_better_than_zone_sample():
    # Hörl RDA matches observed distances; pure attractiveness sampling ignores them,
    # so RDA's total distance deviation should be far lower.
    w = _world(2)
    devs = {}
    for name, cls in [("rda", RelaxationDiscretization), ("zone", ZoneSample)]:
        ctx = run.setup(locations_tuple=w.locations_tuple, solver=cls, rng_seed=1, parameters=_FAST)
        rdf, _, _ = run.solve(ctx=ctx, plans_df=w.plans_df)
        devs[name] = _total_dev(rdf)
    assert devs["rda"] < devs["zone"]


def test_guided_rda_raises_attractiveness():
    # Guidance forces draw secondary activities toward denser / more attractive POI clusters,
    # so chosen-location attractiveness should rise vs pure RDA.
    w = _world(5, n_locations=500, n_persons=120)
    free = set(w.ground_truth[w.ground_truth.to_is_free].unique_leg_id)

    def mean_attr(cls):
        ctx = run.setup(locations_tuple=w.locations_tuple, solver=cls, rng_seed=3, parameters=_FAST)
        rdf, _, _ = run.solve(ctx=ctx, plans_df=w.plans_df)
        return rdf[rdf.unique_leg_id.isin(free)]["to_act_potential"].mean()

    assert mean_attr(RelaxationDiscretizationGuided) > mean_attr(RelaxationDiscretization)


def test_unknown_solver_name_still_errors():
    w = _world()
    try:
        run.setup(locations_tuple=w.locations_tuple, solver="not_a_solver")
        raise AssertionError("expected ValueError for unknown solver name")
    except ValueError:
        pass
