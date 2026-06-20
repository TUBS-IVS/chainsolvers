import numpy as np
import pandas as pd
import pytest

from chainsolvers import run, io


def make_simple_locations():
    """Build a minimal locations dict covering 'home' and 'work'.
    We include a work location whose id is 'w1' at (5, 0) and home 'h1' at (0, 0).
    """
    data = {
        "home": {
            "h1": {"coordinates": [0.0, 0.0], "potential": 1.0, "name": "Home-1"},
        },
        "work": {
            "w0": {"coordinates": [10.0, 0.0], "potential": 1.0, "name": "Work-0"},
            # Intersection point for two 5m radii between (0,0) and (10,0) is exactly (5,0)
            "w1": {"coordinates": [5.0, 0.0], "potential": 3.5, "name": "Work-1"},
        },
    }
    ids, coords, pots = io.build_locations_payload_from_dict(data)
    name_lookup = io.build_name_lookup_from_dict(data)
    return (ids, coords, pots), name_lookup


def test_end_to_end_single_leg_with_enrichment():
    # Locations and lookup
    (ids, coords, pots), name_lookup = make_simple_locations()

    # Single person, single leg: from home to work with known coordinates and identifier
    df = pd.DataFrame([
        {
            "unique_person_id": "p1",
            "unique_leg_id": "l1",
            "to_act_type": "work",
            "distance_meters": 10.0,
            "from_x": 0.0,
            "from_y": 0.0,
            "to_x": 10.0,
            "to_y": 0.0,
            "to_act_identifier": "w0",  # present so enrichment can use it
        }
    ])

    ctx = run.setup(
        locations_tuple=(ids, coords, pots),
        solver="carla",
        rng_seed=42,
    )

    # Inject the name lookup into context by pretending input came from dict
    # The runner would normally set this when using locations_dict/df.
    ctx = run.RunnerContext(
        locations=ctx.locations,
        solver=ctx.solver,
        scorer=ctx.scorer,
        selector=ctx.selector,
        rng=ctx.rng,
        name_lookup=name_lookup,
    )

    result_df, result_plans, valid = run.solve(ctx=ctx, plans_df=df)

    # Valid output
    assert valid is True
    # Required columns exist
    for col in [
        "unique_person_id",
        "unique_leg_id",
        "to_act_type",
        "distance_meters",
        "to_x",
        "to_y",
    ]:
        assert col in result_df.columns

    # Coordinates are finite and match input destination
    row = result_df.iloc[0]
    assert np.isfinite(row.to_x) and np.isfinite(row.to_y)
    assert row.to_x == pytest.approx(10.0)
    assert row.to_y == pytest.approx(0.0)

    # Name and potential enrichment
    assert "to_act_name" in result_df.columns
    assert row.to_act_name == "Work-0"
    assert "to_act_potential" in result_df.columns
    # From make_simple_locations, w0 has potential 1.0
    assert row.to_act_potential == pytest.approx(1.0)

    # When solver returns plans (frozendict) internally, ensure it was converted
    assert isinstance(result_plans, (dict, type(None)))  # frozendict or None


def test_setup_with_unknown_solver_raises():
    (ids, coords, pots), _ = make_simple_locations()
    df = pd.DataFrame([{
        "unique_person_id": "p1",
        "unique_leg_id": "l1",
        "to_act_type": "work",
        "distance_meters": 1.0,
        "from_x": 0.0, "from_y": 0.0,
        "to_x": 0.0, "to_y": 0.0,
        "to_act_identifier": "w1",
    }])

    with pytest.raises(ValueError):
        run.setup(locations_tuple=(ids, coords, pots), solver="does_not_exist")


def test_two_leg_circle_intersection_places_intermediate():
    # Setup locations including intersection candidate w1 at (5, 0)
    (ids, coords, pots), _ = make_simple_locations()

    # Two legs for a single segment: (0,0)->(?,?) distance 5, then (?,?,)->(10,0) distance 5
    df = pd.DataFrame([
        {
            "unique_person_id": "p1",
            "unique_leg_id": "l1",
            "to_act_type": "work",
            "distance_meters": 5.0,
            "from_x": 0.0,
            "from_y": 0.0,
            "to_x": np.nan,  # unknown to be placed
            "to_y": np.nan,
        },
        {
            "unique_person_id": "p1",
            "unique_leg_id": "l2",
            "to_act_type": "home",
            "distance_meters": 5.0,
            "from_x": np.nan,  # will be filled by solver as the same point chosen above
            "from_y": np.nan,
            "to_x": 10.0,
            "to_y": 0.0,
        },
    ])

    ctx = run.setup(
        locations_tuple=(ids, coords, pots),
        solver="carla",
        rng_seed=123,
    )

    result_df, result_plans, valid = run.solve(ctx=ctx, plans_df=df)

    # Valid output
    assert valid is True

    # There should be two rows for p1; the first's to_x/to_y should match the second's from_x/from_y
    assert len(result_df) == 2
    df_sorted = result_df.sort_values("unique_leg_id").reset_index(drop=True)

    # l1 destination equals l2 origin and is at the intersection (5,0)
    l1 = df_sorted.iloc[0]
    l2 = df_sorted.iloc[1]
    assert l1.to_x == pytest.approx(l2.from_x)
    assert l1.to_y == pytest.approx(l2.from_y)
    assert l1.to_x == pytest.approx(5.0)
    assert l1.to_y == pytest.approx(0.0)

