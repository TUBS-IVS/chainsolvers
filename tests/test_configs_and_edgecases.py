import numpy as np
import pandas as pd
import pytest
from collections.abc import Mapping  # added

from chainsolvers import run, io
from chainsolvers.solvers.carla import Carla


def make_locations_for_line(length=20.0, step=5.0):
    """Create locations along x-axis at every `step` meters for a few act types.
    Includes a 'home' at x=0 and x=length, and activities at intermediate points.

    Adds extra low-potential off-line points to satisfy min-candidate thresholds
    without affecting selection (they score poorly).
    """
    acts = ["work", "shop", "leisure"]
    data = {"home": {}}
    # endpoints
    data["home"]["h_start"] = {"coordinates": [0.0, 0.0], "potential": 1.0, "name": "Home-Start"}
    data["home"]["h_end"] = {"coordinates": [length, 0.0], "potential": 1.0, "name": "Home-End"}

    # intermediate on-line candidates for each act type (good ones)
    for i, x in enumerate(np.arange(step, length, step), start=1):
        for act in acts:
            data.setdefault(act, {})
            data[act][f"{act[:1]}{i}"] = {
                "coordinates": [float(x), 0.0],
                "potential": float(1.0 + i),  # strictly higher than noise
            }

    # add off-line "noise" candidates (bad ones) to increase availability
    # Two parallel rows above/below the x-axis, but same potentials
    noise_pot = 1
    y_offsets = (-2.0, 2.0)
    xs = np.linspace(0.0, float(length), int(length) + 1)  # 0..length inclusive
    for act in acts:
        data.setdefault(act, {})
        k = 0
        for x in xs:
            for y in y_offsets:
                key = f"noise_{k}"
                # ensure we don't duplicate any on-line point
                data[act][key] = {"coordinates": [float(x), float(y)], "potential": float(noise_pot)}
                k += 1

    ids, coords, pots = io.build_locations_payload_from_dict(data)
    name_lookup = io.build_name_lookup_from_dict(data)
    return (ids, coords, pots), name_lookup


@pytest.mark.parametrize("anchor_strategy", [
    "lower_middle", "upper_middle", "start"  # keep "end" excluded
])
def test_long_chain_multiple_configs_anchor_strategies(anchor_strategy):
    # Build straight-line chain 0 -> 20 with four 5m legs
    (ids, coords, pots), name_lookup = make_locations_for_line(length=20.0, step=5.0)

    df = pd.DataFrame([
        {"unique_person_id": "p1", "unique_leg_id": "l1", "to_act_type": "work",
         "distance_meters": 5.0, "from_x": 0.0, "from_y": 0.0, "to_x": np.nan, "to_y": np.nan},
        {"unique_person_id": "p1", "unique_leg_id": "l2", "to_act_type": "shop",
         "distance_meters": 5.0, "from_x": np.nan, "from_y": np.nan, "to_x": np.nan, "to_y": np.nan},
        {"unique_person_id": "p1", "unique_leg_id": "l3", "to_act_type": "leisure",
         "distance_meters": 5.0, "from_x": np.nan, "from_y": np.nan, "to_x": np.nan, "to_y": np.nan},
        {"unique_person_id": "p1", "unique_leg_id": "l4", "to_act_type": "home",
         "distance_meters": 5.0, "from_x": np.nan, "from_y": np.nan, "to_x": 20.0, "to_y": 0.0},
    ])

    ctx = run.setup(
        locations_tuple=(ids, coords, pots),
        solver="carla",
        parameters={
            "anchor_strategy": anchor_strategy,
            # With the noise points, we can keep larger thresholds
            "number_of_branches": 5,
            "candidates_complex_case": 20,
            "candidates_two_leg_case": 20,
            "max_iterations_complex_case": 200,
        },
        rng_seed=7,
    )

    # inject names for enrichment
    ctx = run.RunnerContext(
        locations=ctx.locations,
        solver=ctx.solver,
        scorer=ctx.scorer,
        selector=ctx.selector,
        rng=ctx.rng,
        name_lookup=name_lookup,
    )

    result_df, result_plans, valid = run.solve(ctx=ctx, plans_df=df)
    assert valid is True

    # Expect all intermediate placements to sit on the straight line x in {5, 10, 15}, y=0
    df_sorted = result_df.sort_values("unique_leg_id").reset_index(drop=True)
    xs = [df_sorted.iloc[i].to_x for i in range(4)]
    ys = [df_sorted.iloc[i].to_y for i in range(4)]
    assert xs[0] == pytest.approx(5.0)
    assert xs[1] == pytest.approx(10.0)
    assert xs[2] == pytest.approx(15.0)
    assert xs[3] == pytest.approx(20.0)
    assert all(y == pytest.approx(0.0) for y in ys)

    # ensure plans return type normalized and DataFrame enriched with potentials
    assert isinstance(result_plans, (Mapping, type(None)))  # accept frozendict
    assert "to_act_potential" in result_df.columns


def test_setup_defaults_to_first_solver_when_none():
    (ids, coords, pots), _ = make_locations_for_line(length=10.0, step=5.0)
    ctx = run.setup(locations_tuple=(ids, coords, pots), solver=None, rng_seed=1)
    # First in registry is 'carla'
    assert isinstance(ctx.solver, Carla)


def test_missing_distance_allowed_when_flag_false_sets_zero():
    (ids, coords, pots), _ = make_locations_for_line(length=10.0, step=5.0)
    df = pd.DataFrame([
        {
            "unique_person_id": "p1",
            "unique_leg_id": "l1",
            "to_act_type": "home",
            "distance_meters": np.nan,  # missing
            "from_x": 0.0,
            "from_y": 0.0,
            "to_x": 5.0,
            "to_y": 0.0,
        }
    ])
    ctx = run.setup(locations_tuple=(ids, coords, pots), solver="carla", rng_seed=9)

    # default forbids missing distance
    with pytest.raises(ValueError):
        run.solve(ctx=ctx, plans_df=df)

    # When forbidden flag is off, it should proceed and set distance to 0.0 in conversion
    res_df, res_plans, ok = run.solve(ctx=ctx, plans_df=df, forbid_missing_distance=False)
    assert ok is True
    assert res_df.iloc[0].to_x == pytest.approx(5.0)