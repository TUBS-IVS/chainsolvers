# python
import numpy as np
import pandas as pd
import pytest

from chainsolvers import run, io


def make_locations_two_intersections():
    """
    Construct a catalog with several 'work' points clustered near the two
    circle-intersection points for:
      start=(0,0), end=(10,0), distances = 6 and 6
    Intersections are near (5, ±sqrt(11)) ≈ (5, ±3.3166).
    Also add minimal 'home' endpoints.
    """
    y = float(np.sqrt(11.0))
    work_points = [
        ("w1", [5.0,  y], 10.0),
        ("w2", [5.2,  y],  9.0),
        ("w3", [4.8,  y],  8.0),
        ("w4", [5.0, -y], 10.0),
        ("w5", [5.2, -y],  9.0),
        ("w6", [4.8, -y],  8.0),
    ]
    data = {
        "leisure": {
            "h_start": {"coordinates": [0.0, 0.0], "potential": 1.0, "name": "Home-Start"},
            "h_end":   {"coordinates": [10.0, 0.0], "potential": 1.0, "name": "Home-End"},
        },
        "work": {wid: {"coordinates": xy, "potential": pot} for wid, xy, pot in work_points},
    }
    return io.build_locations_payload_from_dict(data)

def test_integration_two_leg_two_intersections_k_gt_1_axis_shape_guard():
    """
    Integration: two-leg case with two circle intersections and k=2.
    If ids are taken along axis=1 in query_closest, this will raise an AxisError.
    With axis=0 for ids/coords/pots, it should pass and place the activity near (5, ±sqrt(11)).
    """
    ids, coords, pots = make_locations_two_intersections()

    df = pd.DataFrame([
        # Leg 1: known start -> unknown work, distance 6
        {"unique_person_id": "p1", "unique_leg_id": "l1", "to_act_type": "work",
         "distance_meters": 6.0, "from_x": 0.0, "from_y": 0.0, "to_x": np.nan, "to_y": np.nan},
        # Leg 2: unknown work -> known home, distance 6
        {"unique_person_id": "p1", "unique_leg_id": "l2", "to_act_type": "home",
         "distance_meters": 6.0, "from_x": np.nan, "from_y": np.nan, "to_x": 10.0, "to_y": 0.0},
    ])

    ctx = run.setup(
        locations_tuple=(ids, coords, pots),
        solver="carla",
        parameters={
            "candidates_two_leg_case": 2,     # force k=2 per intersection
            "selection_strategy_two_leg_case": "top_n",
            "number_of_branches": 1,          # irrelevant for n=2
        },
        rng_seed=123,
    )

    result_df, result_plans, valid = run.solve(ctx=ctx, plans_df=df)
    assert valid is True

    # The placed work should be near either of the intersection clusters
    row = result_df.loc[result_df.unique_leg_id == "l1"].iloc[0]
    assert row.to_x == pytest.approx(5.0, abs=0.5)
    assert abs(row.to_y) == pytest.approx(np.sqrt(11.0), abs=0.6)

