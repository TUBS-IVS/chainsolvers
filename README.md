# chainsolvers

**chainsolvers** is a Python library for solving *point placement along chains* problems — for example, distributing activities along activity chains to feasible locations. It provides pluggable solver routines, together with configurable **scorers** and **selectors**, to flexibly evaluate and select candidate solutions.

## Quickstart

```python
pip install chainsolvers
```

Use the two-step runner: `setup(...) -> RunnerContext` and `solve(ctx=..., plans_df=...)`.

```python
import chainsolvers as cs
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# 1) Candidate locations
# Provide exactly one of: locations_df, locations_dict, or locations_tuple (these are just different ways of representing the same thing)
# df is probably the easiest coming from a csv, geopackage or similar, tuple is the internal format
locations_df = pd.DataFrame([
    # minimal columns: id, act_type, x, y
    # optional: name, potentials (plural, one per possible activity type at this location, set to 0 if not specified)
    {"activities": "work; business; leisure", "id": "1", "x": 15.0, "y": 13.0, "name": "Business Factory"},
    {"activities": "leisure", "id": "2", "x": 10.0, "y": 10.0, "potentials": 100000.0, "name": "Central Park"},
    {"activities": "education; sports", "id": "3", "x": 10.0, "y": 10.0, "potentials": "5000.0; 60", "name": "Big School"},
])

locations_dict = {
    "work": {
        "1": {"coordinates": [15.0, 13.0], "name": "Business Factory"},
    },
    "business": {
        "1": {"coordinates": [15.0, 13.0], "name": "Business Factory"},
    },
    "leisure": {
        "1": {"coordinates": [15.0, 13.0], "name": "Business Factory"},
        "2": {"coordinates": [10.0, 10.0], "potential": 100000.0, "name": "Central Park"}, # potential, singular
    },
    "education": {
        "3": {"coordinates": [10.0, 10.0], "potential": 5000.0, "name": "Big School"},
    },
    "sports": {
        "3": {"coordinates": [10.0, 10.0], "potential": 60.0, "name": "Big School"},
    },
}

locations_tuple = (
    {
        "work":      np.array(["1"], dtype=object),
        "business":  np.array(["1"], dtype=object),
        "leisure":   np.array(["1", "2"], dtype=object),
        "education": np.array(["3"], dtype=object),
        "sports":    np.array(["3"], dtype=object),
    },
    {
        "work":      np.array([[15.0, 13.0]], dtype=float),
        "business":  np.array([[15.0, 13.0]], dtype=float),
        "leisure":   np.array([[15.0, 13.0], [10.0, 10.0]], dtype=float),
        "education": np.array([[10.0, 10.0]], dtype=float),
        "sports":    np.array([[10.0, 10.0]], dtype=float),
    },
    {
        "work":      np.array([0], dtype=float),
        "business":  np.array([0], dtype=float),
        "leisure":   np.array([0, 100000.0], dtype=float),
        "education": np.array([5000.0], dtype=float),
        "sports":    np.array([60.0], dtype=float),
    },
)

# 2) Create a runner context
ctx = cs.setup(
    locations_df=locations_df,  # or locations_dict= or locations_tuple=...
    # --- optional parameters ---
    # solver="carla",           # defaults to "carla"
    # parameters={              # parameters for the solver (uses default values if not specified)
    #     "number_of_branches": 50,
    #     "candidates_complex_case": 100,
    #     "candidates_two_leg_case": 40,
    #     "anchor_strategy": "lower_middle",
    #     "selection_strategy_complex_case": "top_n_spatial_downsample",
    #     "selection_strategy_two_leg_case": "top_n",
    #     "max_iterations_complex_case": 100,
    # },
    # rng_seed=42,              # or pass a numpy Generator
    # scorer=CustomScorer(),    # uses default scorer if not specified
    # selector=CustomSelector() # uses default selector if not specified
    # progress=tqdm,            # for progress bars, use your own if you want, no progress bars shown if not specified
    # visualizer=None,          # "out/maps" will create a default Visualizer(savedir="out/maps") or pass your own CustomVisualizer()
)

# 3) Input plans. Minimum required columns:
#    unique_person_id, unique_leg_id, to_act_type, distance_meters, from_x, from_y, to_x, to_y
plans_df = pd.DataFrame([
    {"unique_person_id": "p1", "unique_leg_id": "p1-1", "to_act_type": "work", "distance_meters": 5000,
     "from_x": 10.0, "from_y": 10.0, "to_x": float("nan"), "to_y": float("nan")},
    {"unique_person_id": "p1", "unique_leg_id": "p1-2", "to_act_type": "home", "distance_meters": 4900,
     "from_x": float("nan"), "from_y": float("nan"), "to_x": 300.0, "to_y": 350.4},
])

# 4) Solve
result_df, result_plans, valid = cs.solve(ctx=ctx, plans_df=plans_df)

print(valid)
print(result_df)
print(result_plans)

```

## Returns
A tuple of three elements (in order):
- **`result_df`**: `pandas.DataFrame` (always returned). Placed plans in same df format as input plans.
- **`result_plans`**: `SegmentedPlans` (`frozendict[str, tuple[Segment, ...]]`) or `None`. Results in the internal `SegmentedPlans` (may be useful, else just ignore).
- **`valid`**: `bool`. Whether the output validation succeeded. 


