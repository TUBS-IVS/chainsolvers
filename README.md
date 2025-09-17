# chainsolvers

⚠️ Experimental/unstable API

**chainsolvers** is a Python library for solving *point placement along chains* problems — for example, distributing activities along activity chains to feasible locations. It provides pluggable solver routines, together with configurable **scorers** and **selectors**, to flexibly evaluate and select candidate solutions.

## Quickstart

Use the two-step runner: `setup(...) -> RunnerContext` and `solve(ctx=..., plans_df=...)`.

```python
import chainsolvers as cs
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


# 1) Candidate locations (choose exactly one source)
locations_df = pd.DataFrame([
    # minimal columns: id, act_type, x, y
    {"activities": "work; business; leisure", "id": "1", "x": 15.0, "y": 13.0, "potential": 100.0, "name": "Business Factory"},
    {"activities": "leisure", "id": "2", "x": 10.0, "y": 10.0, "potential": 100000.0, "name": "Central Park"},
    {"activities": "education; sports", "id": "3", "x": 10.0, "y": 10.0, "potential": 5000.0, "name": "Big School"},
])

# 2) Create a runner context
ctx = cs.setup(
    locations_df=locations_df,     # or: locations_dict=... (nested mapping)
    solver="carla",           # from cs.SOLVER_REGISTRY: {"carla","carla_plus"}
    parameters={"number_of_branches": 20, "candidates_complex_case": 50},  # passed to the chosen solver, unspecified uses default
    # rng_seed=42,                 # or pass a numpy Generator
    # scorer=CustomScorer(),       # optional; defaults provided
    # selector=CustomSelector(),   # optional; defaults provided
)

# 3) Input plans (long format). Required columns use default PlanColumns:
#    person_id, leg_id, to_act_type, distance_m, from_x, from_y, to_x, to_y
plans_df = pd.DataFrame([
    {"unique_person_id": "p1", "unique_leg_id": "p1-1", "to_act_type": "work", "distance_meters": 5000,
     "from_x": 10.0, "from_y": 10.0, "to_x": float("nan"), "to_y": float("nan")},
    {"unique_person_id": "p1", "unique_leg_id": "p1-2", "to_act_type": "home", "distance_meters": 4900,
     "from_x": float("nan"), "from_y": float("nan"), "to_x": 300.0, "to_y": 350.4},
])

# 4) Solve
placed_df, placement_plans, valid = cs.solve(ctx=ctx, plans_df=plans_df)

print(valid)
print(placed_df)
```

## Returns
Tuple:
- **`placement_df`**: `pandas.DataFrame` (always returned). Results in df format.
- **`placed_plans`**: `SegmentedPlans` (`frozendict[str, tuple[Segment, ...]]`) or `None`. Results in the internal `SegmentedPlans` (may be useful, else just ignore).
- **`valid`**: `bool`. Whether the output validation succeeded. 


