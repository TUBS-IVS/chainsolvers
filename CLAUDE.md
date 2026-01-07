# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chainsolvers** is a Python library for solving *point placement along chains* problems. It distributes activities along activity chains to feasible locations using pluggable solver routines with configurable scorers and selectors.

## Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_integration_pipeline.py -v

# Run a specific test
pytest tests/test_integration_pipeline.py::test_name -v
```

### Installation
```bash
# Development install
pip install -e .

# With visualization extras
pip install -e ".[viz]"

# Using uv (if available)
uv pip install -e .
```

## Architecture

### Two-Step Execution Model

The library uses a setup/solve pattern:

1. **Setup Phase**: `cs.setup(...)` → `RunnerContext`
   - Initializes locations, solver, scorer, selector, RNG
   - Builds spatial indices (KD-trees via scipy)
   - Validates input data structures

2. **Solve Phase**: `cs.solve(ctx=..., plans_df=...)` → `(result_df, result_plans, valid)`
   - Validates input plans
   - Converts plans to solver-specific format
   - Executes solver algorithm
   - Enriches results with names & potentials

### Key Modules

- **`run.py`**: Core pipeline orchestrating setup and solve phases
- **`types.py`**: Data structures (`Leg`, `Segment`, `SegmentedPlan`, `PlanColumns`, `LocationColumns`)
- **`locations.py`**: Spatial indexing with KD-trees for nearest-neighbor queries
- **`io.py`**: Input/output conversion between DataFrame and internal formats
- **`scoring_selection.py`**: Scorer and Selector classes for candidate evaluation
- **`solvers/carla.py`**: Main CARLA solver - recursive placement of activity locations
- **`solvers/carla_plus.py`**: Extended solver (placeholder for household-level optimization)

### Data Flow

```
locations (dict/df/tuple) + plans_df
    ↓ cs.setup()
RunnerContext (locations index, solver, scorer, selector)
    ↓ cs.solve()
(result_df, result_plans, valid)
```

### Key Data Structures

- **Leg**: NamedTuple with leg_id, distance, from/to locations, activity type, mode, times
- **Segment**: `Tuple[Leg, ...]` - sequence of legs within a segment
- **SegmentedPlan**: `Tuple[Segment, ...]` - plan for one person
- **SegmentedPlans**: `frozendict[str, SegmentedPlan]` - plans keyed by person_id
- **PlanColumns/LocationColumns**: Dataclasses defining standard column name mappings

### Solver Registry

Solvers are registered in `SOLVER_REGISTRY` and must implement:
- `wanted_format` property: "segmented_plans", "households", or "df"
- `required_df_columns(PlanColumns) -> set[str]` method
- `solve(...)` method

## Dependencies

Core: `numpy`, `pandas`, `frozendict`, `scipy`, `pytest`

Optional (visualization): `folium`, `networkx`, `pyproj`, `matplotlib`
