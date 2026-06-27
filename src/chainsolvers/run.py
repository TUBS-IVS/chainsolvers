# === Runner  ===========================================================
# Design: user selects solver via settings["solver"]; solvers are in-lib classes
# with .solve(...) and a strict boolean property .needs_segmented_plans.
# RNG may be a seed (int), a numpy Generator, or None. Names are enriched on
# the export DataFrame only. Name lookups are built early so big inputs can be
# freed during long solves.

from __future__ import annotations
from typing import Optional, Iterable, Mapping, Any, Type, Tuple, Dict
import os

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from frozendict import frozendict

from .locations import Locations
from .scoring_selection import Scorer, Selector
from .solvers.carla import Carla, CarlaSample
from .solvers.carla_plus import CarlaPlus
from .solvers.dp import Dp, CarlaDp, DpRefine, CarlaDpRefine, DpPotential, DpFull, DpSample, Milp
from .types import PlanColumns, SegmentedPlans, Households
from . import io

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RunnerContext:
    locations: "Locations"
    solver: Any
    scorer: Any
    selector: Any
    rng: np.random.Generator
    name_lookup: dict[str, str]   # for post-enrichment

SOLVER_REGISTRY: dict[str, Type[Any]] = {
    "carla": Carla,
    "carla_sample": CarlaSample,     # same Carla, pre-wired to sample (mnl + 1 branch); greedy-ancestral ablation vs dp_sample
    "carla_plus": CarlaPlus,
    "dp_full": DpFull,               # the pure DP: exact over the full per-type catalog (global-optimum oracle)
    "dp_rings": Dp,                  # pruned DP over ring-envelope candidates
    "dp_carla": CarlaDp,             # pruned DP over CARLA's generation (rings + 2-leg circle-intersection)
    "dp_rings_refine": DpRefine,     # dp_rings + iterative neighbour refinement
    "dp_carla_refine": CarlaDpRefine,# dp_carla + iterative neighbour refinement
    "dp_carla_pot": DpPotential,     # dp_carla_refine + potential-aware pooling (near-exact for combined obj)
    "dp_sample": DpSample,           # generative forward-backward MNL sampler
    "milp": Milp,                    # MILP oracle (== dp_rings)
}

# The input formats a solver may request via its `wanted_format` attribute.
WANTED_FORMATS: frozenset[str] = frozenset({"segmented_plans", "households", "df"})

def _nop_progress(seq: Iterable, **_: Any) -> Iterable:
    """Progress shim: allows solvers to call progress(range(n)) without tqdm."""
    return seq

def _normalize_rng(rng: Optional[Any] = None, rng_seed: Optional[int] = None) -> np.random.Generator:
    """Accept None, int, or numpy Generator and return a numpy Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng_seed is not None:
        return np.random.default_rng(int(rng_seed))
    return np.random.default_rng()


def _instantiate_solver(
    *,
    solver_name: str,
    params: Optional[dict] = None,
    locations: Locations,
    rng: np.random.Generator,
    progress: Optional[Any] = None,
    stats: Optional[Any] = None,
    scorer: Optional[Any] = None,
    selector: Optional[Any] = None,
    visualizer: Optional[Any] = None
) -> Any:

    # `solver_name` may be a registered name (str) or a solver class directly (lets
    # callers — e.g. eval baselines or user code — plug a solver without registering it).
    if isinstance(solver_name, type):
        SolverCls = solver_name
    else:
        try:
            SolverCls = SOLVER_REGISTRY[solver_name]
        except KeyError as e:
            raise ValueError(f"Unknown solver '{solver_name}'. Available: {sorted(SOLVER_REGISTRY)}") from e

    if params is None:
        params = {}

    # Unexpected keys in 'params' will raise TypeError here.
    solver = SolverCls(
        locations=locations,
        rng=rng,
        progress=progress,
        stats=stats,
        scorer=scorer,
        selector=selector,
        visualizer=visualizer,
        **params,
    )

    # Enforce contract early
    if not hasattr(solver, "wanted_format"):
        raise AttributeError(
            f"Solver '{type(solver).__name__}' must define string 'wanted_format' "
            f"(one of {sorted(WANTED_FORMATS)})."
        )
    if solver.wanted_format not in WANTED_FORMATS:
        raise ValueError(
            f"Solver '{type(solver).__name__}' has wanted_format={solver.wanted_format!r}; "
            f"must be one of {sorted(WANTED_FORMATS)}."
        )
    if not hasattr(solver, "required_leg_fields") or not callable(getattr(solver, "required_leg_fields")):
        raise AttributeError(
            f"Solver '{type(solver).__name__}' must implement required_leg_fields(self) -> set[str]."
        )
    if not hasattr(solver, "solve") or not callable(getattr(solver, "solve")):
        raise AttributeError(f"Solver '{type(solver).__name__}' must implement a callable .solve(...).")

    return solver


def setup(
    *,
    # locations sources (one required)
    locations_dict: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
    locations_df: Optional[pd.DataFrame] = None,
    locations_tuple: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,

    # solver selection
    solver: Optional[str] = None,
    parameters: Optional[dict] = None,

    # scoring / selection
    scorer: Optional[Any] = None,
    selector: Optional[Any] = None,

    # infra
    stats: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    rng_seed: Optional[int] = None,
    progress: Optional[Any] = None,  # tqdm-like wrapper or None
    visualizer: Optional[Any] = None,
    visualizer_crs: Any = 25832,  # source CRS when `visualizer` is a path; EPSG code or pyproj spec
) -> RunnerContext:

    progress_fn = progress or _nop_progress
    rng_obj = _normalize_rng(rng=rng, rng_seed=rng_seed)

    if solver is None:
        solver = next(iter(SOLVER_REGISTRY)) # First in dict
        logger.info("No solver name provided; using '%s'.", solver)

    # Optionally construct a visualizer if a path-like or string is provided.
    # We import lazily so projects without viz extras installed are unaffected unless used.
    viz_obj = visualizer
    if isinstance(visualizer, (str, os.PathLike)):
        try:
            from .visualizer import Visualizer as _VizCls
        except ImportError as e:
            raise ImportError(
                "Visualizer requires optional dependencies. Install with: pip install 'chainsolvers[viz]'"
            ) from e
        viz_obj = _VizCls(savedir=str(visualizer), map_prefix=solver, source_crs=visualizer_crs)

    # --- Locations
    if sum(x is not None for x in (locations_dict, locations_df, locations_tuple)) != 1:
        raise ValueError("Provide exactly one of locations_dict, locations_df, or locations_tuple.")

    if locations_dict is not None:
        identifiers, coordinates, potentials = io.build_locations_payload_from_dict(locations_dict)
        name_lookup = io.build_name_lookup_from_dict(locations_dict)
    elif locations_df is not None:
        identifiers, coordinates, potentials = io.build_locations_payload_from_df(locations_df)  # type: ignore[arg-type]
        name_lookup = io.build_name_lookup_from_df(locations_df)  # type: ignore[arg-type]
    elif locations_tuple is not None:
        identifiers, coordinates, potentials = locations_tuple
        name_lookup = {}
    else:
        raise ValueError("Cannot reach this point.")
    locations = Locations(identifiers, coordinates, potentials, stats)

    # --- Scoring / selection
    scor = scorer or Scorer()
    selr = selector or Selector()

    # --- Solver
    solver_obj = _instantiate_solver(
        solver_name=solver,
        params=parameters or {},
        locations=locations,
        rng=rng_obj,
        progress=progress_fn,
        stats=stats,
        scorer=scor,
        selector=selr,
        visualizer=viz_obj
    )

    return RunnerContext(
        locations=locations,
        solver=solver_obj,
        scorer=scor,
        selector=selr,
        rng=rng_obj,
        name_lookup=name_lookup,
    )


def solve(
    *,
    ctx: RunnerContext,
    plans_df: pd.DataFrame,
    forbid_negative_distance: bool = True,
    forbid_missing_distance: bool = True,
    include_extras_on_export: bool = True,
) -> Tuple[pd.DataFrame, Optional["SegmentedPlans"], bool]:

    solver_obj = ctx.solver
    cols = PlanColumns()

    # Ask the solver which leg fields it needs
    required_leg_fields = solver_obj.required_leg_fields()
    fmt = solver_obj.wanted_format

    # Convert to column names for validation
    required_cols = io.get_required_df_columns(required_leg_fields)
    required_cols.add(cols.person_id)  # Always need person_id for grouping
    if fmt == "households":
        required_cols.add(cols.household_id)  # Households format needs household_id

    io.validate_input_plans_df(plans_df, required_cols=required_cols)
    io.summarize_plans_df(plans_df)

    # --- Build input structure based on wanted_format ----------------------------
    if fmt == "segmented_plans":
        plans_in = io.convert_to_segmented_plans(
            plans_df,
            required_leg_fields=required_leg_fields,
            forbid_negative_distance=forbid_negative_distance,
            forbid_missing_distance=forbid_missing_distance,
        )
        plans_in = io.segment_plans(plans_in)
        res = solver_obj.solve(plans=plans_in)

    elif fmt == "households":
        households_in = io.convert_to_households(
            plans_df,
            required_leg_fields=required_leg_fields,
            forbid_negative_distance=forbid_negative_distance,
            forbid_missing_distance=forbid_missing_distance,
        )

        # Find connections if solver requires connected_legs field
        if "connected_legs" in required_leg_fields:
            # Extract connection config from solver
            config = getattr(solver_obj, 'config', None)
            if config:
                households_in = io.find_connected_legs_in_households(
                    households_in,
                    distance_tolerance_m=config.distance_tolerance_m,
                    dep_tolerance_s=config.dep_tolerance_s,
                    arr_tolerance_s=config.arr_tolerance_s,
                    compatible_modes=config.compatible_modes,
                    compatible_activities=config.compatible_activities,
                )

        res = solver_obj.solve(households=households_in)

    elif fmt == "df":
        res = solver_obj.solve(df=plans_df)

    else:
        raise ValueError(f"Unknown wanted_format {fmt!r}")

    # Normalize returns
    result_plans = None
    result_df = None

    if isinstance(res, pd.DataFrame):
        result_df = res
    elif isinstance(res, frozendict):
        result_plans = res
    elif isinstance(res, dict):
        result_plans = frozendict(res)
    else:
        raise TypeError(
            f"Solver returned {type(res).__name__}; expected pandas.DataFrame or SegmentedPlans (frozendict)."
        )

    # Ensure DataFrame + optional name enrichment
    if result_df is None:
        assert result_plans is not None
        result_df = io.segmented_plans_to_dataframe(result_plans, include_extras=include_extras_on_export)

    if ctx.name_lookup:
        result_df = io.enrich_plans_df_with_names(result_df, name_lookup=ctx.name_lookup)
    result_df = io.enrich_plans_df_with_potentials(result_df, locations=ctx.locations)

    valid = io.validate_output_plans_df(result_df)

    # Optional visualization
    viz = getattr(solver_obj, "visualizer", None)
    if viz is not None:
        logger.info("Running visualizer...")
        for meth in ("visualize", "visualize_levels"):
            fn = getattr(viz, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    logger.warning("Visualizer.%s failed: %s", meth, e)

    return result_df, result_plans, valid
