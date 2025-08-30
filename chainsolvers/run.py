# === Runner  ===========================================================
# Design: user selects solver via settings["solver"]; solvers are in-lib classes
# with .solve(...) and a strict boolean property .needs_segmented_plans.
# RNG may be a seed (int), a numpy Generator, or None. Names are enriched on
# the export DataFrame only. Name lookups are built early so big inputs can be
# freed during long solves.

from __future__ import annotations
from typing import Optional, Iterable, Mapping, Any, Type, Tuple

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from frozendict import frozendict

from .locations import Locations
from .scoring_selection import Scorer, Selector
from .solvers.carla import Carla
from .solvers.carla_plus import CarlaPlus
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

SOLVER_REGISTRY: dict[str, Type[Any]] = {"carla": Carla, "carla_plus": CarlaPlus}

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
    solver_name: Optional[str] = None,
    params: Optional[dict] = None,
    locations: Any,
    rng: np.random.Generator,
    progress: Optional[Any] = None,
    stats: Optional[Any] = None,
    scorer: Optional[Any] = None,
    selector: Optional[Any] = None,
    visualizer: Optional[Any] = None
) -> Any:

    if solver_name is None:
        solver_name = next(iter(SOLVER_REGISTRY)) # First in dict
        logger.info("No solver name provided; using '%s'.", solver_name)
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
    if not hasattr(solver, "needs_segmented_plans"):
        raise AttributeError(f"Solver '{type(solver).__name__}' must define boolean 'needs_segmented_plans'.")
    if not hasattr(solver, "solve") or not callable(getattr(solver, "solve")):
        raise AttributeError(f"Solver '{type(solver).__name__}' must implement a callable .solve(...).")

    return solver
#
# def run_population_solver(
#     *,
#     plans_df: pd.DataFrame,
#     locations_dict: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
#     locations_df: Optional[pd.DataFrame] = None,
#     solver: Optional[str] = None,
#     parameters: dict = None,
#
#     scorer: Optional[Any] = None,
#     selector: Optional[Any] = None,
#
#     stats: Optional[Any] = None, # Statstracker object
#     rng: Optional[np.random.Generator] = None,
#     rng_seed: Optional[int] = None,
#     progress: Optional[Iterable] = None,   # e.g., tqdm; if None, a no-op shim is used
#
#     validate_plans: bool = True,
#     forbid_negative_distance: bool = True,
#     forbid_missing_distance: bool = True,
#
#     include_extras_on_export: bool = True,
# ) -> Tuple[Optional[SegmentedPlans], pd.DataFrame]:
#     """
#
#     """
#     progress_fn = progress or _nop_progress
#     rng_obj = _normalize_rng(rng=rng, rng_seed=rng_seed)
#
#     # Instantiate Locations
#     if locations_dict is not None and locations_df is not None:
#         raise ValueError("Cannot provide both locations_dict and locations_df.")
#     elif locations_dict is not None:
#         identifiers, coordinates, potentials = io.build_locations_payload_from_dict(locations_dict)
#     elif locations_df is not None:
#         identifiers, coordinates, potentials = io.build_locations_payload_from_df(locations_df)
#     else:
#         raise ValueError("Must provide either locations_dict or locations_df.")
#     locations = Locations(identifiers, coordinates, potentials, stats)
#
#     # Instantiate scoring and selection
#     scor = scorer or Scorer()
#     selr = selector or Selector()
#
#     # Build name lookup EARLY and drop big inputs locally
#     name_lookup: dict[str, str] = {}
#     if locations_df is not None:
#         name_lookup.update(io.build_name_lookup_from_df(locations_df))
#         del locations_df
#     if locations_dict is not None:
#         name_lookup.update(io.build_name_lookup_from_dict(locations_dict))
#         del locations_dict
#
#     # Instantiate solver
#     solver_obj = _instantiate_solver(
#         solver_name=solver,
#         params=parameters,
#         locations=locations,
#         rng=rng_obj,
#         progress=progress_fn,
#         stats=stats,
#         scorer=scor,
#         selector=selr
#     )
#     if not getattr(solver_obj, "needs_segmented_plans", None) in (True, False):
#         raise AttributeError(f"Solver '{type(solver_obj).__name__}' must define boolean 'needs_segmented_plans'.")
#     if not hasattr(solver_obj, "solve") or not callable(getattr(solver_obj, "solve")):
#         raise AttributeError(f"Solver '{type(solver_obj).__name__}' must implement a callable .solve(...).")
#
#     logger.debug("Initialized solver '%s' (needs_segmented_plans=%s).",
#              type(solver_obj).__name__, solver_obj.needs_segmented_plans)
#
#     # Prepare input and run solver
#     if solver_obj.needs_segmented_plans:
#         logger.debug("Converting input plans DataFrame to SegmentedPlans...")
#         plans_in = io.convert_to_segmented_plans(plans_df,
#             validate=validate_plans,
#             forbid_negative_distance=forbid_negative_distance,
#             forbid_missing_distance=forbid_missing_distance
#         )
#         plans_in = io.segment_plans(plans_in)
#
#         logger.debug("Running solver.solve(plans=...)")
#         res = solver_obj.solve(plans=plans_in)
#     else:
#         logger.debug("Running solver.solve(df=...) (no conversion to SegmentedPlans)")
#         res = solver_obj.solve(df=plans_df)
#
#     # Normalize solver return
#     result_df = None
#     if isinstance(res, pd.DataFrame):
#         result_df = res
#         result_plans = None
#     elif isinstance(res, frozendict):
#         result_plans = res  # already a SegmentedPlans
#     elif isinstance(res, dict):
#         # If a plain dict sneaks through in internal code, normalize to frozendict
#         result_plans = frozendict(res)
#     else:
#         raise TypeError(
#             f"Solver returned unsupported type {type(res).__name__}. "
#             "Expected SegmentedPlans (frozendict) or pandas.DataFrame."
#         )
#
#     # Ensure DataFrame and enrich with names
#     if result_df is None:
#         logger.debug("Converting SegmentedPlans result to DataFrame...")
#         assert result_plans is not None
#         result_df = io.segmented_plans_to_dataframe(
#             result_plans, include_extras=include_extras_on_export
#         )
#
#     if name_lookup:
#         logger.debug("Enriching output DataFrame with human-readable names...")
#         result_df = io.enrich_plans_df_with_names(
#             result_df,
#             name_lookup=name_lookup,
#         )
#
#     return result_plans, result_df

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
    visualizer: Optional[Any] = None
) -> RunnerContext:

    progress_fn = progress or _nop_progress
    rng_obj = _normalize_rng(rng=rng, rng_seed=rng_seed)

    # --- Locations
    if (locations_dict is None) == (locations_df is None) == (locations_tuple is None):
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
        visualizer = visualizer
    )

    needs_flag = getattr(solver_obj, "needs_segmented_plans", None)
    if needs_flag not in (True, False):
        raise AttributeError(f"Solver '{type(solver_obj).__name__}' must define boolean 'needs_segmented_plans'.")
    if not callable(getattr(solver_obj, "solve", None)):
        raise AttributeError(f"Solver '{type(solver_obj).__name__}' must implement .solve(...).")

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

    # validation / conversion options
    validate_plans: bool = True,
    forbid_negative_distance: bool = True,
    forbid_missing_distance: bool = True,

    # export options
    include_extras_on_export: bool = True,
) -> Tuple[Optional["SegmentedPlans"], pd.DataFrame, bool]:

    solver_obj = ctx.solver

    io.validate_input_plans_df(plans_df)

    # Prepare input for solver
    if solver_obj.needs_segmented_plans:
        plans_in = io.convert_to_segmented_plans(
            plans_df,
            forbid_negative_distance=forbid_negative_distance,
            forbid_missing_distance=forbid_missing_distance,
        )
        plans_in = io.segment_plans(plans_in)
        res = solver_obj.solve(plans=plans_in)
    else:
        res = solver_obj.solve(df=plans_df)

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

    valid = io.validate_output_plans_df(result_df)

    return result_df, result_plans, valid

