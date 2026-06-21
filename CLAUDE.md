# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chainsolvers** is a Python library for solving *point placement along chains* problems. It
distributes activities along activity chains (trip sequences) onto feasible real-world locations
using pluggable solver routines with configurable scorers and selectors. The primary algorithm is
**CARLA**, a recursive anchor-based placement solver backed by KD-tree spatial queries.

## Repository Layout

The repo is split into three tiers with hard boundaries:

- **`src/chainsolvers/`** — the **library** (MIT, the only thing that ships). src-layout: the wheel
  packages exactly this dir (`[tool.hatch.build.targets.wheel] packages = ["src/chainsolvers"]`), so
  nothing else can leak into a `pip install`.
- **`research/`** — the **research/eval tier**, a *separate* package `chainsolvers_eval` (imported as
  `chainsolvers_eval`, NOT `chainsolvers.eval`). It depends on the library; the library never depends
  on it (enforced by `tests/unit/test_import_boundaries.py`). Holds `chainsolvers_eval/` (synth,
  calibration, survey, baselines, external, interop, viz), `scripts/` (benchmark/prognosis/survey
  harnesses), `data/` (**gitignored** — MiD HTS CSVs, never committed) and `out/` (**gitignored** —
  generated figures).
- **`_private/`** — **gitignored**, NON-DISTRIBUTED: vendored GPL tools (eqasim) + glue. Keeps GPL out
  of the MIT library.
- **`paper/` / `papers/`** — **gitignored**: the manuscript is not tracked in this repo.
- **`tests/unit/`** library tests, **`tests/research/`** eval tests.

## Commands

### Testing
```bash
pytest tests/unit -v                                   # library tests (default testpath)
pytest tests/research -v                               # eval tests (needs research pkg installed)
pytest tests/unit/test_integration_pipeline.py::test_name -v  # one test
```

### Installation
```bash
pip install -e .                 # library (from src/chainsolvers)
pip install -e ./research        # research/eval package (chainsolvers_eval), depends on the library
pip install -e ".[viz]"          # library + visualization extras (folium, networkx, pyproj, matplotlib)
```

Python >= 3.9. Build backend is `hatchling`.

## Architecture

### Two-Step Execution Model

1. **Setup**: `cs.setup(...)` → `RunnerContext`
   - Loads locations from one of three sources (dict / DataFrame / pre-built tuple)
   - Builds per-activity-type KD-trees (`scipy.spatial.cKDTree`)
   - Instantiates the solver from `SOLVER_REGISTRY` and enforces its interface contract
   - Normalizes RNG, wires up optional scorer/selector/stats/progress/visualizer
   - Returns a frozen `RunnerContext`

2. **Solve**: `cs.solve(ctx=..., plans_df=...)` → `(result_df, result_plans, valid)`
   - Validates input plan columns (derived from the solver's `required_leg_fields()`)
   - Converts the DataFrame into the solver's `wanted_format` (segmented_plans / households / df)
   - Runs the solver
   - Converts internal structures back to a DataFrame, enriches with names & potentials
   - Validates output and (optionally) runs the visualizer

The only public exports (`chainsolvers/__init__.py`) are `setup` and `solve`.

### Key Modules

- **`run.py`**: Pipeline orchestration, `SOLVER_REGISTRY`, solver instantiation + contract checks,
  RNG normalization, `setup`/`solve`, `RunnerContext` dataclass.
- **`types.py`**: Core data structures (`Leg`, `Segment`, `SegmentedPlan`, `SegmentedPlans`,
  `Households`, `PlanColumns`, `LocationColumns`).
- **`io.py`**: DataFrame ⇄ internal-format conversion, input/output validation, leg construction,
  household grouping, connection detection, name/potential enrichment.
- **`locations.py`**: `LocationsIndex` — KD-tree spatial indexing with ring / overlapping-ring /
  circle-intersection candidate generation and the full candidate→score→select pipelines.
- **`scoring_selection.py`**: `Scorer` (scores candidates) and `Selector` (picks indices via
  monte_carlo / top_n / mixed / spatial_downsample / top_n_spatial_downsample).
- **`helpers.py`**: Geometry utilities (euclidean, circle intersections, radii expansion, angle
  sector tests, spatial downsampling) and connection-matching predicates.
- **`solvers/carla.py`**: Main CARLA solver — recursive anchor-based placement.
- **`solvers/carla_plus.py`**: Household-level solver — **currently a placeholder** (all methods
  raise `NotImplementedError`).
- **`solvers/dp.py`**: Exact solvers. Registry key → class: `dp_full`→`DpFull`, `dp_rings`→`Dp`,
  `dp_carla`→`CarlaDp`, `dp_rings_refine`→`DpRefine`, `dp_carla_refine`→`CarlaDpRefine`,
  `dp_carla_pot`→`DpPotential`, `milp`→`Milp`. The separable objective is a layered shortest path; per
  segment they generate candidate facilities per free node via the shared `solve_chain(...)` core.
  **Only `dp_full` is a pure (unpruned) DP** → exact over the **full per-type catalog** (`O(n·N²)`; the
  true global optimum; an oracle, not scalable). Every other DP-family solver is *pruned*: it reuses
  **CARLA's** geometric generation, so it is exact only over the candidates it generates. `dp_rings`
  uses overlapping-ring (triangle-inequality) envelopes; `dp_carla` adds circle-intersection for the
  single-intermediate (two-leg) case (= CARLA's exact generation, so `carla` vs `dp_carla` isolates
  search vs branching). `dp_rings_refine`/`dp_carla_refine` add iterative neighbour-based candidate
  refinement (`_refine`): re-bracket each node by its provisional neighbours, carry the previous choice
  forward (monotone), re-solve until convergence — closing the recall gap one-shot endpoint generation
  leaves. `dp_carla_pot` adds **potential-aware pooling** (`pot_pool_k`): augment every pool with the
  top-K facilities by potential, keeping the pruned DP near-exact for the **combined** (α>0) objective
  the distance envelope would otherwise prune. `Milp` solves the identical assignment as a min-cost-flow
  MILP via `scipy.optimize.milp` (HiGHS) — an exact oracle (== `dp_rings`) for validation / future
  non-separable constraints, not scalable. All read `alpha`/`beta` from the injected `Scorer`
  (mode/weights) so they optimize the *same* objective as CARLA. The pruned solvers are exact over
  their generated candidate set; `dp_full` is exact globally (`milp` == `dp_rings`); `dp_carla_pot`
  and `dp_full` extend exactness toward the combined (α>0) objective.
**`research/chainsolvers_eval/` is a separate, non-distributed package** (imported as
`chainsolvers_eval`, install via `pip install -e ./research`). The library core never depends on it
— core → `chainsolvers_eval` imports are forbidden and enforced by
`tests/unit/test_import_boundaries.py`; `chainsolvers_eval` may import core, e.g. `attr_value` lives
in the library's `solvers/dp.py`. It holds the synthetic-world generator, calibration, and survey
primitives used only for benchmarking/experiments.

- **`research/chainsolvers_eval/synth.py`**: composable synthetic-world generator. `build_topology(...)` (multi-type
  clustered locations + latent structural attractiveness `sizes`), `generate_chains(topo, ...,
  sizes=...)` (gravity-placed chains; potentials are the resulting visit counts), and
  `topology_locations_tuple(topo, values)` (hand a model a chosen attractiveness vector).
  `generate_world(...)` → `SyntheticWorld` composes them. Builds a ground-truth eval scenario:
  ~1000 clustered **multi-type** locations with latent attractiveness; a population of activity
  chains placed on real points via a gravity rule (anchors `home`/`work` are known to the solver,
  secondary activities are to be placed); **potentials are derived from usage** (per-(location,type)
  visit counts). Returns `locations_tuple`, `plans_df` (solver input), and `ground_truth` (true
  facility id/coords per leg). Because activities sit on real points, the global-optimal deviation
  is ~0 (so *recovery* of the true facility is the discriminating metric); `distance_noise` adds
  relative noise to observed distances, creating the ambiguity where usage-based potentials measurably
  improve recovery (`combined` mode, with `pot_weight` scaled to the deviation magnitude).
- **`research/scripts/benchmark_solvers.py`**: Comparison harness. Synthetic by default via `generate_world`
  (reports **recov%** / placement error vs ground truth), or real locations/plans CSVs via
  `--locations-csv`/`--plans-csv`. Reports mean total |Δd|, runtime, pairwise win/tie/loss, and
  (with `--oracle`) `opt%` / `%gap` vs the true-global `dp_full`. `--min-candidates` controls
  DP-family pool size (small values expose recall gaps); plus a chain-length scaling sweep.
- **`research/chainsolvers_eval/calibration.py`**: `fit_location_choice(topo, plans_df, gt)` — maximum-likelihood estimation
  of the structural MNL parameters (`alpha` attractiveness sensitivity, `scale` distance decay)
  from observed choices over the full per-type candidate set (`transform` selects the attractiveness
  form, `log`/`log1p`, and must match the solver's). These are the stable parameters that give the
  model prognosis ability; with the matched `log` form it recovers the true (alpha≈1, scale).
  `dp_sample` takes an `attr_transform` parameter to match.
- **`research/chainsolvers_eval/baselines.py`**: comparison baselines run via `setup(solver=<class>)` (no registry
  entry needed). `RelaxationDiscretization` = faithful Hörl & Axhausen RDA (eqasim): gravity-chain
  continuous relaxation + nearest-facility discretization over an assignment loop keeping the
  lowest-deviation chain (single intermediate via circle-intersection trig). `RelaxationDiscretizationGuided`
  adds the **guidance forces** of Langrognet et al. 2026 (eqasim `force_model`): the relaxation is biased
  by a normalized inverse-cube attraction toward POI-weighted attractors (weights = potentials) — the
  attractiveness-aware RDA, counterpart to CARLA's potential term. Floors: `Nearest`
  (cumulative-share line points + nearest snap), `ZoneSample` (attractiveness-weighted facility,
  ignores distance; PAM-style), `GravityIndependent` (sequential gravity sample, no chain coupling).
  *(Note: `run.setup`/`_instantiate_solver` accept a solver **class** as well as a registry name.)*
- **`research/chainsolvers_eval/external.py`**: `CallableSolver` — wraps a user-supplied per-segment placement
  function (`parameters={"place_fn": fn}`) as a solver, for an in-process same-inputs comparison
  against an external tool (e.g. call eqasim's relaxation solver). chainsolvers imports nothing
  external; the third-party import lives in the user's private/gitignored glue (GPL stays out of
  the MIT library since it's never distributed/vendored). `place_fn(S, E, distances, act_types,
  locations, rng) -> [(id, coord, potential), ...]`.
- **`research/chainsolvers_eval/interop.py`**: export the world to standard CSVs (`facilities.csv`/`plans.csv`) for an
  external tool (e.g. eqasim, run out-of-process so its GPL stays separate), and
  `result_df_from_external(...)` to read its free-leg assignments back into a `result_df` for the
  metrics. Note: a full external run samples its own distances → distributional comparison, not a
  clean same-distances per-instance solver swap (use the in-library reimplementations for that).
- **`research/chainsolvers_eval/survey.py`**: survey-realistic eval primitives (mirrors how MiD-style HTS is used). On a
  large super-region ground truth: `study_window`/`persons_in_window` carve the study sub-region;
  `draw_survey` takes a random person sample; `per_mode_distance_samples` derives the empirical
  free-leg distance distribution per mode; `resample_distances` applies it to a synthetic
  population (status-quo input for argmin solvers that lack observed distances).
- **`research/scripts/survey_experiment.py`**: super-region GT → random survey → two input modes → all
  solvers → validate vs GT. Track 1 (direct chains on surveyed persons: deviation/%gap/recovery);
  Track 3 (survey distance distribution applied to the study population: distance-distribution
  fit + attractiveness + recovery — status-quo resample+argmin vs generative `dp_sample`).
- **`research/scripts/prognosis.py`**: ground-truthed forecasting experiment. Calibrates the structural
  model on a baseline (MLE), boosts a district's attractiveness, regenerates the TRUE counterfactual
  chains, and checks which models predict the resulting shift of trips into the district — only a
  structural (attractiveness-aware) model told the new sizes tracks it; a non-structural/distance
  model has no elasticity. `--elasticity-sweep` varies the boost and reports predicted vs true
  Δ(district share) — the structural model tracks the curve, the non-structural is flat.
  Demonstrates *prognosis* vs mere reproduction of today's distribution.
- **`visualizer.py`**: Optional folium-based rendering of the recursive branching tree
  (hard-coded EPSG:25832 → 4326 transform).

### Data Flow

```
locations (dict | df | tuple) + plans_df
    ↓ cs.setup()
RunnerContext (LocationsIndex, solver, scorer, selector, rng, name_lookup)
    ↓ cs.solve(ctx, plans_df)
(result_df, result_plans, valid)
```

### Key Data Structures (`types.py`)

- **`Leg`** (NamedTuple): `unique_leg_id`, `distance`, `from_location`/`to_location` (each a
  shape-(2,) `np.ndarray` or None), `to_act_type`, `to_act_identifier`, `to_act_is_main_act`,
  `mode`, `dep_time_s`, `arr_time_s`, `connected_legs`, `conn_leader_id`, `conn_to_act`,
  `conn_to_mode`, `extras` (debug only). The connection fields are reserved for future use.
- **`Segment`** = `Tuple[Leg, ...]` — a run of legs bounded by known locations.
- **`SegmentedPlan`** = `Tuple[Segment, ...]` — one person's segments.
- **`SegmentedPlans`** = `frozendict[str, SegmentedPlan]` — keyed by person_id.
- **`Households`** = `frozendict[str, SegmentedPlans]` — keyed by household_id.
- **`PlanColumns`** / **`LocationColumns`**: dataclasses mapping logical fields to DataFrame
  column names (with defaults). `LocationColumns.required()` returns the mandatory column set.

### Solver Registry & Contract

Registered solvers (`SOLVER_REGISTRY` in `run.py`): `carla`, `carla_plus` (stub), and the DP family.
**Only `dp_full` is a pure (unpruned) DP** — exact DP over the full per-type catalog → true global
optimum; oracle, not scalable. Every other DP-family solver is *pruned* (reuses **CARLA's** geometric
generation, so it is exact only over its generated candidate set):
- `dp_rings` — pruned DP over overlapping-ring (triangle-inequality) envelope candidates.
- `dp_carla` — pruned DP over CARLA's generation (ring envelopes + 2-leg circle-intersection); same
  candidates as `carla`, so `carla` vs `dp_carla` isolates search vs branching.
- `dp_rings_refine` / `dp_carla_refine` — the above + iterative neighbour refinement (monotone).
- `dp_carla_pot` — `dp_carla_refine` + potential-aware pooling (augments each pool with the top-K
  facilities by potential, keeping the pruned DP near-exact for the **combined** α>0 objective).
- `dp_sample` — generative MNL forward–backward sampler. `milp` — MILP oracle (== `dp_rings`).
A solver class is instantiated with
`locations, rng, progress, stats, scorer, selector, visualizer, **parameters` and must provide:

- **`wanted_format`**: a string — `"segmented_plans"`, `"households"`, or `"df"`.
- **`required_leg_fields(self) -> set[str]`**: which `Leg` fields the solver consumes.
  `io.get_required_df_columns()` maps these to the DataFrame columns validated on input.
  *(Note: this replaced the old `required_df_columns` method — earlier docs/commits may reference
  the old name.)*
- **`solve(...)`**: callable; receives `plans=`, `households=`, or `df=` per `wanted_format`.

The contract is enforced at instantiation time in `_instantiate_solver` (`run.py`).

## CARLA Algorithm (`solvers/carla.py`)

Recursive divide-and-conquer placement per segment (`solve_segment`):
- **n == 1**: nothing to place.
- **n == 2**: place the single intermediate point via circle-intersection candidates
  (`LocationsIndex.get_best_circle_intersection_locations`).
- **n >= 3**: pick an **anchor** leg (`anchor_strategy`, default `lower_middle`), generate anchor
  candidates from overlapping-ring queries, **branch on the top `number_of_branches` candidates**
  (default 30), recurse on the left and right sub-segments, and keep the branch maximizing
  `score_left + score_right + score_anchor`.

Config (`CarlaConfig`): `number_of_branches=30`, `candidates_complex_case=100`,
`candidates_two_leg_case=40`, `anchor_strategy="lower_middle"`,
`selection_strategy_complex_case="top_n_spatial_downsample"`,
`selection_strategy_two_leg_case="top_n"`, `max_iterations_complex_case=1000`. Override via
`setup(parameters={...})`.

## Dependencies

Core: `numpy`, `pandas`, `frozendict`, `scipy`, `pytest`.
Optional (`viz` extra): `folium`, `networkx`, `pyproj`, `matplotlib`.

## Known Limitations / Gotchas

- **`Scorer.score` default is geometric** (`scoring_selection.py`) — the *default* `ScoreMode.GEOMETRIC`
  ignores `potentials` (distance deviation only), but the `POTENTIAL` and `COMBINED` modes do use
  `potentials`/`pot_weight`. Pass `scorer=Scorer(mode="combined", pot_weight=...)` to drive placement
  with attractiveness. (Earlier docs claimed potentials were entirely unused — no longer true.)
- **`CarlaPlus` is a stub** — selecting `solver="carla_plus"` raises `NotImplementedError`.
- **Exponential branching**: with `number_of_branches=30` and a mid-segment anchor, long chains
  blow up combinatorially; there is no timeout/iteration cap on the recursion itself.
- **CARLA edge checks use `assert`**, which is stripped under `python -O`.
- **Visualizer CRS is hard-coded** to EPSG:25832.
- Significant **commented-out dead code** remains in `helpers.py` and `locations.py`.
