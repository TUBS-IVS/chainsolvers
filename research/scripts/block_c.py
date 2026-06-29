#!/usr/bin/env python
"""Block C — forecasting demand shifts (the ground-truthed counterfactual), TWO channels.

The paper's climax. A says the optimization is *solved* (exact + fast). B says reproducing *today*
doesn't honestly separate models (descriptive fit is fakeable). C is the test that can't be faked:
predict a *changed* world. Because the synthetic DGP is known, we regenerate the TRUE counterfactual
and ask each model to forecast it from anchors+types+modes alone (no observed distances).

Two structural channels — one per parameter of the generative model (utility ~ size^alpha * exp(-d/lambda)):

  attractiveness (alpha channel): multiply a district's structural attractiveness `sizes` x b, regenerate
      true chains. Outcome = the district's SHARE of secondary stops (rises). Argmin in combined mode
      can partially respond via its potential term but UNDERSHOOTS (and no constant weight matches the
      shape -> the weight sweep is the curve-shape engine).

  accessibility (lambda channel): keep the POIs but make them multi-use (each non-home type's candidate
      pool gains a fraction `m` of all non-home POIs -- the 15-minute city). Regenerate true chains:
      nearer opportunities -> shorter trips. Outcome = median secondary-trip distance (falls) / share
      of "local" trips (rises). Argmin is fed BASELINE-resampled distances, so it reproduces the OLD
      trip lengths and is FLAT *at every weight* -- the future isn't in its inputs, and no knob helps.

TIE TO A/B (same runs, one yardstick): every cell at every shock level (incl. the no-shock anchor)
records three metric families -- gap-to-oracle on the objective (A: argmin ~0 throughout), fit
(B: pot_decile_tv + distance-Wasserstein), and the forecast outcome (C). So one frame shows
"optimization-perfect, fit-acceptable, forecast-wrong". C@shock0 (sampled regime) cross-checks B's
sampled column. Model ladder = the full Block B roster.

RAW-FIRST (like B): one per-free-leg parquet per (scenario, world); every metric is a post-hoc replot.
KEY CHOICE: transform="log" (matches the DGP gravity form; recovers alpha~1), NOT B's log1p.

    python research/scripts/block_c.py --quick                       # tiny gauss smoke, both channels
    python research/scripts/block_c.py --scenarios attractiveness accessibility --persons 1500 --jobs 16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from chainsolvers import run
from chainsolvers.scoring_selection import Scorer
from chainsolvers_eval import survey as S
from chainsolvers_eval.calibration import fit_location_choice
from chainsolvers_eval.regen import regenerate_world
from chainsolvers_eval.synth import Topology, topology_locations_tuple
from chainsolvers_eval.worlds import load_world

from block_a import _scored_legs, per_person_dev  # noqa: E402
from block_b import _spec, ARGMIN, MODELS, FLAT_REFS, _pot_decile_tv  # noqa: E402

WORLDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "worlds")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "block_c")
CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "calib_cache")
ORACLE = "dp_full"
TRANSFORM = "log"     # match the DGP (gravity utility prop log(size)); recovers alpha~1 (see prognosis)
GEN_MODELS = list(MODELS)              # dp_sample, carla_sample, gravity_independent
DEFAULT_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]   # argmin pot_weight sweep (0 == geom floor)
LOCAL_M = 1200.0      # "local" trip threshold (~15-min walk @ 5 km/h) for the accessibility outcome
MAX_CAND = 300        # dp_sample candidate-pool cap (spatial downsample) -- keeps diversified pools O(cap^2)
HOME = "home"
ANCHORS = ("home", "work")   # solver-given anchors (work shares its pool with the business alias)
ACCESS_LEVER = "amenity"   # accessibility lever: "amenity" (add near-home POIs) | "multiuse" (dead)
SHOCK_ANCHORS = False      # False = fix anchors (short-run destination-choice forecast; boost only
                           # district SECONDARY facilities so homes/works don't relocate). True =
                           # long-run equilibrium (boost all district facilities; anchors relocate too).

ATTRACT, ACCESS = "attractiveness", "accessibility"
SCENARIOS = (ATTRACT, ACCESS)
DEFAULT_LEVELS = {ATTRACT: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],   # dense -> smooth curves
                  ACCESS:  [0.0, 0.25, 0.5, 0.75, 1.0]}
LEVEL_NAME = {ATTRACT: "boost", ACCESS: "mix"}
# two_zone's dp_sample OOMs at high parallelism (16k catalog) and its diversified pools are huge ->
# run two_zone with --jobs 2 (or the max_candidates cap) until that lands. Smoke is gauss.
OOM_WORLDS = {"two_zone"}


# --------------------------------------------------------------------------- #
# topology helpers
# --------------------------------------------------------------------------- #

def _tag() -> str:
    """Anchor-regime tag for output filenames so the two regimes don't clobber each other."""
    return "full" if SHOCK_ANCHORS else "fixed"


def _shock_mask(topo, district_mask) -> np.ndarray:
    """Locations whose attractiveness the boost multiplies. SHOCK_ANCHORS=False restricts it to
    non-anchor (secondary) district facilities, so home selection (prop sizes[home]) and work
    placement are unchanged -> anchors stay put and the response is pure destination elasticity."""
    if SHOCK_ANCHORS:
        return district_mask
    anchor = np.zeros(topo.coords.shape[0], dtype=bool)
    for t in ANCHORS:
        if t in topo.type_locs:
            anchor[np.asarray(topo.type_locs[t], dtype=int)] = True
    return district_mask & ~anchor


def _district_mask(world_name: str, topo) -> np.ndarray:
    c, box = topo.coords, topo.box
    if world_name == "two_zone":
        cen = box / 2.0
        return np.hypot(c[:, 0] - cen, c[:, 1] - cen) < 0.13 * box   # the dense urban core
    return (c[:, 0] < 0.4 * box) & (c[:, 1] < 0.4 * box)             # gauss / osm: corner quadrant


def diversify_supply(topo, m: float, rng, exclude=(HOME,)) -> Topology:
    """15-minute-city lever: keep POIs but make them multi-use. Each non-home activity type's
    candidate pool is augmented with a fraction `m` of ALL non-home POIs (m=0 baseline; m=1 every
    non-home POI serves every non-home type). Pure type_locs membership change -- coords/sizes/loc_ids
    untouched (a POI brings its location size to whatever it now hosts), so it is reorder-safe."""
    if m <= 0:
        return topo
    pool = np.unique(np.concatenate([topo.type_locs[t] for t in topo.types if t not in exclude]))
    k = int(round(m * len(pool)))
    new_locs = dict(topo.type_locs)
    for t in topo.types:
        if t in exclude:
            continue
        add = pool if k >= len(pool) else rng.choice(pool, size=k, replace=False)
        new_locs[t] = np.unique(np.concatenate([topo.type_locs[t], np.asarray(add, dtype=int)]))
    return Topology(topo.coords, topo.sizes, topo.loc_ids, topo.types, topo.type_idx, new_locs, topo.box)


def add_home_amenities(topo, a: float, rng, jitter_m=150.0) -> Topology:
    """15-min-city lever for worlds with REAL residential geography (osm): ADD new multi-use amenity
    facilities co-located with a fraction `a` of home locations (jittered ~neighbourhood scale),
    serving every non-anchor activity type. These create genuinely-local (d~0) options the baseline
    lacks, so the gravity DGP shortens secondary trips. Unlike `diversify_supply` (which only re-labels
    existing, same-distance POIs and produced no signal), this introduces nearer supply. a=0 baseline;
    a=1 ~ one amenity per home location."""
    home = np.asarray(topo.type_locs[HOME], dtype=int)
    n_add = int(round(a * len(home)))
    if n_add <= 0:
        return topo
    secondary = [t for t in topo.types if t not in (HOME, "work")]
    pool = np.unique(np.concatenate([topo.type_locs[t] for t in secondary])) if secondary else home
    size_val = float(np.median(topo.sizes[pool])) if pool.size else float(np.median(topo.sizes))
    pick = rng.choice(home, size=n_add, replace=n_add > len(home))
    new_coords = topo.coords[pick] + rng.normal(0.0, jitter_m, size=(n_add, 2))
    n0 = topo.coords.shape[0]
    coords = np.vstack([topo.coords, new_coords])
    sizes = np.concatenate([topo.sizes, np.full(n_add, size_val)])
    loc_ids = np.concatenate([topo.loc_ids, np.array([f"amen{i}" for i in range(n_add)], dtype=object)])
    new_idx = np.arange(n0, n0 + n_add)
    type_locs = dict(topo.type_locs)
    for t in secondary:
        type_locs[t] = np.concatenate([np.asarray(topo.type_locs[t], dtype=int), new_idx])
    return Topology(coords, sizes, loc_ids, topo.types, topo.type_idx, type_locs, topo.box)


def apply_access_lever(topo, level: float, rng) -> Topology:
    """Dispatch the accessibility (15-min-city) lever. `amenity` (default) adds near-home amenities --
    the only lever that produced a signal; `multiuse` re-labels existing POIs (dead -- co-located/dense
    worlds show no shortening)."""
    if ACCESS_LEVER == "amenity":
        return add_home_amenities(topo, level, rng)
    return diversify_supply(topo, level, rng)


class _Shim:
    """Stand-in world carrying a (possibly modified) topology + the baked meta, for regenerate_world."""
    __slots__ = ("topology", "meta")

    def __init__(self, topology, meta):
        self.topology, self.meta = topology, meta


def regen(world, world_name, topo, sizes, n_persons, seed):
    """Regenerate TRUE chains on `topo` (possibly shocked sizes and/or diversified pools) via the
    single-sourced baked DGP. Regenerate BOTH baseline and counterfactual through here (load_world
    reorder caveat) -- never mix with the baked baseline plans."""
    w = regenerate_world(_Shim(topo, world.meta), world_name, n_persons, sizes=sizes,
                         rng=np.random.default_rng(seed))
    return w.plans_df, w.ground_truth


def _free_ids(gt) -> set:
    return set(gt.loc[gt["to_is_free"], "unique_leg_id"])


# --------------------------------------------------------------------------- #
# calibration (transform="log", baseline only) -- cached, shared across scenarios
# --------------------------------------------------------------------------- #

def calibrate_cached(world_name, topo, plans_b, gt_b, seed, n_persons, refit=False) -> dict:
    os.makedirs(CALIB_DIR, exist_ok=True)
    path = os.path.join(CALIB_DIR, f"C_{world_name}_seed{seed}_n{n_persons}_{TRANSFORM}.json")
    if not refit and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        print(f"  [{world_name}] calib CACHE HIT alpha={d['alpha']:.3f} scale={d['scale']:.0f}m", flush=True)
        return d
    alpha, scale = fit_location_choice(topo, plans_b, gt_b, transform=TRANSFORM)
    d = {"alpha": float(alpha), "scale": float(scale)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


# --------------------------------------------------------------------------- #
# the model ladder per scenario: (solver, condition, view, alpha)
#   view "shock" = told the intervention (attract: shocked sizes; access: diversified pools)
#   view "base"  = told yesterday (baseline sizes / baseline pools)
# --------------------------------------------------------------------------- #

_MERGE_KEYS = ["scenario", "solver", "condition", "alpha", "level"]


def _merge_on_keys(old: pd.DataFrame, new: pd.DataFrame, keys=_MERGE_KEYS) -> pd.DataFrame:
    """Append `new` over `old`: drop old rows whose (scenario,solver,condition,alpha,level) key
    appears in `new`, then concat. Lets a later run add dp_full (--oracle-only --append) without
    redoing the cheap solvers. alpha NaN (the 'true' rows) is filled so the key compares cleanly."""
    def tuples(d):
        k = d[keys].copy(); k["alpha"] = k["alpha"].fillna(-999.0)
        return list(k.itertuples(index=False, name=None))
    new_keys = set(tuples(new))
    keep = [t not in new_keys for t in tuples(old)]
    return pd.concat([old[keep], new], ignore_index=True)


def model_specs(scenario, alpha_cal, weights) -> List[tuple]:
    specs = []
    no_attr_view = "shock" if scenario == ACCESS else "base"   # access: lambda response survives alpha=0
    flat_view = "shock" if scenario == ACCESS else "base"      # access: tell rda the new supply too
    # dp_full is exact over the FULL per-type catalog (O(n*N^2)); on the accessibility channel the
    # diversified pools blow N up to the whole catalog -> infeasible. Use dp_carla_pot as near-oracle.
    argmin = ARGMIN if scenario == ATTRACT else [s for s in ARGMIN if s != ORACLE]
    for s in GEN_MODELS:
        specs += [(s, "informed", "shock", alpha_cal),
                  (s, "no_attr", no_attr_view, 0.0)]   # blind dropped: under fixed anchors == no_attr
    for s in argmin:
        for w in weights:
            view = "shock" if (w > 0 or scenario == ACCESS) else "base"
            specs.append((s, f"w{w:g}", view, float(w)))
    for s in FLAT_REFS:
        specs.append((s, "dist_only", flat_view, 0.0))
    # control: the exact argmin fed the TRUE counterfactual distances (not resampled) -- isolates
    # whether its share miss is the objective itself or merely the stale distance input.
    if scenario == ATTRACT:
        specs.append((ORACLE, "truedist", "shock", 1.0))
    # solver-mode filter: run cheap roster now (skip_oracle), append the exact dp_full later
    # (oracle_only --append). dp_full is O(n*N^2); deferring it keeps the main pass fast.
    if _SOLVER_MODE == "skip_oracle":
        specs = [sp for sp in specs if sp[0] != ORACLE]
    elif _SOLVER_MODE == "oracle_only":
        specs = [sp for sp in specs if sp[0] == ORACLE]
    return specs


# --------------------------------------------------------------------------- #
# prediction + metrics for one cell
# --------------------------------------------------------------------------- #

def predict(topo_model, sizes_field, plans_in, solver, alpha, seed, cal, cap=0):
    loc = topology_locations_tuple(topo_model, sizes_field)
    cls, params = _spec(solver)
    extra: Dict[str, object] = {}
    if solver == "dp_sample":
        extra.update({"default_scale": float(cal["scale"]), "attr_transform": TRANSFORM})
        if cap > 0:                                  # only cap huge (diversified) pools; an aggressive
            extra["max_candidates"] = cap            # cap on normal pools makes dp_sample over-concentrate
    elif solver == "gravity_independent":
        extra["scale"] = float(cal["scale"])
    elif solver == "dp_carla_pot":
        extra["pot_pool_k"] = _POTK
    # carla_sample reads attr_transform via the Scorer + sharpness via pot_weight (no decay-scale param)
    params = {**(params or {}), **extra} or None
    scorer = Scorer(mode="combined", pot_weight=float(alpha), attr_transform=TRANSFORM)
    ctx = run.setup(locations_tuple=loc, solver=cls, rng_seed=seed, scorer=scorer, parameters=params)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=plans_in)
    return rdf, bool(valid)


def _tvc(topo_model, base_sizes, free_df):
    """{type: (size potential vec, realized visit-count vec)} aligned to the model's per-type pool --
    for pot_decile_tv (B-tie fit metric: do placed visits match the attractiveness distribution)."""
    out = {}
    counts_by_type = {t: free_df[free_df["to_act_type"] == t]["to_act_identifier"].value_counts()
                      for t in free_df["to_act_type"].unique()}
    for t in topo_model.types:
        idx = topo_model.type_locs[t]
        if t not in counts_by_type or len(idx) < 10:
            continue
        ids = topo_model.loc_ids[idx]
        pot = np.asarray(base_sizes, float)[idx]
        vc = counts_by_type[t]
        counts = np.array([vc.get(fid, 0) for fid in ids], dtype=float)
        if counts.sum() > 0 and pot.sum() > 0:
            out[t] = (pot, counts)
    return out


def evaluate_cell(scenario, topo_base, base_sizes, ld, solver, condition, view, alpha, seed, cal):
    if scenario == ATTRACT:
        sizes_field = ld["sizes_shock"] if view == "shock" else base_sizes
        topo_model = topo_base
        pot_sizes = base_sizes
    else:
        # the amenity lever EXTENDS the topology, so use the topo's own (extended) sizes vector;
        # base_sizes has the original length and would overflow the diversified type_locs.
        topo_model = ld["topo_div"] if view == "shock" else topo_base
        sizes_field = pot_sizes = np.asarray(topo_model.sizes, dtype=float)

    cap = MAX_CAND if scenario == ACCESS else 0     # only the diversified pools need capping
    plans = ld["plans_true"] if condition == "truedist" else ld["plans_in"]  # control: true CF distances
    rdf, valid = predict(topo_model, sizes_field, plans, solver, alpha, seed, cal, cap=cap)
    free = rdf[rdf["unique_leg_id"].isin(ld["free_ids"])].copy()
    ach = np.hypot(free["to_x"].to_numpy(float) - free["from_x"].to_numpy(float),
                   free["to_y"].to_numpy(float) - free["from_y"].to_numpy(float))
    inp = free["distance_meters"].to_numpy(float)

    # --- forecast outcome (scenario-specific) ---
    if scenario == ATTRACT:
        ids = free.set_index("unique_leg_id")["to_act_identifier"]
        main = float(ids.isin(ld["district_ids"]).mean()) if len(ids) else float("nan")
        frac_local = float("nan")
    else:
        main = float(np.nanmedian(ach)) if ach.size else float("nan")          # median secondary distance
        frac_local = float(np.mean(ach < LOCAL_M)) if ach.size else float("nan")
    delta = main - ld["base_main"]

    # --- fit (B-tie) + gap inputs (A-tie) + recovery (diagnostic) ---
    dist_w = float(wasserstein_distance(ach[np.isfinite(ach)], ld["true_dists"])) \
        if np.isfinite(ach).any() and ld["true_dists"].size else float("nan")
    pot_tv = _pot_decile_tv(_tvc(topo_model, pot_sizes, free))
    # Objective on the BOTH-SIDED scored leg set (per_person_dev) -- the set the DP actually minimizes
    # (every leg touching a free node, incl. the outgoing one), so gap-to-dp_full is well-defined
    # (>=0). Free-leg-only accounting omits outgoing-leg deviation and can spuriously invert the gap.
    dev_pp = per_person_dev(rdf, ld["plans_in"], ld["gt"], ld["scored"])
    pid_of = ld["plans_in"].set_index("unique_leg_id")["unique_person_id"]
    sz = np.clip(free["to_act_potential"].to_numpy(float), 1e-9, None)
    pot_pp = (pd.Series(np.log(sz), index=free["unique_leg_id"].map(pid_of).to_numpy())
              .groupby(level=0).sum().reindex(dev_pp.index).fillna(0.0))
    combined_cost = float(np.nanmean(dev_pp - float(alpha) * pot_pp))   # beta=1; matches block_b
    gt_free = ld["gt"].loc[ld["gt"]["to_is_free"], ["unique_leg_id", "true_to_identifier"]]
    m = free[["unique_leg_id", "to_act_identifier"]].merge(gt_free, on="unique_leg_id", how="inner")
    recovery = 100.0 * float((m["to_act_identifier"].astype(str) == m["true_to_identifier"].astype(str)
                              ).mean()) if len(m) else float("nan")

    median_ach = float(np.nanmedian(ach)) if np.isfinite(ach).any() else float("nan")
    row = {"scenario": scenario, "solver": solver, "condition": condition, "view": view,
           "alpha": float(alpha), "level": ld["level"], "outcome_main": main, "outcome_delta": delta,
           "true_main": ld["true_main"], "base_main": ld["base_main"], "frac_local": frac_local,
           "dist_w_m": dist_w, "median_achieved_m": median_ach, "true_median_dist_m": ld["true_median"],
           "pot_decile_tv": pot_tv, "combined_cost": combined_cost,
           "recovery_pct": recovery, "valid": valid}
    pid_of = ld["plans_in"].set_index("unique_leg_id")["unique_person_id"]
    raw = pd.DataFrame({
        "scenario": scenario, "solver": solver, "condition": condition, "alpha": float(alpha),
        "level": ld["level"], "unique_person_id": free["unique_leg_id"].map(pid_of).to_numpy(),
        "unique_leg_id": free["unique_leg_id"].to_numpy(), "to_act_type": free["to_act_type"].to_numpy(),
        "chosen_loc_id": free["to_act_identifier"].to_numpy(),
        "in_district": (free["to_act_identifier"].isin(ld["district_ids"]).to_numpy()
                        if scenario == ATTRACT else np.zeros(len(free), bool)),
        "achieved_dist_m": ach, "input_dist_m": inp, "is_local": ach < LOCAL_M,
        "chosen_size": free["to_act_potential"].to_numpy(float)})
    return {"row": row, "raw": raw}


# --------------------------------------------------------------------------- #
# parallel worker
# --------------------------------------------------------------------------- #

_TOPO = _BASE_SIZES = _LEVELS = _CAL = None
_SCENARIO = None
_POTK = 100


def _init_worker(scenario, topo, base_sizes, levels, cal, potk):
    global _SCENARIO, _TOPO, _BASE_SIZES, _LEVELS, _CAL, _POTK
    _SCENARIO, _TOPO, _BASE_SIZES, _LEVELS, _CAL, _POTK = scenario, topo, base_sizes, levels, cal, potk


def _cell(task):
    level, solver, condition, view, alpha, seed = task
    ld = _LEVELS[level]
    try:
        return evaluate_cell(_SCENARIO, _TOPO, _BASE_SIZES, ld, solver, condition, view, alpha, seed, _CAL)
    except Exception as e:
        print(f"      [{solver}/{condition} {LEVEL_NAME[_SCENARIO]}={level}] failed: {e!r}", flush=True)
        return {"row": {"scenario": _SCENARIO, "solver": solver, "condition": condition,
                        "alpha": float(alpha), "level": level, "valid": False}, "raw": None}


# --------------------------------------------------------------------------- #
# per-scenario driver
# --------------------------------------------------------------------------- #

def _outcome_true(scenario, plans_cf, gt_cf, free_ids, district_ids):
    """The TRUE (DGP) outcome at a shock level: district share (attract) or median free-leg distance
    (access). True distances = the regenerated chains' own distance_meters (distance_noise=0)."""
    if scenario == ATTRACT:
        ids = gt_cf.set_index("unique_leg_id")["true_to_identifier"]
        return float(ids[ids.index.isin(free_ids)].isin(district_ids).mean())
    d = plans_cf[plans_cf["unique_leg_id"].isin(free_ids)]["distance_meters"].to_numpy(float)
    return float(np.nanmedian(d))


def run_scenario(world, world_name, scenario, topo, base_sizes, district_mask, cal,
                 samples_by_mode, base_main, levels, n_persons, seed, out_dir, jobs, potk):
    anchored = "anchors-relocate" if SHOCK_ANCHORS else "anchors-fixed"
    print(f"--- scenario={scenario} ({anchored})  base_{LEVEL_NAME[scenario]}_outcome={base_main:.4g} ---",
          flush=True)
    district_ids = set(topo.loc_ids[district_mask])
    shock_mask = _shock_mask(topo, district_mask)
    weights = _WEIGHTS

    level_data = {}
    for lv in levels:
        if scenario == ATTRACT:
            sizes_shock = base_sizes.copy(); sizes_shock[shock_mask] *= lv
            topo_div = None
            plans_cf, gt_cf = regen(world, world_name, topo, sizes_shock, n_persons, seed + 1)
        else:
            sizes_shock = None
            topo_div = apply_access_lever(topo, lv, np.random.default_rng(seed + 7))
            plans_cf, gt_cf = regen(world, world_name, topo_div, topo_div.sizes, n_persons, seed + 1)
        free_ids = _free_ids(gt_cf)
        true_main = _outcome_true(scenario, plans_cf, gt_cf, free_ids, district_ids)
        plans_in = S.resample_distances(plans_cf, gt_cf, samples_by_mode,
                                        np.random.default_rng(seed + 100), feasible=True)
        true_d = plans_cf[plans_cf["unique_leg_id"].isin(free_ids)]["distance_meters"].to_numpy(float)
        true_d = true_d[np.isfinite(true_d)]
        level_data[lv] = {"level": lv, "plans_in": plans_in, "plans_true": plans_cf,
                          "gt": gt_cf, "free_ids": free_ids, "scored": _scored_legs(plans_in, gt_cf),
                          "district_ids": district_ids, "true_main": true_main, "base_main": base_main,
                          "true_dists": true_d, "true_median": float(np.median(true_d)) if true_d.size
                          else float("nan"), "sizes_shock": sizes_shock, "topo_div": topo_div}
        tag = f"share={true_main*100:.1f}%" if scenario == ATTRACT else f"median={true_main:.0f}m"
        print(f"  [{world_name}/{scenario}] {LEVEL_NAME[scenario]}={lv:<4g} true {tag} "
              f"(d {true_main-base_main:+.3g})  free legs={len(free_ids)}", flush=True)

    specs = model_specs(scenario, cal["alpha"], weights)
    tasks = [(lv, s, cond, view, a, seed) for lv in levels for (s, cond, view, a) in specs]
    t0 = time.perf_counter()
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(scenario, topo, base_sizes, level_data, cal, potk)) as ex:
            results = list(ex.map(_cell, tasks))
    else:
        _init_worker(scenario, topo, base_sizes, level_data, cal, potk)
        results = [_cell(t) for t in tasks]
    print(f"  [{world_name}/{scenario}] solved {len(tasks)} cells in {time.perf_counter()-t0:.1f}s",
          flush=True)

    rows = [r["row"] for r in results if r is not None]
    raws = [r["raw"] for r in results if r is not None and r.get("raw") is not None]
    rows += [{"scenario": scenario, "solver": "true", "condition": "true", "view": "true",
              "alpha": float("nan"), "level": lv, "outcome_main": level_data[lv]["true_main"],
              "outcome_delta": level_data[lv]["true_main"] - base_main,
              "true_main": level_data[lv]["true_main"], "base_main": base_main,
              "true_median_dist_m": level_data[lv]["true_median"], "valid": True}
             for lv in levels]
    df = pd.DataFrame(rows)
    df["world"] = world_name; df["alpha_cal"] = cal["alpha"]; df["anchor"] = _tag()
    os.makedirs(out_dir, exist_ok=True)
    tag = _tag()
    csv_path = os.path.join(out_dir, f"prognosis_{scenario}_{tag}_{world_name}.csv")
    raw_path = os.path.join(out_dir, f"raw_legs_{scenario}_{tag}_{world_name}.parquet")
    if APPEND and os.path.exists(csv_path):                  # merge new cells over old (e.g. add dp_full)
        df = _merge_on_keys(pd.read_csv(csv_path), df)
        print(f"  [{world_name}/{scenario}] appended -> {df['solver'].nunique()} solvers in CSV", flush=True)
    df.to_csv(csv_path, index=False)
    if raws:
        raw_df = pd.concat(raws, ignore_index=True); raw_df["world"] = world_name
        if APPEND and os.path.exists(raw_path):
            raw_df = _merge_on_keys(pd.read_parquet(raw_path), raw_df)
        raw_df.to_parquet(raw_path, index=False)
    print(f"  [{world_name}/{scenario}] rows={len(df)} -> {out_dir}", flush=True)
    return df


def run_true_curves(world_name, scenarios, n_persons, seed, out_dir):
    """Decision data (no solver cells): regenerate the TRUE outcome across levels -- median secondary
    distance / %-local / district share -- per world, plus the two_zone rural subpop. Cheap; used to
    decide which world showcases each channel (e.g. does the lambda/accessibility signal need the
    supply-constrained two_zone rural ring rather than dense gauss)."""
    print(f"[true-curve] world={world_name} scenarios={list(scenarios)} persons={n_persons}", flush=True)
    world = load_world(os.path.join(WORLDS_DIR, world_name))
    topo = world.topology
    base_sizes = np.asarray(topo.sizes, float).copy()
    district_mask = _district_mask(world_name, topo)
    district_ids = set(topo.loc_ids[district_mask])
    rows = []
    for sc in scenarios:
        levels = DEFAULT_LEVELS[sc] if _LEVELS_OVR is None else _LEVELS_OVR.get(sc, DEFAULT_LEVELS[sc])
        for lv in levels:
            if sc == ATTRACT:
                sizes = base_sizes.copy(); sizes[_shock_mask(topo, district_mask)] *= lv
                plans, gt = regen(world, world_name, topo, sizes, n_persons, seed + 1)
            else:
                tdiv = apply_access_lever(topo, lv, np.random.default_rng(seed + 7))
                plans, gt = regen(world, world_name, tdiv, tdiv.sizes, n_persons, seed + 1)
            fid = _free_ids(gt)
            fp = plans[plans["unique_leg_id"].isin(fid)]
            d = fp["distance_meters"].to_numpy(float)
            share = float(gt.set_index("unique_leg_id")["true_to_identifier"]
                          .reindex(list(fid)).isin(district_ids).mean())
            row = {"world": world_name, "scenario": sc, "level": lv, "n_free": len(fid),
                   "median_dist_m": float(np.nanmedian(d)), "frac_local": float(np.mean(d < LOCAL_M)),
                   "district_share": share}
            rural = fp[fp["unique_person_id"].astype(str).str.startswith("r")]
            if len(rural):
                dr = rural["distance_meters"].to_numpy(float)
                row["median_dist_rural_m"] = float(np.nanmedian(dr))
                row["frac_local_rural"] = float(np.mean(dr < LOCAL_M))
            rows.append(row)
            print(f"  [{world_name}/{sc}] {LEVEL_NAME[sc]}={lv:<4g} median={row['median_dist_m']:.0f}m "
                  f"local={row['frac_local']*100:.0f}% share={share*100:.1f}%"
                  + (f"  rural_median={row.get('median_dist_rural_m'):.0f}m"
                     if "median_dist_rural_m" in row else ""), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"true_curves_{world_name}.csv"), index=False)
    return df


def run_world(world_name, scenarios, n_persons, seed, out_dir, jobs, potk, refit):
    print(f"world={world_name}  persons={n_persons}  jobs={jobs}  scenarios={list(scenarios)}", flush=True)
    if world_name in OOM_WORLDS and jobs > 2:
        print(f"  [{world_name}] WARNING: dp_sample OOMs on this catalog at jobs={jobs}; use --jobs 2.",
              flush=True)
    world = load_world(os.path.join(WORLDS_DIR, world_name))
    topo = world.topology
    base_sizes = np.asarray(topo.sizes, float).copy()
    district_mask = _district_mask(world_name, topo)
    print(f"  [{world_name}] district = {int(district_mask.sum())}/{len(topo.loc_ids)} locations", flush=True)

    plans_b, gt_b = regen(world, world_name, topo, base_sizes, n_persons, seed)
    cal = calibrate_cached(world_name, topo, plans_b, gt_b, seed, n_persons, refit=refit)
    print(f"  [{world_name}] calibrated (log MLE) alpha={cal['alpha']:.3f} scale={cal['scale']:.0f}m",
          flush=True)
    samples_by_mode = S.per_mode_distance_samples(plans_b, gt_b)
    free_b = _free_ids(gt_b)
    district_ids = set(topo.loc_ids[district_mask])
    base_share = _outcome_true(ATTRACT, plans_b, gt_b, free_b, district_ids)
    base_median = _outcome_true(ACCESS, plans_b, gt_b, free_b, district_ids)
    print(f"  [{world_name}] baseline district share={base_share*100:.1f}%  median secondary={base_median:.0f}m",
          flush=True)

    out = {}
    for sc in scenarios:
        base_main = base_share if sc == ATTRACT else base_median
        out[sc] = run_scenario(world, world_name, sc, topo, base_sizes, district_mask, cal,
                               samples_by_mode, base_main, DEFAULT_LEVELS[sc] if _LEVELS_OVR is None
                               else _LEVELS_OVR.get(sc, DEFAULT_LEVELS[sc]),
                               n_persons, seed, out_dir, jobs, potk)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

_WEIGHTS = DEFAULT_WEIGHTS
_LEVELS_OVR: Optional[dict] = None
_SOLVER_MODE = "all"      # "all" | "skip_oracle" (drop dp_full) | "oracle_only" (only dp_full)
APPEND = False            # merge into existing CSV/parquet instead of overwriting


def main(argv=None):
    global _WEIGHTS, _LEVELS_OVR, _POTK, ACCESS_LEVER, SHOCK_ANCHORS, _SOLVER_MODE, APPEND, TRANSFORM
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", nargs="+", default=["gauss_hannover"])
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--persons", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="argmin pot_weight sweep (the curve-shape engine; 0 == geom floor)")
    p.add_argument("--boosts", type=float, nargs="+", default=None, help="attractiveness levels")
    p.add_argument("--mixes", type=float, nargs="+", default=None, help="accessibility (mixed-use) levels")
    p.add_argument("--pot-pool-k", type=int, default=100)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--refit", action="store_true")
    p.add_argument("--transform", default="log", choices=["log", "log1p", "linear"],
                   help="generative attractiveness form (calib-form ablation: log=matched DGP)")
    p.add_argument("--skip-oracle", action="store_true", help="drop dp_full (defer the O(n*N^2) oracle)")
    p.add_argument("--oracle-only", action="store_true", help="run ONLY dp_full (use with --append)")
    p.add_argument("--append", action="store_true", help="merge into existing CSV/parquet (add dp_full later)")
    p.add_argument("--access-lever", choices=["amenity", "multiuse"], default="amenity",
                   help="accessibility lever: amenity (add near-home POIs) | multiuse (re-label, dead)")
    p.add_argument("--shock-anchors", action="store_true",
                   help="long-run variant: boost ALL district facilities so anchors relocate too "
                        "(default: fix anchors, boost only secondary facilities -> pure dest. elasticity)")
    p.add_argument("--true-curve", action="store_true",
                   help="decision data only: regenerate TRUE outcomes across levels (no solver cells)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke: 400 persons, short level/weight grids, gauss, both scenarios")
    args = p.parse_args(argv)

    persons = 400 if args.quick else args.persons
    ACCESS_LEVER = args.access_lever
    TRANSFORM = args.transform
    _SOLVER_MODE = "oracle_only" if args.oracle_only else ("skip_oracle" if args.skip_oracle else "all")
    APPEND = args.append
    SHOCK_ANCHORS = args.shock_anchors
    _WEIGHTS = args.weights if args.weights is not None else ([0.0, 1.0, 5.0] if args.quick
                                                              else DEFAULT_WEIGHTS)
    _POTK = args.pot_pool_k
    ovr = {}
    if args.quick:
        ovr = {ATTRACT: [1.0, 2.0, 4.0, 8.0], ACCESS: [0.0, 0.5, 1.0]}
    if args.boosts is not None:
        ovr[ATTRACT] = args.boosts
    if args.mixes is not None:
        ovr[ACCESS] = args.mixes
    _LEVELS_OVR = ovr or None
    worlds = ["gauss_hannover"] if args.quick else args.worlds

    t_all = time.perf_counter()
    if args.true_curve:
        dfs = {wn: run_true_curves(wn, args.scenarios, persons, args.seed, args.out) for wn in worlds}
        if not args.no_plot:
            from plot_block_c import plot_true_curves
            for sc in args.scenarios:
                plot_true_curves(dfs, sc, args.out)
        print(f"\nTRUE-CURVE DONE {time.perf_counter()-t_all:.1f}s -> {args.out}", flush=True)
        return

    for wn in worlds:
        out = run_world(wn, args.scenarios, persons, args.seed, args.out, args.jobs, args.pot_pool_k,
                        args.refit)
        if not args.no_plot:
            from plot_block_c import plot_all
            for sc, df in out.items():
                plot_all(df, sc, wn, args.out, tag=_tag())
    print(f"\nALL DONE {time.perf_counter()-t_all:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
