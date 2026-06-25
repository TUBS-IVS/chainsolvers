#!/usr/bin/env python
"""Block A — solver-quality evaluation (the distance / gap-to-oracle lens).

Every result is read as **metres-above-oracle** (solver deviation - dp_full deviation, per
person) so the metric means the same thing in every regime: how far from the best achievable
the solver lands. alpha=0 (distance only) throughout; recovery is NOT used (it is a diagnostic,
and a red flag if high — see eval-philosophy). The same `--persons` sample (fixed seed) is used
for every solver and regime (common random numbers), via the canonical `survey.sample_persons`.

Five results, all on the baked worlds (gauss_hannover primary; osm_hannover / two_zone for the
robustness echo):

  1. gap x difficulty   — gap-to-oracle as inputs degrade (true -> dist-noise/sampled;
                          true -> anchor-disturb -> anchor-remove). The easy end is the
                          0-vs-0 sanity floor; the degraded end is where it separates.
                          NOTE: anchor-disturb is thin in worlds where `work` rarely bounds a
                          secondary segment (most chains are home-anchored excursions) — its
                          signal lives in the commuter-with-errands subpopulation, so the
                          all-persons mean understates it.
  2. quality-runtime    — each solver's quality knob swept -> deviation vs runtime frontier.
  3. chain-length scale  — runtime vs chain length: CARLA exponential, DP linear.
  4. generation-vs-search — gap decomposition (carla->dp_carla = search, dp_carla->dp_full =
                          generation/recall). NOTE: with the CARLA paper config (candidates=10)
                          and dp_carla's default min_candidates=50, the two use slightly different
                          candidate budgets (a sub-metre confound here); kept as-is per the paper.
  5. dp_full N-wall      — runtime vs catalog size N: pruned ~flat, dp_full O(n*N^2).

RDA / RDA-guided auto-use the vendored real eqasim (`_private/eqasim_glue.py`) when present
(paper numbers; see memory: paper-rda-use-vendored), else the MIT remakes.

    python research/scripts/block_a.py --quick                 # fast smoke run
    python research/scripts/block_a.py --persons 1000 --world gauss_hannover
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers_eval import survey as S
from chainsolvers_eval.baselines import RelaxationDiscretization, RelaxationDiscretizationGuided
from chainsolvers_eval.external import CallableSolver
from chainsolvers_eval.worlds import load_world

WORLDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "worlds")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "block_a")
ORACLE = "dp_full"
_BASELINES = frozenset({"rda", "rda_guided", "dp_sample"})  # off-scale refs (generative/RDA): in raw, off plot 1


def _vendored_rda():
    """Real eqasim RDA via the private glue, as `place_fn`s for CallableSolver. Returns {} if the
    vendored eqasim isn't present (then we fall back to the MIT remakes). The RDA `place_fn` runs
    eqasim's GravityChainSolver in the keep-best assignment loop (n_restarts) on our fixed
    distances — i.e. the real relaxation core; n_restarts is the assignment-iteration knob."""
    priv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "_private"))
    if not os.path.exists(os.path.join(priv, "eqasim_glue.py")):
        return {}
    if priv not in sys.path:
        sys.path.insert(0, priv)
    try:
        from eqasim_glue import (eqasim_place_guided, GravityChainSolver, _problem, _discretize)  # noqa
    except Exception as e:
        print(f"  (vendored eqasim unavailable: {e!r}; using MIT RDA remakes)", flush=True)
        return {}

    def rda_keepbest(S_, E_, distances, act_types, locations, rng, n_restarts=20):
        solver = GravityChainSolver(random=rng)
        prob, dist = _problem(S_, E_, act_types), np.asarray(distances, float)
        best, best_dev = None, np.inf
        for _ in range(n_restarts):
            chosen = _discretize(np.asarray(solver.solve(prob, dist)["locations"]), act_types, locations)
            coords = [np.asarray(S_, float).ravel()] + [c[1] for c in chosen] + [np.asarray(E_, float).ravel()]
            legs = np.linalg.norm(np.diff(np.asarray(coords), axis=0), axis=1)
            dev = float(np.abs(legs - dist).sum())
            if dev < best_dev:
                best, best_dev = chosen, dev
        return best

    def make_rda(n_restarts):  # frontier knob: assignment restarts (eqasim AssignmentSolver iters)
        return lambda a, b, d, t, loc, rng: rda_keepbest(a, b, d, t, loc, rng, n_restarts=n_restarts)

    return {"rda": (CallableSolver, {"place_fn": rda_keepbest}),
            "rda_guided": (CallableSolver, {"place_fn": eqasim_place_guided}),
            "_make_rda": make_rda}


# CARLA base config from Petre et al. 2025 (Table 3); the frontier sweeps number_of_branches.
CARLA_CFG = {"number_of_branches": 50, "candidates_complex_case": 10, "candidates_two_leg_case": 20,
             "anchor_strategy": "lower_middle", "selection_strategy_complex_case": "mixed",
             "selection_strategy_two_leg_case": "top_n"}

# name -> registry key (str) | solver class | (class, params). dp_sample is the generative MNL
# sampler: it doesn't optimize the argmin objective, so it's an off-scale reference baseline (in
# _BASELINES) rather than a competitor on gap-to-oracle.
SOLVERS: Dict[str, object] = {
    "carla": ("carla", CARLA_CFG), "dp_rings": "dp_rings", "dp_carla": "dp_carla",
    "dp_rings_refine": "dp_rings_refine", "dp_carla_refine": "dp_carla_refine", "dp_full": "dp_full",
    "dp_sample": "dp_sample",
    "rda": RelaxationDiscretization, "rda_guided": RelaxationDiscretizationGuided,  # MIT fallback
}
_VENDORED = _vendored_rda()
_MAKE_RDA = _VENDORED.pop("_make_rda", None)
SOLVERS.update(_VENDORED)  # real eqasim RDA/guided if the private glue is present (paper numbers)
RDA_IS_VENDORED = bool(_VENDORED)
DIST_SOLVERS = [s for s in SOLVERS if s != "rda_guided"]  # guided RDA is broken+heavy -> report-only


def _frontier_spec(solver, knob, val):
    """(cls, params) for one frontier point. RDA's knob is restart/assignment iterations; for the
    vendored solver that means a fresh place_fn, for the MIT remake `assignment_iterations`."""
    if solver == "rda":
        if RDA_IS_VENDORED:
            return CallableSolver, {"place_fn": _MAKE_RDA(val)}
        return RelaxationDiscretization, {"assignment_iterations": val}
    cls, base = _spec(solver)
    return cls, {**(base or {}), knob: val}  # keep base config (e.g. CARLA_CFG), override the knob


def _spec(name):
    """Return (solver, params) for run.setup, unpacking the optional (class, params) form."""
    s = SOLVERS[name]
    return (s[0], s[1]) if isinstance(s, tuple) else (s, None)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #

def load(name: str):
    return load_world(os.path.join(WORLDS_DIR, name))


class _LocOnly:
    """Minimal world stand-in exposing only `locations_tuple` — all `_solve_timed` needs. Lets the
    parallel workers (result 1) skip loading the multi-GB plans/gt parquets, which they never use;
    without this, 16 workers x the full two_zone world (~3 GB each) OOMs a 36 GB box."""
    __slots__ = ("locations_tuple", "meta")

    def __init__(self, locations_tuple, meta=None):
        self.locations_tuple = locations_tuple
        self.meta = meta or {}  # carries anchor_types etc. for synth_chain_plans in parallel workers


def _load_locations_only(name: str) -> _LocOnly:
    import json
    d = os.path.join(WORLDS_DIR, name)
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    fac = pd.read_parquet(os.path.join(d, "facilities.parquet"))
    types = list(meta.get("types") or fac["type"].drop_duplicates().tolist())
    ids_d, coords_d, pots_d = {}, {}, {}
    for t in types:
        sub = fac[fac["type"] == t]
        ids_d[t] = sub["loc_id"].to_numpy(dtype=object)
        coords_d[t] = np.column_stack([sub["x"].to_numpy(float), sub["y"].to_numpy(float)])
        pots_d[t] = sub["potential"].to_numpy(float)
    return _LocOnly((ids_d, coords_d, pots_d), meta)


def _solve_timed(world, plans, solver, seed: int, extra: Optional[dict] = None) -> Tuple[pd.DataFrame, float, bool]:
    cls, params = _spec(solver)
    params = {**(params or {}), **(extra or {})} or None
    ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=seed, parameters=params)
    t0 = time.perf_counter()
    rdf, _, valid = run.solve(ctx=ctx, plans_df=plans)
    return rdf, time.perf_counter() - t0, bool(valid)


def _scored_legs(plans: pd.DataFrame, gt: pd.DataFrame) -> set:
    """Leg ids whose geometry a placement controls = every leg touching a free node on EITHER
    side (to_is_free, OR from is the previous leg's free to-node, in travel order). This is the
    objective dp_full/CARLA/RDA all minimize; scoring only `to_is_free` legs under-counts the
    outgoing leg and lets a solver appear to beat the oracle."""
    free_to = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    scored = set()
    for _pid, idx in plans.groupby("unique_person_id", sort=False).groups.items():
        prev_free = False
        for lid in plans.loc[idx, "unique_leg_id"]:           # row order = travel order
            tf = lid in free_to
            if tf or prev_free:
                scored.add(lid)
            prev_free = tf
    return scored


def per_person_dev(rdf: pd.DataFrame, plans: pd.DataFrame, gt: pd.DataFrame,
                   scored: Optional[set] = None) -> pd.Series:
    """Total |observed - achieved| over each person's free-segment legs (the alpha=0 objective).
    A person is scored only if all its scored legs are placed finitely; else NaN (a coverage
    failure), so a solver that fails legs cannot appear to beat the oracle by summing fewer terms."""
    if scored is None:
        scored = _scored_legs(plans, gt)
    pid_of = plans.set_index("unique_leg_id")["unique_person_id"]
    expected = pid_of[pid_of.index.isin(scored)].value_counts()
    sub = rdf[rdf["unique_leg_id"].isin(scored)].copy()
    ach = np.hypot(sub["to_x"] - sub["from_x"], sub["to_y"] - sub["from_y"])
    sub["dev"] = (sub["distance_meters"] - ach).abs()
    sub["pid"] = sub["unique_leg_id"].map(pid_of)
    grp = sub.groupby("pid")["dev"]
    total = grp.sum(min_count=1)
    n_finite = grp.apply(lambda x: int(x.notna().sum()))
    return total.where(n_finite == expected.reindex(n_finite.index))


def gaps_over_oracle(world, plans, gt, solvers: List[str], seed: int) -> Dict[str, dict]:
    """Run the oracle + each solver on the SAME plans; return per-solver mean metres-above-oracle
    (mean +/- SE over persons), runtime, validity. Oracle is dp_full unless absent."""
    orc_rdf, orc_t, orc_valid = _solve_timed(world, plans, ORACLE, seed)
    orc = per_person_dev(orc_rdf, plans, gt)
    out: Dict[str, dict] = {}
    for s in solvers:
        if s == ORACLE:
            gap = orc - orc  # zero by construction
            rt, valid = orc_t, orc_valid
        else:
            rdf, rt, valid = _solve_timed(world, plans, s, seed)
            gap = (per_person_dev(rdf, plans, gt) - orc).reindex(orc.index)
        g = gap.to_numpy(float)
        out[s] = {"gap_mean": float(np.nanmean(g)),
                  "gap_se": float(np.nanstd(g) / max(1, np.sqrt(np.sum(~np.isnan(g))))),
                  "runtime_s": rt, "valid": valid, "n": int(np.sum(~np.isnan(g)))}
    return out


# --------------------------------------------------------------------------- #
# difficulty regimes (GT preserved; distance axis + anchor axis)
# --------------------------------------------------------------------------- #

def make_regimes(plans, gt, world, rng, *, dist_noise=0.15, anchor_sigma=500.0) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(label, plans, gt) per regime. Anchor axis uses TRUE distances (OFAT)."""
    samples = S.per_mode_distance_samples(world.plans_df, world.ground_truth)

    def add_noise(pl):
        free = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
        out = pl.copy(); d = out["distance_meters"].to_numpy(float).copy(); ids = out["unique_leg_id"].to_numpy()
        for i in range(len(out)):
            if ids[i] in free:
                d[i] = max(1.0, d[i] * (1.0 + float(rng.normal(0, dist_noise))))
        out["distance_meters"] = d
        return out

    dem_pl, dem_gt = S.demote_anchor(plans, gt, anchor="work")
    return [
        ("true", plans, gt),
        (f"dist_noise={dist_noise}", add_noise(plans), gt),
        ("dist_sampled", S.resample_distances(plans, gt, samples, rng, feasible=True), gt),
        (f"anchor_disturb={anchor_sigma:.0f}m", S.disturb_anchor(plans, anchor="work", noise_m=anchor_sigma, rng=rng), gt),
        ("anchor_remove", dem_pl, dem_gt),
    ]


def _save_raw(dev: Dict, out_dir: str, fname: str, key_cols: Tuple[str, ...]) -> None:
    """Persist FULL per-person raw deviations (no aggregation): one row per (cell, person). `dev`
    maps a key (scalar or tuple matching `key_cols`) -> per-person dev Series (NaN = unplaced).
    Every mean/SE plotted elsewhere is recomputable from this; nothing is pre-summarised."""
    frames = []
    for key, series in dev.items():
        kv = key if isinstance(key, tuple) else (key,)
        cols = {c: v for c, v in zip(key_cols, kv)}
        cols["unique_person_id"] = np.asarray(series.index)
        cols["dev_m"] = series.to_numpy(float)
        frames.append(pd.DataFrame(cols))
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=[*key_cols, "unique_person_id", "dev_m"]))
    out.to_csv(os.path.join(out_dir, fname), index=False)


# --------------------------------------------------------------------------- #
# result 1: gap x difficulty
# --------------------------------------------------------------------------- #

_W = None  # per-worker world (parallel path)


def _init_worker(world_name):
    global _W
    _W = _load_locations_only(world_name)  # locations only: workers never touch plans/gt (passed via args)


def _cell(args):
    """One (solver, regime) cell. `solver` is a NAME (worker rebuilds its own spec, incl. the
    vendored-RDA closures) so nothing unpicklable crosses the process boundary."""
    solver, label, plans, gt, scored, seed = args
    rdf, rt, valid = _solve_timed(_W, plans, solver, seed)
    return solver, label, per_person_dev(rdf, plans, gt, scored), rt, valid


def result_gap_difficulty(world, n_persons, solvers, seed, out_dir, world_name=None, jobs=1,
                          anchor_sigma=1000.0) -> pd.DataFrame:
    assert ORACLE in solvers, f"result 1 needs {ORACLE!r} in the solver list (it is the gap baseline)"
    w = S.sample_persons(world, n_persons, seed=seed)
    rng = np.random.default_rng(seed)
    regimes = make_regimes(w.plans_df, w.ground_truth, world, rng, anchor_sigma=anchor_sigma)
    tasks, regime_data = [], {}
    for label, pl, gt in regimes:
        scored = _scored_legs(pl, gt)
        regime_data[label] = (pl, gt, scored)
        tasks += [(s, label, pl, gt, scored, seed) for s in solvers]

    if jobs > 1 and world_name:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(world_name,)) as ex:
            cells = list(ex.map(_cell, tasks))
    else:
        globals()["_W"] = world
        cells = [_cell(t) for t in tasks]

    dev, rt, valid = {}, {}, {}
    for s, label, d, t, v in cells:
        dev[(s, label)], rt[(s, label)], valid[(s, label)] = d, t, v

    _save_raw(dev, out_dir, "1_gap_raw.csv", ("solver", "regime"))
    pd.DataFrame([{"solver": s, "regime": l, "runtime_s": rt[(s, l)], "valid": valid[(s, l)]}
                  for (s, l) in dev]).to_csv(os.path.join(out_dir, "1_gap_meta.csv"), index=False)

    rows = []
    for label in regime_data:
        orc = dev[(ORACLE, label)]
        for s in solvers:
            g = (dev[(s, label)] - orc).reindex(orc.index).to_numpy(float)
            ok = ~np.isnan(g)
            se = float(np.nanstd(g, ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
            rows.append({"regime": label, "solver": s, "gap_mean": float(np.nanmean(g)), "gap_se": se,
                         "runtime_s": rt[(s, label)], "valid": valid[(s, label)], "n": int(ok.sum())})
    df = pd.DataFrame(rows)
    _plot_gap_difficulty(df, os.path.join(out_dir, "1_gap_difficulty.png"))
    return df


def _plot_gap_difficulty(df, path):
    import matplotlib.pyplot as plt
    regimes = df["regime"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in df["solver"].drop_duplicates():
        if s in _BASELINES:  # rda/guided are ~20-300x larger -> they squash the placement solvers
            continue          # to ~0; the raw CSV keeps them for a separate baseline panel
        d = df[df["solver"] == s].set_index("regime").reindex(regimes)
        ax.errorbar(range(len(regimes)), d["gap_mean"], yerr=d["gap_se"], marker="o", capsize=2, label=s)
    ax.set_xticks(range(len(regimes))); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylabel("metres above oracle (per person)")
    ax.set_title("Block A.1 — gap-to-oracle x difficulty (placement solvers; rda baselines omitted)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 2: quality-runtime frontier (swept at a degraded regime so gaps exist)
# --------------------------------------------------------------------------- #

_KNOBS = {
    "carla": ("number_of_branches", [2, 5, 10, 20, 40]),
    "dp_carla": ("min_candidates", [5, 10, 25, 50, 100]),
    "dp_carla_refine": ("refine_passes", [0, 1, 2, 5, 10]),
    "rda": ("n_restarts", [1, 3, 10, 20, 40]),
}


def result_frontier(world, n_persons, seed, out_dir) -> pd.DataFrame:
    w = S.sample_persons(world, n_persons, seed=seed)
    rng = np.random.default_rng(seed)
    # degraded regime (sampled distances) so deviation > 0 and a quality/runtime tradeoff exists
    samples = S.per_mode_distance_samples(world.plans_df, world.ground_truth)
    # feasible=True mirrors result-1's dist_sampled regime: infeasible draws collapse CARLA's
    # candidate pool below num_mc and crash the monte_carlo selector (and unfairly handicap argmin).
    pl = S.resample_distances(w.plans_df, w.ground_truth, samples, rng, feasible=True); gt = w.ground_truth
    scored = _scored_legs(pl, gt)
    rows, raw = [], {}
    for s, (knob, vals) in _KNOBS.items():
        for v in vals:
            cls, params = _frontier_spec(s, knob, v)
            ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=seed, parameters=params)
            t0 = time.perf_counter(); rdf, _, _ = run.solve(ctx=ctx, plans_df=pl); dt = time.perf_counter() - t0
            per = per_person_dev(rdf, pl, gt, scored)
            raw[(s, knob, v)] = per
            rows.append({"solver": s, "knob": knob, "val": v, "runtime_s": dt, "mean_dev_m": float(per.mean())})
    df = pd.DataFrame(rows)
    _save_raw(raw, out_dir, "2_frontier_raw.csv", ("solver", "knob", "val"))
    df.drop(columns=["mean_dev_m"]).to_csv(os.path.join(out_dir, "2_frontier_meta.csv"), index=False)
    _plot_frontier(df, os.path.join(out_dir, "2_frontier.png"))
    return df


def _plot_frontier(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in df["solver"].drop_duplicates():
        d = df[df["solver"] == s].sort_values("runtime_s")
        ax.plot(d["runtime_s"], d["mean_dev_m"], marker="o", label=s)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("runtime [s] (log)"); ax.set_ylabel("mean deviation [m] (log)")
    ax.set_title("Block A.2 — quality-runtime frontier"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 3: chain-length scaling (bin the natural sample by #free legs)
# --------------------------------------------------------------------------- #

def synth_chain_plans(world, n_free: int, k: int, seed: int, leg_m: float = 3000.0):
    """k synthetic persons, each a 2-anchor chain with `n_free` secondary nodes (chain length =
    n_free+1 legs), on the world's facility types — used to drive the chain-length scaling cleanly
    (the natural sample's long-chain tail is too thin). Distances are feasible by construction
    (legs ~leg_m, end anchor within ~leg_m of start)."""
    rng = np.random.default_rng(seed)
    # place only SECONDARY (free) types: anchors (home/work) are never placed, and their catalogs
    # differ across worlds (home 15k vs 67k), which would otherwise leak a world artefact into the
    # scaling curve. The secondary catalogs (shop/leisure/education/other) are matched across worlds.
    anchors = set(world.meta.get("anchor_types", ("home", "work")))
    types = [t for t in world.locations_tuple[0] if t not in anchors]
    allc = np.vstack([np.asarray(world.locations_tuple[1][t], float) for t in types])
    lo, hi = allc.min(0), allc.max(0)
    rows, gtr = [], []
    n_legs = n_free + 1
    for p in range(k):
        start = rng.uniform(lo, hi)
        end = start + rng.normal(0, leg_m, 2)
        for j in range(n_legs):
            lid = f"n{n_free}-{p}-l{j+1}"
            rows.append({"unique_person_id": f"n{n_free}-{p}", "unique_leg_id": lid,
                         "to_act_type": types[j % len(types)], "mode": "car",
                         "distance_meters": float(rng.uniform(0.8, 1.2) * leg_m),
                         "from_x": start[0] if j == 0 else np.nan, "from_y": start[1] if j == 0 else np.nan,
                         "to_x": end[0] if j == n_legs - 1 else np.nan, "to_y": end[1] if j == n_legs - 1 else np.nan})
            gtr.append({"unique_leg_id": lid, "to_is_free": j < n_legs - 1, "true_to_identifier": None})
    return pd.DataFrame(rows), pd.DataFrame(gtr)


def result_scaling(world, n_persons, seed, out_dir,
                   lengths=(1, 2, 3, 4, 6, 8, 10, 12, 14), cap_s=45.0) -> pd.DataFrame:
    """Runtime vs chain length on synthetic chains. k (synthetic draws averaged per point) shrinks
    as n grows but stays large enough to smooth per-chain timing variance; a solver is dropped for
    longer n once a single point exceeds `cap_s` (CARLA's exponential blowup is the point)."""
    rows = []
    for solver in ["carla", "dp_carla", "dp_carla_refine"]:
        for n in lengths:
            k = 60 if n <= 4 else 30 if n <= 8 else 12  # more draws -> smoother curve (was 25/8/3)
            pl, _ = synth_chain_plans(world, n, k, seed)
            cls, params = _spec(solver)
            ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=seed, parameters=params)
            t0 = time.perf_counter(); run.solve(ctx=ctx, plans_df=pl); dt = time.perf_counter() - t0
            rows.append({"solver": solver, "n_free": n, "k": k, "dt_s": dt, "ms_per_person": 1000 * dt / k})
            if dt > cap_s:  # blew up -> stop pushing this solver to longer chains
                break
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "3_scaling.csv"), index=False)
    _plot_scaling(df, os.path.join(out_dir, "3_scaling.png"))
    return df


def _plot_scaling(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in df["solver"].drop_duplicates():
        d = df[df["solver"] == s].sort_values("n_free")
        ax.plot(d["n_free"], d["ms_per_person"], marker="o", label=s)
    ax.set_yscale("log"); ax.set_xlabel("chain length (free nodes)"); ax.set_ylabel("ms / person (log)")
    ax.set_title("Block A.3 — runtime vs chain length"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 4: generation, not search, is binding (min_candidates sweep)
# --------------------------------------------------------------------------- #

def result_generation(world, n_persons, seed, out_dir) -> pd.DataFrame:
    """Decompose the gap-to-oracle along the solver ladder at TRUE distances:
    carla->dp_carla = value of SEARCH (~0, solved); dp_carla->dp_full = value of GENERATION,
    closed by neighbour refinement (dp_carla_refine). Search is solved; generation binds."""
    w = S.sample_persons(world, n_persons, seed=seed)
    pl, gt = w.plans_df, w.ground_truth
    scored = _scored_legs(pl, gt)
    ladder = [s for s in ["carla", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", "dp_full"]
              if s in SOLVERS]
    dev, meta = {}, []
    for s in ladder:                                  # one solve per ladder rung; keep per-person raw
        rdf, rt, valid = _solve_timed(world, pl, s, seed)
        dev[s] = per_person_dev(rdf, pl, gt, scored)
        meta.append({"solver": s, "runtime_s": rt, "valid": valid})
    _save_raw(dev, out_dir, "4_generation_raw.csv", ("solver",))
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, "4_generation_meta.csv"), index=False)
    orc = dev[ORACLE]
    rows = []
    for s in ladder:
        g = (dev[s] - orc).reindex(orc.index).to_numpy(float)
        ok = ~np.isnan(g)
        se = float(np.nanstd(g, ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
        rows.append({"solver": s, "gap_m": float(np.nanmean(g)), "se": se})
    df = pd.DataFrame(rows)
    _plot_generation(df, ladder, os.path.join(out_dir, "4_generation.png"))
    return df


def _plot_generation(df, order, path):
    import matplotlib.pyplot as plt
    d = df.set_index("solver").reindex([s for s in order if s in set(df["solver"])])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(d)), d["gap_m"], yerr=d["se"], capsize=3,
           color=["#bbb", "#9cf", "#6ad", "#48b", "#27a", "#2a2"][:len(d)])
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, rotation=20, ha="right")
    ax.set_ylabel("metres above oracle"); ax.set_title("Block A.4 — search is solved; generation binds")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 5: the dp_full N-wall (pruned ~flat in N; oracle is O(n*N^2))
# --------------------------------------------------------------------------- #

def _subsample_locations(locations_tuple, n_per_type, seed):
    ids, coords, pots = locations_tuple
    rng = np.random.default_rng(seed)
    nids, ncoords, npots = {}, {}, {}
    for t in ids:
        N = len(ids[t]); k = min(n_per_type, N); idx = rng.choice(N, k, replace=False)
        nids[t] = np.asarray(ids[t])[idx]
        ncoords[t] = np.asarray(coords[t], float)[idx]
        npots[t] = np.asarray(pots[t], float)[idx]
    return nids, ncoords, npots


def result_nwall(world, n_persons, seed, out_dir, levels=(250, 500, 1000, 2000, 4000)) -> pd.DataFrame:
    """Runtime vs catalog size N (per type), timed on a small fixed person set. Pruned solvers are
    ~flat (KD-tree query capped at min_candidates); dp_full is O(n*N^2) -> the wall that motivates
    dp_carla_pot as the scalable near-oracle."""
    w = S.sample_persons(world, min(n_persons, 60), seed=seed)
    npers = w.plans_df["unique_person_id"].nunique()
    # cap levels at the largest PLACEABLE (free) catalog — past it dp_full's cost driver saturates
    # and the x-axis would be mislabeled (e.g. shop=3759 < 4000).
    free_types = set(w.plans_df.loc[w.plans_df["unique_leg_id"].isin(
        set(w.ground_truth.loc[w.ground_truth["to_is_free"], "unique_leg_id"])), "to_act_type"])
    max_sec = max((len(world.locations_tuple[0][t]) for t in free_types if t in world.locations_tuple[0]), default=10**9)
    levels = tuple(N for N in levels if N <= max_sec) or (min(max_sec, max(levels)),)
    # all algorithms (rda_guided excluded — broken & slow): only dp_full should show the N^2 wall.
    nwall_solvers = [s for s in ["carla", "dp_rings", "dp_carla", "dp_rings_refine",
                                 "dp_carla_refine", "rda", "dp_full"] if s in SOLVERS]
    rows = []
    for N in levels:
        loc = _subsample_locations(world.locations_tuple, N, seed)
        for solver in nwall_solvers:
            cls, params = _spec(solver)
            ctx = run.setup(locations_tuple=loc, solver=cls, rng_seed=seed, parameters=params)
            t0 = time.perf_counter()
            try:
                run.solve(ctx=ctx, plans_df=w.plans_df); dt = (time.perf_counter() - t0)
            except Exception:
                dt = float("nan")
            rows.append({"N_per_type": N, "solver": solver, "npersons": npers, "dt_s": dt,
                         "ms_per_person": 1000 * dt / npers})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "5_nwall.csv"), index=False)
    _plot_nwall(df, os.path.join(out_dir, "5_nwall.png"))
    return df


def _plot_nwall(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in df["solver"].drop_duplicates():
        d = df[df["solver"] == s].sort_values("N_per_type")
        ax.plot(d["N_per_type"], d["ms_per_person"], marker="o", label=s)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("facilities per type N (log)")
    ax.set_ylabel("ms / person (log)"); ax.set_title("Block A.5 — the dp_full N-wall (pruned stays flat)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 6: does a bigger generated pool close the recall (generation) gap?
# --------------------------------------------------------------------------- #

def result_recall(world, n_persons, seed, out_dir, mins=(10, 25, 50, 100, 200, 400)) -> pd.DataFrame:
    """Sweep dp_carla's min_candidates (the generated-pool floor) at TRUE distances and measure the
    gap above dp_full. Gap shrinks with the pool -> the generation gap is a BUDGET artifact; gap
    stays flat -> a true geometry limit of CARLA's ring/circle generation (two_zone is where it
    bites; gauss/osm sit ~0 already, showing their pool is sufficient)."""
    w = S.sample_persons(world, n_persons, seed=seed)
    pl, gt = w.plans_df, w.ground_truth
    scored = _scored_legs(pl, gt)
    orc = per_person_dev(_solve_timed(world, pl, ORACLE, seed)[0], pl, gt, scored)
    dev, rows = {}, []
    for mc in mins:
        rdf, rt, _ = _solve_timed(world, pl, "dp_carla", seed, extra={"min_candidates": mc})
        gap = (per_person_dev(rdf, pl, gt, scored) - orc).reindex(orc.index)
        dev[mc] = gap
        g = gap.to_numpy(float); ok = ~np.isnan(g)
        rows.append({"min_candidates": mc, "gap_m": float(np.nanmean(g)),
                     "se": float(np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"),
                     "runtime_s": rt})
    _save_raw(dev, out_dir, "6_recall_raw.csv", ("min_candidates",))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "6_recall.csv"), index=False)
    _plot_recall(df, os.path.join(out_dir, "6_recall.png"))
    return df


def _plot_recall(df, path):
    import matplotlib.pyplot as plt
    d = df.sort_values("min_candidates")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(d["min_candidates"], d["gap_m"], yerr=d["se"], marker="o", capsize=3, color="#27a")
    ax.set_xscale("log"); ax.set_xlabel("min_candidates (pruned-DP pool floor, log)")
    ax.set_ylabel("dp_carla gap above oracle [m]")
    ax.set_title("Block A.6 — does a bigger pool close the generation gap?")
    ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# result 7: the density trade -- gap AND runtime vs facility density
# --------------------------------------------------------------------------- #

def result_density(world, n_persons, seed, out_dir, levels=(250, 500, 1000, 2000, 4000)) -> pd.DataFrame:
    """At each facilities-per-type N, the gap-to-oracle AND runtime for carla vs dp_carla vs the full
    oracle (oracle recomputed per N -> gap is vs the N-specific global optimum). dp_full's runtime
    walls up O(N^2) while carla's stays flat; if carla's GAP climbs with N (a fixed beam saturating
    among more candidates) the better time/deviation tradeoff shifts to the heuristic in dense
    catalogs -- the density axis A5 only timed."""
    w = S.sample_persons(world, min(n_persons, 300), seed=seed)
    pl, gt = w.plans_df, w.ground_truth
    scored = _scored_legs(pl, gt)
    npers = pl["unique_person_id"].nunique()
    free_types = set(pl.loc[pl["unique_leg_id"].isin(
        set(gt.loc[gt["to_is_free"], "unique_leg_id"])), "to_act_type"])
    max_sec = max((len(world.locations_tuple[0][t]) for t in free_types if t in world.locations_tuple[0]), default=10**9)
    levels = tuple(N for N in levels if N <= max_sec) or (min(max_sec, max(levels)),)
    rows = []
    for N in levels:
        loc = _subsample_locations(world.locations_tuple, N, seed)

        def solve(s):
            cls, params = _spec(s)
            ctx = run.setup(locations_tuple=loc, solver=cls, rng_seed=seed, parameters=params)
            t0 = time.perf_counter(); rdf, _, _ = run.solve(ctx=ctx, plans_df=pl)
            return rdf, time.perf_counter() - t0

        orc_rdf, orc_rt = solve(ORACLE)
        orc = per_person_dev(orc_rdf, pl, gt, scored)
        rows.append({"N_per_type": N, "solver": ORACLE, "gap_m": 0.0, "se": 0.0, "runtime_ms": 1000 * orc_rt / npers})
        for s in ["carla", "dp_carla"]:
            rdf, rt = solve(s)
            g = (per_person_dev(rdf, pl, gt, scored) - orc).reindex(orc.index).to_numpy(float)
            ok = ~np.isnan(g)
            rows.append({"N_per_type": N, "solver": s, "gap_m": float(np.nanmean(g)),
                         "se": float(np.nanstd(g[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"),
                         "runtime_ms": 1000 * rt / npers})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "7_density.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #
# result 8: density x length -- runtime AND gap for the whole DP family + carla
# --------------------------------------------------------------------------- #

_DL_W = None  # per-worker locations-only world for the result-8 grid


def _dl_init(world_name):
    global _DL_W
    _DL_W = _load_locations_only(world_name)


def _dl_cell(args):
    """One (n, N) cell: synth chains of length n, catalog subsampled to N, then oracle + every
    solver. Self-contained so it parallelizes; returns the rows for this cell."""
    n, N, k, seed = args
    world = _DL_W
    pl, gt = synth_chain_plans(world, n, k, seed)
    scored = _scored_legs(pl, gt)
    npers = pl["unique_person_id"].nunique()
    loc = _subsample_locations(world.locations_tuple, N, seed)
    solvers = [s for s in ["carla", "rda", "dp_rings", "dp_carla", "dp_rings_refine", "dp_carla_refine", ORACLE]
               if s in SOLVERS]

    def solve(s):
        cls, params = _spec(s)
        ctx = run.setup(locations_tuple=loc, solver=cls, rng_seed=seed, parameters=params)
        t0 = time.perf_counter(); rdf, _, _ = run.solve(ctx=ctx, plans_df=pl)
        return rdf, time.perf_counter() - t0

    orc_rdf, orc_rt = solve(ORACLE)
    orc = per_person_dev(orc_rdf, pl, gt, scored)
    rows = [{"n_free": n, "N_per_type": N, "solver": ORACLE, "runtime_ms": 1000 * orc_rt / npers, "gap_m": 0.0}]
    for s in solvers:
        if s == ORACLE:
            continue
        rdf, rt = solve(s)
        g = (per_person_dev(rdf, pl, gt, scored) - orc).reindex(orc.index).to_numpy(float)
        ok = ~np.isnan(g)
        rows.append({"n_free": n, "N_per_type": N, "solver": s, "runtime_ms": 1000 * rt / npers,
                     "gap_m": float(np.nanmean(g)) if ok.any() else float("nan")})
    return rows


def result_density_length(world, n_persons, seed, out_dir, lengths=(2, 6, 10),
                          levels=(250, 500, 1000, 2000, 4000), k=20, jobs=1, world_name=None) -> pd.DataFrame:
    """Density x length grid (synthetic chains): runtime AND gap-to-oracle for the whole DP family
    (+ carla + rda). dp_full pays O(n N^2), so its density wall is ~n x taller at long chains; the
    geometrically pruned DPs cap K by generation -> flat in N at every length (linear in n only);
    carla is exponential in n; rda scales with n (relaxation) but is flat in N. Each (n, N) cell is
    independent, so the grid parallelizes across `jobs` workers (every cell runs to completion --- no
    cap/skip, so the grid is always full)."""
    # taper draws up at short/cheap lengths (smoother heavy-tailed gap) without exploding the costly
    # long-chain cells: n<=2 -> 60, n<=6 -> 40, else the floor k (20). carla at n=10 dominates cost.
    def _k(n):
        return max(k, 60 if n <= 2 else 40 if n <= 6 else 0)
    cells = [(n, N, _k(n), seed) for n in lengths for N in levels]
    if jobs > 1 and world_name:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, initializer=_dl_init, initargs=(world_name,)) as ex:
            results = list(ex.map(_dl_cell, cells))
    else:
        globals()["_DL_W"] = world
        results = [_dl_cell(c) for c in cells]
    df = pd.DataFrame([r for cell in results for r in cell])
    df.to_csv(os.path.join(out_dir, "8_density_length.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", default="gauss_hannover")
    p.add_argument("--persons", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--results", nargs="+", default=["1", "2", "3", "4", "5", "6", "7"])
    p.add_argument("--jobs", type=int, default=1, help="parallel workers for result 1 (persons/cells)")
    p.add_argument("--anchor-sigma", type=float, default=1000.0, help="work-anchor jitter sigma (m) for the anchor-disturb regime")
    p.add_argument("--quick", action="store_true", help="tiny run (40 persons, subset of solvers)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    persons = 40 if args.quick else args.persons
    solvers = (["carla", "dp_carla", "dp_carla_refine", "dp_full", "rda"] if args.quick else DIST_SOLVERS)

    world = load(args.world)
    print(f"world={args.world}  full_persons={world.meta.get('n_persons')}  facilities={world.meta.get('n_locations')}  "
          f"-> sampling {persons} (seed {args.seed})  RDA={'vendored eqasim' if RDA_IS_VENDORED else 'MIT remake'}", flush=True)

    if "1" in args.results:
        t0 = time.perf_counter()
        df = result_gap_difficulty(world, persons, solvers, args.seed, args.out, world_name=args.world,
                                   jobs=args.jobs, anchor_sigma=args.anchor_sigma)
        order = df["regime"].drop_duplicates().tolist()  # true -> degraded (insertion order)
        print(f"[1] gap x difficulty  ({time.perf_counter()-t0:.1f}s)\n",
              df.pivot(index="regime", columns="solver", values="gap_mean").reindex(order).round(1).to_string(), "\n", flush=True)
    if "2" in args.results:
        t0 = time.perf_counter(); result_frontier(world, persons, args.seed, args.out)
        print(f"[2] frontier  ({time.perf_counter()-t0:.1f}s)", flush=True)
    if "3" in args.results:
        t0 = time.perf_counter(); df = result_scaling(world, persons, args.seed, args.out)
        print(f"[3] scaling  ({time.perf_counter()-t0:.1f}s)\n",
              df.pivot(index="n_free", columns="solver", values="ms_per_person").round(2).to_string(), "\n", flush=True)
    if "4" in args.results:
        t0 = time.perf_counter(); df = result_generation(world, persons, args.seed, args.out)
        print(f"[4] generation-vs-search  ({time.perf_counter()-t0:.1f}s)\n", df.round(1).to_string(index=False), flush=True)
    if "5" in args.results:
        t0 = time.perf_counter(); df = result_nwall(world, persons, args.seed, args.out)
        print(f"[5] dp_full N-wall  ({time.perf_counter()-t0:.1f}s)\n",
              df.pivot(index="N_per_type", columns="solver", values="ms_per_person").round(2).to_string(), flush=True)
    if "6" in args.results:
        t0 = time.perf_counter(); df = result_recall(world, persons, args.seed, args.out)
        print(f"[6] recall vs min_candidates  ({time.perf_counter()-t0:.1f}s)\n",
              df.round(2).to_string(index=False), flush=True)
    if "7" in args.results:
        t0 = time.perf_counter(); df = result_density(world, persons, args.seed, args.out)
        print(f"[7] density trade (gap+runtime vs N)  ({time.perf_counter()-t0:.1f}s)\n",
              df.round(2).to_string(index=False), flush=True)
    if "8" in args.results:  # opt-in (heavy): density x length grid for the whole DP family
        t0 = time.perf_counter()
        df = result_density_length(world, persons, args.seed, args.out, jobs=args.jobs, world_name=args.world)
        print(f"[8] density x length  ({time.perf_counter()-t0:.1f}s)\n",
              df.round(2).to_string(index=False), flush=True)
    print(f"\nplots -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
