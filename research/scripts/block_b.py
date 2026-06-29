#!/usr/bin/env python
"""Block B — the eval, now with potentials: the alpha sweep (full design, parallel + raw-saving).

A is alpha=0 (distance only); B turns attractiveness on and SWEEPS it. ONE knob — the injected
`Scorer.pot_weight` (= alpha) — drives every solver: the argmin DP family reads it as the potential
weight in their objective; the generative `dp_sample` reads it as its MNL attractiveness coefficient.

The combined objective uses the **log1p** attractiveness form (`attr_transform="log1p"`), matching the
calibrated MNL and `dp_sample` — so the argmin optimizes the correctly-specified objective (its
over-concentration is then purely the argmax-vs-sample effect, not objective mis-specification).
Block A (alpha=0) is untouched: the potential term vanishes at alpha=0.

Axes (cartesian product): alpha (the sweep) x work {anchored, demoted} x input {true, sampled} x world.

RAW-FIRST: we persist one per-free-leg table (`raw_legs_<world>.parquet`) holding, per cell,
(person, leg, type, chosen_loc_id, observed/achieved distance, chosen potential). EVERY metric
(decile-TV, facility-TV, zone-TV, Spearman, mean deviation, Wasserstein, cost decomposition) and
bootstrap CI is then a post-hoc replot on that raw — no re-solve to change a metric or add a solver.

Aggregate metrics recorded per cell (also derivable from the raw): combined_cost (on the log1p
objective) + its parts (dist_dev_m, pot_captured); potential-fit as decile-TV (headline),
facility-TV + zone-TV (diagnostics that mislead — see memory), and visits~potential Spearman;
distance realism as Wasserstein vs the TRUE distribution; recovery (diagnostic). The cost GAP to the
oracle is computed offline (plot_block_b.add_eff_gap) from combined_cost, so cells are independent
and parallelize cleanly.

    python research/scripts/block_b.py --quick
    python research/scripts/block_b.py --persons 500 --jobs 16 --worlds gauss_hannover two_zone
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

from chainsolvers import run
from chainsolvers.scoring_selection import Scorer
from chainsolvers.solvers.dp import DpSample  # for the candidate-capped sampler variant
from chainsolvers_eval import survey as S
from chainsolvers_eval.calibration import fit_location_choice, fit_mode_kernels
from chainsolvers_eval.baselines import GravityIndependent  # no-coupling gravity sampler (floor)
from chainsolvers_eval.depletion import visit_potential_fit
from chainsolvers_eval.worlds import load_world

from block_a import _scored_legs, per_person_dev, SOLVERS, _load_locations_only  # noqa: E402
from block_a import _spec as _spec_a  # noqa: E402

WORLDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "worlds")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "block_b")
CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "calib_cache")
ORACLE = "dp_full"
ATTR_TRANSFORM = "log1p"  # combined-objective attractiveness form (matches calibration + dp_sample)
SAMPLE_CAP = 3000         # dp_sample_capped: max candidate pool per free node (bounds cdist to cap^2)

ARGMIN = ["carla", "dp_carla", "dp_carla_refine", "dp_carla_pot", "dp_full"]
# sampler ladder (realism panels): exact-joint, greedy-ancestral D&C, no-coupling floor
MODELS = ["dp_sample", "carla_sample", "gravity_independent"]
FLAT_REFS = ["rda", "rda_guided"]

_CLASS_SOLVERS = {"gravity_independent": GravityIndependent,  # baseline classes (not registry keys)
                  "dp_sample_capped": DpSample}               # dp_sample + max_candidates cap (fast variant)


def _spec(name):
    if name in _CLASS_SOLVERS:
        return _CLASS_SOLVERS[name], None
    if name in SOLVERS:
        return _spec_a(name)
    return name, None


def load(name: str):
    return load_world(os.path.join(WORLDS_DIR, name))


# --------------------------------------------------------------------------- #
# calibration (serial, once per world)
# --------------------------------------------------------------------------- #

def calibrate(world, plans, gt, max_persons=400) -> Dict[str, object]:
    alpha_cal, scale_cal = fit_location_choice(world.topology, plans, gt, transform="log1p",
                                               max_persons=max_persons)
    _, decay_scales, tail_weights, tail_scale_factors, pooled = fit_mode_kernels(
        world.topology, plans, gt, transform="log1p", max_persons=max_persons)
    scale_p, w_p, tf_p = pooled
    sample_params = {
        "decay_scales": decay_scales, "default_scale": scale_p,
        "tail_weights": tail_weights, "tail_weight": w_p,
        "tail_scale_factors": tail_scale_factors, "tail_scale_factor": tf_p,
        "attr_transform": "log1p",
    }
    return {"alpha_cal": alpha_cal, "scale_cal": scale_cal, "sample_params": sample_params}


def calibrate_cached(world, world_name, plans, gt, seed, n_persons, max_persons=400, refit=False) -> dict:
    """Calibration is deterministic given (world, seed, n_persons, max_persons, transform) and the
    worlds never change, so cache it to disk (research/data/calib_cache) and reuse — the MLE costs
    ~minutes/world otherwise. `--refit` forces a fresh fit."""
    key = f"{world_name}_seed{seed}_n{n_persons}_mp{max_persons}_log1p.json"
    path = os.path.join(CALIB_DIR, key)
    if not refit and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        print(f"  [{world_name}] calibration CACHE HIT (alpha={d['alpha_cal']:.3f}, "
              f"scale={d['scale_cal']:.0f}m)", flush=True)
        return d
    cal = calibrate(world, plans, gt, max_persons)
    os.makedirs(CALIB_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal, f)
    return cal


def _merge_on_keys(old: pd.DataFrame, new: pd.DataFrame,
                   keys=("work", "input", "solver", "alpha")) -> pd.DataFrame:
    """Append `new` over `old`: drop old rows whose (work,input,solver,alpha) appear in `new`
    (so re-running a solver/cell updates it), then concat. Works for the aggregate CSV (one row
    per cell) and the raw-legs table (many rows per cell)."""
    keys = list(keys)
    new_keys = set(map(tuple, new[keys].itertuples(index=False, name=None)))
    keep = [t not in new_keys for t in old[keys].itertuples(index=False, name=None)]
    return pd.concat([old[keep], new], ignore_index=True)


# --------------------------------------------------------------------------- #
# metrics (all also recomputable from raw_legs)
# --------------------------------------------------------------------------- #

def _dist_wasserstein(rdf, free_ids, ref_dists: np.ndarray) -> float:
    df = rdf[rdf["unique_leg_id"].isin(free_ids)]
    real = np.hypot(df["to_x"].to_numpy(float) - df["from_x"].to_numpy(float),
                    df["to_y"].to_numpy(float) - df["from_y"].to_numpy(float))
    real = real[np.isfinite(real)]
    if real.size == 0 or ref_dists.size == 0:
        return float("nan")
    return float(wasserstein_distance(real, ref_dists))


def _type_visit_counts(world, rdf, free_ids):
    """{type: (potential vec, realized visit-count vec)} aligned to the per-type catalog order."""
    ids, _, pots = world.locations_tuple
    df = rdf[rdf["unique_leg_id"].isin(free_ids)].dropna(subset=["to_act_identifier"])
    out = {}
    for t in ids:
        pot = np.asarray(pots[t], dtype=float)
        if pot.sum() <= 0:
            continue
        id2i = {fid: i for i, fid in enumerate(ids[t])}
        counts = np.zeros(len(pot))
        sub = df[df["to_act_type"] == t] if "to_act_type" in df else df
        for fid in sub["to_act_identifier"]:
            i = id2i.get(fid)
            if i is not None:
                counts[i] += 1
        out[t] = (pot, counts)
    return out


def _pot_decile_tv(tvc, n_bins=10) -> float:
    """Equal-POTENTIAL-MASS bins: partition each type's facilities (sorted by potential) so each bin
    holds ~1/n_bins of the total potential, then TV between visit-share-per-bin and
    potential-share-per-bin (per type, visit-weighted). Target shares are ~uniform (1/n_bins), so
    over-concentration in the high-potential tail crosses bin boundaries (detected, unlike
    equal-count deciles that lump the tail into one bin) and bins stay dense (no per-facility
    sparsity floor). Perfect generative draw -> ~0; argmax over-concentration -> high; alpha=0
    (visits ignore potential) -> high. Headline potential-fit metric."""
    tvs, weights = [], []
    for pot, counts in tvc.values():
        tot = float(pot.sum())
        if tot <= 0 or counts.sum() == 0 or len(pot) < n_bins:
            continue
        order = np.argsort(pot)
        cum = np.cumsum(pot[order]) / tot                       # cumulative mass share (ascending)
        b = np.minimum((cum * n_bins).astype(int), n_bins - 1)  # equal-mass bin index per facility
        P = np.zeros(n_bins); Q = np.zeros(n_bins)
        np.add.at(P, b, pot[order]); np.add.at(Q, b, counts[order])
        P /= P.sum(); Q /= Q.sum()
        tvs.append(0.5 * float(np.abs(P - Q).sum())); weights.append(float(counts.sum()))
    return float(np.average(tvs, weights=weights)) if tvs else float("nan")


def _pot_facility_tv(tvc) -> float:
    """Per-facility TV (visit-share vs potential-share), per type, visit-weighted. DIAGNOSTIC ONLY:
    sparsity-floored (~0.9 at 500 persons) and rewards spread over correct concentration."""
    tvs, weights = [], []
    for pot, counts in tvc.values():
        if counts.sum() == 0:
            continue
        p, q = pot / pot.sum(), counts / counts.sum()
        tvs.append(0.5 * float(np.abs(p - q).sum())); weights.append(float(counts.sum()))
    return float(np.average(tvs, weights=weights)) if tvs else float("nan")


def _pot_spearman(tvc) -> float:
    """Rank correlation of per-facility realized visits vs potential, visit-weighted over types.
    Higher = attractive places get proportionally more visits. (Ties-heavy at small N — a candidate;
    recompute from raw if needed.)"""
    rhos, weights = [], []
    for pot, counts in tvc.values():
        if counts.sum() == 0 or len(pot) < 3:
            continue
        rho = spearmanr(counts, pot).correlation
        if np.isfinite(rho):
            rhos.append(float(rho)); weights.append(float(counts.sum()))
    return float(np.average(rhos, weights=weights)) if rhos else float("nan")


def _zone_pot_fit(world, rdf, free_ids, n_cells=20) -> float:
    """Coarse spatial-grid TV (pooled over types). DIAGNOSTIC ONLY: distance already gets the zone
    right, so it washes out the attractiveness signal."""
    _, coords, pots = world.locations_tuple
    allc = np.vstack([np.asarray(coords[t], float) for t in coords])
    allp = np.concatenate([np.asarray(pots[t], float) for t in pots])
    lo, hi = allc.min(0), allc.max(0)
    xe = np.linspace(lo[0], hi[0], n_cells + 1); ye = np.linspace(lo[1], hi[1], n_cells + 1)
    trueH, _, _ = np.histogram2d(allc[:, 0], allc[:, 1], bins=[xe, ye], weights=allp)
    df = rdf[rdf["unique_leg_id"].isin(free_ids)].dropna(subset=["to_x", "to_y"])
    visH, _, _ = np.histogram2d(df["to_x"].to_numpy(float), df["to_y"].to_numpy(float), bins=[xe, ye])
    p, q = trueH.ravel(), visH.ravel()
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * float(np.abs(p - q).sum())


def _recovery_pct(rdf, gt) -> float:
    free = gt.loc[gt["to_is_free"], ["unique_leg_id", "true_to_identifier"]]
    m = rdf[["unique_leg_id", "to_act_identifier"]].merge(free, on="unique_leg_id", how="inner")
    if not len(m):
        return float("nan")
    ok = m["to_act_identifier"].astype(str) == m["true_to_identifier"].astype(str)
    return 100.0 * float(ok.mean())


# --------------------------------------------------------------------------- #
# one (solver, alpha) cell -> aggregate row + raw per-leg frame
# --------------------------------------------------------------------------- #

def _solve(world, plans, solver, seed, scorer, extra: Optional[dict] = None):
    cls, params = _spec(solver)
    params = {**(params or {}), **(extra or {})} or None
    ctx = run.setup(locations_tuple=world.locations_tuple, solver=cls, rng_seed=seed,
                    parameters=params, scorer=scorer)
    t0 = time.perf_counter()
    rdf, _, valid = run.solve(ctx=ctx, plans_df=plans)
    return ctx, rdf, time.perf_counter() - t0, bool(valid)


def evaluate(world, plans, gt, solver, alpha, beta, seed, *, scored, free_ids, pid_of,
             ref_dists, cal, pot_pool_k) -> dict:
    scorer = Scorer(mode="combined", pot_weight=float(alpha), dist_dev_weight=float(beta),
                    attr_transform=ATTR_TRANSFORM)
    extra: Dict[str, object] = {}
    if solver in ("dp_sample", "dp_sample_capped"):
        extra.update(cal["sample_params"])
    if solver == "dp_sample_capped":
        extra["max_candidates"] = SAMPLE_CAP    # capped variant: bounded candidate pool (fast)
    if solver == "gravity_independent":
        extra["scale"] = cal["scale_cal"]               # calibrated decay for the no-coupling floor
    if solver == "dp_carla_pot" and pot_pool_k:
        extra["pot_pool_k"] = int(pot_pool_k)
    ctx, rdf, rt, valid = _solve(world, plans, solver, seed, scorer, extra or None)

    dev = per_person_dev(rdf, plans, gt, scored)                              # distance part / person
    free = rdf[rdf["unique_leg_id"].isin(free_ids)].copy()
    free["pid"] = free["unique_leg_id"].map(pid_of)
    pot_raw = free.groupby("pid")["to_act_potential"].sum().reindex(dev.index).fillna(0.0)
    pot_log = (free.assign(_l=np.log1p(free["to_act_potential"].to_numpy(float)))
               .groupby("pid")["_l"].sum().reindex(dev.index).fillna(0.0))
    cost = beta * dev - float(alpha) * pot_log                               # objective IS log1p-form

    tvc = _type_visit_counts(world, rdf, free_ids)
    row = {
        "solver": solver, "alpha": float(alpha), "runtime_s": rt, "valid": valid,
        "dist_dev_m": float(np.nanmean(dev)),
        "pot_captured": float(np.nanmean(pot_raw)),
        "combined_cost": float(np.nanmean(cost)),
        "pot_decile_tv": _pot_decile_tv(tvc),
        "pot_fit_tv": _pot_facility_tv(tvc),
        "pot_fit_zone_tv": _zone_pot_fit(world, rdf, free_ids),
        "pot_spearman": _pot_spearman(tvc),
        "dist_w_m": _dist_wasserstein(rdf, free_ids, ref_dists),
        "recovery_pct": _recovery_pct(rdf, gt),
    }
    ach = np.hypot(free["to_x"].to_numpy(float) - free["from_x"].to_numpy(float),
                   free["to_y"].to_numpy(float) - free["from_y"].to_numpy(float))
    raw = pd.DataFrame({
        "solver": solver, "alpha": float(alpha),
        "unique_person_id": free["pid"].to_numpy(),
        "unique_leg_id": free["unique_leg_id"].to_numpy(),
        "to_act_type": free["to_act_type"].to_numpy(),
        "chosen_loc_id": free["to_act_identifier"].to_numpy(),
        "observed_dist_m": free["distance_meters"].to_numpy(float),
        "achieved_dist_m": ach,
        "chosen_potential": free["to_act_potential"].to_numpy(float),
    })
    return {"row": row, "raw": raw}


# --------------------------------------------------------------------------- #
# regimes (GT preserved): work axis + input axis
# --------------------------------------------------------------------------- #

def make_work(plans, gt, mode):
    if mode == "anchored":
        return plans, gt
    if mode == "demoted":
        return S.demote_anchor(plans, gt, anchor="work")
    raise ValueError(mode)


def make_input(plans, gt, mode, world, rng):
    if mode == "true":
        return plans
    if mode == "sampled":
        samples = S.per_mode_distance_samples(world.plans_df, world.ground_truth)
        return S.resample_distances(plans, gt, samples, rng, feasible=True)  # feasibility-handled
    raise ValueError(mode)


def build_combo(world, w, work_mode, input_mode, seed) -> dict:
    rng = np.random.default_rng(seed)
    plans0, gt = make_work(w.plans_df, w.ground_truth, work_mode)
    plans = make_input(plans0, gt, input_mode, world, rng)
    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    ref = plans0[plans0["unique_leg_id"].isin(free_ids)]["distance_meters"].to_numpy(float)
    return {"plans": plans, "gt": gt, "scored": _scored_legs(plans, gt), "free_ids": free_ids,
            "pid_of": plans.set_index("unique_leg_id")["unique_person_id"],
            "ref_dists": ref[np.isfinite(ref)]}


# --------------------------------------------------------------------------- #
# parallel cell worker (globals set by initializer; serial path sets them directly)
# --------------------------------------------------------------------------- #

_W = None       # locations (per worker)
_COMBOS = None  # {(work, input): combo dict}
_CAL = None
_BETA = 1.0
_POTK = 100


def _init_worker(world_name, combos, cal, beta, potk, world=None):
    global _W, _COMBOS, _CAL, _BETA, _POTK
    _W = world if world is not None else _load_locations_only(world_name)
    if isinstance(combos, str):           # path to a pickled combos dict (parallel path; see sweep)
        import pickle
        with open(combos, "rb") as fh:
            combos = pickle.load(fh)
    _COMBOS, _CAL, _BETA, _POTK = combos, cal, beta, potk


def _cell(args):
    work, input_mode, solver, alpha, seed = args
    if work == "demoted" and solver == ORACLE:
        return None  # full-chain dp_full O(n*N^2) blows up; dp_carla_pot is the near-oracle there
    cd = _COMBOS[(work, input_mode)]
    try:
        res = evaluate(_W, cd["plans"], cd["gt"], solver, alpha, _BETA, seed, scored=cd["scored"],
                       free_ids=cd["free_ids"], pid_of=cd["pid_of"], ref_dists=cd["ref_dists"],
                       cal=_CAL, pot_pool_k=_POTK)
        res["row"].update(work=work, input=input_mode)
        res["raw"]["work"] = work; res["raw"]["input"] = input_mode
        return res
    except Exception as e:
        print(f"      [{solver} a={alpha} {work}/{input_mode}] failed: {e!r}", flush=True)
        return {"row": {"solver": solver, "alpha": float(alpha), "valid": False,
                        "work": work, "input": input_mode}, "raw": None}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def sweep(world, world_name, n_persons, seed, out_dir, *, alphas, beta, pot_pool_k, solvers,
          works, inputs, jobs, append=False, refit=False) -> pd.DataFrame:
    w = S.sample_persons(world, n_persons, seed=seed)
    cal = calibrate_cached(world, world_name, w.plans_df, w.ground_truth, seed, n_persons, refit=refit)
    print(f"  [{world_name}] alpha={cal['alpha_cal']:.3f}  scale={cal['scale_cal']:.0f}m "
          f"(combined objective uses {ATTR_TRANSFORM})", flush=True)
    combos = {(wk, ip): build_combo(world, w, wk, ip, seed) for wk in works for ip in inputs}
    tasks = [(wk, ip, s, a, seed) for wk in works for ip in inputs for s in solvers for a in alphas]
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        import pickle
        # Ship `combos` (can be GBs at large N) to workers via a temp file, NOT initargs: Windows
        # spawn pickles initargs through a pipe and DEADLOCKS past a few GB (the large-N hang).
        # Workers load it from disk in _init_worker. Unique name (pid) so parallel sweeps don't clash.
        combos_path = os.path.join(
            out_dir, f"_combos_{world_name}_{'-'.join(works)}_{'-'.join(inputs)}_{os.getpid()}.pkl")
        with open(combos_path, "wb") as fh:
            pickle.dump(combos, fh, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                     initargs=(world_name, combos_path, cal, beta, pot_pool_k)) as ex:
                results = list(ex.map(_cell, tasks))
        finally:
            try:
                os.remove(combos_path)
            except OSError:
                pass
    else:
        _init_worker(world_name, combos, cal, beta, pot_pool_k, world=world)  # reuse loaded world
        results = [_cell(t) for t in tasks]
    print(f"  [{world_name}] solved {len(tasks)} cells in {time.perf_counter()-t0:.1f}s", flush=True)

    rows = [r["row"] for r in results if r is not None]
    raws = [r["raw"] for r in results if r is not None and r.get("raw") is not None]
    df = pd.DataFrame(rows); df["world"] = world_name; df["alpha_cal"] = cal["alpha_cal"]
    raw_df = pd.concat(raws, ignore_index=True) if raws else None
    if raw_df is not None:
        raw_df["world"] = world_name

    csv_path = os.path.join(out_dir, f"alpha_sweep_{world_name}.csv")
    raw_path = os.path.join(out_dir, f"raw_legs_{world_name}.parquet")
    if append and os.path.exists(csv_path):                       # merge into existing (e.g. add dp_full)
        df = _merge_on_keys(pd.read_csv(csv_path), df)
        print(f"  [{world_name}] appended -> {df['solver'].nunique()} solvers in CSV", flush=True)
    df.to_csv(csv_path, index=False)
    if raw_df is not None:
        if append and os.path.exists(raw_path):
            raw_df = _merge_on_keys(pd.read_parquet(raw_path), raw_df)
        raw_df.to_parquet(raw_path, index=False)

    from plot_block_b import add_eff_gap, plot_all  # local import: matplotlib only when plotting
    dfe = add_eff_gap(df)
    dfe.to_csv(os.path.join(out_dir, f"alpha_sweep_{world_name}_eff.csv"), index=False)
    plot_all(dfe, world_name, out_dir, float(df["alpha_cal"].iloc[0]))
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", nargs="+", default=["gauss_hannover"])
    p.add_argument("--persons", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--alphas", type=float, nargs="+", default=None)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--pot-pool-k", type=int, default=100)
    p.add_argument("--jobs", type=int, default=1, help="parallel workers over (work,input,solver,alpha) cells")
    p.add_argument("--work", nargs="+", default=["anchored", "demoted"], choices=["anchored", "demoted"])
    p.add_argument("--input", nargs="+", default=["true", "sampled"], choices=["true", "sampled"])
    p.add_argument("--solvers", nargs="+", default=None,
                   help="explicit solver list (overrides the default set; keep dp_carla_pot as the "
                        "cost-gap baseline when dp_full is excluded).")
    p.add_argument("--no-sampler", action="store_true")
    p.add_argument("--no-refs", action="store_true", help="skip rda / rda_guided flat references")
    p.add_argument("--append", action="store_true",
                   help="merge results into the existing CSV/parquet (replacing matching cells) instead "
                        "of overwriting — e.g. add dp_full later without re-running the others")
    p.add_argument("--refit", action="store_true", help="ignore the calibration cache and refit")
    p.add_argument("--quick", action="store_true", help="tiny run (40 persons, short grid, gauss only)")
    args = p.parse_args(argv)

    persons = 40 if args.quick else args.persons
    alphas = args.alphas if args.alphas is not None else (
        [0.0, 1.0, 5.0] if args.quick
        else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    works = ["anchored"] if args.quick else args.work
    inputs = ["true"] if args.quick else args.input
    if args.solvers:
        solvers = list(args.solvers)
    else:
        solvers = list(ARGMIN)            # _spec resolves registry keys, block_a names, and class baselines
        if not args.no_sampler:
            solvers += MODELS
        if not (args.quick or args.no_refs):
            solvers += FLAT_REFS
    worlds = ["gauss_hannover"] if args.quick else args.worlds

    t_all = time.perf_counter()
    for wn in worlds:
        print(f"world={wn}  persons={persons}  jobs={args.jobs}  works={works}  inputs={inputs}  "
              f"alphas={alphas}", flush=True)
        df = sweep(load(wn), wn, persons, args.seed, args.out, alphas=alphas, beta=args.beta,
                   pot_pool_k=args.pot_pool_k, solvers=solvers, works=works, inputs=inputs, jobs=args.jobs,
                   append=args.append, refit=args.refit)
        print(f"[{wn}] rows={len(df)}", flush=True)
    print(f"\nALL DONE {time.perf_counter()-t_all:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
