"""Survey-realistic evaluation backbone (mirrors how MiD-style HTS data is actually used).

Reality: a (national) household travel survey samples a small random fraction of the
population; per-mode/purpose distance distributions are derived from that sample and
applied to a synthetic population of a (smaller) study region, where secondary activity
locations are then assigned. Structural attractiveness (e.g. a business register) is
available everywhere.

This module provides the survey primitives on top of `synth`:
  - `sample_persons`: the canonical "sample N persons" subset of a world (count + seed,
    deterministic) — used by every benchmark/experiment harness so all solvers/regimes share the
    byte-identical population.
  - `study_window` / `persons_in_window`: carve a study sub-region out of a super-region (spatial).
  - `draw_survey`: random person sample by fraction (the "survey"; the rng-driven sibling of
    `sample_persons`).
  - `per_mode_distance_samples`: empirical free-leg distance distribution per mode.

The eval keeps the resident's ground-truth chain and anchors throughout (so recovery and
gap-to-oracle stay measurable everywhere) and parameterizes realistic difficulty as two
controlled degradations of the *true* inputs the solver sees:
  - distance quality  (`resample_distances`): true -> survey-distribution distances. The
    status-quo argmin input; pairs recovery-collapse against surviving distance-fit.
  - anchor quality (`disturb_anchor` / `demote_anchor`): perturb the work anchor (imperfect
    commuting model) or demote it to a free node (no commuting model — the harder, near-
    single-anchor instance). GT is retained, so both are honest dose-response sweeps and
    neither needs a primary-activity placer.

`transplant_survey_chains` is a footnote: the faithful HTS-donation regime (whole donor chain
on a synthetic home), but it forfeits the resident ground truth and, on two-anchor spans,
collapses into the same input class as `resample_distances` on real anchors — so it is not a
headline track.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def study_window(topo, frac_side: float = 0.5) -> Tuple[float, float]:
    """Centered square study window covering `frac_side` of the box per axis -> (lo, hi)."""
    lo = topo.box * (1.0 - frac_side) / 2.0
    hi = topo.box * (1.0 + frac_side) / 2.0
    return lo, hi


def persons_in_window(plans_df: pd.DataFrame, lo: float, hi: float) -> List[str]:
    """Persons whose home (first leg's known from-anchor) lies in the study window."""
    firsts = (plans_df.sort_values("unique_leg_id")
              .groupby("unique_person_id", sort=False).head(1))
    inside = ((firsts["from_x"] >= lo) & (firsts["from_x"] <= hi)
              & (firsts["from_y"] >= lo) & (firsts["from_y"] <= hi))
    return firsts.loc[inside, "unique_person_id"].tolist()


def _subset(plans_df, gt, pids):
    """Shared core for every person subsample: restrict plans + ground truth to `pids` (and the
    legs they own). `sample_persons` and `draw_survey` are the two front doors onto this."""
    pid_set = set(pids)
    p = plans_df[plans_df["unique_person_id"].isin(pid_set)].copy()
    g = gt[gt["unique_leg_id"].isin(set(p["unique_leg_id"]))].copy()
    return p, g


def sample_persons(world, n: int, *, seed: int):
    """Canonical "sample N persons" primitive: a uniform random draw of `n` person ids from a
    (large/baked) `SyntheticWorld`, returning a new `SyntheticWorld` with only those persons'
    `plans_df`/`ground_truth`. The full-population `locations_tuple` (ids/coords/potentials) is
    kept VERBATIM — subsampling persons must not thin the dense usage/attractiveness field — and
    `meta`'s `n_persons`/`n_legs`/`n_free_legs` are refreshed to the subset.

    Deterministic in `seed`: it builds a fresh `np.random.default_rng(seed)` per call rather than
    advancing a shared rng, so the draw never depends on call order. Every benchmark/experiment run
    that passes the same `(world, n, seed)` therefore gets the byte-identical population — hence the
    SAME persons for every solver and every regime. If `n >= ` the population size, the whole world
    is returned (all persons).

    Relationship to `draw_survey`: they differ ONLY in count-vs-fraction (`n` here, `frac` there)
    and seed-vs-rng (deterministic-in-seed here, shared advancing rng there); both delegate to
    `_subset` and both draw UNIFORMLY over persons, so each preserves the chain-length / mode /
    free-leg-distance marginals in expectation (no stratification). `study_window`/
    `persons_in_window` are a different, spatial selection and are unaffected.

    Caveat: a uniform draw under-represents the long-chain tail in any single sample. Callers
    plotting the scaling / recall tail (runtime or recall vs chain length) should OVERSAMPLE long
    chains explicitly rather than rely on this draw; the default sampler stays unstratified on
    purpose."""
    pids = world.plans_df["unique_person_id"].unique()
    chosen = pids if n >= len(pids) else np.random.default_rng(seed).choice(pids, size=n, replace=False)
    pl, gt = _subset(world.plans_df, world.ground_truth, chosen)
    meta = {**world.meta, "n_persons": int(len(chosen)), "n_legs": len(pl),
            "n_free_legs": int(gt["to_is_free"].sum())}
    return dataclasses.replace(world, plans_df=pl, ground_truth=gt, meta=meta)


def draw_survey(plans_df, gt, frac: float, rng: np.random.Generator):
    """Random person sample of the population -> (survey_plans, survey_gt). The survey-emulation
    sibling of `sample_persons`: it samples a FRACTION of persons with a caller-supplied advancing
    `rng` (mirroring an HTS sampling design) instead of a fixed count with a seed, but routes
    through the same `_subset` core and is likewise uniform over persons (no stratification)."""
    pids = plans_df["unique_person_id"].unique()
    k = max(1, int(round(frac * len(pids))))
    chosen = rng.choice(pids, size=k, replace=False)
    return _subset(plans_df, gt, chosen)


def per_mode_distance_samples(plans_df, gt) -> Dict[Optional[str], np.ndarray]:
    """Empirical free (secondary) leg distance arrays per mode (plus a pooled `None` key),
    to be sampled from when applying the survey to a synthetic population."""
    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    free = plans_df[plans_df["unique_leg_id"].isin(free_ids)]
    out: Dict[Optional[str], np.ndarray] = {None: free["distance_meters"].to_numpy()}
    if "mode" in free.columns:
        for m, grp in free.groupby("mode"):
            out[m] = grp["distance_meters"].to_numpy()
    return out


def decay_from_survey(plans_df, gt, *, by=("mode", "to_act_type")) -> Dict[tuple, float]:
    """Calibrate a `(mode, to_act)` → median free-leg distance (metres) table from the survey,
    usable as the `to_decay` fallback for generative solvers applied to another region. This
    is the survey-transfer analogue of the baked MiD constants: stats come from the sample,
    geometry comes from wherever the model is then applied."""
    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    free = plans_df[plans_df["unique_leg_id"].isin(free_ids)]
    out: Dict[tuple, float] = {}
    for key, g in free.groupby(list(by)):
        out[tuple(key) if isinstance(key, tuple) else (key,)] = float(g["distance_meters"].median())
    return out


def _chain_slack(distances: np.ndarray, direct: float) -> float:
    """eqasim feasibility slack of a segment's leg `distances` given the anchor `direct` distance
    (0 == feasible): no leg may exceed direct + sum(other legs), and direct <= sum(legs)."""
    d = np.asarray(distances, float); tot = float(d.sum())
    delta = float(np.max(d - direct - (tot - d))) if len(d) else 0.0
    return max(delta, direct - tot, 0.0)


def _anchor_segments(plans_df):
    """Yield (row-labels, direct_distance) per anchor-bounded segment (a run from a known `from`
    to the next known `to`) that contains >= 1 leg. Relies on travel-order rows."""
    for _pid, idx in plans_df.groupby("unique_person_id", sort=False).groups.items():
        seg, start = [], None
        for i in idx:
            if not seg:
                start = (plans_df.at[i, "from_x"], plans_df.at[i, "from_y"])
            seg.append(i)
            tox, toy = plans_df.at[i, "to_x"], plans_df.at[i, "to_y"]
            if np.isfinite(tox) and np.isfinite(toy):  # reached an anchor -> segment closes
                direct = float(np.hypot(tox - start[0], toy - start[1])) if np.all(np.isfinite(start)) else 0.0
                yield seg, direct
                seg, start = [], None


def resample_distances(plans_df, gt, samples_by_mode, rng: np.random.Generator,
                       *, feasible: bool = False, max_tries: int = 200) -> pd.DataFrame:
    """Return a copy of `plans_df` with each FREE leg's `distance_meters` replaced by a draw
    from the survey distribution for its mode (pooled fallback). Anchor-leg distances (e.g.
    home->work commutes) are left untouched. This is the status-quo input for argmin solvers
    applied to a synthetic population without its own observed distances.

    With ``feasible=True`` the draw is, per anchor-bounded segment, REJECTION-SAMPLED until the
    chain can actually be realised given the anchor geometry (eqasim Alg. 3 / Hörl & Axhausen):
    each leg <= direct + sum(other legs) and direct <= sum(legs). This is what the real RDA does
    and what the B realism arm needs — independent per-leg draws otherwise hand the argmin
    geometrically infeasible targets, inflating its deviation for reasons that aren't the method.
    If no feasible draw is found in `max_tries`, the lowest-slack draw is kept (and the chain is
    left near-feasible, matching eqasim's "best so far" behaviour).

    Note: keeps each resident's own chain skeleton and ground-truth facilities, swapping only the
    per-leg distance marginals. The fully realistic regime is `transplant_survey_chains`."""
    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    pooled = samples_by_mode.get(None)
    out = plans_df.copy()
    for col in ("distance_meters",):
        out[col] = out[col].astype(float)

    def _draw(i):
        pool = samples_by_mode.get(out.at[i, "mode"] if "mode" in out.columns else None, pooled)
        if pool is None or len(pool) == 0:
            pool = pooled
        return float(rng.choice(pool))

    for i in out.index:
        if out.at[i, "unique_leg_id"] in free_ids:
            out.at[i, "distance_meters"] = _draw(i)

    if feasible:
        for seg, direct in _anchor_segments(out):
            free_here = [i for i in seg if out.at[i, "unique_leg_id"] in free_ids]
            if not free_here:
                continue
            best = None
            for _ in range(max_tries):
                for i in free_here:
                    out.at[i, "distance_meters"] = _draw(i)
                d = out.loc[seg, "distance_meters"].to_numpy(float)
                slack = _chain_slack(d, direct)
                if best is None or slack < best[0]:
                    best = (slack, d.copy())
                if slack <= 1e-6:
                    break
            out.loc[seg, "distance_meters"] = best[1]  # feasible draw, or lowest-slack fallback
    return out


def home_coords(plans_df: pd.DataFrame) -> np.ndarray:
    """(m,2) array of each person's home coordinate = the first leg's known from-anchor.
    Relies on travel-order rows (as produced by `synth.generate_chains`)."""
    firsts = plans_df.groupby("unique_person_id", sort=False).head(1)
    return firsts[["from_x", "from_y"]].to_numpy(dtype=float)


def transplant_survey_chains(
    region_homes: np.ndarray,
    survey_plans: pd.DataFrame,
    *,
    anchor_types: Tuple[str, ...] = ("home", "work"),
    n_persons: Optional[int] = None,
    person_prefix: str = "t",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign WRONG-person plans to a study region — the realistic synthetic-population regime.

    With a real synthetic population we never know the true person living at a home, so each
    synthetic resident is handed a *donor* activity chain sampled from the survey: its activity
    skeleton, free-leg distances and mode, rigidly relocated so the donor's home lands on a freshly
    drawn region home (the whole anchor frame is translated by home_new - home_donor; secondary
    coords stay unknown/NaN, exactly as in any solver input). Because the donor chain never
    occurred at this home, there is NO ground-truth facility for the free legs — so this track is
    scored by %gap-to-oracle and distance-distribution fit, NEVER recovery.

    Returns (plans_df, free_frame); `free_frame` carries `unique_leg_id` + `to_is_free` so the
    existing metric helpers (`_deviation`, `metrics.free_leg_distances`) work unchanged. Donor leg
    rows are consumed in travel order (do not pre-sort by `unique_leg_id` — it is lexicographic)."""
    rng = rng or np.random.default_rng()
    region_homes = np.asarray(region_homes, dtype=float)
    anchors = set(anchor_types)
    donors = [g for _, g in survey_plans.groupby("unique_person_id", sort=False)]
    if region_homes.size == 0 or not donors:
        raise ValueError("need at least one region home and one survey donor")
    n = int(n_persons) if n_persons is not None else len(region_homes)

    rows: List[dict] = []
    free_rows: List[dict] = []
    for i in range(n):
        home_xy = region_homes[int(rng.integers(len(region_homes)))]
        dplan = donors[int(rng.integers(len(donors)))]
        donor_home = dplan.iloc[0][["from_x", "from_y"]].to_numpy(dtype=float)
        tx, ty = home_xy[0] - donor_home[0], home_xy[1] - donor_home[1]

        def _shift(vx, vy):
            # Translate known anchor coords; leave unknown (NaN) secondary coords unknown.
            if vx is None or vy is None or not (np.isfinite(vx) and np.isfinite(vy)):
                return np.nan, np.nan
            return vx + tx, vy + ty

        for k, (_, leg) in enumerate(dplan.iterrows(), start=1):
            lid = f"{person_prefix}{i}-l{k}"
            to_free = str(leg["to_act_type"]) not in anchors
            fx, fy = _shift(leg.get("from_x", np.nan), leg.get("from_y", np.nan))
            ox, oy = _shift(leg.get("to_x", np.nan), leg.get("to_y", np.nan))
            rows.append({
                "unique_person_id": f"{person_prefix}{i}", "unique_leg_id": lid,
                "to_act_type": leg["to_act_type"], "distance_meters": float(leg["distance_meters"]),
                "mode": leg.get("mode"),
                "from_x": fx, "from_y": fy, "to_x": ox, "to_y": oy,
            })
            free_rows.append({"unique_leg_id": lid, "to_is_free": to_free})
    return pd.DataFrame(rows), pd.DataFrame(free_rows)


def _person_leg_order(plans_df: pd.DataFrame):
    """Yield (person_id, [row-labels in travel order]) — relies on synth's travel-order rows."""
    for pid, idx in plans_df.groupby("unique_person_id", sort=False).groups.items():
        yield pid, list(idx)


def disturb_anchor(
    plans_df: pd.DataFrame,
    *,
    anchor: str = "work",
    noise_m: float,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Anchor-quality degradation: jitter the (known, GT) `anchor` location by isotropic Gaussian
    noise of σ=`noise_m` metres — an imperfect commuting/OD model. Ground truth is untouched
    (scoring is against the resident's true facility), so this drives a clean robustness curve:
    secondary placement degradation vs anchor error, with the solver held fixed.

    A node is shared by consecutive legs (leg k's `to` == leg k+1's `from`), so each anchor
    occurrence is moved by ONE delta applied to both sides to keep the chain geometry consistent."""
    rng = rng or np.random.default_rng()
    out = plans_df.copy()
    for col in ("from_x", "from_y", "to_x", "to_y"):
        out[col] = out[col].astype(float)
    for _pid, labels in _person_leg_order(out):
        for r, i in enumerate(labels):
            if str(out.at[i, "to_act_type"]) == anchor and np.isfinite(out.at[i, "to_x"]):
                dx, dy = rng.normal(0.0, noise_m), rng.normal(0.0, noise_m)
                out.at[i, "to_x"] += dx
                out.at[i, "to_y"] += dy
                if r + 1 < len(labels):  # propagate to the next leg's from (same node)
                    j = labels[r + 1]
                    if np.isfinite(out.at[j, "from_x"]):
                        out.at[j, "from_x"] += dx
                        out.at[j, "from_y"] += dy
    return out


def demote_anchor(
    plans_df: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    anchor: str = "work",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Anchor-quality degradation (extreme): demote `anchor` from a KNOWN anchor to a FREE node
    the solver must place — no commuting model at all. Blanks the anchor's coords (its `to`-side
    and the shared `from`-side of the next leg) so it joins the free segment, and flips
    `to_is_free` in a COPY of `gt` so the still-recorded truth scores it. This is the harder,
    near-single-anchor instance (the CARLA 'main not placed' scenario) with GT retained — no
    primary placer required, since we hide a known point rather than synthesize one.

    Returns (plans_df, gt) copies."""
    out = plans_df.copy()
    g = gt.copy()
    work_ids = set(out.loc[out["to_act_type"].astype(str) == anchor, "unique_leg_id"])
    g.loc[g["unique_leg_id"].isin(work_ids), "to_is_free"] = True
    for col in ("from_x", "from_y", "to_x", "to_y"):
        out[col] = out[col].astype(float)
    for _pid, labels in _person_leg_order(out):
        for r, i in enumerate(labels):
            if str(out.at[i, "to_act_type"]) == anchor:
                out.at[i, "to_x"] = np.nan
                out.at[i, "to_y"] = np.nan
                if r + 1 < len(labels):
                    j = labels[r + 1]
                    out.at[j, "from_x"] = np.nan
                    out.at[j, "from_y"] = np.nan
    return out, g
