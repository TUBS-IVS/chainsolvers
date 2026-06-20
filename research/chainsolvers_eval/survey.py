"""Survey-realistic evaluation backbone (mirrors how MiD-style HTS data is actually used).

Reality: a (national) household travel survey samples a small random fraction of the
population; per-mode/purpose distance distributions are derived from that sample and
applied to a synthetic population of a (smaller) study region, where secondary activity
locations are then assigned. Structural attractiveness (e.g. a business register) is
available everywhere.

This module provides the survey primitives on top of `synth`:
  - `study_window` / `persons_in_window`: carve a study sub-region out of a super-region.
  - `draw_survey`: random person sample (the "survey").
  - `per_mode_distance_samples`: empirical free-leg distance distribution per mode.
  - `resample_distances`: replace a target population's free-leg distances with draws from
    the survey distribution — the status-quo input for distance-matching solvers applied to
    a synthetic population that has no observed distances of its own.
"""

from __future__ import annotations

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
    pid_set = set(pids)
    p = plans_df[plans_df["unique_person_id"].isin(pid_set)].copy()
    g = gt[gt["unique_leg_id"].isin(set(p["unique_leg_id"]))].copy()
    return p, g


def draw_survey(plans_df, gt, frac: float, rng: np.random.Generator):
    """Random person sample of the population -> (survey_plans, survey_gt)."""
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


def resample_distances(plans_df, gt, samples_by_mode, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of `plans_df` with each FREE leg's `distance_meters` replaced by a draw
    from the survey distribution for its mode (pooled fallback). Anchor-leg distances (e.g.
    home->work commutes) are left untouched. This is the status-quo input for argmin solvers
    applied to a synthetic population without its own observed distances."""
    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    pooled = samples_by_mode.get(None)
    out = plans_df.copy()
    d = out["distance_meters"].to_numpy(dtype=float).copy()
    modes = out["mode"].to_numpy() if "mode" in out.columns else np.array([None] * len(out))
    leg_ids = out["unique_leg_id"].to_numpy()
    for i in range(len(out)):
        if leg_ids[i] in free_ids:
            pool = samples_by_mode.get(modes[i], pooled)
            if pool is None or len(pool) == 0:
                pool = pooled
            d[i] = float(rng.choice(pool))
    out["distance_meters"] = d
    return out
