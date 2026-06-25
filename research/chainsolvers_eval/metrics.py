"""Distribution-fit metrics — the field-standard, ground-truth-free evaluation.

Where %gap/recovery need the true facility (only available in a synthetic world), the
*distance-distribution fit* asks the operational question used in practice (eqasim, Hörl &
Axhausen 2023; Ding & Balać hEART 2026): does the placed population reproduce the survey's
free-leg distance distribution, per mode / purpose, up to ~the 95th percentile? This runs on
a synthetic solution *or* on real survey output, so it bridges the synthetic benchmark to
real MiD data.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max abs CDF difference), no SciPy needed."""
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def distribution_fit(realized: np.ndarray, reference: np.ndarray, *,
                     clip_q: float = 0.95,
                     qs: Sequence[float] = (0.25, 0.5, 0.75, 0.9)) -> Dict[str, float]:
    """Compare a `realized` distance distribution to a `reference` (survey) one. Both are
    clipped at the `clip_q` percentile of the reference (extreme outliers aren't modelled;
    the literature reports fit up to ~p95). Returns KS, the mean/median relative error, and
    absolute quantile errors (metres)."""
    realized = np.asarray(realized, dtype=float)
    reference = np.asarray(reference, dtype=float)
    realized = realized[np.isfinite(realized)]
    reference = reference[np.isfinite(reference)]
    if realized.size == 0 or reference.size == 0:
        return {"ks": float("nan"), "n_realized": int(realized.size), "n_reference": int(reference.size)}
    hi = float(np.quantile(reference, clip_q))
    r = realized[realized <= hi]; f = reference[reference <= hi]
    out: Dict[str, float] = {
        "ks": ks_statistic(r, f),
        "n_realized": int(realized.size), "n_reference": int(reference.size),
        "mean_rel_err": float(abs(r.mean() - f.mean()) / f.mean()) if f.mean() else float("nan"),
        "median_rel_err": (float(abs(np.median(r) - np.median(f)) / np.median(f))
                           if np.median(f) else float("nan")),
    }
    for q in qs:
        out[f"q{int(q * 100)}_abs_err_m"] = float(abs(np.quantile(r, q) - np.quantile(f, q)))
    return out


def grouped_distribution_fit(realized_by: Dict[object, np.ndarray],
                             reference_by: Dict[object, np.ndarray],
                             **kw) -> pd.DataFrame:
    """`distribution_fit` per key (e.g. per mode or per purpose); rows for shared keys."""
    rows = []
    for k in sorted(set(realized_by) & set(reference_by), key=str):
        rows.append({"group": k, **distribution_fit(realized_by[k], reference_by[k], **kw)})
    return pd.DataFrame(rows)


def free_leg_distances(result_df: pd.DataFrame, ground_truth: pd.DataFrame, *,
                       by: Optional[str] = None) -> Dict[object, np.ndarray] | np.ndarray:
    """Realized (achieved) straight-line distances of the placed FREE legs in a solution,
    computed from the result coordinates. With `by` (e.g. 'mode' or 'to_act_type') returns a
    dict of arrays keyed by that column; otherwise a single pooled array."""
    free_ids = set(ground_truth.loc[ground_truth["to_is_free"], "unique_leg_id"])
    df = result_df[result_df["unique_leg_id"].isin(free_ids)].copy()
    d = np.hypot(df["to_x"].to_numpy(float) - df["from_x"].to_numpy(float),
                 df["to_y"].to_numpy(float) - df["from_y"].to_numpy(float))
    df = df.assign(_d=d)
    if by is None:
        return df["_d"].to_numpy()
    return {k: g["_d"].to_numpy() for k, g in df.groupby(by)}
