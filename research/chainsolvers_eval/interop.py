"""Interop with external location-assignment tools (e.g. eqasim) without vendoring them.

Export our world to standard CSVs, run the external tool *out of process* (so its
license stays separate — running a tool and reading its output is not a derivative work),
then read its assignments back into a `result_df` the comparison harness understands.

Caveat on fairness: a full external run (e.g. eqasim) typically *samples* its own target
distances from fitted distributions, so it compares at the **distributional** level (same
facilities + chains), not as a clean same-distances per-instance solver swap. For the
latter, use our in-library reimplementations (which hold the given distances fixed).

Contract for the external tool's output (`assignments.csv`):
  - one row per *free* (secondary) leg, keyed by `unique_leg_id`,
  - columns `to_x`, `to_y` (chosen facility coordinates), optional `to_act_identifier`.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from chainsolvers.types import PlanColumns


def export_for_external(locations_tuple, plans_df, out_dir: str,
                        ground_truth: Optional[pd.DataFrame] = None) -> dict:
    """Write facilities.csv (id, activity, x, y, potential — one row per (location, type))
    and plans.csv (the solver input) for an external tool. Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    ids, coords, pots = locations_tuple
    rows = []
    for t in ids:
        for i in range(len(ids[t])):
            rows.append({"id": ids[t][i], "activity": t,
                         "x": float(coords[t][i][0]), "y": float(coords[t][i][1]),
                         "potential": float(pots[t][i])})
    fac_path = os.path.join(out_dir, "facilities.csv")
    pd.DataFrame(rows).to_csv(fac_path, index=False)

    plans_path = os.path.join(out_dir, "plans.csv")
    plans_df.to_csv(plans_path, index=False)

    paths = {"facilities": fac_path, "plans": plans_path}
    if ground_truth is not None:
        gt_path = os.path.join(out_dir, "ground_truth.csv")
        ground_truth.to_csv(gt_path, index=False)
        paths["ground_truth"] = gt_path
    return paths


def result_df_from_external(plans_df, assignments, cols: PlanColumns = PlanColumns()) -> pd.DataFrame:
    """Reconstruct a chain-consistent `result_df` from an external tool's free-leg
    assignments. `assignments` is a DataFrame keyed by `unique_leg_id` with `to_x`,`to_y`
    (+ optional `to_act_identifier`). Anchor legs keep their known coordinates; each free
    leg's chosen location fills its `to_*` and the next leg's `from_*`.
    """
    a = assignments.set_index(cols.unique_leg_id)
    has_id = cols.to_act_identifier in assignments.columns
    out_rows = []
    for _, grp in plans_df.groupby(cols.person_id, sort=False):
        prev = (float(grp.iloc[0][cols.from_x]), float(grp.iloc[0][cols.from_y]))  # home start
        for _, leg in grp.iterrows():
            lid = leg[cols.unique_leg_id]
            if lid in a.index:  # free leg -> external assignment
                tox, toy = float(a.loc[lid, cols.to_x]), float(a.loc[lid, cols.to_y])
                tid = a.loc[lid, cols.to_act_identifier] if has_id else leg.get(cols.to_act_identifier)
            else:               # anchor leg -> known location from plans
                tox, toy = float(leg[cols.to_x]), float(leg[cols.to_y])
                tid = leg.get(cols.to_act_identifier)
            out_rows.append({
                cols.person_id: leg[cols.person_id],
                cols.unique_leg_id: lid,
                cols.to_act_type: leg[cols.to_act_type],
                cols.leg_distance_m: leg[cols.leg_distance_m],
                cols.from_x: prev[0], cols.from_y: prev[1],
                cols.to_x: tox, cols.to_y: toy,
                cols.to_act_identifier: tid,
            })
            prev = (tox, toy)
    return pd.DataFrame(out_rows)
