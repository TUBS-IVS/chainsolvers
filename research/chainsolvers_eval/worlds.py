"""Persist a built `synth.SyntheticWorld` to disk as a fixed, reloadable snapshot.

A baked world is three parquet tables + a JSON sidecar in one directory::

    <dir>/facilities.parquet   one row per (location, type): loc_id, type, x, y, size, potential
    <dir>/plans.parquet        the solver input (plans_df)
    <dir>/ground_truth.parquet the true facility per leg (for the recovery / %gap metrics)
    <dir>/meta.json            world.meta + anchor_types (+ box for topology reconstruction)

`save_world`/`load_world` round-trip everything needed to *solve* (locations_tuple, plans_df) and
*score* (ground_truth), plus enough to re-render the viz (the topology is rebuilt from the
facilities table: per-location coords, latent `size`, and per-type membership). Parquet keeps the
full-population tables (millions of legs) fast to load and compact; the files are large local
artifacts (research/data is gitignored) — the committed *recipe* (research/scripts/bake_worlds.py)
is what makes them regenerable.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .synth import SyntheticWorld, Topology


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def save_world(world: SyntheticWorld, out_dir: str) -> str:
    """Write `world` to `out_dir` as facilities/plans/ground_truth parquet + meta.json.
    Returns `out_dir`. The latent `size` column is filled from `world.topology` when present
    (NaN otherwise — solving/scoring never need it, only the viz panel 1 does)."""
    os.makedirs(out_dir, exist_ok=True)
    ids, coords, pots = world.locations_tuple
    types = list(world.meta.get("types") or ids.keys())

    lid2size = {}
    if world.topology is not None:
        lid2size = {str(l): float(s) for l, s in
                    zip(world.topology.loc_ids, np.asarray(world.topology.sizes, float))}

    fac = []
    for t in types:
        idt, cot, pot = ids[t], np.asarray(coords[t], float), np.asarray(pots[t], float)
        for i in range(len(idt)):
            lid = str(idt[i])
            fac.append({"loc_id": lid, "type": t, "x": float(cot[i][0]), "y": float(cot[i][1]),
                        "size": lid2size.get(lid, np.nan), "potential": float(pot[i])})
    pd.DataFrame(fac, columns=["loc_id", "type", "x", "y", "size", "potential"]) \
        .to_parquet(os.path.join(out_dir, "facilities.parquet"), index=False)
    world.plans_df.to_parquet(os.path.join(out_dir, "plans.parquet"), index=False)
    world.ground_truth.to_parquet(os.path.join(out_dir, "ground_truth.parquet"), index=False)

    meta = dict(world.meta)
    meta["types"] = types
    meta["anchor_types"] = list(world.anchor_types)
    if world.topology is not None:
        meta["box"] = float(world.topology.box)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return out_dir


def load_world(out_dir: str) -> SyntheticWorld:
    """Reload a `save_world` snapshot into a `SyntheticWorld` (topology rebuilt from facilities)."""
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    anchor_types = tuple(meta.get("anchor_types", []))
    fac = pd.read_parquet(os.path.join(out_dir, "facilities.parquet"))
    plans_df = pd.read_parquet(os.path.join(out_dir, "plans.parquet"))
    ground_truth = pd.read_parquet(os.path.join(out_dir, "ground_truth.parquet"))

    types = list(meta.get("types") or fac["type"].drop_duplicates().tolist())
    # locations_tuple: per type, in stored file order (matches the original within-type order).
    ids_d, coords_d, pots_d = {}, {}, {}
    for t in types:
        sub = fac[fac["type"] == t]
        ids_d[t] = sub["loc_id"].to_numpy(dtype=object)
        coords_d[t] = np.column_stack([sub["x"].to_numpy(float), sub["y"].to_numpy(float)])
        pots_d[t] = sub["potential"].to_numpy(float)

    # topology: one entry per unique location (first appearance), with latent size + type membership.
    uniq = fac.drop_duplicates("loc_id")
    loc_ids = uniq["loc_id"].to_numpy(dtype=object)
    coords = np.column_stack([uniq["x"].to_numpy(float), uniq["y"].to_numpy(float)])
    sizes = uniq["size"].to_numpy(float)
    lid2idx = {l: i for i, l in enumerate(loc_ids)}
    type_idx = {t: j for j, t in enumerate(types)}
    type_locs = {t: np.array([lid2idx[l] for l in fac.loc[fac["type"] == t, "loc_id"]], dtype=int)
                 for t in types}
    box = float(meta.get("box", float(coords.max()) if len(coords) else 0.0))
    topo = Topology(coords, sizes, loc_ids, types, type_idx, type_locs, box)

    return SyntheticWorld(
        locations_tuple=(ids_d, coords_d, pots_d), plans_df=plans_df,
        ground_truth=ground_truth, anchor_types=anchor_types, topology=topo, meta=meta,
    )
