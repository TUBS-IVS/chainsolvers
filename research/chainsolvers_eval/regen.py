"""Counterfactual chain regeneration on **baked** worlds.

A baked snapshot (``worlds.save_world``) round-trips everything needed to *solve* and *score*
a world — topology, plans, ground truth, anchors — but it does **not** store the chain-generating
DGP params (``templates``, the per-(mode,from→to) decay tables, ``decay_inflate``, ``mode_split``,
the heavy-tail knobs, …). Those live only in the builder source constants (``synth.CITY_PRESETS``,
``synth.URBAN_BENCH``/``RURAL_BENCH``, ``osm.MID_INFLATE_OSM`` + the shared ``MID_*`` tables).

This module is the missing plumbing: given a **reloaded** ``SyntheticWorld`` and the canonical bake
recipe name, it re-exposes that DGP and regenerates the activity chains on the *same* (reloaded)
topology — optionally under a shocked structural-attractiveness field ``sizes`` (the prognosis
counterfactual). It reads only; it changes nothing about world creation, the bake process, or any
baked artifact, and re-uses the exact same source objects the builders use (no duplicated data).

Recipe names match ``scripts/bake_worlds.py`` RECIPES:

    gauss_hannover  -> single-population DGP (synth.city_world("hannover"))
    osm_hannover    -> single-population DGP (osm.hannover_osm_world)
    two_zone        -> two-population DGP (synth.two_zone_world(heavy_tail=True))

Reproducibility rule (the reorder caveat): ``worlds.load_world`` rebuilds the topology in
type-grouped file order, which differs from the original construction order, so the RNG stream
maps onto different facilities. Therefore regenerate **both** the baseline (``sizes=None`` =>
``topo.sizes``) and the counterfactual through this module on the *same* reloaded topology — never
compare a regenerated counterfactual against the baked baseline plans (they will not be
byte-identical even at ``boost=1``). See ``block_*``/prognosis usage.

    from chainsolvers_eval.worlds import load_world
    from chainsolvers_eval.regen import regenerate_world
    w = load_world("research/data/worlds/gauss_hannover")
    base = regenerate_world(w, "gauss_hannover", n_persons=2000, rng=np.random.default_rng(1))
    shocked = w.topology.sizes.copy(); shocked[district_mask] *= 6.0
    cf   = regenerate_world(w, "gauss_hannover", n_persons=2000, sizes=shocked,
                            rng=np.random.default_rng(1))
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# The keyword params of `synth.world_from_topology`/`generate_chains` that drive chain generation
# (everything that is NOT topology construction). Used to slice a `generate_world` preset (e.g.
# CITY_PRESETS["hannover"]) down to its chain-side bundle. `type_aliases` is intentionally absent:
# it is a *topology* mutation (add_type_alias) that is already baked into the facilities table and
# rebuilt by load_world, so the reloaded topology already carries the alias type.
CHAIN_PARAM_KEYS = frozenset({
    "gravity_scale", "distance_noise", "assign_modes", "mode_decay", "pair_decay", "to_decay",
    "mode_split", "decay_inflate", "tail_frac", "tail_scale", "car_ownership", "bike_ownership",
    "anchors", "templates",
})

# Which baked worlds are single-population (one world_from_topology call) vs the two-zone DGP.
# `heavy_tail` mirrors scripts/bake_worlds.py RECIPES["two_zone"]["kwargs"] (the canonical bake).
RECIPES: Dict[str, dict] = {
    "gauss_hannover": {"kind": "single"},
    "osm_hannover":   {"kind": "single"},
    "two_zone":       {"kind": "two_zone", "heavy_tail": True},
}


def chain_params_for(recipe: str) -> dict:
    """Return the chain-generating DGP bundle for a single-population baked `recipe`, sourced
    directly from the builder's own constants (so it cannot drift from the bake). Raises for
    `two_zone`, whose DGP is two zone-specific bundles — use `regenerate_world` for that."""
    if recipe == "gauss_hannover":
        from .synth import CITY_PRESETS
        # The gauss preset IS a generate_world kwargs bundle; the chain side is exactly its
        # CHAIN_PARAM_KEYS subset (topology keys n_locations/density/n_clusters/spread/
        # type_prevalence and the topology-mutating type_aliases are dropped). Zero drift.
        return {k: v for k, v in CITY_PRESETS["hannover"].items() if k in CHAIN_PARAM_KEYS}
    if recipe == "osm_hannover":
        # osm.hannover_osm_world assembles this inline from the shared MiD tables + MID_INFLATE_OSM;
        # we reference the same named objects so the data is single-sourced.
        from .osm import MID_INFLATE_OSM
        from .synth import (MID_MODE_DECAY_M, MID_PAIR_DECAY_M, MID_TO_DECAY_M, MID_TOUR_MODE_SPLIT,
                            MID_URBAN_TEMPLATES)
        return dict(templates=MID_URBAN_TEMPLATES, pair_decay=MID_PAIR_DECAY_M,
                    to_decay=MID_TO_DECAY_M, mode_decay=MID_MODE_DECAY_M,
                    mode_split=MID_TOUR_MODE_SPLIT, decay_inflate=dict(MID_INFLATE_OSM))
    if recipe == "two_zone":
        raise ValueError(
            "two_zone is a two-population DGP (urban core + rural ring with distinct bundles); "
            "there is no single chain_params bundle. Call regenerate_world(world, 'two_zone', ...).")
    raise KeyError(f"unknown recipe {recipe!r}; have {sorted(RECIPES)}")


def two_zone_bundles(heavy_tail: bool = True):
    """Return ``(urban_params, rural_params)`` exactly as ``synth.two_zone_world`` assembles them
    (URBAN_BENCH/RURAL_BENCH, plus the heavy-tail kernel knobs when ``heavy_tail``). Mirrors the
    builder's selection using its own named constants."""
    from .synth import (HEAVY_TAIL_RURAL_INFLATE, HEAVY_TAIL_URBAN_INFLATE, RURAL_BENCH,
                        TWO_ZONE_TAIL_FRAC_RURAL, TWO_ZONE_TAIL_FRAC_URBAN, TWO_ZONE_TAIL_SCALE,
                        URBAN_BENCH)
    up = dict(URBAN_BENCH)
    rp = dict(RURAL_BENCH)
    if heavy_tail:
        up.update(tail_frac=TWO_ZONE_TAIL_FRAC_URBAN, tail_scale=TWO_ZONE_TAIL_SCALE,
                  decay_inflate=HEAVY_TAIL_URBAN_INFLATE)
        rp.update(tail_frac=TWO_ZONE_TAIL_FRAC_RURAL, tail_scale=TWO_ZONE_TAIL_SCALE,
                  decay_inflate=HEAVY_TAIL_RURAL_INFLATE)
    return up, rp


def _world_from_chains(topo, plan_rows, gt_rows, visits, anchor_types, base_meta: dict):
    """Assemble a SyntheticWorld from generated chains on `topo` (same shape as the builders'
    tail), carrying `base_meta` forward with refreshed counts."""
    from .synth import SyntheticWorld
    ids_d, coords_d, pots_d = {}, {}, {}
    for t in topo.types:
        idx = topo.type_locs[t]
        ids_d[t] = topo.loc_ids[idx]
        coords_d[t] = topo.coords[idx].astype(float)
        pots_d[t] = visits[idx, topo.type_idx[t]].astype(float)
    meta = dict(base_meta)
    meta["n_legs"] = len(plan_rows)
    meta["n_free_legs"] = int(sum(r["to_is_free"] for r in gt_rows))
    return SyntheticWorld(
        locations_tuple=(ids_d, coords_d, pots_d),
        plans_df=pd.DataFrame(plan_rows), ground_truth=pd.DataFrame(gt_rows),
        anchor_types=tuple(anchor_types), topology=topo, meta=meta,
    )


def _regenerate_two_zone(world, n_persons: int, *, sizes, rng, heavy_tail: bool,
                         distance_noise: float, rural_share: Optional[float]):
    """Reproduce ``synth.two_zone_world``'s two-population DGP on the reloaded topology: split the
    home pool at ``meta['n_core_fac']`` (core homes vs ring homes), draw urban persons with the
    urban bundle and rural persons with the rural bundle, then merge. The urban/rural split defaults
    to the baked ratio (``meta['n_urban']`` : ``meta['n_rural']``) so the regenerated mix matches
    the baked one; pass ``rural_share`` to override."""
    from .synth import DEFAULT_ANCHORS, generate_chains
    rng = rng or np.random.default_rng()
    topo = world.topology
    meta = world.meta
    if "n_core_fac" not in meta:
        raise ValueError("world.meta lacks 'n_core_fac'; this is not a two_zone snapshot.")
    n_core = int(meta["n_core_fac"])
    # Core/ring home split by CONSTRUCTION order, recovered from the loc id. `merge_topologies`
    # numbers facilities "loc{i}" with the core block first (i < n_core_fac), the ring after, and
    # these ids survive save/load. The builder splits on the topology's *positional* index
    # (`home < n_core`), which is valid only on the freshly-merged topo: `load_world` rebuilds the
    # topology in TYPE-GROUPED order, so positional index no longer encodes core-vs-ring and the
    # index test silently mis-splits (it labels the first n_core homes core regardless of zone).
    # Splitting on the stable loc-id integer reproduces the builder's split exactly on the reload.
    try:
        locnum = np.array([int(str(s)[3:]) for s in topo.loc_ids], dtype=np.int64)
    except ValueError as e:
        raise ValueError(
            "two_zone regen needs construction-ordered 'loc<i>' ids to recover the core/ring home "
            f"split on the reloaded topology, but a loc id is not 'loc<int>' ({e}).") from e
    home = topo.type_locs["home"]
    is_core = locnum[home] < n_core
    core_home, ring_home = home[is_core], home[~is_core]

    if rural_share is None:
        nub, nrb = int(meta.get("n_urban", 0)), int(meta.get("n_rural", 0))
        rural_share = (nrb / (nub + nrb)) if (nub + nrb) else 0.3
    n_rural = int(round(rural_share * n_persons))
    n_urban = n_persons - n_rural

    up, rp = two_zone_bundles(heavy_tail)
    rows_u, gt_u, vis_u = generate_chains(topo, n_urban, sizes=sizes, home_pool=core_home,
                                          person_prefix="u", distance_noise=distance_noise,
                                          rng=rng, **up)
    rows_r, gt_r, vis_r = generate_chains(topo, n_rural, sizes=sizes, home_pool=ring_home,
                                          person_prefix="r", distance_noise=distance_noise,
                                          rng=rng, **rp)
    visits = vis_u + vis_r
    base_meta = dict(meta)
    base_meta.update(n_persons=n_persons, n_urban=n_urban, n_rural=n_rural)
    return _world_from_chains(topo, rows_u + rows_r, gt_u + gt_r, visits,
                              DEFAULT_ANCHORS, base_meta)


def regenerate_world(world, recipe: str, n_persons: int, *, sizes=None,
                     rng: Optional[np.random.Generator] = None, distance_noise: float = 0.0,
                     heavy_tail: Optional[bool] = None, rural_share: Optional[float] = None):
    """Regenerate the activity chains of a **reloaded** baked `world` on its own topology, under the
    structural attractiveness `sizes` (``None`` => ``topo.sizes``, the baseline; pass a shocked
    vector indexed by topology location for the counterfactual).

    `recipe` is the canonical bake name (see ``RECIPES``). Returns a fresh ``SyntheticWorld`` that
    shares the same topology/facility geometry but has newly generated ``plans_df``,
    ``ground_truth`` and usage-derived potentials. `distance_noise` matches the bakes' default (0).

    For prognosis: call this for the baseline (``sizes=None``) **and** each shocked `sizes`, with
    the same `n_persons`, on the same reloaded `world` — never mix a regenerated counterfactual with
    the baked baseline (the reorder caveat in the module docstring)."""
    if world.topology is None:
        raise ValueError("world.topology is None; load the snapshot via worlds.load_world, which "
                         "rebuilds the topology from the facilities table.")
    if recipe not in RECIPES:
        raise KeyError(f"unknown recipe {recipe!r}; have {sorted(RECIPES)}")

    if RECIPES[recipe]["kind"] == "two_zone":
        ht = RECIPES[recipe]["heavy_tail"] if heavy_tail is None else heavy_tail
        return _regenerate_two_zone(world, n_persons, sizes=sizes, rng=rng, heavy_tail=ht,
                                    distance_noise=distance_noise, rural_share=rural_share)

    from .synth import world_from_topology
    params = chain_params_for(recipe)
    # distance_noise is handled explicitly (it is a runtime knob, not part of the static bundle);
    # the single-population bundles never include it, so there is no kwargs collision.
    return world_from_topology(world.topology, n_persons, sizes=sizes, rng=rng,
                               distance_noise=distance_noise, **params)
