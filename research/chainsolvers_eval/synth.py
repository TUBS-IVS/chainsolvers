"""Synthetic world generator for testing, evaluation, and prognosis experiments.

Builds a self-consistent ground-truth scenario:

1. A *topology* (`build_topology`) of `n_locations` real points (clustered), each
   offering one or more activity types (multi-type facilities) and carrying a latent
   *structural attractiveness* (`sizes`) — the exogenous driver (think: employment).
2. A population of activity *chains* (`generate_chains`) placed on those points via a
   gravity rule P(loc) ∝ size·exp(-d/scale): prefer near + attractive, starting/ending
   at the person's home, through anchor (fixed/known) and secondary (to-place) activities.
3. *Potentials* are derived from the chains (per-(location,type) visit counts) — an
   *outcome* of the simulation. Structural `sizes` drive choice; visits are the load.

`generate_world` composes these and exposes potentials = visits (usage). For prognosis
experiments use `build_topology` once, then `generate_chains` twice with different
`sizes` (baseline vs shocked), and `topology_locations_tuple` to hand a model the
*structural* attractiveness it is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_TYPE_PREVALENCE: Dict[str, float] = {
    "home": 0.55, "work": 0.25, "shop": 0.20, "leisure": 0.15,
    "education": 0.06, "other": 0.12,
}
DEFAULT_ANCHORS: Tuple[str, ...] = ("home", "work")

# --- MiD 2017 reference values -----------------------------------------------------------
# "Mobilität in Deutschland 2017" (BMVI / infas / DLR / IVT / infas 360), national, trip
# level. See research/synthetic_world_calibration.md for the full derivation + sources.
#   3.1 trips/person/day; mean trip length by main mode (km): foot 1.1, bike 3.3,
#   car ~11, public transport ~7.5. Modal split (trips): foot 22 %, bike 11 %,
#   car (driver+passenger) 57 %, public transport 10 %. Trip purposes (share of trips):
#   work 21, business 13, shopping 16, leisure 28, education 9, accompany 13.
# `mode_decay[m]` is the gravity decay length of P(loc) ∝ size·exp(-d/scale); it maps
# directly to the mean trip length of mode `m` (an exponential's mean is its scale).
DEFAULT_MODE_DECAY: Dict[str, float] = {"walk": 1100.0, "bike": 3300.0, "car": 11000.0, "pt": 7500.0}
# Urban (intra-city) decay: in-town car/PT trips are far shorter than the national mean,
# which is inflated by inter-urban travel. Hannover-city preset uses these.
URBAN_MODE_DECAY: Dict[str, float] = {"walk": 1000.0, "bike": 3300.0, "car": 7000.0, "pt": 6000.0}
DEFAULT_TEMPLATES: List[Tuple[Tuple[str, ...], float]] = [
    (("home", "work", "home"), 3.0),
    (("home", "work", "shop", "home"), 2.0),
    (("home", "shop", "home"), 2.0),
    (("home", "work", "shop", "leisure", "home"), 1.5),
    (("home", "shop", "leisure", "home"), 1.0),
    (("home", "education", "home"), 0.6),
    (("home", "work", "leisure", "shop", "other", "home"), 0.7),
]


@dataclass
class Topology:
    coords: np.ndarray                  # (n,2)
    sizes: np.ndarray                   # (n,) latent structural attractiveness
    loc_ids: np.ndarray                 # (n,) object ids
    types: List[str]
    type_idx: Dict[str, int]
    type_locs: Dict[str, np.ndarray]    # type -> indices of locations offering it
    box: float


@dataclass
class SyntheticWorld:
    locations_tuple: Tuple[dict, dict, dict]
    plans_df: pd.DataFrame
    ground_truth: pd.DataFrame
    anchor_types: Tuple[str, ...]
    topology: Optional[Topology] = None
    meta: dict = field(default_factory=dict)


def _weighted_choice(rng: np.random.Generator, w: np.ndarray) -> int:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return int(rng.integers(0, len(w)))
    return int(rng.choice(len(w), p=w / s))


def build_topology(
    *,
    n_locations: int = 1000,
    box: Optional[float] = None,
    density_per_km2: Optional[float] = None,
    n_clusters: int = 8,
    type_prevalence: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Topology:
    rng = rng or np.random.default_rng()
    # Box sizing (when not given explicitly):
    #  - `density_per_km2`: pin facility density to a real value -> box = sqrt(n/density).
    #    A real German city carries ~40-50 activity facilities/km2 (Hannover: ~9500 over
    #    204 km2; see research/synthetic_world_calibration.md). The legacy default below
    #    is ~2.5/km2 -- a deliberately sparse abstraction, ~18x coarser than reality.
    #  - else legacy auto-size: ~constant density as n grows (20 km box at 1000 locations).
    if box is None and density_per_km2 is not None:
        box = 1000.0 * float(np.sqrt(n_locations / float(density_per_km2)))
    if box is None:
        box = 20000.0 * float(np.sqrt(n_locations / 1000.0))
    type_prevalence = dict(type_prevalence or DEFAULT_TYPE_PREVALENCE)
    types = list(type_prevalence.keys())
    type_idx = {t: j for j, t in enumerate(types)}

    # Keep cluster centres off the border so the Gaussian blobs can spread freely
    # without piling points up against a hard boundary (no clipping -> no edge artefact).
    margin = box * 0.12
    centers = rng.uniform(margin, box - margin, size=(n_clusters, 2))
    which = rng.integers(0, n_clusters, size=n_locations)
    spread = box / (n_clusters * 1.5)
    coords = centers[which] + rng.normal(0, spread, size=(n_locations, 2))
    sizes = np.exp(rng.normal(0, 0.7, size=n_locations))

    offers = {t: rng.random(n_locations) < type_prevalence[t] for t in types}
    none_mask = ~np.any(np.column_stack([offers[t] for t in types]), axis=1)
    offers["home"][none_mask] = True
    for t in types:
        if not offers[t].any():
            offers[t][int(rng.integers(0, n_locations))] = True

    loc_ids = np.array([f"loc{i}" for i in range(n_locations)], dtype=object)
    type_locs = {t: np.flatnonzero(offers[t]) for t in types}
    return Topology(coords, sizes, loc_ids, types, type_idx, type_locs, box)


def topology_locations_tuple(topo: Topology, values: np.ndarray) -> Tuple[dict, dict, dict]:
    """Build a (ids, coords, potentials) payload from a per-location `values` vector
    (e.g. structural sizes, possibly shocked) — the attractiveness a model may see."""
    ids_d, coords_d, pots_d = {}, {}, {}
    for t in topo.types:
        idx = topo.type_locs[t]
        ids_d[t] = topo.loc_ids[idx]
        coords_d[t] = topo.coords[idx].astype(float)
        pots_d[t] = np.asarray(values, dtype=float)[idx]
    return ids_d, coords_d, pots_d


def generate_chains(
    topo: Topology,
    n_persons: int,
    *,
    sizes: Optional[np.ndarray] = None,
    gravity_scale: float = 4000.0,
    distance_noise: float = 0.0,
    assign_modes: bool = True,
    mode_decay: Optional[Dict[str, float]] = None,
    car_ownership: float = 0.6,
    bike_ownership: float = 0.5,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    templates: Optional[List[Tuple[Tuple[str, ...], float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[dict], List[dict], np.ndarray]:
    """Generate activity chains on `topo` using attractiveness `sizes` (defaults to
    topo.sizes). Returns (plan_rows, ground_truth_rows, visits[(n_loc, n_type)])."""
    rng = rng or np.random.default_rng()
    sizes = topo.sizes if sizes is None else np.asarray(sizes, dtype=float)
    mode_decay = dict(mode_decay or DEFAULT_MODE_DECAY)
    templates = templates or DEFAULT_TEMPLATES
    anchors = tuple(anchors)
    coords, type_locs, type_idx, loc_ids = topo.coords, topo.type_locs, topo.type_idx, topo.loc_ids

    def _draw_tour_mode() -> str:
        owns_car = rng.random() < car_ownership
        owns_bike = rng.random() < bike_ownership
        cand = {"walk": 1.0, "pt": 0.6}
        if owns_bike:
            cand["bike"] = 1.0
        if owns_car:
            cand["car"] = 1.5
        modes = list(cand)
        w = np.array([cand[m] for m in modes], dtype=float)
        return modes[int(rng.choice(len(modes), p=w / w.sum()))]

    visits = np.zeros((coords.shape[0], len(topo.types)))
    tmpl_seqs = [seq for seq, _ in templates]
    tmpl_w = np.array([w for _, w in templates], dtype=float)
    tmpl_w /= tmpl_w.sum()

    plan_rows: List[dict] = []
    gt_rows: List[dict] = []
    for pi in range(n_persons):
        seq = tmpl_seqs[rng.choice(len(tmpl_seqs), p=tmpl_w)]
        tour_mode = _draw_tour_mode() if assign_modes else None
        scale = mode_decay[tour_mode] if assign_modes else gravity_scale
        home_pool = type_locs["home"]
        home_i = int(home_pool[_weighted_choice(rng, sizes[home_pool])])

        placed_coord: List[np.ndarray] = []
        placed_id: List[int] = []
        prev = home_i
        for t in seq:
            if t == "home":
                li = home_i
            else:
                pool = type_locs[t]
                d = np.hypot(coords[pool, 0] - coords[prev, 0], coords[pool, 1] - coords[prev, 1])
                w = sizes[pool] * np.exp(-d / scale)
                li = int(pool[_weighted_choice(rng, w)])
            placed_id.append(li)
            placed_coord.append(coords[li])
            visits[li, type_idx[t]] += 1
            prev = li

        for k in range(1, len(seq)):
            frm, to = placed_coord[k - 1], placed_coord[k]
            from_anchor = seq[k - 1] in anchors
            to_anchor = seq[k] in anchors
            lid = f"p{pi}-l{k}"
            true_d = float(np.hypot(to[0] - frm[0], to[1] - frm[1]))
            obs_d = true_d
            if distance_noise > 0:
                obs_d = max(1.0, true_d * (1.0 + float(rng.normal(0, distance_noise))))
            plan_rows.append({
                "unique_person_id": f"p{pi}", "unique_leg_id": lid, "to_act_type": seq[k],
                "distance_meters": obs_d, "mode": tour_mode,
                "from_x": frm[0] if from_anchor else np.nan,
                "from_y": frm[1] if from_anchor else np.nan,
                "to_x": to[0] if to_anchor else np.nan,
                "to_y": to[1] if to_anchor else np.nan,
            })
            gt_rows.append({
                "unique_leg_id": lid, "true_to_identifier": loc_ids[placed_id[k]],
                "true_to_x": float(to[0]), "true_to_y": float(to[1]), "to_is_free": not to_anchor,
            })
    return plan_rows, gt_rows, visits


def single_chain_plans(
    topo: Topology,
    n_legs: int,
    *,
    gravity_scale: float = 4000.0,
    distance_noise: float = 0.0,
    secondary: Sequence[str] = ("shop", "leisure", "other"),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One person's `n_legs`-leg chain on `topo` (home -> secondary... -> work): fixed
    endpoints, all intermediates free. A controlled single-chain fixture for runtime /
    chain-length scaling probes on the *same* gravity world as the population benchmark
    (so no separate uniform generator is needed). Returns (plans_df, ground_truth)."""
    if n_legs < 2:
        raise ValueError("n_legs must be >= 2.")
    rng = rng or np.random.default_rng()
    inter = tuple(secondary[i % len(secondary)] for i in range(n_legs - 1))
    seq = ("home", *inter, "work")  # n_legs+1 activities -> n_legs legs; home/work anchored
    plan_rows, gt_rows, _ = generate_chains(
        topo, 1, sizes=topo.sizes, gravity_scale=gravity_scale,
        distance_noise=distance_noise, assign_modes=False,
        templates=[(seq, 1.0)], rng=rng,
    )
    return pd.DataFrame(plan_rows), pd.DataFrame(gt_rows)


def generate_world(
    *,
    n_locations: int = 1000,
    n_persons: int = 500,
    box: Optional[float] = None,
    density_per_km2: Optional[float] = None,
    gravity_scale: float = 4000.0,
    n_clusters: int = 8,
    distance_noise: float = 0.0,
    assign_modes: bool = True,
    mode_decay: Optional[Dict[str, float]] = None,
    car_ownership: float = 0.6,
    bike_ownership: float = 0.5,
    type_prevalence: Optional[Dict[str, float]] = None,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    templates: Optional[List[Tuple[Tuple[str, ...], float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticWorld:
    """Compose a topology + chains into a ready-to-use world. Potentials = visit counts
    (usage). `distance_noise` adds relative noise to observed distances; `assign_modes`
    gives each tour an ownership-consistent mode with mode-specific decay."""
    rng = rng or np.random.default_rng()
    topo = build_topology(n_locations=n_locations, box=box, density_per_km2=density_per_km2,
                          n_clusters=n_clusters, type_prevalence=type_prevalence, rng=rng)
    plan_rows, gt_rows, visits = generate_chains(
        topo, n_persons, gravity_scale=gravity_scale, distance_noise=distance_noise,
        assign_modes=assign_modes, mode_decay=mode_decay, car_ownership=car_ownership,
        bike_ownership=bike_ownership, anchors=anchors, templates=templates, rng=rng,
    )

    ids_d, coords_d, pots_d = {}, {}, {}
    for t in topo.types:
        idx = topo.type_locs[t]
        ids_d[t] = topo.loc_ids[idx]
        coords_d[t] = topo.coords[idx].astype(float)
        pots_d[t] = visits[idx, topo.type_idx[t]].astype(float)

    return SyntheticWorld(
        locations_tuple=(ids_d, coords_d, pots_d),
        plans_df=pd.DataFrame(plan_rows),
        ground_truth=pd.DataFrame(gt_rows),
        anchor_types=tuple(anchors),
        topology=topo,
        meta={"n_locations": n_locations, "n_persons": n_persons, "types": topo.types,
              "box": topo.box, "density_per_km2": n_locations / (topo.box / 1000.0) ** 2,
              "n_legs": len(plan_rows),
              "n_free_legs": int(sum(r["to_is_free"] for r in gt_rows))},
    )


# --- City presets ------------------------------------------------------------------------
# Calibrated to a concrete real city so the synthetic world reproduces its facility density,
# spatial footprint and intra-city trip lengths. Full derivation + sources (OSM/MiD/Destatis)
# in research/synthetic_world_calibration.md. Each preset is a kwargs bundle for
# `generate_world`; `city_world(name, **overrides)` applies it.
CITY_PRESETS: Dict[str, dict] = {
    # Hannover (Landeshauptstadt): ~535 k inhabitants over ~204 km^2. Calibrated to MEASURED
    # OSM trip-destination densities (Overpass, area wikidata Q1715, 2026-06-09; see the doc):
    #   shop 18.6 /km^2 (3804) | leisure-destinations 11.9 (2421, gastronomy+sport+culture,
    #   excl. green space) | education 3.5 (714, incl. kindergarten) | other-services 5.6 (1140).
    # Anchoring shop at prevalence 0.20 fixes total facility density = 18.6/0.20 = 93 /km^2
    # -> n_locations = 93*204 ~ 19000, box ~ 14.3 km (~204 km^2). Each prevalence = its measured
    # density / 93, so the placeable types reproduce the real counts (shop ~3800, leisure ~2470,
    # education ~720, other ~1140). `work`/`home` are solver anchors -> density less critical;
    # `work` set to a plausible distinct-workplace pool (~15/km^2, between offices-only and all
    # establishments). Urban modal split is less car-bound than national MiD, so car ownership is
    # dialled down / bike up; decay lengths use the intra-city values.
    "hannover": dict(
        n_locations=19000,
        density_per_km2=93.0,          # -> box ~ sqrt(19000/93) ~ 14.3 km, ~204 km^2
        n_clusters=10,                 # core + outer districts (Stadtbezirke)
        type_prevalence={"home": 0.55, "work": 0.16, "shop": 0.20, "leisure": 0.13,
                         "education": 0.038, "other": 0.060},  # = measured /km^2 ÷ 93
        mode_decay=dict(URBAN_MODE_DECAY),
        car_ownership=0.65,            # urban: realized split ~ walk 35 / car 27 / pt 21 / bike 16
        bike_ownership=0.55,           # (the simple tour-mode heuristic under-weights car vs ~38% MIV)
    ),
}


def city_world(name: str = "hannover", *, n_persons: int = 500,
               rng: Optional[np.random.Generator] = None, **overrides) -> SyntheticWorld:
    """Build a `generate_world` calibrated to a real city (see `CITY_PRESETS`).

    `city_world("hannover")` reproduces Hannover's facility density (~46/km^2), footprint
    (~204 km^2) and intra-city trip lengths. Any preset field can be overridden via kwargs,
    e.g. `city_world("hannover", n_persons=2000, distance_noise=0.1)`."""
    if name not in CITY_PRESETS:
        raise KeyError(f"unknown city preset {name!r}; have {sorted(CITY_PRESETS)}")
    params = {**CITY_PRESETS[name], "n_persons": n_persons, **overrides}
    return generate_world(rng=rng, **params)
