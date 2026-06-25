"""Build a `synth.Topology` on **real OSM facility coordinates** instead of Gaussian blobs.

The synthetic world matches a city's facility *counts/densities* but not its *geometry* (it
uses clustered Gaussian blobs). This loader swaps in real OSM points so chains are placed on
the actual street/district layout — the strongest available decoupling of the test geometry
from any solver's search. It consumes a small POI table (produced offline by
research/scripts/fetch_osm_topology.py via Overpass); the library itself needs no network.

A POI table has columns: ``type`` (home/work/shop/leisure/education/other), ``lat``, ``lon``.
Attractiveness `sizes` are drawn at random (lognormal, like the Gaussian world) — the real
geometry already produces a realistic centre-high *potential* gradient via accessibility (more
reachable homes), so all three worlds share one attractiveness model and differ only in geometry.
An optional ``weight_col`` lets a caller supply a real capacity proxy instead.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .synth import (MID_MODE_DECAY_M, MID_PAIR_DECAY_M, MID_TO_DECAY_M, MID_TOUR_MODE_SPLIT,
                    MID_URBAN_TEMPLATES, Topology, add_type_alias, world_from_topology)

# Equirectangular metres around a reference point — accurate enough over a single city (~20 km),
# dependency-free (no pyproj). x east, y north.
_M_PER_DEG_LAT = 110_540.0

# Cached real Hannover POI snapshot (gitignored; produced by research/scripts/fetch_osm_topology.py).
DEFAULT_HANNOVER_POIS = os.path.join(os.path.dirname(__file__), "..", "data", "hannover_pois.csv")
# Real residential buildings + resident counts (research/scripts/extract_homes.py from the
# Niedersachsen synthetic-population GeoPackage); homes weighted by residents, not 1-per-building.
# Sum of weights = true Hannover population (~506 k residents); the mobile subset (~84%, MiD 2023)
# is the natural full-population build size (see MID_URBAN_MOBILITY_RATE).
DEFAULT_HANNOVER_HOMES = os.path.join(os.path.dirname(__file__), "..", "data", "hannover_homes.csv")

# decay_inflate tuned on the REAL OSM geometry (17.9 km box) so realized per-mode leg medians
# match MiD: walk/bike land on target (~1.0); car/pt saturate at ~0.75/0.81x — raising the inflate
# past ~2 no longer helps because the tightly-clustered real facilities cannot reach the urban-
# large-city median (which includes trips leaving Hannover), a harder geometric cap than the
# Gaussian preset's. (Re-tuned for the rawN>=10 + business template mix.)
MID_INFLATE_OSM = {"walk": 0.63, "bike": 0.77, "car": 2.2, "pt": 2.1}


def project_latlon(lat: np.ndarray, lon: np.ndarray,
                   lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    x = (lon - lon0) * (111_320.0 * np.cos(np.radians(lat0)))
    y = (lat - lat0) * _M_PER_DEG_LAT
    return x, y


def topology_from_pois(df: pd.DataFrame, *, weight_col: Optional[str] = None,
                       rng: Optional[np.random.Generator] = None) -> Topology:
    """Build a Topology from a projected POI table with columns ``type``, ``x``, ``y``. Each row
    is one (location, type) offering; a facility serving several purposes appears once per type
    (so it lands in several ``type_locs``). `sizes` are random lognormal by default (the geometry
    drives the potential gradient); pass ``weight_col`` to use a real attractiveness column."""
    need = {"type", "x", "y"}
    if not need.issubset(df.columns):
        raise ValueError(f"POI table needs columns {need}; got {set(df.columns)}")
    df = df.reset_index(drop=True)
    coords = np.column_stack([df["x"].to_numpy(float), df["y"].to_numpy(float)])
    # Shift to a non-negative box (the generator assumes a [0, box]^2 frame).
    coords = coords - coords.min(axis=0)
    rng = rng or np.random.default_rng()
    sizes = np.exp(rng.normal(0.0, 0.7, size=len(df)))       # default: lognormal, like build_topology
    if weight_col is not None and weight_col in df.columns:
        # use the real capacity where given (e.g. households for homes), random elsewhere (NaN)
        w = df[weight_col].to_numpy(float)
        real = np.isfinite(w) & (w > 0)
        sizes = np.where(real, w, sizes)
    types = sorted(df["type"].astype(str).unique())
    type_idx = {t: j for j, t in enumerate(types)}
    type_locs = {t: np.flatnonzero(df["type"].to_numpy() == t) for t in types}
    loc_ids = np.array([f"osm{i}" for i in range(len(df))], dtype=object)
    box = float(coords.max()) if len(df) else 0.0
    return Topology(coords, sizes, loc_ids, types, type_idx, type_locs, box)


def load_poi_csv(path: str, *, lat0: Optional[float] = None, lon0: Optional[float] = None,
                 weight_col: Optional[str] = None, rng: Optional[np.random.Generator] = None) -> Topology:
    """Load a POI CSV and build a Topology. If the table has ``lat``/``lon`` (not ``x``/``y``)
    it is projected to metres around (lat0, lon0), defaulting to the data centroid."""
    df = pd.read_csv(path)
    if "x" not in df.columns or "y" not in df.columns:
        lat0 = float(df["lat"].mean()) if lat0 is None else lat0
        lon0 = float(df["lon"].mean()) if lon0 is None else lon0
        df["x"], df["y"] = project_latlon(df["lat"], df["lon"], lat0, lon0)
    return topology_from_pois(df, weight_col=weight_col, rng=rng)


def hannover_topology(*, pois_csv: str = DEFAULT_HANNOVER_POIS,
                      homes_csv: Optional[str] = DEFAULT_HANNOVER_HOMES,
                      rng: Optional[np.random.Generator] = None) -> Topology:
    """Real-Hannover topology: OSM secondary/work facilities (random `sizes`) + real residential
    buildings weighted by **household count** (`homes_csv`) instead of one-per-building. Both are
    lat/lon and projected together. Falls back to the OSM building homes if `homes_csv` is absent."""
    rng = rng or np.random.default_rng()
    df = pd.read_csv(pois_csv)
    if homes_csv and os.path.exists(homes_csv):
        df = df[df["type"] != "home"]                      # drop OSM building homes
        df = pd.concat([df, pd.read_csv(homes_csv)], ignore_index=True)   # add household-weighted homes
    lat0, lon0 = float(df["lat"].mean()), float(df["lon"].mean())
    df = df.assign(**dict(zip(("x", "y"), project_latlon(df["lat"], df["lon"], lat0, lon0))))
    return topology_from_pois(df, weight_col="weight", rng=rng)


# Share of residents who are mobile on the reporting day (W-weighted MiD 2023 `mobil==1`,
# renormalized over known cases, RegioStaRGem5 51/52 large cities). The ~16% immobile make zero
# trips, so the generated *traveling* population is this fraction of total residents — every
# generated person travels (templates are mobile-only), so we scale the count, not the templates.
MID_URBAN_MOBILITY_RATE = 0.84


def hannover_osm_world(n_persons: Optional[int] = None, *, pois_csv: str = DEFAULT_HANNOVER_POIS,
                       homes_csv: Optional[str] = DEFAULT_HANNOVER_HOMES,
                       mobility_rate: float = MID_URBAN_MOBILITY_RATE,
                       distance_noise: float = 0.0, rng=None, **overrides):
    """Hannover world on **real OSM geometry**: real facility coordinates + real resident-weighted
    homes + MiD-2023 chains, per-(mode,from→to) decay and observed mode-split, with `decay_inflate`
    tuned on the real layout (`MID_INFLATE_OSM`). The real-geometry counterpart to
    `synth.city_world("hannover")`. Needs the cached snapshots in research/data/ (gitignored).

    `n_persons=None` builds the **true mobile population**: total residents (sum of home weights,
    ≈ 506 k) times `mobility_rate` (≈ 0.84 for an urban large city in MiD 2023) — only the mobile
    subset makes trips; the immobile ~16% produce no legs/visits and are simply absent. Pass a
    smaller int for quick runs (it is taken as the mobile count directly, no further scaling), or
    set `mobility_rate=1.0` to place every resident. Solve a sampled subset (`survey.draw_survey`)
    rather than the whole population."""
    rng = rng or np.random.default_rng()
    topo = hannover_topology(pois_csv=pois_csv, homes_csv=homes_csv, rng=rng)
    topo = add_type_alias(topo, "business", "work")   # MiD business trips -> work catalog
    if n_persons is None:   # true mobile population = total residents * mobility rate
        residents = float(topo.sizes[topo.type_locs["home"]].sum())
        n_persons = int(round(residents * mobility_rate))
    params = dict(templates=MID_URBAN_TEMPLATES, pair_decay=MID_PAIR_DECAY_M,
                  to_decay=MID_TO_DECAY_M, mode_decay=MID_MODE_DECAY_M,
                  mode_split=MID_TOUR_MODE_SPLIT, decay_inflate=dict(MID_INFLATE_OSM),
                  distance_noise=distance_noise)
    params.update(overrides)
    return world_from_topology(topo, n_persons, rng=rng, **params)
