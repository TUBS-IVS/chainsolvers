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
# Type mix calibrated to MEASURED Hannover OSM destination densities (= per-type /km^2 ÷ 93; see
# CITY_PRESETS["hannover"] and synthetic_world_calibration.md). `work` is the real OSM workplace
# count (~1,583 over 204 km^2 -> 7.8/km^2 -> 0.083), matching osm.hannover_topology; `business`
# is placed against the work pool. Shared by the gauss city preset and the two-zone world so both
# city geographies carry the same (real-OSM-matched) type proportions; the two-zone ring then
# inherits this mix at its own (sparse) `ring_density`, giving urban-dense / rural-sparse work.
HANNOVER_TYPE_PREVALENCE: Dict[str, float] = {
    "home": 0.55, "work": 0.083, "shop": 0.20, "leisure": 0.13,
    "education": 0.038, "other": 0.060,
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

# --- MiD 2023 urban calibration (baked constants) ----------------------------------------
# Derived offline from the MiD 2023 microdata (Wege file) by research/scripts/mid_extract.py,
# restricted to large cities (RegioStaRGem5 ∈ {51,52}), weighted by W_GEW. The raw CSV is
# licensed and gitignored (research/data/); only these small constants are committed.
# Used by the `hannover` city preset: real activity-chain skeletons + per-(mode, from→to)
# gravity decay + observed tour-mode split. See synthetic_world_calibration.md.

# Home-anchored activity-type chains, each backed by >= 10 MiD diaries (a disclosure floor,
# not a top-N rank cut); 437 templates cover ~93 % of urban tour weight; normalized.
# `business` (zweck 2, dienstlich) is its own type, placed against the WORK catalog; `home`
# also closes returns (zweck 9). See synthetic_world_calibration.md.
MID_URBAN_TEMPLATES: List[Tuple[Tuple[str, ...], float]] = [
    (("home", "work", "home"), 0.1164),
    (("home", "other", "home"), 0.0993),
    (("home", "leisure", "home"), 0.0950),
    (("home", "education", "home"), 0.0721),
    (("home", "shop", "home"), 0.0621),
    (("home", "leisure", "leisure", "home"), 0.0236),
    (("home", "education", "home", "leisure", "home"), 0.0214),
    (("home", "other", "other", "home"), 0.0181),
    (("home", "work", "home", "leisure", "home"), 0.0173),
    (("home", "shop", "home", "leisure", "home"), 0.0157),
    (("home", "other", "home", "other", "home"), 0.0149),
    (("home", "leisure", "home", "leisure", "home"), 0.0142),
    (("home", "other", "shop", "home"), 0.0126),
    (("home", "work", "home", "shop", "home"), 0.0124),
    (("home", "other", "home", "leisure", "home"), 0.0110),
    (("home", "work", "shop", "home"), 0.0104),
    (("home", "education", "home", "other", "home"), 0.0103),
    (("home", "shop", "home", "other", "home"), 0.0095),
    (("home", "work", "home", "other", "home"), 0.0095),
    (("home", "other", "home", "home"), 0.0092),
    (("home", "work", "leisure", "home"), 0.0086),
    (("home", "work", "other", "home"), 0.0086),
    (("home", "other", "home", "shop", "home"), 0.0084),
    (("home", "other", "other", "other", "home"), 0.0078),
    (("home", "other", "leisure", "home"), 0.0073),
    (("home", "shop", "shop", "home"), 0.0068),
    (("home", "education", "leisure", "home"), 0.0067),
    (("home", "education", "other", "home"), 0.0067),
    (("home", "leisure", "shop", "home"), 0.0060),
    (("home", "leisure", "other", "home"), 0.0059),
    (("home", "shop", "leisure", "home"), 0.0059),
    (("home", "home", "home"), 0.0058),
    (("home", "shop", "home", "home"), 0.0058),
    (("home", "shop", "home", "shop", "home"), 0.0056),
    (("home", "leisure", "home", "home"), 0.0055),
    (("home", "leisure", "leisure", "leisure", "home"), 0.0047),
    (("home", "shop", "other", "home"), 0.0044),
    (("home", "other", "work", "home"), 0.0041),
    (("home", "leisure", "home", "other", "home"), 0.0039),
    (("home", "leisure", "home", "shop", "home"), 0.0037),
    (("home", "work", "home", "home"), 0.0035),
    (("home", "education", "home", "shop", "home"), 0.0034),
    (("home", "business", "home"), 0.0033),
    (("home", "home", "leisure", "home"), 0.0031),
    (("home", "education", "shop", "home"), 0.0030),
    (("home", "work", "other", "home", "home"), 0.0029),
    (("home", "other", "other", "other", "other", "home"), 0.0026),
    (("home", "education", "home", "home"), 0.0026),
    (("home", "leisure", "leisure", "home", "home"), 0.0025),
    (("home", "other", "other", "leisure", "home"), 0.0024),
    (("home", "other", "home", "other", "home", "other", "home"), 0.0024),
    (("home", "other", "home", "other", "other", "home"), 0.0022),
    (("home", "other", "work", "other", "home"), 0.0022),
    (("home", "work", "shop", "home", "leisure", "home"), 0.0021),
    (("home", "shop", "home", "other", "home", "leisure", "home"), 0.0021),
    (("home", "home", "other", "home"), 0.0021),
    (("home", "education", "education", "home"), 0.0020),
    (("home", "shop", "home", "leisure", "leisure", "home"), 0.0020),
    (("home", "other", "other", "shop", "home"), 0.0020),
    (("home", "other", "shop", "shop", "home"), 0.0019),
    (("home", "education", "home", "education", "home"), 0.0019),
    (("home", "work", "shop", "home", "home"), 0.0018),
    (("home", "leisure", "leisure", "leisure", "leisure", "home"), 0.0018),
    (("home", "work", "other", "other", "home"), 0.0018),
    (("home", "work", "home", "shop", "home", "leisure", "home"), 0.0017),
    (("home", "other", "home", "leisure", "leisure", "home"), 0.0017),
    (("home", "leisure", "work", "home"), 0.0016),
    (("home", "other", "other", "home", "other", "home"), 0.0016),
    (("home", "shop", "home", "work", "home"), 0.0016),
    (("home", "work", "other", "shop", "home"), 0.0016),
    (("home", "shop", "home", "leisure", "home", "leisure", "home"), 0.0016),
    (("home", "home", "shop", "home"), 0.0016),
    (("home", "other", "other", "home", "home"), 0.0015),
    (("home", "home", "home", "home"), 0.0015),
    (("home", "other", "home", "shop", "home", "leisure", "home"), 0.0015),
    (("home", "work", "business", "home"), 0.0014),
    (("home", "shop", "home", "other", "home", "other", "home"), 0.0014),
    (("home", "other", "shop", "home", "other", "home"), 0.0014),
    (("home", "leisure", "leisure", "home", "leisure", "home"), 0.0014),
    (("home", "shop", "home", "shop", "home", "leisure", "home"), 0.0014),
    (("home", "other", "home", "work", "home"), 0.0014),
    (("home", "leisure", "home", "leisure", "leisure", "home"), 0.0013),
    (("home", "leisure", "shop", "home", "leisure", "home"), 0.0012),
    (("home", "education", "home", "leisure", "leisure", "home"), 0.0012),
    (("home", "work", "home", "other", "shop", "home"), 0.0012),
    (("home", "other", "home", "shop", "home", "other", "home"), 0.0012),
    (("home", "other", "leisure", "leisure", "home"), 0.0012),
    (("home", "other", "shop", "home", "leisure", "home"), 0.0012),
    (("home", "work", "home", "work", "home"), 0.0012),
    (("home", "leisure", "leisure", "other", "home"), 0.0011),
    (("home", "work", "home", "leisure", "leisure", "home"), 0.0011),
    (("home", "other", "home", "other", "home", "leisure", "home"), 0.0011),
    (("home", "other", "leisure", "other", "home"), 0.0011),
    (("home", "other", "other", "home", "leisure", "home"), 0.0011),
    (("home", "education", "other", "other", "home"), 0.0011),
    (("home", "other", "shop", "home", "home"), 0.0011),
    (("home", "other", "shop", "other", "home"), 0.0011),
    (("home", "leisure", "leisure", "shop", "home"), 0.0010),
    (("home", "other", "other", "home", "shop", "home"), 0.0010),
    (("home", "shop", "shop", "home", "other", "home"), 0.0010),
    (("home", "leisure", "other", "other", "home"), 0.0009),
    (("home", "work", "home", "other", "other", "home"), 0.0009),
    (("home", "work", "business", "business", "home"), 0.0009),
    (("home", "other", "leisure", "home", "home"), 0.0009),
    (("home", "work", "leisure", "leisure", "home"), 0.0009),
    (("home", "other", "home", "other", "home", "shop", "home"), 0.0009),
    (("home", "leisure", "shop", "home", "home"), 0.0009),
    (("home", "other", "home", "other", "home", "home"), 0.0009),
    (("home", "work", "education", "home"), 0.0009),
    (("home", "work", "work", "work", "home", "home", "home"), 0.0009),
    (("home", "other", "home", "other", "shop", "home"), 0.0009),
    (("home", "shop", "shop", "shop", "home"), 0.0009),
    (("home", "shop", "shop", "home", "leisure", "home"), 0.0009),
    (("home", "work", "home", "other", "leisure", "home"), 0.0009),
    (("home", "leisure", "home", "work", "home"), 0.0009),
    (("home", "shop", "other", "shop", "home"), 0.0008),
    (("home", "work", "other", "home", "leisure", "home"), 0.0008),
    (("home", "work", "home", "other", "home", "leisure", "home"), 0.0008),
    (("home", "shop", "other", "leisure", "home"), 0.0008),
    (("home", "other", "home", "other", "leisure", "home"), 0.0008),
    (("home", "leisure", "home", "other", "home", "other", "home"), 0.0008),
    (("home", "other", "education", "home"), 0.0008),
    (("home", "education", "leisure", "leisure", "home"), 0.0008),
    (("home", "work", "other", "leisure", "home"), 0.0008),
    (("home", "other", "shop", "home", "other", "shop", "home"), 0.0008),
    (("home", "leisure", "leisure", "leisure", "home", "home"), 0.0008),
    (("home", "work", "work", "home", "home"), 0.0008),
    (("home", "other", "other", "other", "other", "other", "home"), 0.0008),
    (("home", "leisure", "other", "home", "home"), 0.0008),
    (("home", "work", "work", "home"), 0.0008),
    (("home", "education", "home", "leisure", "home", "leisure", "home"), 0.0008),
    (("home", "work", "business", "work", "home"), 0.0008),
    (("home", "education", "home", "other", "home", "leisure", "home"), 0.0007),
    (("home", "other", "leisure", "shop", "home"), 0.0007),
    (("home", "other", "education", "home", "leisure", "home"), 0.0007),
    (("home", "shop", "shop", "home", "shop", "home"), 0.0007),
    (("home", "shop", "other", "other", "home"), 0.0007),
    (("home", "shop", "home", "other", "other", "home"), 0.0007),
    (("home", "shop", "shop", "leisure", "home"), 0.0007),
    (("home", "work", "shop", "home", "other", "home"), 0.0007),
    (("home", "leisure", "home", "shop", "home", "leisure", "home"), 0.0007),
    (("home", "education", "other", "home", "home"), 0.0007),
    (("home", "education", "home", "other", "leisure", "home"), 0.0007),
    (("home", "work", "other", "other", "home", "home"), 0.0007),
    (("home", "home", "other", "other", "home"), 0.0007),
    (("home", "work", "leisure", "home", "home"), 0.0007),
    (("home", "shop", "home", "leisure", "other", "home"), 0.0006),
    (("home", "shop", "home", "other", "home", "home"), 0.0006),
    (("home", "shop", "work", "home"), 0.0006),
    (("home", "leisure", "home", "leisure", "home", "leisure", "home"), 0.0006),
    (("home", "other", "home", "leisure", "home", "home"), 0.0006),
    (("home", "education", "leisure", "home", "leisure", "home"), 0.0006),
    (("home", "shop", "shop", "other", "home"), 0.0006),
    (("home", "education", "education", "other", "home"), 0.0006),
    (("home", "other", "home", "shop", "home", "shop", "home"), 0.0006),
    (("home", "work", "other", "work", "home"), 0.0006),
    (("home", "leisure", "shop", "home", "other", "home"), 0.0006),
    (("home", "shop", "home", "shop", "home", "other", "home"), 0.0006),
    (("home", "work", "shop", "other", "home"), 0.0006),
    (("home", "shop", "leisure", "leisure", "home", "leisure", "home"), 0.0006),
    (("home", "business", "home", "leisure", "home"), 0.0006),
    (("home", "work", "shop", "shop", "home"), 0.0006),
    (("home", "work", "home", "leisure", "shop", "home"), 0.0006),
    (("home", "education", "work", "home"), 0.0006),
    (("home", "work", "other", "home", "other", "home"), 0.0005),
    (("home", "education", "leisure", "other", "home"), 0.0005),
    (("home", "leisure", "leisure", "home", "shop", "home"), 0.0005),
    (("home", "work", "other", "home", "shop", "home"), 0.0005),
    (("home", "other", "shop", "home", "shop", "home"), 0.0005),
    (("home", "shop", "leisure", "home", "home"), 0.0005),
    (("home", "other", "home", "home", "leisure", "home"), 0.0005),
    (("home", "education", "home", "leisure", "other", "home"), 0.0005),
    (("home", "leisure", "other", "leisure", "home"), 0.0005),
    (("home", "shop", "leisure", "leisure", "home"), 0.0005),
    (("home", "education", "home", "other", "other", "home"), 0.0005),
    (("home", "education", "home", "work", "home"), 0.0005),
    (("home", "leisure", "home", "other", "home", "leisure", "home"), 0.0005),
    (("home", "other", "home", "leisure", "home", "other", "home"), 0.0005),
    (("home", "other", "work", "home", "other", "home"), 0.0005),
    (("home", "leisure", "home", "shop", "leisure", "home"), 0.0005),
    (("home", "education", "home", "leisure", "home", "home"), 0.0005),
    (("home", "education", "other", "home", "leisure", "home"), 0.0005),
    (("home", "shop", "home", "other", "home", "shop", "home"), 0.0005),
    (("home", "work", "home", "shop", "shop", "home"), 0.0005),
    (("home", "shop", "other", "other", "other", "home"), 0.0005),
    (("home", "work", "other", "home", "home", "leisure", "home"), 0.0005),
    (("home", "leisure", "leisure", "home", "other", "home"), 0.0005),
    (("home", "work", "leisure", "shop", "home"), 0.0005),
    (("home", "leisure", "home", "shop", "home", "other", "home"), 0.0005),
    (("home", "other", "home", "leisure", "shop", "home"), 0.0005),
    (("home", "other", "other", "other", "home", "other", "home"), 0.0005),
    (("home", "education", "education", "leisure", "home"), 0.0004),
    (("home", "shop", "home", "other", "leisure", "home"), 0.0004),
    (("home", "other", "home", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "other", "home", "business", "home"), 0.0004),
    (("home", "other", "shop", "leisure", "home"), 0.0004),
    (("home", "shop", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "other", "home", "home", "home"), 0.0004),
    (("home", "education", "home", "leisure", "home", "other", "home"), 0.0004),
    (("home", "leisure", "home", "other", "other", "home"), 0.0004),
    (("home", "education", "home", "other", "shop", "home"), 0.0004),
    (("home", "leisure", "shop", "shop", "home"), 0.0004),
    (("home", "education", "education", "home", "leisure", "home"), 0.0004),
    (("home", "other", "home", "shop", "home", "home"), 0.0004),
    (("home", "education", "home", "other", "home", "other", "home"), 0.0004),
    (("home", "shop", "home", "leisure", "home", "home"), 0.0004),
    (("home", "work", "home", "other", "home", "other", "home"), 0.0004),
    (("home", "other", "other", "work", "home"), 0.0004),
    (("home", "education", "other", "home", "other", "home"), 0.0004),
    (("home", "leisure", "work", "home", "leisure", "home"), 0.0004),
    (("home", "other", "shop", "home", "other", "home", "other", "home"), 0.0004),
    (("home", "leisure", "home", "leisure", "other", "home"), 0.0004),
    (("home", "leisure", "leisure", "leisure", "leisure", "leisure", "home"), 0.0004),
    (("home", "other", "home", "leisure", "other", "home"), 0.0004),
    (("home", "business", "work", "home"), 0.0004),
    (("home", "business", "business", "home"), 0.0004),
    (("home", "other", "work", "other", "home", "home"), 0.0004),
    (("home", "education", "leisure", "home", "home"), 0.0004),
    (("home", "leisure", "other", "home", "leisure", "home"), 0.0004),
    (("home", "shop", "home", "leisure", "home", "other", "home"), 0.0004),
    (("home", "work", "home", "other", "home", "shop", "home"), 0.0004),
    (("home", "business", "leisure", "home"), 0.0004),
    (("home", "other", "leisure", "home", "other", "home"), 0.0004),
    (("home", "other", "home", "shop", "other", "home"), 0.0004),
    (("home", "leisure", "home", "other", "leisure", "home"), 0.0004),
    (("home", "work", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "home", "home", "home", "home"), 0.0004),
    (("home", "leisure", "shop", "leisure", "home"), 0.0004),
    (("home", "leisure", "leisure", "home", "leisure", "leisure", "home"), 0.0004),
    (("home", "other", "home", "leisure", "home", "shop", "home"), 0.0004),
    (("home", "other", "other", "other", "home", "home", "home"), 0.0004),
    (("home", "leisure", "shop", "other", "home"), 0.0004),
    (("home", "work", "home", "shop", "home", "other", "home"), 0.0004),
    (("home", "other", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "other", "home", "other", "other", "other", "home"), 0.0004),
    (("home", "business", "home", "other", "home"), 0.0004),
    (("home", "work", "home", "shop", "leisure", "home"), 0.0004),
    (("home", "work", "home", "shop", "home", "home"), 0.0003),
    (("home", "leisure", "leisure", "leisure", "leisure", "home", "home"), 0.0003),
    (("home", "other", "work", "shop", "home"), 0.0003),
    (("home", "other", "home", "shop", "shop", "home"), 0.0003),
    (("home", "leisure", "leisure", "leisure", "leisure", "leisure", "leisure", "home"), 0.0003),
    (("home", "business", "home", "shop", "home"), 0.0003),
    (("home", "work", "shop", "leisure", "home"), 0.0003),
    (("home", "education", "home", "education", "home", "leisure", "home"), 0.0003),
    (("home", "other", "work", "home", "leisure", "home"), 0.0003),
    (("home", "other", "other", "other", "home", "home"), 0.0003),
    (("home", "work", "business", "home", "home"), 0.0003),
    (("home", "leisure", "other", "shop", "home"), 0.0003),
    (("home", "leisure", "home", "leisure", "home", "home"), 0.0003),
    (("home", "home", "leisure", "leisure", "home"), 0.0003),
    (("home", "shop", "home", "leisure", "shop", "home"), 0.0003),
    (("home", "other", "work", "home", "shop", "home"), 0.0003),
    (("home", "education", "other", "leisure", "home"), 0.0003),
    (("home", "other", "business", "home"), 0.0003),
    (("home", "education", "home", "leisure", "shop", "home"), 0.0003),
    (("home", "work", "shop", "home", "home", "leisure", "home"), 0.0003),
    (("home", "shop", "home", "shop", "home", "shop", "home"), 0.0003),
    (("home", "home", "home", "home", "home", "home"), 0.0003),
    (("home", "other", "other", "shop", "other", "home"), 0.0003),
    (("home", "other", "home", "education", "home"), 0.0003),
    (("home", "leisure", "other", "home", "other", "home"), 0.0003),
    (("home", "other", "other", "home", "other", "other", "home"), 0.0003),
    (("home", "other", "work", "home", "home"), 0.0003),
    (("home", "education", "shop", "home", "home"), 0.0003),
    (("home", "business", "other", "home"), 0.0003),
    (("home", "leisure", "education", "home"), 0.0003),
    (("home", "leisure", "home", "leisure", "leisure", "leisure", "home"), 0.0003),
    (("home", "other", "other", "other", "other", "leisure", "home"), 0.0003),
    (("home", "work", "other", "other", "other", "home"), 0.0003),
    (("home", "other", "work", "other", "home", "other", "home"), 0.0003),
    (("home", "other", "home", "home", "other", "home"), 0.0003),
    (("home", "other", "shop", "shop", "home", "leisure", "home"), 0.0003),
    (("home", "other", "shop", "home", "other", "home", "shop", "home"), 0.0003),
    (("home", "work", "shop", "work", "home"), 0.0003),
    (("home", "education", "leisure", "home", "other", "home"), 0.0003),
    (("home", "leisure", "leisure", "leisure", "home", "home", "home"), 0.0003),
    (("home", "work", "business", "home", "other", "home"), 0.0003),
    (("home", "education", "other", "education", "home"), 0.0003),
    (("home", "home", "shop", "home", "leisure", "home"), 0.0003),
    (("home", "other", "shop", "shop", "shop", "home"), 0.0003),
    (("home", "home", "home", "leisure", "home"), 0.0003),
    (("home", "work", "other", "other", "shop", "home"), 0.0003),
    (("home", "other", "work", "leisure", "home"), 0.0003),
    (("home", "other", "home", "other", "other", "leisure", "home"), 0.0003),
    (("home", "education", "home", "shop", "home", "leisure", "home"), 0.0003),
    (("home", "work", "shop", "home", "shop", "home"), 0.0003),
    (("home", "other", "home", "work", "home", "other", "home"), 0.0003),
    (("home", "other", "other", "shop", "home", "other", "home"), 0.0003),
    (("home", "leisure", "home", "home", "home"), 0.0003),
    (("home", "work", "home", "leisure", "other", "home"), 0.0003),
    (("home", "other", "home", "other", "other", "home", "other", "home"), 0.0003),
    (("home", "work", "other", "home", "home", "shop", "home"), 0.0003),
    (("home", "other", "education", "leisure", "home"), 0.0003),
    (("home", "education", "other", "home", "home", "leisure", "home"), 0.0002),
    (("home", "leisure", "leisure", "leisure", "home", "leisure", "home"), 0.0002),
    (("home", "other", "other", "other", "leisure", "home"), 0.0002),
    (("home", "leisure", "work", "leisure", "home"), 0.0002),
    (("home", "work", "shop", "home", "home", "other", "home"), 0.0002),
    (("home", "leisure", "home", "leisure", "home", "shop", "home"), 0.0002),
    (("home", "shop", "leisure", "other", "home"), 0.0002),
    (("home", "shop", "shop", "shop", "home", "leisure", "home"), 0.0002),
    (("home", "shop", "shop", "home", "home"), 0.0002),
    (("home", "shop", "other", "home", "home"), 0.0002),
    (("home", "education", "education", "education", "home"), 0.0002),
    (("home", "other", "other", "other", "shop", "home"), 0.0002),
    (("home", "shop", "other", "home", "leisure", "home"), 0.0002),
    (("home", "business", "home", "home"), 0.0002),
    (("home", "leisure", "shop", "leisure", "leisure", "home"), 0.0002),
    (("home", "shop", "home", "leisure", "home", "shop", "home"), 0.0002),
    (("home", "leisure", "home", "other", "home", "shop", "home"), 0.0002),
    (("home", "other", "other", "other", "other", "home", "home"), 0.0002),
    (("home", "leisure", "leisure", "shop", "home", "home"), 0.0002),
    (("home", "other", "work", "other", "home", "leisure", "home"), 0.0002),
    (("home", "work", "leisure", "other", "home"), 0.0002),
    (("home", "shop", "other", "home", "other", "home"), 0.0002),
    (("home", "leisure", "shop", "home", "shop", "home"), 0.0002),
    (("home", "business", "shop", "home"), 0.0002),
    (("home", "other", "work", "other", "home", "shop", "home"), 0.0002),
    (("home", "home", "work", "home"), 0.0002),
    (("home", "shop", "home", "shop", "home", "leisure", "leisure", "home"), 0.0002),
    (("home", "work", "other", "work", "home", "shop", "home"), 0.0002),
    (("home", "other", "leisure", "leisure", "leisure", "home"), 0.0002),
    (("home", "other", "home", "shop", "leisure", "home"), 0.0002),
    (("home", "business", "business", "business", "home"), 0.0002),
    (("home", "shop", "home", "business", "home"), 0.0002),
    (("home", "shop", "home", "other", "home", "leisure", "leisure", "home"), 0.0002),
    (("home", "work", "home", "shop", "other", "home"), 0.0002),
    (("home", "education", "education", "home", "home"), 0.0002),
    (("home", "education", "home", "other", "home", "home"), 0.0002),
    (("home", "other", "other", "leisure", "leisure", "home"), 0.0002),
    (("home", "other", "home", "other", "home", "other", "other", "home"), 0.0002),
    (("home", "shop", "home", "leisure", "leisure", "leisure", "home"), 0.0002),
    (("home", "work", "home", "shop", "home", "shop", "home"), 0.0002),
    (("home", "other", "shop", "other", "other", "home"), 0.0002),
    (("home", "shop", "home", "home", "other", "home"), 0.0002),
    (("home", "leisure", "home", "shop", "shop", "home"), 0.0002),
    (("home", "leisure", "leisure", "leisure", "other", "home"), 0.0002),
    (("home", "work", "leisure", "home", "shop", "home"), 0.0002),
    (("home", "shop", "home", "other", "shop", "home"), 0.0002),
    (("home", "other", "home", "other", "other", "home", "leisure", "home"), 0.0002),
    (("home", "leisure", "other", "home", "shop", "home"), 0.0002),
    (("home", "leisure", "home", "other", "shop", "home"), 0.0002),
    (("home", "work", "business", "business", "home", "leisure", "home"), 0.0002),
    (("home", "other", "shop", "home", "work", "home"), 0.0002),
    (("home", "other", "other", "shop", "home", "home"), 0.0002),
    (("home", "work", "home", "leisure", "home", "home"), 0.0002),
    (("home", "education", "education", "home", "other", "home"), 0.0002),
    (("home", "other", "other", "other", "home", "leisure", "home"), 0.0002),
    (("home", "shop", "home", "home", "leisure", "home"), 0.0002),
    (("home", "leisure", "shop", "home", "leisure", "leisure", "home"), 0.0002),
    (("home", "other", "shop", "other", "shop", "home"), 0.0002),
    (("home", "work", "home", "home", "home"), 0.0002),
    (("home", "shop", "other", "home", "shop", "home"), 0.0002),
    (("home", "home", "shop", "leisure", "home"), 0.0002),
    (("home", "work", "home", "other", "home", "home"), 0.0002),
    (("home", "home", "other", "home", "home"), 0.0002),
    (("home", "work", "leisure", "work", "home"), 0.0002),
    (("home", "work", "home", "leisure", "home", "other", "home"), 0.0002),
    (("home", "other", "other", "shop", "shop", "home"), 0.0002),
    (("home", "work", "home", "leisure", "home", "shop", "home"), 0.0002),
    (("home", "other", "leisure", "home", "shop", "home"), 0.0002),
    (("home", "work", "business", "home", "shop", "home"), 0.0002),
    (("home", "home", "home", "other", "home"), 0.0002),
    (("home", "shop", "leisure", "shop", "home"), 0.0002),
    (("home", "other", "shop", "home", "leisure", "leisure", "home"), 0.0002),
    (("home", "shop", "home", "shop", "home", "home"), 0.0002),
    (("home", "work", "home", "leisure", "home", "leisure", "home"), 0.0002),
    (("home", "other", "work", "other", "leisure", "home"), 0.0002),
    (("home", "home", "other", "home", "other", "home"), 0.0002),
    (("home", "leisure", "leisure", "leisure", "shop", "home"), 0.0001),
    (("home", "other", "shop", "home", "other", "other", "home"), 0.0001),
    (("home", "leisure", "home", "shop", "home", "home"), 0.0001),
    (("home", "other", "leisure", "other", "home", "leisure", "home"), 0.0001),
    (("home", "shop", "other", "leisure", "leisure", "home"), 0.0001),
    (("home", "shop", "shop", "other", "home", "other", "home"), 0.0001),
    (("home", "other", "other", "shop", "home", "leisure", "home"), 0.0001),
    (("home", "shop", "home", "other", "other", "other", "home"), 0.0001),
    (("home", "leisure", "leisure", "home", "home", "home"), 0.0001),
    (("home", "other", "shop", "home", "other", "home", "leisure", "home"), 0.0001),
    (("home", "home", "leisure", "home", "leisure", "home"), 0.0001),
    (("home", "other", "other", "leisure", "shop", "home"), 0.0001),
    (("home", "work", "home", "work", "home", "leisure", "home"), 0.0001),
    (("home", "leisure", "other", "other", "home", "home"), 0.0001),
    (("home", "work", "home", "business", "home"), 0.0001),
    (("home", "shop", "shop", "shop", "shop", "home"), 0.0001),
    (("home", "other", "other", "leisure", "home", "home"), 0.0001),
    (("home", "other", "work", "other", "other", "home"), 0.0001),
    (("home", "shop", "home", "leisure", "leisure", "home", "home"), 0.0001),
    (("home", "other", "other", "home", "other", "home", "other", "home"), 0.0001),
    (("home", "shop", "leisure", "home", "other", "home"), 0.0001),
    (("home", "work", "work", "shop", "home"), 0.0001),
    (("home", "education", "shop", "home", "leisure", "home"), 0.0001),
    (("home", "shop", "home", "shop", "other", "home"), 0.0001),
    (("home", "other", "home", "leisure", "leisure", "leisure", "home"), 0.0001),
    (("home", "work", "other", "home", "home", "other", "home"), 0.0001),
    (("home", "other", "other", "leisure", "other", "home"), 0.0001),
    (("home", "leisure", "home", "shop", "home", "shop", "home"), 0.0001),
    (("home", "other", "shop", "shop", "home", "home"), 0.0001),
    (("home", "shop", "home", "leisure", "leisure", "home", "leisure", "home"), 0.0001),
    (("home", "work", "home", "work", "shop", "home"), 0.0001),
    (("home", "work", "home", "other", "other", "other", "home"), 0.0001),
    (("home", "work", "other", "work", "other", "home"), 0.0001),
    (("home", "leisure", "home", "leisure", "shop", "home"), 0.0001),
    (("home", "work", "other", "other", "leisure", "home"), 0.0001),
    (("home", "work", "other", "work", "home", "leisure", "home"), 0.0001),
    (("home", "home", "leisure", "shop", "home"), 0.0001),
    (("home", "shop", "leisure", "leisure", "leisure", "home"), 0.0001),
    (("home", "other", "other", "home", "other", "home", "home"), 0.0001),
    (("home", "shop", "home", "shop", "shop", "home"), 0.0001),
    (("home", "leisure", "home", "leisure", "home", "other", "home"), 0.0001),
    (("home", "work", "leisure", "home", "other", "home"), 0.0001),
    (("home", "other", "home", "work", "other", "home"), 0.0001),
    (("home", "other", "home", "work", "home", "shop", "home"), 0.0001),
    (("home", "work", "business", "home", "leisure", "home"), 0.0001),
    (("home", "work", "shop", "work", "leisure", "home"), 0.0001),
    (("home", "shop", "home", "home", "home"), 0.0001),
    (("home", "home", "other", "leisure", "home"), 0.0001),
    (("home", "other", "leisure", "other", "other", "home"), 0.0001),
    (("home", "other", "shop", "shop", "home", "other", "home"), 0.0001),
    (("home", "work", "home", "other", "other", "leisure", "home"), 0.0001),
    (("home", "work", "other", "other", "home", "leisure", "home"), 0.0001),
    (("home", "other", "other", "other", "other", "other", "other", "home"), 0.0001),
    (("home", "other", "other", "home", "work", "home"), 0.0001),
    (("home", "leisure", "home", "other", "home", "home"), 0.0001),
    (("home", "other", "other", "other", "other", "shop", "home"), 0.0001),
    (("home", "other", "other", "home", "other", "home", "leisure", "home"), 0.0001),
    (("home", "shop", "home", "work", "home", "leisure", "home"), 0.0001),
    (("home", "leisure", "home", "home", "leisure", "home"), 0.0001),
    (("home", "shop", "shop", "shop", "home", "home"), 0.0001),
    (("home", "other", "other", "leisure", "home", "leisure", "home"), 0.0001),
    (("home", "business", "home", "business", "home"), 0.0001),
    (("home", "other", "shop", "shop", "other", "home"), 0.0001),
    (("home", "work", "shop", "home", "leisure", "leisure", "home"), 0.0001),
    (("home", "education", "shop", "other", "home"), 0.0001),
    (("home", "leisure", "home", "leisure", "leisure", "home", "home"), 0.0001),
    (("home", "shop", "shop", "home", "leisure", "leisure", "home"), 0.0000),
]

# Gravity decay length (metres) = median observed leg distance, keyed by (mode, from, to).
# Cells with >=200 weighted obs; sparser cells fall back to MID_TO_DECAY_M then MID_MODE_DECAY_M.
MID_PAIR_DECAY_M: Dict[Tuple[str, str, str], float] = {
    ("bike", "education", "home"): 1960,
    ("bike", "home", "education"): 1960,
    ("bike", "home", "home"): 2940,
    ("bike", "home", "leisure"): 2940,
    ("bike", "home", "other"): 1960,
    ("bike", "home", "shop"): 1760,
    ("bike", "home", "work"): 3920,
    ("bike", "leisure", "home"): 2940,
    ("bike", "leisure", "leisure"): 2940,
    ("bike", "other", "home"): 1960,
    ("bike", "other", "leisure"): 2450,
    ("bike", "other", "other"): 1960,
    ("bike", "other", "shop"): 1960,
    ("bike", "other", "work"): 3140,
    ("bike", "shop", "home"): 1470,
    ("bike", "work", "home"): 3920,
    ("bike", "work", "other"): 2450,
    ("bike", "work", "shop"): 1960,
    ("car", "business", "business"): 10766,
    ("car", "business", "home"): 10450,
    ("car", "education", "home"): 2850,
    ("car", "education", "other"): 5073,
    ("car", "home", "business"): 11400,
    ("car", "home", "education"): 2850,
    ("car", "home", "home"): 7600,
    ("car", "home", "leisure"): 7600,
    ("car", "home", "other"): 5700,
    ("car", "home", "shop"): 3330,
    ("car", "home", "work"): 10450,
    ("car", "leisure", "home"): 7600,
    ("car", "leisure", "leisure"): 6650,
    ("car", "leisure", "other"): 5700,
    ("car", "leisure", "shop"): 3800,
    ("car", "other", "home"): 5700,
    ("car", "other", "leisure"): 7600,
    ("car", "other", "other"): 5700,
    ("car", "other", "shop"): 3800,
    ("car", "other", "work"): 7600,
    ("car", "shop", "home"): 3230,
    ("car", "shop", "leisure"): 6195,
    ("car", "shop", "other"): 3844,
    ("car", "shop", "shop"): 3800,
    ("car", "work", "home"): 10930,
    ("car", "work", "leisure"): 9500,
    ("car", "work", "other"): 6180,
    ("car", "work", "shop"): 4245,
    ("pt", "business", "home"): 7200,
    ("pt", "education", "home"): 4500,
    ("pt", "home", "education"): 4500,
    ("pt", "home", "home"): 8873,
    ("pt", "home", "leisure"): 6300,
    ("pt", "home", "other"): 4500,
    ("pt", "home", "shop"): 3150,
    ("pt", "home", "work"): 9000,
    ("pt", "leisure", "home"): 6750,
    ("pt", "leisure", "leisure"): 6300,
    ("pt", "other", "home"): 5400,
    ("pt", "other", "leisure"): 6300,
    ("pt", "other", "other"): 4500,
    ("pt", "other", "shop"): 3600,
    ("pt", "shop", "home"): 3600,
    ("pt", "work", "home"): 9000,
    ("pt", "work", "leisure"): 7078,
    ("pt", "work", "other"): 5400,
    ("walk", "education", "home"): 980,
    ("walk", "education", "other"): 780,
    ("walk", "home", "education"): 780,
    ("walk", "home", "home"): 980,
    ("walk", "home", "leisure"): 1470,
    ("walk", "home", "other"): 980,
    ("walk", "home", "shop"): 780,
    ("walk", "home", "work"): 980,
    ("walk", "leisure", "home"): 980,
    ("walk", "leisure", "leisure"): 2450,
    ("walk", "leisure", "other"): 980,
    ("walk", "leisure", "shop"): 980,
    ("walk", "other", "home"): 980,
    ("walk", "other", "leisure"): 1179,
    ("walk", "other", "other"): 980,
    ("walk", "other", "shop"): 590,
    ("walk", "shop", "home"): 780,
    ("walk", "shop", "leisure"): 1234,
    ("walk", "shop", "other"): 980,
    ("walk", "shop", "shop"): 590,
    ("walk", "work", "home"): 1270,
    ("walk", "work", "leisure"): 980,
    ("walk", "work", "other"): 980,
    ("walk", "work", "shop"): 780,
}
MID_TO_DECAY_M: Dict[Tuple[str, str], float] = {  # fallback by (mode, to)
    ("bike", "education"): 1960,
    ("bike", "home"): 2450,
    ("bike", "leisure"): 2940,
    ("bike", "other"): 1960,
    ("bike", "shop"): 1845,
    ("bike", "work"): 3920,
    ("car", "business"): 10671,
    ("car", "education"): 2850,
    ("car", "home"): 6650,
    ("car", "leisure"): 7600,
    ("car", "other"): 5700,
    ("car", "shop"): 3710,
    ("car", "work"): 9500,
    ("pt", "business"): 8456,
    ("pt", "education"): 4500,
    ("pt", "home"): 5850,
    ("pt", "leisure"): 6300,
    ("pt", "other"): 4500,
    ("pt", "shop"): 3600,
    ("pt", "work"): 8820,
    ("walk", "business"): 980,
    ("walk", "education"): 780,
    ("walk", "home"): 980,
    ("walk", "leisure"): 1470,
    ("walk", "other"): 980,
    ("walk", "shop"): 780,
    ("walk", "work"): 980,
}
MID_MODE_DECAY_M: Dict[str, float] = {"bike": 2450, "car": 5910, "pt": 5400, "walk": 980}
# Tour main-mode split (mode of the longest leg), urban — for one-mode-per-tour sampling.
MID_TOUR_MODE_SPLIT: Dict[str, float] = {"car": 0.393, "pt": 0.241, "walk": 0.202, "bike": 0.164}

# --- MiD 2023 RURAL calibration (RegioStaRGem5 == 55, villages) ---------------------------
# Same extraction, rural filter. Contrast vs urban: car median 9.5 km (vs 5.9),
# car tour share 72 % (vs 39 %) — longer, car-dominated. Rural ring of the two-zone world.
MID_RURAL_TEMPLATES: List[Tuple[Tuple[str, ...], float]] = [
    (("home", "work", "home"), 0.1514),
    (("home", "other", "home"), 0.1228),
    (("home", "leisure", "home"), 0.0932),
    (("home", "education", "home"), 0.0769),
    (("home", "shop", "home"), 0.0635),
    (("home", "education", "home", "leisure", "home"), 0.0244),
    (("home", "other", "other", "home"), 0.0213),
    (("home", "leisure", "leisure", "home"), 0.0187),
    (("home", "other", "home", "other", "home"), 0.0172),
    (("home", "work", "home", "leisure", "home"), 0.0166),
    (("home", "other", "home", "home"), 0.0163),
    (("home", "shop", "home", "leisure", "home"), 0.0133),
    (("home", "other", "shop", "home"), 0.0119),
    (("home", "other", "home", "leisure", "home"), 0.0119),
    (("home", "education", "home", "other", "home"), 0.0113),
    (("home", "work", "home", "shop", "home"), 0.0111),
    (("home", "work", "other", "home"), 0.0111),
    (("home", "work", "home", "other", "home"), 0.0110),
    (("home", "leisure", "home", "leisure", "home"), 0.0103),
    (("home", "work", "shop", "home"), 0.0103),
    (("home", "shop", "home", "other", "home"), 0.0096),
    (("home", "other", "leisure", "home"), 0.0092),
    (("home", "other", "other", "other", "home"), 0.0081),
    (("home", "shop", "home", "home"), 0.0075),
    (("home", "leisure", "other", "home"), 0.0074),
    (("home", "work", "leisure", "home"), 0.0066),
    (("home", "leisure", "home", "other", "home"), 0.0064),
    (("home", "leisure", "home", "home"), 0.0064),
    (("home", "shop", "shop", "home"), 0.0058),
    (("home", "education", "leisure", "home"), 0.0057),
    (("home", "work", "home", "work", "home"), 0.0053),
    (("home", "other", "home", "shop", "home"), 0.0051),
    (("home", "work", "home", "home"), 0.0050),
    (("home", "leisure", "shop", "home"), 0.0050),
    (("home", "home", "home"), 0.0048),
    (("home", "education", "other", "home"), 0.0041),
    (("home", "other", "home", "other", "home", "other", "home"), 0.0040),
    (("home", "leisure", "leisure", "leisure", "home"), 0.0039),
    (("home", "other", "other", "home", "home"), 0.0038),
    (("home", "education", "home", "home"), 0.0037),
    (("home", "shop", "other", "home"), 0.0037),
    (("home", "other", "work", "home"), 0.0032),
    (("home", "shop", "leisure", "home"), 0.0031),
    (("home", "business", "home"), 0.0030),
    (("home", "work", "shop", "home", "home"), 0.0029),
    (("home", "shop", "home", "shop", "home"), 0.0027),
    (("home", "education", "home", "shop", "home"), 0.0027),
    (("home", "other", "other", "shop", "home"), 0.0027),
    (("home", "other", "shop", "home", "home"), 0.0025),
    (("home", "other", "home", "work", "home"), 0.0024),
    (("home", "work", "business", "business", "home"), 0.0024),
    (("home", "education", "education", "home", "home"), 0.0023),
    (("home", "other", "home", "other", "home", "leisure", "home"), 0.0022),
    (("home", "other", "shop", "home", "leisure", "home"), 0.0021),
    (("home", "other", "other", "home", "other", "home"), 0.0020),
    (("home", "leisure", "home", "shop", "home"), 0.0020),
    (("home", "other", "work", "other", "home"), 0.0018),
    (("home", "home", "leisure", "home"), 0.0018),
    (("home", "other", "other", "leisure", "home"), 0.0018),
    (("home", "education", "education", "home"), 0.0018),
    (("home", "home", "other", "home"), 0.0017),
    (("home", "leisure", "leisure", "home", "home"), 0.0017),
    (("home", "shop", "leisure", "leisure", "home"), 0.0017),
    (("home", "other", "home", "other", "other", "home"), 0.0017),
    (("home", "home", "home", "home"), 0.0017),
    (("home", "shop", "home", "other", "home", "shop", "home"), 0.0016),
    (("home", "work", "shop", "home", "leisure", "home"), 0.0016),
    (("home", "other", "other", "other", "other", "home"), 0.0016),
    (("home", "education", "other", "other", "home"), 0.0016),
    (("home", "work", "business", "home"), 0.0015),
    (("home", "shop", "shop", "shop", "home"), 0.0015),
    (("home", "other", "home", "other", "home", "home"), 0.0015),
    (("home", "shop", "home", "other", "home", "leisure", "home"), 0.0014),
    (("home", "work", "other", "other", "home"), 0.0014),
    (("home", "work", "work", "home", "home"), 0.0013),
    (("home", "work", "business", "home", "home"), 0.0013),
    (("home", "shop", "shop", "home", "other", "home"), 0.0013),
    (("home", "leisure", "work", "home"), 0.0013),
    (("home", "leisure", "leisure", "other", "home"), 0.0013),
    (("home", "other", "shop", "other", "home"), 0.0013),
    (("home", "work", "home", "other", "other", "home"), 0.0013),
    (("home", "shop", "home", "leisure", "home", "other", "home"), 0.0013),
    (("home", "shop", "home", "work", "home"), 0.0013),
    (("home", "work", "other", "home", "other", "home"), 0.0012),
    (("home", "shop", "other", "home", "leisure", "home"), 0.0012),
    (("home", "leisure", "leisure", "shop", "home"), 0.0012),
    (("home", "education", "home", "other", "other", "home"), 0.0012),
    (("home", "work", "other", "home", "home"), 0.0012),
    (("home", "work", "home", "other", "home", "leisure", "home"), 0.0011),
    (("home", "other", "home", "shop", "home", "leisure", "home"), 0.0011),
    (("home", "education", "home", "leisure", "home", "leisure", "home"), 0.0011),
    (("home", "work", "home", "shop", "home", "leisure", "home"), 0.0011),
    (("home", "other", "other", "home", "leisure", "home"), 0.0011),
    (("home", "shop", "home", "other", "home", "other", "home"), 0.0010),
    (("home", "other", "shop", "home", "other", "home"), 0.0010),
    (("home", "education", "home", "leisure", "home", "other", "home"), 0.0010),
    (("home", "shop", "home", "leisure", "leisure", "home"), 0.0010),
    (("home", "education", "home", "other", "home", "leisure", "home"), 0.0010),
    (("home", "education", "other", "home", "home"), 0.0010),
    (("home", "leisure", "other", "home", "home"), 0.0010),
    (("home", "other", "shop", "shop", "home"), 0.0010),
    (("home", "other", "home", "other", "home", "shop", "home"), 0.0009),
    (("home", "shop", "other", "leisure", "home"), 0.0009),
    (("home", "work", "work", "home"), 0.0009),
    (("home", "education", "shop", "home"), 0.0009),
    (("home", "education", "home", "leisure", "leisure", "home"), 0.0008),
    (("home", "work", "home", "other", "home", "other", "home"), 0.0008),
    (("home", "leisure", "home", "shop", "home", "leisure", "home"), 0.0008),
    (("home", "other", "home", "business", "home"), 0.0008),
    (("home", "other", "home", "shop", "home", "other", "home"), 0.0008),
    (("home", "shop", "other", "shop", "home"), 0.0008),
    (("home", "leisure", "shop", "home", "home"), 0.0008),
    (("home", "leisure", "leisure", "leisure", "leisure", "home"), 0.0008),
    (("home", "other", "leisure", "leisure", "home"), 0.0008),
    (("home", "other", "other", "work", "home"), 0.0008),
    (("home", "leisure", "home", "leisure", "home", "leisure", "home"), 0.0008),
    (("home", "work", "home", "home", "home"), 0.0008),
    (("home", "shop", "other", "other", "home"), 0.0008),
    (("home", "shop", "home", "other", "other", "home"), 0.0007),
    (("home", "shop", "other", "home", "home"), 0.0007),
    (("home", "business", "home", "leisure", "home"), 0.0007),
    (("home", "leisure", "other", "other", "home"), 0.0007),
    (("home", "leisure", "home", "other", "home", "leisure", "home"), 0.0007),
    (("home", "other", "leisure", "other", "home"), 0.0007),
    (("home", "work", "home", "shop", "home", "home"), 0.0007),
    (("home", "education", "education", "home", "leisure", "home"), 0.0007),
    (("home", "leisure", "leisure", "home", "leisure", "home"), 0.0007),
    (("home", "leisure", "home", "work", "home"), 0.0007),
    (("home", "other", "home", "leisure", "leisure", "home"), 0.0006),
    (("home", "work", "home", "leisure", "shop", "home"), 0.0006),
    (("home", "business", "business", "home"), 0.0006),
    (("home", "other", "home", "leisure", "home", "home"), 0.0006),
    (("home", "work", "leisure", "leisure", "home"), 0.0006),
    (("home", "other", "education", "home"), 0.0006),
    (("home", "work", "other", "home", "leisure", "home"), 0.0006),
    (("home", "leisure", "home", "leisure", "home", "other", "home"), 0.0006),
    (("home", "education", "other", "home", "leisure", "home"), 0.0006),
    (("home", "work", "home", "other", "leisure", "home"), 0.0006),
    (("home", "work", "other", "shop", "home"), 0.0006),
    (("home", "education", "home", "leisure", "home", "home"), 0.0006),
    (("home", "shop", "shop", "home", "home"), 0.0006),
    (("home", "other", "other", "other", "home", "home"), 0.0006),
    (("home", "work", "home", "other", "shop", "home"), 0.0006),
    (("home", "work", "home", "shop", "home", "other", "home"), 0.0006),
    (("home", "other", "home", "home", "home"), 0.0006),
    (("home", "shop", "shop", "other", "home"), 0.0006),
    (("home", "other", "other", "other", "other", "other", "home"), 0.0005),
    (("home", "other", "home", "work", "home", "other", "home"), 0.0005),
    (("home", "leisure", "other", "home", "leisure", "home"), 0.0005),
    (("home", "shop", "leisure", "shop", "home"), 0.0005),
    (("home", "leisure", "shop", "home", "other", "home"), 0.0005),
    (("home", "shop", "shop", "leisure", "home"), 0.0005),
    (("home", "work", "other", "home", "shop", "home"), 0.0005),
    (("home", "other", "shop", "home", "shop", "home"), 0.0005),
    (("home", "education", "home", "leisure", "other", "home"), 0.0005),
    (("home", "home", "shop", "home"), 0.0005),
    (("home", "other", "home", "leisure", "other", "home"), 0.0005),
    (("home", "business", "business", "business", "home"), 0.0005),
    (("home", "work", "home", "other", "home", "home"), 0.0005),
    (("home", "leisure", "home", "leisure", "leisure", "home"), 0.0005),
    (("home", "other", "work", "home", "other", "home"), 0.0005),
    (("home", "shop", "home", "other", "home", "home"), 0.0005),
    (("home", "work", "home", "shop", "leisure", "home"), 0.0005),
    (("home", "work", "home", "leisure", "leisure", "home"), 0.0004),
    (("home", "work", "home", "leisure", "other", "home"), 0.0004),
    (("home", "work", "other", "work", "home"), 0.0004),
    (("home", "other", "home", "other", "leisure", "home"), 0.0004),
    (("home", "work", "shop", "home", "other", "home"), 0.0004),
    (("home", "shop", "home", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "other", "other", "other", "home", "other", "home"), 0.0004),
    (("home", "education", "education", "leisure", "home"), 0.0004),
    (("home", "other", "home", "leisure", "home", "other", "home"), 0.0004),
    (("home", "education", "home", "education", "home"), 0.0004),
    (("home", "work", "home", "leisure", "home", "leisure", "home"), 0.0004),
    (("home", "leisure", "shop", "home", "leisure", "home"), 0.0004),
    (("home", "other", "other", "home", "shop", "home"), 0.0004),
    (("home", "work", "home", "shop", "shop", "home"), 0.0004),
    (("home", "other", "home", "other", "other", "other", "home"), 0.0003),
    (("home", "shop", "work", "home"), 0.0003),
    (("home", "leisure", "leisure", "leisure", "home", "leisure", "home"), 0.0003),
    (("home", "other", "leisure", "home", "home"), 0.0003),
    (("home", "other", "work", "home", "leisure", "home"), 0.0003),
    (("home", "leisure", "leisure", "leisure", "leisure", "leisure", "home"), 0.0003),
    (("home", "home", "home", "home", "home"), 0.0003),
    (("home", "leisure", "other", "shop", "home"), 0.0003),
    (("home", "shop", "home", "leisure", "other", "home"), 0.0003),
    (("home", "leisure", "home", "other", "home", "other", "home"), 0.0003),
    (("home", "leisure", "shop", "shop", "home"), 0.0003),
    (("home", "shop", "home", "leisure", "home", "home"), 0.0003),
    (("home", "shop", "home", "shop", "home", "other", "home"), 0.0003),
    (("home", "work", "home", "work", "home", "leisure", "home"), 0.0003),
    (("home", "work", "home", "business", "home"), 0.0003),
    (("home", "other", "leisure", "home", "other", "home"), 0.0003),
    (("home", "leisure", "other", "home", "other", "home"), 0.0003),
    (("home", "leisure", "other", "leisure", "home"), 0.0003),
    (("home", "other", "home", "leisure", "home", "leisure", "home"), 0.0003),
    (("home", "other", "home", "other", "shop", "home"), 0.0002),
    (("home", "home", "other", "home", "other", "home"), 0.0002),
    (("home", "shop", "other", "home", "other", "home"), 0.0002),
    (("home", "shop", "shop", "home", "leisure", "home"), 0.0002),
    (("home", "other", "leisure", "shop", "home"), 0.0002),
    (("home", "other", "home", "shop", "home", "home"), 0.0002),
    (("home", "work", "leisure", "shop", "home"), 0.0002),
    (("home", "other", "shop", "leisure", "home"), 0.0002),
    (("home", "work", "shop", "other", "home"), 0.0002),
    (("home", "education", "other", "leisure", "home"), 0.0002),
    (("home", "work", "home", "shop", "other", "home"), 0.0002),
    (("home", "shop", "home", "shop", "home", "shop", "home"), 0.0002),
    (("home", "leisure", "leisure", "home", "other", "home"), 0.0002),
    (("home", "education", "home", "other", "home", "other", "home"), 0.0002),
    (("home", "work", "shop", "shop", "home"), 0.0002),
    (("home", "work", "home", "other", "home", "shop", "home"), 0.0002),
    (("home", "leisure", "home", "other", "other", "home"), 0.0002),
    (("home", "other", "other", "other", "leisure", "home"), 0.0002),
    (("home", "other", "home", "leisure", "leisure", "leisure", "home"), 0.0002),
    (("home", "shop", "home", "shop", "home", "leisure", "home"), 0.0001),
    (("home", "business", "home", "other", "home"), 0.0001),
    (("home", "other", "shop", "shop", "home", "other", "home"), 0.0001),
]
MID_RURAL_PAIR_DECAY_M: Dict[Tuple[str, str, str], float] = {
    ("bike", "education", "home"): 1960,
    ("bike", "home", "education"): 1749,
    ("bike", "home", "home"): 1960,
    ("bike", "home", "leisure"): 1960,
    ("bike", "home", "other"): 2021,
    ("bike", "home", "shop"): 1260,
    ("bike", "home", "work"): 2450,
    ("bike", "leisure", "home"): 1960,
    ("bike", "other", "home"): 2160,
    ("bike", "shop", "home"): 1270,
    ("bike", "work", "home"): 2450,
    ("car", "business", "business"): 23750,
    ("car", "business", "home"): 12350,
    ("car", "education", "home"): 5700,
    ("car", "education", "other"): 7600,
    ("car", "home", "business"): 14250,
    ("car", "home", "education"): 4750,
    ("car", "home", "home"): 9500,
    ("car", "home", "leisure"): 9500,
    ("car", "home", "other"): 7600,
    ("car", "home", "shop"): 6180,
    ("car", "home", "work"): 14250,
    ("car", "leisure", "home"): 9500,
    ("car", "leisure", "leisure"): 10450,
    ("car", "leisure", "other"): 9500,
    ("car", "leisure", "shop"): 4750,
    ("car", "other", "home"): 8550,
    ("car", "other", "leisure"): 8550,
    ("car", "other", "other"): 6650,
    ("car", "other", "shop"): 3800,
    ("car", "other", "work"): 14250,
    ("car", "shop", "home"): 6650,
    ("car", "shop", "leisure"): 4750,
    ("car", "shop", "other"): 4750,
    ("car", "shop", "shop"): 3800,
    ("car", "work", "business"): 19050,
    ("car", "work", "home"): 15200,
    ("car", "work", "leisure"): 13286,
    ("car", "work", "other"): 10450,
    ("car", "work", "shop"): 4750,
    ("pt", "education", "education"): 8798,
    ("pt", "education", "home"): 9000,
    ("pt", "home", "education"): 9000,
    ("pt", "home", "home"): 9900,
    ("pt", "home", "leisure"): 18000,
    ("pt", "home", "other"): 10800,
    ("pt", "home", "work"): 22500,
    ("pt", "leisure", "home"): 18000,
    ("pt", "other", "home"): 10800,
    ("pt", "work", "home"): 18000,
    ("walk", "education", "home"): 880,
    ("walk", "home", "education"): 980,
    ("walk", "home", "home"): 980,
    ("walk", "home", "leisure"): 1370,
    ("walk", "home", "other"): 980,
    ("walk", "home", "shop"): 590,
    ("walk", "home", "work"): 980,
    ("walk", "leisure", "home"): 980,
    ("walk", "leisure", "leisure"): 2940,
    ("walk", "other", "home"): 980,
    ("walk", "other", "leisure"): 1960,
    ("walk", "other", "other"): 980,
    ("walk", "other", "shop"): 490,
    ("walk", "shop", "home"): 690,
    ("walk", "work", "home"): 690,
}
MID_RURAL_TO_DECAY_M: Dict[Tuple[str, str], float] = {
    ("bike", "education"): 1701,
    ("bike", "home"): 1960,
    ("bike", "leisure"): 1960,
    ("bike", "other"): 1960,
    ("bike", "shop"): 1180,
    ("bike", "work"): 2450,
    ("car", "business"): 17100,
    ("car", "education"): 4750,
    ("car", "home"): 9500,
    ("car", "leisure"): 9500,
    ("car", "other"): 7600,
    ("car", "shop"): 5320,
    ("car", "work"): 14257,
    ("pt", "education"): 9000,
    ("pt", "home"): 10800,
    ("pt", "leisure"): 13500,
    ("pt", "other"): 9000,
    ("pt", "work"): 20700,
    ("walk", "education"): 980,
    ("walk", "home"): 980,
    ("walk", "leisure"): 1598,
    ("walk", "other"): 980,
    ("walk", "shop"): 590,
    ("walk", "work"): 980,
}
MID_RURAL_MODE_DECAY_M: Dict[str, float] = {"bike": 1960, "car": 9500, "pt": 10800, "walk": 980}
MID_RURAL_TOUR_MODE_SPLIT: Dict[str, float] = {"car": 0.717, "walk": 0.129, "pt": 0.094, "bike": 0.06}


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
    heterogeneity: float = 0.0,
    cluster_centers: Optional[np.ndarray] = None,
    spread_scale: float = 1.0,
    type_prevalence: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Topology:
    """Build a clustered multi-type topology. `heterogeneity` ∈ [0,1] makes the world a
    dense, attractive **core in a sparse surround** (vs the default ~uniform clustering):
    cluster centres are pulled toward the box centre and latent `sizes` decay radially, so a
    centred study window sits in a city-like core. 0 = homogeneous (legacy behaviour).
    `cluster_centers` ((k,2) array) places the blobs explicitly (e.g. towns on a ring around a
    city) instead of uniform-random; overrides `n_clusters`/`heterogeneity` centre placement."""
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
    het = float(np.clip(heterogeneity, 0.0, 1.0))
    gc = box / 2.0
    if cluster_centers is not None:
        centers = np.asarray(cluster_centers, dtype=float)
        n_clusters = len(centers)
    else:
        centers = rng.uniform(margin, box - margin, size=(n_clusters, 2))
        if het > 0:
            centers = centers + het * (gc - centers)   # pull clusters toward the centre
    which = rng.integers(0, n_clusters, size=n_locations)
    # `spread_scale` > 1 widens each blob so neighbouring clusters merge into one contiguous mass
    # (a single city) rather than reading as separate villages.
    spread = spread_scale * box / (n_clusters * 1.5)
    coords = centers[which] + rng.normal(0, spread, size=(n_locations, 2))
    sizes = np.exp(rng.normal(0, 0.7, size=n_locations))
    if het > 0:
        # Decay attractiveness with distance from the centre -> central core is more attractive.
        r = np.hypot(coords[:, 0] - gc, coords[:, 1] - gc) / (box / 2.0)
        sizes *= np.exp(-2.0 * het * r)

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


def add_type_alias(topo: Topology, new: str, like: str) -> Topology:
    """Return a copy of `topo` with activity type `new` added, sharing `like`'s candidate pool
    (same location indices). Used to place ``business`` (dienstlich) trips against the ``work``
    catalog: a business stop is a workplace visit, so its facilities ARE the work facilities, but
    it is a distinct chain/decay identity and accrues its own visit-count potential column."""
    if new in topo.type_idx:
        return topo
    if like not in topo.type_locs:
        raise KeyError(f"cannot alias {new!r} to absent type {like!r}")
    types = list(topo.types) + [new]
    type_idx = {**topo.type_idx, new: len(topo.types)}
    type_locs = {**topo.type_locs, new: topo.type_locs[like]}
    return Topology(topo.coords, topo.sizes, topo.loc_ids, types, type_idx, type_locs, topo.box)


def generate_chains(
    topo: Topology,
    n_persons: int,
    *,
    sizes: Optional[np.ndarray] = None,
    gravity_scale: float = 4000.0,
    distance_noise: float = 0.0,
    assign_modes: bool = True,
    mode_decay: Optional[Dict[str, float]] = None,
    pair_decay: Optional[Dict[Tuple[str, str, str], float]] = None,
    to_decay: Optional[Dict[Tuple[str, str], float]] = None,
    mode_split: Optional[Dict[str, float]] = None,
    decay_inflate: "float | Dict[str, float]" = 1.0,
    tail_frac: Optional[Dict[str, float]] = None,
    tail_scale: Optional[Dict[str, float]] = None,
    car_ownership: float = 0.6,
    bike_ownership: float = 0.5,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    templates: Optional[List[Tuple[Tuple[str, ...], float]]] = None,
    home_pool: Optional[np.ndarray] = None,
    person_prefix: str = "p",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[dict], List[dict], np.ndarray]:
    """Generate activity chains on `topo` using attractiveness `sizes` (defaults to
    topo.sizes). Returns (plan_rows, ground_truth_rows, visits[(n_loc, n_type)]).

    `home_pool` restricts home selection to a subset of location indices (e.g. one zone of a
    multi-zone topology); defaults to all home-offering locations. `person_prefix` namespaces
    the person/leg ids so several sub-populations can be merged without collision.

    Distance model: each free stop is gravity-placed P(loc) ∝ size·exp(-d/scale), `scale` from
    `pair_decay`/`to_decay`/`mode_decay` × `decay_inflate` (or flat `mode_decay`). `mode_split`
    sets the tour mode (else the ownership heuristic).

    `tail_frac`/`tail_scale` add a heavy tail (local-vs-global sampler, à la eqasim/hEART): with
    per-mode probability `tail_frac[mode]` a stop is drawn with the long-range `tail_scale[mode]`
    instead of its local decay, producing realistic long car/PT trips the thin exponential kernel
    can't. Needs a big-enough box to have far destinations. `None` => local-only (legacy)."""
    rng = rng or np.random.default_rng()
    sizes = topo.sizes if sizes is None else np.asarray(sizes, dtype=float)
    mode_decay = dict(mode_decay or DEFAULT_MODE_DECAY)
    templates = templates or DEFAULT_TEMPLATES
    anchors = tuple(anchors)
    coords, type_locs, type_idx, loc_ids = topo.coords, topo.type_locs, topo.type_idx, topo.loc_ids

    def _draw_tour_mode() -> str:
        if mode_split is not None:
            modes = list(mode_split)
            p = np.array([mode_split[m] for m in modes], dtype=float)
            return modes[int(rng.choice(len(modes), p=p / p.sum()))]
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

    def _inflate(mode: Optional[str]) -> float:
        return decay_inflate.get(mode, 1.0) if isinstance(decay_inflate, dict) else decay_inflate

    def _leg_scale(mode: Optional[str], frm: str, to: str, base: float) -> float:
        if pair_decay is None:
            return base
        s = pair_decay.get((mode, frm, to))
        if s is None and to_decay is not None:
            s = to_decay.get((mode, to))
        if s is None:
            s = mode_decay.get(mode, base)
        return float(s) * _inflate(mode)

    visits = np.zeros((coords.shape[0], len(topo.types)))
    tmpl_seqs = [seq for seq, _ in templates]
    tmpl_w = np.array([w for _, w in templates], dtype=float)
    tmpl_w /= tmpl_w.sum()
    home_idx = type_locs["home"] if home_pool is None else np.asarray(home_pool, dtype=int)

    plan_rows: List[dict] = []
    gt_rows: List[dict] = []
    for pi in range(n_persons):
        seq = tmpl_seqs[rng.choice(len(tmpl_seqs), p=tmpl_w)]
        tour_mode = _draw_tour_mode() if assign_modes else None
        scale = mode_decay[tour_mode] if assign_modes else gravity_scale
        home_i = int(home_idx[_weighted_choice(rng, sizes[home_idx])])

        placed_coord: List[np.ndarray] = []
        placed_id: List[int] = []
        prev = home_i
        prev_type = "home"  # templates start at home
        for t in seq:
            if t == "home":
                li = home_i
            else:
                pool = type_locs[t]
                d = np.hypot(coords[pool, 0] - coords[prev, 0], coords[pool, 1] - coords[prev, 1])
                leg_scale = _leg_scale(tour_mode, prev_type, t, scale)
                if tail_frac is not None and rng.random() < tail_frac.get(tour_mode, 0.0):
                    # long-range draw: far-biased (d^2) so it overcomes the dense central pool and
                    # reaches distant destinations, capped by tail_scale to stay within the box.
                    ts = (tail_scale or {}).get(tour_mode, 40000.0)
                    w = sizes[pool] * d * d * np.exp(-d / ts)
                else:
                    w = sizes[pool] * np.exp(-d / leg_scale)
                li = int(pool[_weighted_choice(rng, w)])
            placed_id.append(li)
            placed_coord.append(coords[li])
            visits[li, type_idx[t]] += 1
            prev = li
            prev_type = t

        for k in range(1, len(seq)):
            frm, to = placed_coord[k - 1], placed_coord[k]
            from_anchor = seq[k - 1] in anchors
            to_anchor = seq[k] in anchors
            lid = f"{person_prefix}{pi}-l{k}"
            true_d = float(np.hypot(to[0] - frm[0], to[1] - frm[1]))
            obs_d = true_d
            if distance_noise > 0:
                obs_d = max(1.0, true_d * (1.0 + float(rng.normal(0, distance_noise))))
            plan_rows.append({
                "unique_person_id": f"{person_prefix}{pi}", "unique_leg_id": lid, "to_act_type": seq[k],
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
    heterogeneity: float = 0.0,
    spread_scale: float = 1.0,
    distance_noise: float = 0.0,
    assign_modes: bool = True,
    mode_decay: Optional[Dict[str, float]] = None,
    pair_decay: Optional[Dict[Tuple[str, str, str], float]] = None,
    to_decay: Optional[Dict[Tuple[str, str], float]] = None,
    mode_split: Optional[Dict[str, float]] = None,
    decay_inflate: "float | Dict[str, float]" = 1.0,
    tail_frac: Optional[Dict[str, float]] = None,
    tail_scale: Optional[Dict[str, float]] = None,
    car_ownership: float = 0.6,
    bike_ownership: float = 0.5,
    type_prevalence: Optional[Dict[str, float]] = None,
    type_aliases: Optional[Dict[str, str]] = None,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    templates: Optional[List[Tuple[Tuple[str, ...], float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticWorld:
    """Compose a topology + chains into a ready-to-use world. Potentials = visit counts
    (usage). `distance_noise` adds relative noise to observed distances; `assign_modes`
    gives each tour a mode. `pair_decay`/`to_decay`/`mode_split` enable the MiD-calibrated
    per-(mode,from→to) distance + observed mode-share path (see `generate_chains`).
    `type_aliases` (e.g. ``{"business": "work"}``) adds placement-alias types sharing another
    type's candidate pool — needed when templates reference a type with no facilities of its own."""
    rng = rng or np.random.default_rng()
    topo = build_topology(n_locations=n_locations, box=box, density_per_km2=density_per_km2,
                          n_clusters=n_clusters, heterogeneity=heterogeneity,
                          spread_scale=spread_scale, type_prevalence=type_prevalence, rng=rng)
    for new, like in (type_aliases or {}).items():
        topo = add_type_alias(topo, new, like)
    return world_from_topology(
        topo, n_persons, gravity_scale=gravity_scale, distance_noise=distance_noise,
        assign_modes=assign_modes, mode_decay=mode_decay, pair_decay=pair_decay,
        to_decay=to_decay, mode_split=mode_split, decay_inflate=decay_inflate,
        tail_frac=tail_frac, tail_scale=tail_scale, car_ownership=car_ownership,
        bike_ownership=bike_ownership, anchors=anchors, templates=templates, rng=rng,
    )


def world_from_topology(
    topo: Topology,
    n_persons: int = 500,
    *,
    sizes: Optional[np.ndarray] = None,
    gravity_scale: float = 4000.0,
    distance_noise: float = 0.0,
    assign_modes: bool = True,
    mode_decay: Optional[Dict[str, float]] = None,
    pair_decay: Optional[Dict[Tuple[str, str, str], float]] = None,
    to_decay: Optional[Dict[Tuple[str, str], float]] = None,
    mode_split: Optional[Dict[str, float]] = None,
    decay_inflate: "float | Dict[str, float]" = 1.0,
    tail_frac: Optional[Dict[str, float]] = None,
    tail_scale: Optional[Dict[str, float]] = None,
    car_ownership: float = 0.6,
    bike_ownership: float = 0.5,
    anchors: Sequence[str] = DEFAULT_ANCHORS,
    templates: Optional[List[Tuple[Tuple[str, ...], float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticWorld:
    """Generate chains on a **pre-built** topology and assemble a `SyntheticWorld`. Lets a
    caller hand in a real-geometry topology (e.g. `osm.topology_from_pois`) instead of the
    Gaussian-blob one `generate_world` builds. Potentials = realized visit counts."""
    rng = rng or np.random.default_rng()
    plan_rows, gt_rows, visits = generate_chains(
        topo, n_persons, sizes=sizes, gravity_scale=gravity_scale, distance_noise=distance_noise,
        assign_modes=assign_modes, mode_decay=mode_decay, pair_decay=pair_decay,
        to_decay=to_decay, mode_split=mode_split, decay_inflate=decay_inflate,
        tail_frac=tail_frac, tail_scale=tail_scale, car_ownership=car_ownership,
        bike_ownership=bike_ownership, anchors=anchors, templates=templates, rng=rng,
    )
    ids_d, coords_d, pots_d = {}, {}, {}
    for t in topo.types:
        idx = topo.type_locs[t]
        ids_d[t] = topo.loc_ids[idx]
        coords_d[t] = topo.coords[idx].astype(float)
        pots_d[t] = visits[idx, topo.type_idx[t]].astype(float)

    n_loc = topo.coords.shape[0]
    return SyntheticWorld(
        locations_tuple=(ids_d, coords_d, pots_d),
        plans_df=pd.DataFrame(plan_rows),
        ground_truth=pd.DataFrame(gt_rows),
        anchor_types=tuple(anchors),
        topology=topo,
        meta={"n_locations": n_loc, "n_persons": n_persons, "types": topo.types,
              "box": topo.box,
              "density_per_km2": n_loc / (topo.box / 1000.0) ** 2 if topo.box else float("nan"),
              "n_legs": len(plan_rows),
              "n_free_legs": int(sum(r["to_is_free"] for r in gt_rows))},
    )


# --- Two-zone (urban core + rural ring) world --------------------------------------------
# Per-zone MiD behaviour bundles. decay_inflate is tuned per zone (see two_zone_world docstring
# / synthetic_world_calibration.md) so each zone's realized leg medians match its MiD subset.
# Tuned on the 45 km two-zone world so each zone's realized per-mode leg medians match its MiD
# subset: urban core -> walk 992/980, bike 2498/2450, car 5969/5910, pt 5125/5400; rural ring ->
# walk 980, bike 1972/1960, car 9830/9500, pt 11853/10800. The big box hosts the long rural trips,
# so (unlike the single-city presets) car/pt are not truncated.
TWO_ZONE_URBAN_INFLATE: Dict[str, float] = {"walk": 0.69, "bike": 1.08, "car": 1.78, "pt": 1.46}
TWO_ZONE_RURAL_INFLATE: Dict[str, float] = {"walk": 0.55, "bike": 0.68, "car": 1.0, "pt": 1.0}
# Heavy-tail (long car/PT trips) params for the local-vs-global sampler, tuned on an ~80 km
# super-region so realized car/PT p90 approach MiD (~27-31 km vs MiD 31; full p95=52 km needs a
# bigger box still). OPT-IN via two_zone_world(heavy_tail=True): adding inter-regional trips lifts
# the median, so the local-only median calibration above is for heavy_tail=False. Needs a box big
# enough to hold the long trips (>=~70 km); the flag bumps the default box accordingly.
TWO_ZONE_TAIL_SCALE: Dict[str, float] = {"car": 60000.0, "pt": 60000.0}
# heavy_tail uses a 120 km box (the long trips land mid-ring, not against the wall -> no border
# pile-up) with the core held at ~16 km. The bigger box lets car/PT run longer, so inflate + tail
# are jointly retuned at 120 km (walk/bike reuse the local values). Result: BOTH the median and the
# p90 match MiD — urban car p50 6.0/5.8, p90 30/31; rural car 8.7/8.6, 33/33; pt likewise.
HEAVY_TAIL_BOX = 120000.0
HEAVY_TAIL_CORE_FRAC = 0.13
HEAVY_TAIL_RING_DENSITY = 4.0
HEAVY_TAIL_URBAN_INFLATE: Dict[str, float] = {"walk": 0.58, "bike": 0.83, "car": 1.34, "pt": 1.32}
HEAVY_TAIL_RURAL_INFLATE: Dict[str, float] = {"walk": 0.55, "bike": 0.62, "car": 0.88, "pt": 0.60}
TWO_ZONE_TAIL_FRAC_URBAN: Dict[str, float] = {"car": 0.094, "pt": 0.079}
TWO_ZONE_TAIL_FRAC_RURAL: Dict[str, float] = {"car": 0.075, "pt": 0.069}
URBAN_BENCH = dict(templates=MID_URBAN_TEMPLATES, pair_decay=MID_PAIR_DECAY_M,
                   to_decay=MID_TO_DECAY_M, mode_decay=MID_MODE_DECAY_M,
                   mode_split=MID_TOUR_MODE_SPLIT, decay_inflate=TWO_ZONE_URBAN_INFLATE)
RURAL_BENCH = dict(templates=MID_RURAL_TEMPLATES, pair_decay=MID_RURAL_PAIR_DECAY_M,
                   to_decay=MID_RURAL_TO_DECAY_M, mode_decay=MID_RURAL_MODE_DECAY_M,
                   mode_split=MID_RURAL_TOUR_MODE_SPLIT, decay_inflate=TWO_ZONE_RURAL_INFLATE)


def merge_topologies(*topos: Topology, box: Optional[float] = None) -> Topology:
    """Concatenate topologies that already share a coordinate frame into one (re-indexing
    type_locs). Used to embed a dense core inside a sparse surround."""
    coords = np.vstack([t.coords for t in topos])
    sizes = np.concatenate([t.sizes for t in topos])
    types = list(dict.fromkeys(t for topo in topos for t in topo.types))
    type_locs: Dict[str, list] = {t: [] for t in types}
    off = 0
    for topo in topos:
        for t in topo.types:
            type_locs[t].append(topo.type_locs[t] + off)
        off += topo.coords.shape[0]
    type_locs = {t: (np.concatenate(v) if v else np.array([], dtype=int)) for t, v in type_locs.items()}
    loc_ids = np.array([f"loc{i}" for i in range(coords.shape[0])], dtype=object)
    box = float(box if box is not None else coords.max())
    return Topology(coords, sizes, loc_ids, types, {t: j for j, t in enumerate(types)}, type_locs, box)


def two_zone_world(
    *,
    n_persons: Optional[int] = None,
    box: Optional[float] = None,
    core_frac_side: Optional[float] = None,
    core_density: float = 93.0,   # urban facilities/km^2 (Hannover-like core)
    ring_density: Optional[float] = None,   # rural facilities/km^2 (sparse surround)
    rural_pop_share: float = 0.3,
    urban_mobile_density: float = 2203.0,   # mobile persons/km^2 in the core (Hannover-like)
    rural_mobile_density: float = 80.0,     # mobile persons/km^2 in the ring (rural Niedersachsen)
    heavy_tail: bool = False,
    urban_params: Optional[dict] = None,
    rural_params: Optional[dict] = None,
    n_clusters_core: int = 8,
    n_clusters_ring: int = 28,
    type_prevalence: Optional[Dict[str, float]] = None,
    distance_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticWorld:
    """A **dense urban core embedded in a sparse rural ring**, each calibrated to its own MiD
    subset: core persons (homes in the centre) use the urban templates/decay/mode-split, ring
    persons (homes in the surround) use the rural ones (longer, car-dominated). Facility density
    steps from `core_density` (centre) to `ring_density` (everywhere). A national survey over this
    world is a true urban+rural mixture, so applying it to the urban study core is a genuine
    transfer mismatch — the realistic version of the heterogeneity gradient. `decay_inflate` per
    zone (`TWO_ZONE_URBAN_INFLATE`/`TWO_ZONE_RURAL_INFLATE`) makes each zone's realized leg medians
    match its MiD subset.

    `n_persons=None` (default) builds the **full mobile population**: each zone's area times its
    mobile persons/km^2 (`urban_mobile_density` core, `rural_mobile_density` ring), so the realized
    urban/rural split follows real density (the ring dominates by area — a national survey over it
    is genuinely rural-heavy, the realistic transfer mismatch) and the potentials are a dense usage
    field. Draw a survey subset to solve. Passing an explicit `n_persons` instead splits it by
    `rural_pop_share` (a small sample build, decoupled from facility density).

    `heavy_tail=True` adds the long car/PT inter-regional tail (`TWO_ZONE_TAIL_*`) so realized p90s
    approach MiD (~30 km); it defaults `box` to 80 km (the long trips need somewhere to go) and
    mildly lifts the median. `box` defaults to 45 km otherwise."""
    rng = rng or np.random.default_rng()
    # heavy_tail switches to the enlarged super-region (120 km, ~16 km core, sparse ring) with its
    # own car/PT inflate + the long-range tail; otherwise the 45 km city-region.
    if box is None:
        box = HEAVY_TAIL_BOX if heavy_tail else 45000.0
    if core_frac_side is None:
        core_frac_side = HEAVY_TAIL_CORE_FRAC if heavy_tail else 0.33
    if ring_density is None:
        ring_density = HEAVY_TAIL_RING_DENSITY if heavy_tail else 8.0
    # Calibrated (real-OSM-matched) type mix in BOTH zones; per-zone density differs via
    # core_density vs ring_density, so work is urban-dense / rural-sparse. (DEFAULT_TYPE_PREVALENCE
    # would put work at 0.25 — 3x the real proportion — so we default to the Hannover mix.)
    if type_prevalence is None:
        type_prevalence = HANNOVER_TYPE_PREVALENCE
    if heavy_tail:
        tail_u = {"tail_frac": TWO_ZONE_TAIL_FRAC_URBAN, "tail_scale": TWO_ZONE_TAIL_SCALE,
                  "decay_inflate": HEAVY_TAIL_URBAN_INFLATE}
        tail_r = {"tail_frac": TWO_ZONE_TAIL_FRAC_RURAL, "tail_scale": TWO_ZONE_TAIL_SCALE,
                  "decay_inflate": HEAVY_TAIL_RURAL_INFLATE}
    else:
        tail_u = tail_r = {}
    up = {**URBAN_BENCH, **tail_u, **(urban_params or {})}
    rp = {**RURAL_BENCH, **tail_r, **(rural_params or {})}
    core_side = box * core_frac_side
    core_n = max(1, round(core_density * (core_side / 1000.0) ** 2))
    ring_n = max(1, round(ring_density * (box / 1000.0) ** 2))

    # Homogeneous core (its own blobs slightly spread so the central city reads as one soft mass).
    core = build_topology(n_locations=core_n, box=core_side, n_clusters=n_clusters_core,
                          spread_scale=1.3, type_prevalence=type_prevalence, rng=rng)
    off = (box - core_side) / 2.0
    core = Topology(core.coords + off, core.sizes, core.loc_ids, core.types, core.type_idx,
                    core.type_locs, box)
    # Rural ring: MANY uniformly-scattered towns (not a tight annulus) so facilities cover all
    # radii continuously -> the distance distribution stays smooth (a sparse annulus leaves a
    # radial gap that makes distances bimodal), while reading as a dense central city dotted with
    # towns rather than a streak. `spread_scale` blends the towns a little.
    ring = build_topology(n_locations=ring_n, box=box, n_clusters=n_clusters_ring, spread_scale=1.4,
                          type_prevalence=type_prevalence, rng=rng)
    topo = merge_topologies(core, ring, box=box)
    topo = add_type_alias(topo, "business", "work")   # MiD business trips -> work catalog
    n_core = core.coords.shape[0]
    home = topo.type_locs["home"]
    core_home, ring_home = home[home < n_core], home[home >= n_core]

    core_area_km2 = (core_side / 1000.0) ** 2
    ring_area_km2 = (box / 1000.0) ** 2 - core_area_km2
    if n_persons is None:   # full mobile population: each zone's area * its mobile persons/km^2
        n_urban = int(round(core_area_km2 * urban_mobile_density))
        n_rural = int(round(ring_area_km2 * rural_mobile_density))
        n_persons = n_urban + n_rural
    else:                   # explicit total -> split by rural_pop_share (sample build)
        n_rural = int(round(rural_pop_share * n_persons))
        n_urban = n_persons - n_rural
    rows_u, gt_u, vis_u = generate_chains(topo, n_urban, home_pool=core_home, person_prefix="u",
                                          distance_noise=distance_noise, rng=rng, **up)
    rows_r, gt_r, vis_r = generate_chains(topo, n_rural, home_pool=ring_home, person_prefix="r",
                                          distance_noise=distance_noise, rng=rng, **rp)
    visits = vis_u + vis_r
    plan_rows, gt_rows = rows_u + rows_r, gt_u + gt_r

    ids_d, coords_d, pots_d = {}, {}, {}
    for t in topo.types:
        idx = topo.type_locs[t]
        ids_d[t] = topo.loc_ids[idx]
        coords_d[t] = topo.coords[idx].astype(float)
        pots_d[t] = visits[idx, topo.type_idx[t]].astype(float)
    return SyntheticWorld(
        locations_tuple=(ids_d, coords_d, pots_d),
        plans_df=pd.DataFrame(plan_rows), ground_truth=pd.DataFrame(gt_rows),
        anchor_types=DEFAULT_ANCHORS, topology=topo,
        meta={"n_locations": topo.coords.shape[0], "n_persons": n_persons, "types": topo.types,
              "box": box, "core_side": core_side, "n_core_fac": n_core,
              "n_urban": n_urban, "n_rural": n_rural,
              "core_density": core_density, "ring_density": ring_density,
              "n_legs": len(plan_rows), "n_free_legs": int(sum(r["to_is_free"] for r in gt_rows))},
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
    # establishments). Chains, per-(mode,from→to) distance decay and tour-mode split are the
    # MiD 2023 urban constants (real activity tours + observed leg distances + observed mode
    # share); `decay_inflate` corrects the gravity scale so realized leg medians match MiD.
    "hannover": dict(
        n_locations=19000,
        density_per_km2=93.0,          # -> box ~ sqrt(19000/93) ~ 14.3 km, ~204 km^2
        # Mobile persons/km^2 for the full-population build (n_persons=None): real Hannover
        # ~535 k residents / 204 km^2 = 2622/km^2, times the MiD-2023 mobile share 0.84 = ~2203.
        # -> full pop ~ 204 * 2203 ~ 450 k mobile persons (matches osm.hannover_osm_world). Sample a
        # survey subset (survey.draw_survey) to solve; the full build only fixes dense potentials.
        mobile_density_per_km2=2203.0,
        n_clusters=10,                 # district centres (10 -> more uniform, lower car/PT seed variance) ...
        spread_scale=1.85,             # ... widened so they read as one city with visible structure
        type_prevalence=HANNOVER_TYPE_PREVALENCE,  # = measured /km^2 ÷ 93; work 0.083 matches the
        # real OSM-Hannover workplace count (~1,583), so business (=work pool) and work density are
        # comparable across the two city worlds. Shared with the two-zone core (see the constant).
        type_aliases={"business": "work"},   # dienstlich trips placed against the work catalog
        templates=MID_URBAN_TEMPLATES,
        pair_decay=MID_PAIR_DECAY_M,
        to_decay=MID_TO_DECAY_M,
        mode_decay=MID_MODE_DECAY_M,
        mode_split=MID_TOUR_MODE_SPLIT,
        # Per-mode correction so realized leg medians ≈ MiD: walk/bike land on target; car/pt
        # are pushed as far as the 204 km^2 footprint allows — they cap ~0.75-0.9x of MiD
        # 10 blobs + spread 1.85: low car/PT seed variance, but tighter blobs re-truncate car/PT to
        # ~0.85x MiD (walk 980, bike ~2400, car ~5000/5910, pt ~4900/5400) — the small-box cap; car
        # inflate saturates ~2.x. Walk/bike on target.
        decay_inflate={"walk": 0.56, "bike": 0.80, "car": 2.2, "pt": 1.7},
    ),
}


def city_world(name: str = "hannover", *, n_persons: Optional[int] = None,
               rng: Optional[np.random.Generator] = None, **overrides) -> SyntheticWorld:
    """Build a `generate_world` calibrated to a real city (see `CITY_PRESETS`).

    `city_world("hannover")` reproduces Hannover's facility density (~93/km^2), footprint
    (~204 km^2) and intra-city trip lengths. Any preset field can be overridden via kwargs,
    e.g. `city_world("hannover", n_persons=2000, distance_noise=0.1)`.

    `n_persons=None` (default) builds the **full mobile population** (footprint km^2 times the
    preset `mobile_density_per_km2`, ~450 k for Hannover) so the potentials are a realistic dense
    usage field; draw a survey subset (`survey.draw_survey`) to actually solve. Pass an int for a
    small sample build."""
    if name not in CITY_PRESETS:
        raise KeyError(f"unknown city preset {name!r}; have {sorted(CITY_PRESETS)}")
    params = {**CITY_PRESETS[name], **overrides}
    mobile_density = params.pop("mobile_density_per_km2", None)
    if n_persons is None:   # full mobile population = footprint area * mobile persons/km^2
        area_km2 = params["n_locations"] / params["density_per_km2"]
        if mobile_density is None:
            raise ValueError(f"city preset {name!r} has no mobile_density_per_km2; pass n_persons.")
        n_persons = int(round(area_km2 * mobile_density))
    params["n_persons"] = n_persons
    return generate_world(rng=rng, **params)
