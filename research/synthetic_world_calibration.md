# Calibrating the synthetic world to a real city (Hannover)

*Scratch research note — not part of the distributed library. Drives the `hannover` preset
in [`chainsolvers/eval/synth.py`](../chainsolvers/eval/synth.py).*

## Why

The synthetic-world generator (`eval/synth.py`) invents a topology of activity facilities and
a population of trip chains on it. Until now its defaults were dimensionally plausible but not
tied to any real place: **~1000 facilities in a 20 km box ≈ 2.5 facilities/km²**, roughly
**18× sparser** than an actual European city. Candidate density per activity type is the single
most important knob for a *point-placement* benchmark (it sets how many facilities compete for
each leg), so an unrealistic density makes recovery/% gap numbers hard to read against reality.

This note collects real figures for **Hannover** (Landeshauptstadt) and the German national
travel survey, derives generator parameters from them, and records the realized output.

## Target city: Hannover (Landeshauptstadt)

| Quantity | Value | Source |
|---|---|---|
| Population (city proper) | ~535,000 (535,932 in 2021) | citypopulation.de, Wikipedia |
| Area (city proper) | ~204 km² | Wikipedia (Hanover) |
| Population density | ~2,600 /km² | derived |
| Region Hannover (wider) | ~1.16 M over ~2,290 km² | Wirtschaftsförderung Hannover |

The model targets the **city proper** (204 km²), not the Region.

## How many POIs / facilities does a real city have?

There is no single published "POI count for Hannover"; the reproducible way is an Overpass
query (see bottom). For calibration we estimate the **activity-relevant facility stock** from
German per-capita rates and scale to Hannover's 535 k inhabitants:

| Activity type | German rate | Hannover estimate | Density | Basis |
|---|---|---|---|---|
| **shop** (retail) | ~3.6 stores / 1,000 | ~1,900 | ~9.3 /km² | Germany ≈ 300 k retail businesses / 83 M (Destatis/HDE order of magnitude) |
| **leisure** (gastronomy, sport, culture) | ~2.7 / 1,000 | ~1,400 | ~7 /km² | Germany ≈ 222 k hospitality establishments / 83 M (Destatis/Statista) |
| **work** (establishments) | — | ~2,400 | ~12 /km² | proxy from ~330 k jobs in the city; establishments dense in core |
| **education** (schools + tertiary) | — | ~140 | ~0.7 /km² | ~120 schools + ~20 tertiary/vocational |
| **other** (services, health, admin) | — | ~1,200 | ~6 /km² | residual service POIs |
| **home** (residential buildings) | — | ~150 k buildings | ~735 /km² | ~280 k households; anchors only (see note) |

**Total non-residential activity POIs ≈ 25,000–35,000** for a city this size — consistent in
order of magnitude with the standardized European OSM POI dataset (McCarty & Kim 2023:
7.2 M POIs across Europe; finest hex8 grid ≈ 0.7 km² cells concentrated in urban centres).

### Modelling choice

The generator collapses all facilities into `n_locations` multi-type points and represents
type availability with `type_prevalence` (fraction of locations offering each type). It does
**not** model the true ~735/km² residential density — `home`/`work` are *anchors* (given to the
solver), so what matters for the placement benchmark is the **secondary-activity** density
(shop/leisure/education/other). We therefore size `n_locations` to hit the **shop** density and
let prevalence distribute the rest:

```
shop density 9.3/km² over 204 km² → ~1,900 shops
shops are 20 % of locations        → n_locations ≈ 1,900 / 0.20 ≈ 9,500
overall facility density            ≈ 9,500 / 204 ≈ 46 /km²
box                                 = sqrt(9,500 / 46) ≈ 14.4 km  (≈ 204 km²)
```

> **Superseded:** this used the *per-capita estimate* (shop 9.3/km²). The live Overpass
> measurement below found shop density is ~2× higher (18.6/km²), so the same method now yields
> **n_locations ≈ 19,000 at 93/km²**. See *Measured vs estimated → Retuned*. The narrative here
> is kept to show the derivation; the actual preset uses the measured numbers.

## Trip distances and modal split: MiD 2017

National travel survey **"Mobilität in Deutschland 2017"** (BMVI / infas / DLR / IVT), trip
level:

| Metric | MiD 2017 (national) | Source |
|---|---|---|
| Trips / person / day | 3.1 | MiD 2017 short report |
| Mean trip length — **foot** | ~1.1 km | MiD 2017 |
| Mean trip length — **bike** | ~3.3 km | MiD 2017 (newer waves ~3.7) |
| Mean trip length — **car** | ~11 km | MiD 2017 |
| Mean trip length — **public transport** | ~7.5 km | MiD 2017 (~7–8) |
| Modal split (trips) | foot 22 % · bike 11 % · car (drv+pax) 57 % · PT 10 % | MiD 2017 |
| Trip purposes (trips) | work 21 · business 13 · shopping 16 · leisure 28 · education 9 · accompany 13 | MiD 2017 |

These national means are inflated by **inter-urban** travel. For an **intra-city** model (a
14 km box) car and PT trips are much shorter, so the `hannover` preset uses urban decay lengths.

| `mode_decay` (gravity scale = mean trip length) | National default | Hannover (urban) |
|---|---|---|
| walk | 1100 m | 1000 m |
| bike | 3300 m | 3300 m |
| car | 11000 m | 7000 m |
| pt | 7500 m | 6000 m |

(`P(loc) ∝ size·exp(-d/scale)`; an exponential's mean equals its scale, so `mode_decay[m]` ≈
the mean trip length of mode `m`.)

## What changed in `synth.py`

1. **`DEFAULT_MODE_DECAY`** retuned to MiD-2017 national means (`1100 / 3300 / 11000 / 7500`).
2. **`URBAN_MODE_DECAY`** added (intra-city values) + a sourced MiD reference comment block.
3. **`build_topology(..., density_per_km2=...)`** — new knob: pins facility density to a real
   value (`box = sqrt(n / density)·1000`). Threaded through `generate_world`. `None` keeps the
   legacy ~2.5/km² sparse abstraction so existing callers/tests are unchanged.
4. **`CITY_PRESETS` + `city_world(name="hannover", ...)`** — a calibrated kwargs bundle
   (`n_locations=19,000`, `density_per_km2=93`, OSM-matched prevalences) and a convenience
   constructor. `meta` now also reports realized `density_per_km2`.

```python
from chainsolvers_eval.synth import city_world
w = city_world("hannover", n_persons=4000)          # calibrated Hannover world
w = city_world("hannover", n_persons=2000, distance_noise=0.1)   # overrides allowed
```

## Realized output vs target — RETUNED to OSM (seed 0, 4000 persons)

The preset is now calibrated to the **measured OSM destination densities** (next section), not
the per-capita estimates. The placeable types reproduce the real counts to within ~1–4 %:

| Property | Target (OSM/real) | Realized | OK? |
|---|---|---|---|
| Area | 204 km² | 204 km² | ✓ |
| Facility density | 93 /km² | 93 /km² | ✓ |
| **shop** | 3,804 (18.6/km²) | 3,759 (18.4) | ✓ |
| **leisure** (destinations) | 2,421 (11.9/km²) | 2,463 (12.1) | ✓ |
| **education** (incl. kindergarten) | 714 (3.5/km²) | 689 (3.4) | ✓ |
| **other** (services) | 1,140 (5.6/km²) | 1,137 (5.6) | ✓ |
| mean trip — walk | ~1.1 km | 1.7 km | ~ (high) |
| mean trip — bike | ~3.3 km | 3.7 km | ✓ |
| mean trip — car | ~5–7 km (urban) | 4.9 km | ✓ |
| mean trip — PT | ~5–6 km (urban) | 4.6 km | ✓ |
| modal split | walk 26 / bike 19 / car 38 / PT 17 (Hannover SrV approx) | walk 36 / bike 15 / car 28 / PT 21 | ~ |

### Known residuals / limitations

- **Realized mean ≠ decay scale exactly.** The gravity decay sets a *target*; the realized mean
  is shaped by facility availability and by truncation at the 14 km city edge. Walk runs long
  (1.7 vs 1.1 km) because every leg (including home-anchored ones) draws from the same gravity
  rule rather than a short-trip-only walk regime.
- **Modal split is car-light.** `_draw_tour_mode` gives walk a constant high weight and only
  adds car when owned, so car share tops out ~27–30 % even at high ownership — below Hannover's
  ~38 % MIV. Fixing this properly needs a tour-length-conditioned mode model; out of scope here.
- **`work`/`home` are anchors**, so their densities are not OSM-calibrated: `work` is set to a
  plausible distinct-workplace pool (~15/km²), `home` lands ~73/km² (still ≪ the real ~219/km²
  mapped residential, but irrelevant to secondary-activity placement — the benchmark scores only
  the placeable legs).

## Measured vs estimated — live Overpass counts (2026-06-09)

Ran Overpass against `area["wikidata"="Q1715"]` (city of Hannover, ~204 km²) to replace the
per-capita guesses with **measured OSM counts**. **These now drive the preset** (retuned
2026-06-09 — see the realized table above and the *Retuned* note below).

| Category | Estimate | est /km² | **OSM count** | **OSM /km²** | ratio |
|---|---:|---:|---:|---:|---:|
| shop (retail + services) | 1,900 | 9.3 | **3,804** | **18.6** | 2.0× |
| gastronomy | 1,400 | 6.9 | **1,609** | **7.9** | 1.1× |
| education (incl. kindergarten) | 140 | 0.7 | **714** | **3.5** | 5.1× |
| office | 2,400 | 11.8 | **1,581** | **7.8** | 0.7× |
| leisure (incl. parks/pitches) | 1,400 | 6.9 | **4,228** | **20.7** | 3.0× |
| health | 1,200 | 5.9 | **565** | **2.8** | 0.5× |
| residential buildings | 150,000 | 735 | **44,641** | **219** | 0.3× |
| **non-residential core (sum)** | | | **12,501** | **61.3** | |

### Retuned (2026-06-09) — what changed in the preset

The raw `leisure` tag (20.7/km²) is inflated by green space (parks, playgrounds, pitches) that
aren't trip destinations, so a **refined destination query** (gastronomy + sport/fitness + culture
+ tourism museums/galleries, *excluding* parks) was used for the `leisure` target, and a
**`other`-services** query (health + financial + civic) for `other`:

| Model type | Mapped from | Measured | /km² | Preset prevalence | Realized count |
|---|---|---:|---:|---:|---:|
| shop | OSM `shop` (retail + consumer services) | 3,804 | 18.6 | **0.20** (anchor) | 3,759 |
| leisure | gastronomy + sport/fitness + culture + tourism | 2,421 | 11.9 | **0.13** | 2,463 |
| education | schools + universities + kindergarten/childcare | 714 | 3.5 | **0.038** | 689 |
| other | health + bank/post/library/townhall/police/… | 1,140 | 5.6 | **0.060** | 1,137 |

**Method:** anchor `shop` at its real prevalence (0.20) ⇒ total facility density = 18.6 / 0.20 =
**93/km²** ⇒ `n_locations = 93 × 204 ≈ 19,000`, `box ≈ 14.3 km` (still ~204 km²). Every other
prevalence = its measured density ÷ 93, so the placeable types fall out at the real counts. Changes:
`n_locations 9,500 → 19,000`, `density_per_km2 46 → 93`, `education 0.015 → 0.038`,
`leisure 0.15 → 0.13`, `other 0.12 → 0.060`, `work 0.25 → 0.16`. Trip lengths / modal split /
templates were already calibrated and unchanged.

**Not OSM-calibrated:** `work` (~15/km², anchor — distinct-workplace pool between offices-only 7.8
and all-establishments) and `home` (~73/km², anchor — far below the real 219/km² mapped residential,
but anchors don't affect placement). Densities (estimate → measured): `office` was over-estimated
(2,400 → 1,581; my "work" proxy mixed all workplaces) and `health` (1,200 → 565) is only a slice of
`other`; `gastronomy` matched (1,400 → 1,609).

**Caveats:** OSM tag scope ≠ the model's conceptual types (mapping above is deliberate); `nwr`
counts nodes+ways+relations (a facility mapped as both a way and a node can double-count); OSM
completeness varies by category. Residential is under-mapped (44,641 buildings tagged
`house/residential/…`; many real buildings are just `building=yes`).

### Queries used

Core category counts:

```overpassql
[out:json][timeout:240];
area["wikidata"="Q1715"]->.a;                  // city of Hannover (Landeshauptstadt)
( nwr["shop"](area.a); ); out count;
( nwr["amenity"~"^(restaurant|cafe|bar|pub|fast_food|biergarten|ice_cream|food_court)$"](area.a); ); out count;
( nwr["amenity"~"^(school|university|college|kindergarten|childcare)$"](area.a); ); out count;
( nwr["office"](area.a); ); out count;
( nwr["leisure"](area.a); ); out count;
( nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|dentist)$"](area.a); ); out count;
( nwr["building"~"^(house|residential|detached|apartments)$"](area.a); ); out count;
```

Refined `leisure`-destinations and `other`-services (used for the retune):

```overpassql
[out:json][timeout:240];
area["wikidata"="Q1715"]->.a;
( nwr["amenity"~"^(restaurant|cafe|bar|pub|fast_food|biergarten|ice_cream|food_court)$"](area.a);
  nwr["leisure"~"^(sports_centre|fitness_centre|fitness_station|stadium|sports_hall|bowling_alley|dance|ice_rink|water_park|golf_course|miniature_golf|escape_game|amusement_arcade|swimming_pool)$"](area.a);
  nwr["amenity"~"^(cinema|theatre|nightclub|arts_centre|casino|community_centre|social_centre)$"](area.a);
  nwr["tourism"~"^(museum|gallery|zoo|theme_park|attraction|aquarium)$"](area.a);
); out count;                                  // leisure destinations = 2,421
( nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|dentist|bank|post_office|library|townhall|police|fire_station|place_of_worship|fuel|veterinary|courthouse)$"](area.a); ); out count;  // other = 1,140
```

## Sources

- [Hanover — Wikipedia](https://en.wikipedia.org/wiki/Hanover) (area ~204 km², population)
- [Hannover — citypopulation.de](https://www.citypopulation.de/en/germany/niedersachsen/region_hannover/03241001__hannover/)
- [Wirtschaftsförderung Hannover — Trends & Facts](https://www.wirtschaftsfoerderung-hannover.de/de/Microsites/Trends_und_Fakten/index_en.php) (Region ~1.18 M; retail strength)
- [MiD 2017 short report (BMVI/infas/DLR/IVT)](https://www.bmv.de/SharedDocs/DE/Anlage/G/mid-2017-short-report.pdf?__blob=publicationFile) — trips/day, distances, modal split, purposes
- [Mobility in Germany — summary (Messe Frankfurt)](https://automotive.messefrankfurt.com/global/en/facts-figures/mobility-in-germany.html) — 3.1 trips/day; purpose shares
- [Clean Energy Wire — modal split in German cities](https://www.cleanenergywire.org/news/car-dominance-decreasing-german-cities-use-bicycles-and-footpaths) — walk 22 % / bike 11 % (2017)
- [McCarty & Kim 2023 — standardized European hexagon OSM POI dataset (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439266/) — 7.2 M POIs, density grids, 10 categories
- [Destatis — Wholesale & retail trade](https://www.destatis.de/EN/Themes/Economy/Wholesale-Trade-Retail-Trade/_node.html) and [Accommodation & food services](https://www.destatis.de/EN/Themes/Economic-Sectors-Enterprises/Accommodation-Food-Services-Activities-Tourism/_node.html) — establishment counts

*Note: facility counts are per-capita estimates, not measured Overpass extracts; treat the
densities as calibrated order-of-magnitude targets, not census figures.*
