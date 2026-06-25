"""Fetch a real Hannover POI table from Overpass and cache it as a CSV for `osm.py`.

One row per (location, type) with lat/lon. `home`/`work` are anchor types; the placeable
secondary types (shop/leisure/education/other) carry the real destination geometry. Output
goes to research/data/ (gitignored). Run occasionally to refresh the snapshot:

    uv run python research/scripts/fetch_osm_topology.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "hannover_pois.csv")
ENDPOINT = "https://overpass-api.de/api/interpreter"
AREA = 'area["wikidata"="Q1715"]->.a;'   # Landeshauptstadt Hannover

# model type -> Overpass selector(s) inside area .a
SELECTORS = {
    "shop": ['nwr["shop"](area.a);'],
    "leisure": ['nwr["amenity"~"^(restaurant|cafe|bar|pub|fast_food|biergarten|ice_cream|food_court)$"](area.a);',
                'nwr["leisure"~"^(sports_centre|fitness_centre|fitness_station|stadium|sports_hall|bowling_alley|dance|ice_rink|water_park|golf_course|swimming_pool)$"](area.a);',
                'nwr["amenity"~"^(cinema|theatre|nightclub|arts_centre|casino|community_centre)$"](area.a);',
                'nwr["tourism"~"^(museum|gallery|zoo|theme_park|attraction|aquarium)$"](area.a);'],
    "education": ['nwr["amenity"~"^(school|university|college|kindergarten|childcare)$"](area.a);'],
    "other": ['nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|dentist|bank|post_office|library|townhall|police|fire_station|place_of_worship|fuel|veterinary|courthouse)$"](area.a);'],
    "work": ['nwr["office"](area.a);'],
    "home": ['nwr["building"~"^(apartments|house|residential|detached)$"](area.a);'],
}


def _query(selector_block: str, *, retries: int = 5) -> list:
    q = f"[out:json][timeout:300];{AREA}({selector_block});out center;"
    for attempt in range(retries):
        req = urllib.request.Request(
            ENDPOINT, data=urllib.parse.urlencode({"data": q}).encode(),
            headers={"User-Agent": "chainsolvers-research/1.0 (felixpetre@gmail.com)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=320) as r:
                return json.load(r)["elements"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 504) and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  {e.code}; backing off {wait}s")
                time.sleep(wait)
                continue
            raise


def main() -> None:
    rows = []
    for typ, sels in SELECTORS.items():
        els = _query("".join(sels))
        for e in els:
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            if lat is not None and lon is not None:
                rows.append({"type": typ, "lat": lat, "lon": lon})
        print(f"{typ:10s} {len(els):6d} elements")
        time.sleep(8)   # be polite to the public endpoint / avoid 429
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} POIs)\n", df["type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
