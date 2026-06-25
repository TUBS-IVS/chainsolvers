"""Extract Hannover residential buildings (home density) from the Niedersachsen synthetic-
population GeoPackage into a light CSV for `osm.py` — same heavy-source -> small-CSV pattern as
the MiD extracts. We take only the **household count** per building (`hh_count`, the "number"),
not the per-person `mid_data` records.

    uv run --with geopandas --with pyogrio python research/scripts/extract_homes.py
"""
from __future__ import annotations

import json
import os

import geopandas as gpd

GPKG = os.path.join(os.path.dirname(__file__), "..", "buildings_with_households_NI_260128.gpkg")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "hannover_homes.csv")
AGS_HANNOVER = "03241001"   # Landeshauptstadt Hannover (city proper)


def main() -> None:
    g = gpd.read_file(GPKG, where=f"AGS = '{AGS_HANNOVER}'", engine="pyogrio")
    g = g[g["has_home"].astype(bool) & (g["hh_count"].fillna(0) > 0)].copy()

    def n_persons(mid: str) -> int:
        if not mid:
            return 0
        return sum(len(h.get("persons", [])) for h in json.loads(mid))

    persons = g["mid_data"].apply(n_persons).to_numpy(float)
    # representative point (inside each building) in the projected CRS, then to lat/lon so it
    # shares the OSM POIs' frame (the osm loader projects lat/lon -> local metres).
    pts = gpd.GeoSeries(g.geometry.representative_point(), crs=g.crs).to_crs(4326)
    pd_out = gpd.pd.DataFrame({
        "type": "home", "lat": pts.y.to_numpy(), "lon": pts.x.to_numpy(),
        "weight": persons,                       # residents per building = real home density
        "hh_count": g["hh_count"].to_numpy(float),
    })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd_out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(pd_out)} residential buildings, "
          f"{pd_out['hh_count'].sum():.0f} households, {persons.sum():.0f} persons "
          f"(true population; lat/lon from {g.crs})")


if __name__ == "__main__":
    main()
