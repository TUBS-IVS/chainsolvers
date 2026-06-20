"""Extract weighted distance / mode / purpose distributions from MiD 2023 microdata.

Reads only the needed columns from the big Wege (trips) CSV in chunks, filters to an
urban subset (RegioStaR), weights by W_GEW, and prints distributions we can calibrate the
synthetic world against. Scratch research tool — kept in gitignored research/.

    uv run python research/mid_extract.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

WEGE = os.path.join(os.path.dirname(__file__), "MiD2023_Wege.csv")

# MiD codings (2023):
HVM = {1: "walk", 2: "bike", 3: "car_pax", 4: "car_drv", 5: "pt"}
ZWECK = {1: "work", 2: "business", 3: "education", 4: "shopping", 5: "errand",
         6: "leisure", 7: "accompany", 8: "home", 9: "other", 10: "other"}
COLS = ["hvm_imp", "zweck", "wegkm_imp", "wegmin", "W_GEW", "RegioStaR7", "RegioStaRGem5"]


def wq(values, weights, qs):
    """Weighted quantiles."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    order = np.argsort(v); v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, v)


def main():
    parts = []
    for chunk in pd.read_csv(WEGE, usecols=COLS, chunksize=1_000_000, low_memory=False):
        # urban subset: large cities (Metropole/Regiopole+Großstadt) at the Gemeinde level
        chunk = chunk[chunk["RegioStaRGem5"].isin([51, 52])]
        # plausible distances only (drop missing/implausible codes and absurd values)
        chunk = chunk[(chunk["wegkm_imp"] > 0) & (chunk["wegkm_imp"] < 200)]
        chunk = chunk[chunk["hvm_imp"].isin(HVM)]
        parts.append(chunk[COLS])
    df = pd.concat(parts, ignore_index=True)
    w = df["W_GEW"].to_numpy()
    km = df["wegkm_imp"].to_numpy()
    print(f"urban trips (RegioStaRGem5 in 51/52): n={len(df):,}  weighted={w.sum():,.0f}\n")

    # --- modal split (weighted) ---------------------------------------------------------
    print("MODAL SPLIT (weighted, urban):")
    ms = df.assign(mode=df["hvm_imp"].map(HVM)).groupby("mode")["W_GEW"].sum()
    ms = (ms / ms.sum() * 100).round(1)
    print("  " + "  ".join(f"{k} {v}%" for k, v in ms.items()))
    # collapse car_pax+car_drv
    car = ms.get("car_drv", 0) + ms.get("car_pax", 0)
    print(f"  -> walk {ms.get('walk',0)} / bike {ms.get('bike',0)} / car {car:.1f} / pt {ms.get('pt',0)}\n")

    # --- distance by mode ---------------------------------------------------------------
    print("DISTANCE by MODE (km, weighted) — mean / p25 / p50 / p75 / p90:")
    for code, name in HVM.items():
        m = df["hvm_imp"] == code
        if m.sum() == 0:
            continue
        ww, kk = w[m], km[m]
        mean = np.average(kk, weights=ww)
        p = wq(kk, ww, [0.25, 0.5, 0.75, 0.90])
        print(f"  {name:8s} mean {mean:5.2f}  p25 {p[0]:4.2f}  p50 {p[1]:4.2f}  p75 {p[2]:5.2f}  p90 {p[3]:6.2f}")
    print()

    # --- distance by purpose ------------------------------------------------------------
    print("DISTANCE by PURPOSE (km, weighted) — share% / mean / p50:")
    df = df.assign(z=df["zweck"].map(ZWECK).fillna("other"))
    tot = w.sum()
    for name in ["work", "business", "education", "shopping", "errand", "leisure", "accompany", "home", "other"]:
        m = (df["z"] == name).to_numpy()
        if m.sum() == 0:
            continue
        ww, kk = w[m], km[m]
        share = ww.sum() / tot * 100
        mean = np.average(kk, weights=ww)
        p50 = wq(kk, ww, [0.5])[0]
        print(f"  {name:10s} share {share:5.1f}%  mean {mean:5.2f}  p50 {p50:5.2f}")


if __name__ == "__main__":
    main()
