"""Overlay each baked world's realized leg-distance distributions on the MiD empirical ones,
per mode and per (mode, home -> activity). The "does the world reproduce MiD?" check, run off
the *baked* snapshots (research/data/worlds/<name>/plans.parquet — full population) rather than a
fresh small build.

Single-zone worlds (gauss_hannover, osm_hannover) are compared to the urban MiD subset; the
two-zone super-region is split into its urban core (person ids "u...") vs rural ring ("r...") and
each compared to its own MiD subset (RegioStaRGem5 51/52 vs 55).

    uv run --project research python research/scripts/plot_worlds_vs_mid.py
Out: research/data/worlds/<world>/vs_mid_permode.png and vs_mid_pairs.png (beside each world.png).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
WEGE = os.path.join(HERE, "..", "data", "MiD2023_Wege.csv")
WORLDS_DIR = os.path.join(HERE, "..", "data", "worlds")
MODE = {1: "walk", 2: "bike", 3: "car", 4: "car", 5: "pt"}
ZTYPE = {1: "work", 2: "business", 3: "education", 4: "shop", 5: "other",
         6: "other", 7: "leisure", 8: "home", 9: "home", 10: "other"}
ZONES = {"urban": [51, 52], "rural": [55]}
MODES = ["walk", "bike", "car", "pt"]
COLS = ["HP_ID", "W_ID", "zweck", "hvm_imp", "wegkm_imp", "RegioStaRGem5"]

# Each baked world -> the zones it spans (mapped to MiD subsets for the overlay).
WORLD_ZONES = {
    "gauss_hannover": ["urban"],
    "osm_hannover": ["urban"],
    "two_zone": ["urban", "rural"],
}


def mid_samples():
    """Per-zone empirical leg distances (km) by mode and by (mode, from, to)."""
    parts = []
    for ch in pd.read_csv(WEGE, usecols=COLS, chunksize=1_000_000, low_memory=False):
        parts.append(ch[ch["RegioStaRGem5"].isin([51, 52, 55])])
    df = pd.concat(parts, ignore_index=True)
    by_mode = {z: defaultdict(list) for z in ZONES}
    by_pair = {z: defaultdict(list) for z in ZONES}
    zone_of = {c: z for z, cs in ZONES.items() for c in cs}
    for _, g in df.groupby("HP_ID", sort=False):
        g = g.sort_values("W_ID")
        z = g["zweck"].to_numpy(); hv = g["hvm_imp"].to_numpy(); km = g["wegkm_imp"].to_numpy()
        zone = zone_of.get(int(g["RegioStaRGem5"].iloc[0]))
        if zone is None or np.any(~np.isin(z, list(ZTYPE))) or np.any(~np.isin(hv, list(MODE))):
            continue
        if np.any(~np.isfinite(km)) or np.any(km <= 0) or np.any(km >= 200):
            continue
        chain = ("home", *[ZTYPE[int(v)] for v in z])
        if chain[-1] != "home":
            continue
        for i in range(1, len(chain)):
            m = MODE[int(hv[i - 1])]
            by_mode[zone][m].append(float(km[i - 1]))
            by_pair[zone][(m, chain[i - 1], chain[i])].append(float(km[i - 1]))
    return by_mode, by_pair


def world_samples(name: str, zones: list):
    """Realized leg distances (km) from a baked world's plans.parquet, per zone/mode/pair."""
    df = pd.read_parquet(os.path.join(WORLDS_DIR, name, "plans.parquet"))
    if zones == ["urban"]:
        df["zone"] = "urban"
    else:                                       # two-zone: core ids start "u", ring start "r"
        df["zone"] = np.where(df["unique_person_id"].str.startswith("u"), "urban", "rural")
    df["k"] = df["unique_leg_id"].str.rsplit("-l", n=1).str[1].astype(int)
    df = df.sort_values(["unique_person_id", "k"])
    df["from_act"] = df.groupby("unique_person_id")["to_act_type"].shift(1).fillna("home")
    df["km"] = df["distance_meters"] / 1000.0
    by_mode = {z: {m: df[(df.zone == z) & (df["mode"] == m)]["km"].to_numpy() for m in MODES}
               for z in zones}
    by_pair = {z: {(m, fr, to): sub["km"].to_numpy()
                   for (m, fr, to), sub in df[df.zone == z].groupby(["mode", "from_act", "to_act_type"])}
               for z in zones}
    return by_mode, by_pair


def _dens(ax, world, mid, label):
    ref = np.asarray(mid, float)
    if ref.size == 0 or len(world) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", fontsize=7); ax.set_title(label, fontsize=8)
        ax.set_yticks([]); return
    hi = np.quantile(ref, 0.95)
    bins = np.linspace(0, max(hi, 0.1), 36)
    ax.hist(np.clip(ref, 0, hi), bins=bins, density=True, color="0.6", alpha=0.6, label="MiD")
    ax.hist(np.clip(world, 0, hi), bins=bins, density=True, histtype="step", color="C3", lw=1.8,
            label="world")
    ax.axvline(np.median(ref), color="0.3", ls="--", lw=0.8)
    ax.axvline(np.median(world), color="C3", ls="--", lw=0.8)
    ax.set_title(label, fontsize=8); ax.set_yticks([])


def plot_world(name: str, zones: list, mid_mode, mid_pair):
    w_mode, w_pair = world_samples(name, zones)

    # --- per mode (rows) x zone (cols) ---
    fig, axes = plt.subplots(len(MODES), len(zones), figsize=(4.6 * len(zones), 11), squeeze=False)
    for r, m in enumerate(MODES):
        for c, z in enumerate(zones):
            _dens(axes[r, c], w_mode[z][m], mid_mode[z][m], f"{z} · {m}")
            if r == 0 and c == 0:
                axes[r, c].legend(fontsize=7)
    fig.suptitle(f"{name} vs MiD — leg-distance density per mode (clipped at MiD p95; "
                 "dashed = medians) · km", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p1 = os.path.join(WORLDS_DIR, name, "vs_mid_permode.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # --- per (mode, home->to) for the urban study zone ---
    tos = ["work", "shop", "leisure", "other"]
    fig, axes = plt.subplots(len(tos), len(MODES), figsize=(13, 10), squeeze=False)
    for r, to in enumerate(tos):
        for c, m in enumerate(MODES):
            key = (m, "home", to)
            _dens(axes[r, c], w_pair["urban"].get(key, np.array([])),
                  mid_pair["urban"].get(key, []), f"{m} · home→{to}")
    fig.suptitle(f"{name} (urban) — leg-distance density per (mode, home→activity): world vs MiD",
                 y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p2 = os.path.join(WORLDS_DIR, name, "vs_mid_pairs.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(WORLD_ZONES), help="plot just this world")
    args = ap.parse_args()
    names = [args.only] if args.only else list(WORLD_ZONES)
    names = [n for n in names if os.path.exists(os.path.join(WORLDS_DIR, n, "plans.parquet"))]
    if not names:
        raise SystemExit("no baked worlds found; run research/scripts/bake_worlds.py first")
    print("extracting MiD samples ...", flush=True)
    mid_mode, mid_pair = mid_samples()
    for name in names:
        print(f"plotting {name} ...", flush=True)
        p1, p2 = plot_world(name, WORLD_ZONES[name], mid_mode, mid_pair)
        print("  wrote", p1, "and", p2, flush=True)


if __name__ == "__main__":
    main()
