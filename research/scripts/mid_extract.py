"""Derive synthetic-world calibration constants from MiD 2023 microdata.

Reads the big Wege (trips) CSV in chunks, restricts to an urban subset (RegioStaR large
cities), reconstructs each person's home-anchored activity chain, and emits BAKE-READY
Python literals for research/chainsolvers_eval/synth.py:

  * MID_URBAN_TEMPLATES   activity-type chains backed by >= MINRESP diaries -> chain skeletons (C)
  * MID_PAIR_DECAY_M      per-(mode, from_act, to_act) median leg distance -> gravity scale (A)
  * MID_TO_DECAY_M        per-(mode, to_act) fallback
  * MID_MODE_DECAY_M      per-mode fallback
  * MID_TOUR_MODE_SPLIT   tour main-mode shares                            -> mode sampling (A)

The raw CSV lives in research/data/ (gitignored, licensed); only the small derived
constants below get pasted into the tracked eval package.

    uv run python research/scripts/mid_extract.py
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
WEGE = os.path.join(HERE, "..", "data", "MiD2023_Wege.csv")

# MiD 2023 codings (confirmed against MiD2023_Codepläne_B1_Standard_v1.1.xlsx):
#   hvm: 1 zu Fuß, 2 Fahrrad, 3 MIV-Mitfahrer, 4 MIV-Fahrer, 5 ÖPV (9 k.A.)
#   zweck: 1 Arbeit, 2 dienstlich, 3 Ausbildung, 4 Einkauf, 5 Erledigung,
#          6 Bringen/Holen/Begleiten, 7 Freizeit, 8 nach Hause, 9 Rückweg, 10 anderer (99 k.A.)
# zweck 2 (dienstlich/business) -> its own type, placed against the WORK catalog (a business trip
#   visits a workplace); this pulls the largest, best-defined chunk out of the catch-all `other`.
# zweck 9 (Rückweg) -> home: a tour-closure approximation (returns usually go home); the MiD-faithful
#   alternative is recoding to the previous trip's purpose, but it is 1.9% of trips and mapping it to
#   `other` would break the home-closure filter, so we keep -> home.
MODE = {1: "walk", 2: "bike", 3: "car", 4: "car", 5: "pt"}
ZTYPE = {1: "work", 2: "business", 3: "education", 4: "shop", 5: "other",
         6: "other", 7: "leisure", 8: "home", 9: "home", 10: "other"}
COLS = ["HP_ID", "W_ID", "zweck", "hvm_imp", "wegkm_imp", "W_GEW", "RegioStaRGem5"]
MINOBS = 200          # min weighted-cell observations to trust a (mode,from,to) decay
MINRESP = 10          # min distinct respondents (diaries) backing a chain template (disclosure floor)
MAXLEN = 8            # cap activities per chain (incl. home endpoints)


def wmedian(vals, wts):
    v = np.asarray(vals, float); w = np.asarray(wts, float)
    o = np.argsort(v); v, w = v[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(0.5, cw / w.sum(), v))


# RegioStaRGem5 (Gemeinde type): 51 Metropole, 52 Regiopole/Großstadt, 53 zentrale Stadt/
# Mittelstadt, 54 städtischer Raum, 55 kleinstädtischer/dörflicher Raum (rural village).
REGIONS = {
    "urban": ([51, 52], "MID", "MID_URBAN_TEMPLATES"),    # large cities (the default Hannover bench)
    "rural": ([55], "MID_RURAL", "MID_RURAL_TEMPLATES"),  # villages / small rural settlements
}


def main(region: str = "urban"):
    codes, prefix, tmpl_name = REGIONS[region]
    parts = []
    for chunk in pd.read_csv(WEGE, usecols=COLS, chunksize=1_000_000, low_memory=False):
        parts.append(chunk[chunk["RegioStaRGem5"].isin(codes)])
    df = pd.concat(parts, ignore_index=True)
    print(f"# {region} trips n={len(df):,}  persons={df['HP_ID'].nunique():,}  (RegioStaRGem5 {codes})")

    tmpl_w = defaultdict(float)                       # chain -> weight
    tmpl_n = defaultdict(int)                          # chain -> distinct respondents (disclosure floor)
    split_w = defaultdict(float)                      # tour main mode -> weight
    pair = defaultdict(lambda: ([], []))              # (mode,from,to) -> (km[], w[])
    n_tours = 0
    for _, g in df.groupby("HP_ID", sort=False):
        g = g.sort_values("W_ID")
        z = g["zweck"].to_numpy(); hv = g["hvm_imp"].to_numpy(); km = g["wegkm_imp"].to_numpy()
        gw = g["W_GEW"].to_numpy()
        if np.any(~np.isin(z, list(ZTYPE))) or np.any(~np.isin(hv, list(MODE))):
            continue                                  # drop incomplete (unknown mode/purpose)
        if np.any(~np.isfinite(km)) or np.any(km <= 0) or np.any(km >= 200):
            continue
        types = [ZTYPE[int(v)] for v in z]
        chain = ("home", *types)
        if chain[-1] != "home" or not (3 <= len(chain) <= MAXLEN):
            continue
        wt = float(np.mean(gw))
        tmpl_w[chain] += wt
        tmpl_n[chain] += 1                            # one diary (respondent-day) for this chain
        n_tours += 1
        main = MODE[int(hv[np.argmax(km)])]           # tour mode = mode of longest leg
        split_w[main] += wt
        for i in range(1, len(chain)):                # legs: chain[i-1] -> chain[i]
            key = (MODE[int(hv[i - 1])], chain[i - 1], chain[i])
            pair[key][0].append(float(km[i - 1])); pair[key][1].append(float(gw[i - 1]))

    # --- templates ----------------------------------------------------------------------
    # Disclosure floor: keep every chain backed by >= MINRESP distinct diaries (a k-anonymity-style
    # cell threshold), not an arbitrary top-N rank cut. Each emitted constant is then a genuine
    # aggregate; single-diary (potentially identifying) day-patterns never leave the secure store.
    kept = sorted([ch for ch in tmpl_w if tmpl_n[ch] >= MINRESP],
                  key=lambda ch: tmpl_w[ch], reverse=True)
    tot = sum(tmpl_w[ch] for ch in kept)
    stops = lambda ch: sum(1 for a in ch if a not in ("home", "work"))
    cov = tot / sum(tmpl_w.values()) * 100
    mean_stops = sum(stops(ch) * tmpl_w[ch] for ch in kept) / tot
    f3 = sum(tmpl_w[ch] for ch in kept if stops(ch) >= 3) / tot * 100
    max_legs = max(len(ch) - 1 for ch in kept)
    print(f"\n# tours kept={n_tours:,}; {len(kept)} templates (>= {MINRESP} resp.) cover {cov:.0f}% "
          f"of weight; mean free-stops={mean_stops:.2f}; >=3 stops={f3:.1f}%; max legs={max_legs}\n"
          f"{tmpl_name} = [")
    for ch in kept:
        print(f"    ({ch!r}, {tmpl_w[ch] / tot:.4f}),")
    print("]")

    # --- coverage-vs-cutoff table -> tables/mid_chains.tex ------------------------------
    # RESPONDENT-based (we SELECT by distinct diaries, not by weight): rank chains by respondent
    # count; cover/stops/>=2/>=3 are all over diaries. min resp = fewest diaries in the cutoff.
    by_resp = sorted(tmpl_w, key=lambda c: tmpl_n[c], reverse=True)
    tot_resp = sum(tmpl_n.values())
    print("\n% mid_chains.tex rows (respondent-based: topN cover stops >=2 >=3 maxlegs minresp):")
    for c in [30, 200, 500, 1000, len(by_resp)]:
        top = by_resp[:c]; nr = sum(tmpl_n[ch] for ch in top)
        cov = nr / tot_resp * 100
        ms = sum(stops(ch) * tmpl_n[ch] for ch in top) / nr
        ge2 = sum(tmpl_n[ch] for ch in top if stops(ch) >= 2) / nr * 100
        ge3 = sum(tmpl_n[ch] for ch in top if stops(ch) >= 3) / nr * 100
        ml = max(len(ch) - 1 for ch in top); mr = min(tmpl_n[ch] for ch in top)
        cc = f"{c:,}".replace(",", "\\,")
        print(f"{cc} & {cov:.0f} & {ms:.2f} & {ge2:.1f} & {ge3:.1f} & {ml} & {mr} \\\\")

    # --- decay tables (median leg distance in metres) -----------------------------------
    mode_km = defaultdict(lambda: ([], []))
    to_km = defaultdict(lambda: ([], []))
    pair_out = {}
    for (mo, fr, to), (kk, ww) in pair.items():
        mode_km[mo][0].extend(kk); mode_km[mo][1].extend(ww)
        to_km[(mo, to)][0].extend(kk); to_km[(mo, to)][1].extend(ww)
        if sum(ww) >= MINOBS:
            pair_out[(mo, fr, to)] = (round(wmedian(kk, ww) * 1000), int(len(kk)))
    print(f"\n{prefix}_PAIR_DECAY_M = {{  # (mode, from, to): metres   # n legs")
    for k in sorted(pair_out):
        m, n = pair_out[k]
        print(f"    {k!r}: {m},   # n={n}")
    print("}")
    print(f"{prefix}_TO_DECAY_M = {{  # (mode, to): metres  fallback")
    for k in sorted(to_km):
        kk, ww = to_km[k]
        if sum(ww) >= MINOBS:
            print(f"    {k!r}: {round(wmedian(kk, ww) * 1000)},")
    print("}")
    print(f"{prefix}_MODE_DECAY_M = {{  # per-mode fallback")
    for k in sorted(mode_km):
        kk, ww = mode_km[k]
        print(f"    {k!r}: {round(wmedian(kk, ww) * 1000)},")
    print("}")

    # --- tour mode split ----------------------------------------------------------------
    s = sum(split_w.values())
    print(f"{prefix}_TOUR_MODE_SPLIT = {{", ", ".join(f"{k!r}: {v / s:.3f}" for k, v in
          sorted(split_w.items(), key=lambda kv: -kv[1])), "}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=list(REGIONS), default="urban")
    main(ap.parse_args().region)
