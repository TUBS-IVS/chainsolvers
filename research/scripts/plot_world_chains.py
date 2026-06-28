"""Small-multiples gallery of example activity chains read off a *baked* world snapshot
(research/data/worlds/<name>/{plans,ground_truth,facilities}.parquet).

This is the per-chain analog of panel 3 in `chainsolvers_eval.viz.plot_world`: instead of overlaying a
handful of chains on the whole city map (where they tangle), every selected chain gets its own little
panel, zoomed to its own bounding box over a faint local facility backdrop. Squares are known anchors
(home/work), hollow circles are the secondary activities the solver must PLACE, the line is the trip
sequence, and each node is annotated with its leg order + activity type.

Crucially we only show chains with **>0 free nodes** — the bulk of the population is trivial
home->work->home (0 to place), which tells you nothing about the placement problem. To surface the
variety that actually exists, rows are grouped by the number of free nodes (1, 2, 3, 4...), so you can
see at a glance how a 1-free shop-hop differs from a 4-free errand tour.

    uv run --project research python research/scripts/plot_world_chains.py
    uv run --project research python research/scripts/plot_world_chains.py --worlds gauss_hannover \
        --free 1 2 3 4 --cols 5 --out papers/chainsolvers-journal/figures
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
WORLDS_DIR = os.path.join(HERE, "..", "data", "worlds")
DEFAULT_OUT = os.path.join(HERE, "..", "out")

WORLDS = [
    ("gauss_hannover", "Gauss-Hannover"),
    ("osm_hannover", "OSM-Hannover"),
    ("two_zone", "Two-zone super-region"),
]

ANCHOR_COL = "#1b1b1b"   # known home/work
FREE_COL = "#d1495b"     # secondary activity to place


def _leg_order(leg_id: str) -> int:
    """Sort key for 'p{pi}-l{k}' so l10 follows l9 (string sort would not)."""
    try:
        return int(leg_id.rsplit("-l", 1)[1])
    except (IndexError, ValueError):
        return 0


def _chain_geometry(grp: pd.DataFrame, gti: pd.DataFrame):
    """(xs, ys, is_anchor, types) for one person's chain, using the TRUE facility coords.
    Node 0 is the home start (first leg's from_*); each subsequent node is a leg's to-location.

    Consecutive duplicate stops are collapsed: the world generator emits spurious zero-distance
    `home->home` legs and a stop is often revisited back-to-back, which would otherwise stack several
    nodes on one point and inflate the visible leg count. (A *non*-consecutive revisit — e.g. the home
    in home->shop->home->other->home — is kept, so genuine multi-tour structure still shows.)"""
    grp = grp.sort_values("unique_leg_id", key=lambda s: s.map(_leg_order))
    first = grp.iloc[0]
    xs = [float(first["from_x"])]
    ys = [float(first["from_y"])]
    is_anchor = [True]            # home start is always known
    types = ["home"]
    for _, leg in grp.iterrows():
        row = gti.loc[leg["unique_leg_id"]]
        x, y = float(row["true_to_x"]), float(row["true_to_y"])
        if abs(x - xs[-1]) < 1e-6 and abs(y - ys[-1]) < 1e-6:  # zero-length step -> same node
            is_anchor[-1] = is_anchor[-1] and (not bool(row["to_is_free"]))
            continue
        xs.append(x); ys.append(y)
        is_anchor.append(not bool(row["to_is_free"]))
        types.append(str(leg["to_act_type"]))
    return np.array(xs), np.array(ys), np.array(is_anchor, bool), types


def _max_free_run(is_anchor: np.ndarray) -> int:
    """Longest run of consecutive free (non-anchor) nodes = the longest segment CARLA must place as a
    coupled chain. 1 = trivial single-intermediate (circle-intersection); >=2 = the branching case."""
    best = cur = 0
    for a in is_anchor:
        cur = 0 if a else cur + 1
        best = max(best, cur)
    return best


def _person_max_free_run(plans: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Per-person longest consecutive free-leg run, vectorized over the whole table (Series indexed by
    person; persons with no free legs are absent). This is the difficulty axis the gallery buckets on —
    total free-node count is misleading because two independent out-and-backs (home->shop->home->
    other->home) score 2 but are two trivial single-intermediate placements, not one hard chain."""
    leg2pid = plans.set_index("unique_leg_id")["unique_person_id"]
    g = gt.assign(pid=gt["unique_leg_id"].map(leg2pid),
                  k=gt["unique_leg_id"].str.rsplit("-l", n=1).str[1].astype(int))
    g = g.sort_values(["pid", "k"])
    blk = (~g["to_is_free"]).cumsum()                       # free legs share a block until broken by an anchor
    runs = g[g["to_is_free"]].groupby([g["pid"], blk]).size()
    return runs.groupby(level=0).max()


def _select(pool_series: pd.Series, buckets, cols, rng):
    """For each requested bucket value k, pick up to `cols` person ids whose axis value == k.
    Returns a list of (k, [pid, ...]) rows in bucket order."""
    rows = []
    for k in buckets:
        pool = pool_series.index[pool_series.values == k].to_numpy()
        pick = list(rng.choice(pool, size=min(cols, pool.size), replace=False)) if pool.size else []
        rows.append((k, pick))
    return rows


def _panel(ax, xs, ys, is_anchor, types, fac_xy, pid):
    # faint local facility backdrop, cropped to a padded chain bbox
    pad = max((xs.max() - xs.min()), (ys.max() - ys.min()), 500.0) * 0.35
    x0, x1 = xs.min() - pad, xs.max() + pad
    y0, y1 = ys.min() - pad, ys.max() + pad
    if fac_xy is not None:
        m = ((fac_xy[:, 0] >= x0) & (fac_xy[:, 0] <= x1) &
             (fac_xy[:, 1] >= y0) & (fac_xy[:, 1] <= y1))
        ax.scatter(fac_xy[m, 0], fac_xy[m, 1], s=2, c="0.82", lw=0, zorder=1, rasterized=True)

    ax.plot(xs, ys, "-", color="0.45", lw=1.1, zorder=2)
    for x, y, anc, t in zip(xs, ys, is_anchor, types):
        if anc:
            ax.scatter([x], [y], marker="s", s=48, facecolor=ANCHOR_COL,
                       edgecolors="white", linewidths=0.6, zorder=4)
        else:
            ax.scatter([x], [y], marker="o", s=52, facecolor="none",
                       edgecolors=FREE_COL, linewidths=1.6, zorder=4)
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="0.25", zorder=5)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    n_free = int((~is_anchor).sum())
    ax.set_title(f"{pid}  ·  {len(xs) - 1} stops · {n_free} free", fontsize=7)


def gallery(name: str, title: str, free_counts, cols, seed, out_dir, dpi, ext="pdf"):
    d = os.path.join(WORLDS_DIR, name)
    plans = pd.read_parquet(os.path.join(d, "plans.parquet"))
    gt = pd.read_parquet(os.path.join(d, "ground_truth.parquet"))
    fac = pd.read_parquet(os.path.join(d, "facilities.parquet")).drop_duplicates("loc_id")
    fac_xy = fac[["x", "y"]].to_numpy(float)

    axis_series = _person_max_free_run(plans, gt)   # difficulty = longest consecutive free-leg run

    rng = np.random.default_rng(seed)
    rows = _select(axis_series, free_counts, cols, rng)

    # subset to the handful of selected persons BEFORE grouping (the full table is millions of legs)
    chosen = {pid for _, pids in rows for pid in pids}
    sub = plans[plans["unique_person_id"].isin(chosen)]
    gti = gt[gt["unique_leg_id"].isin(set(sub["unique_leg_id"]))].set_index("unique_leg_id")
    plans_by_pid = {pid: g for pid, g in sub.groupby("unique_person_id")}

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, cols, figsize=(2.3 * cols, 2.4 * nrows), squeeze=False)
    for r, (k, pids) in enumerate(rows):
        lbl = (f"{k} free in a row\n(single-intermediate)" if k == 1
               else f"{k} free in a row\n(chained segment)")
        axes[r, 0].set_ylabel(lbl, fontsize=9, rotation=90, labelpad=8)
        for c in range(cols):
            ax = axes[r, c]
            if c < len(pids):
                pid = pids[c]
                xs, ys, anc, types = _chain_geometry(plans_by_pid[pid], gti)
                _panel(ax, xs, ys, anc, types, fac_xy, pid)
            else:
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

    fig.suptitle(f"{title} — example chains, grouped by longest run of consecutive free nodes\n"
                 "□ known anchor (home/work)   ○ secondary activity to place   "
                 "(consecutive duplicate stops collapsed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"chains_{name}.{ext}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print("wrote", out, flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", nargs="+", default=[n for n, _ in WORLDS])
    ap.add_argument("--free", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="longest-consecutive-free-run values to show, one row per value "
                         "(1 = trivial single-intermediate; >=2 = chained segment, the hard case)")
    ap.add_argument("--cols", type=int, default=5, help="example chains per row/bucket")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args(argv)

    titles = dict(WORLDS)
    for n in args.worlds:
        if not os.path.exists(os.path.join(WORLDS_DIR, n, "plans.parquet")):
            print(f"skip {n}: no baked snapshot", flush=True)
            continue
        gallery(n, titles.get(n, n), args.free, args.cols, args.seed, args.out, args.dpi, args.ext)


if __name__ == "__main__":
    main()
