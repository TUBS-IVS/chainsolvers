"""Visualize a synthetic ground-truth world (`synth.generate_world`).

One PNG, four panels, answering "does the generated world make sense?":

  1. Topology      — every facility, marker size ∝ latent structural attractiveness
                     (the exogenous driver the solver does not see). Shows clustering.
  2. Realized use  — facilities that were actually visited, size/colour ∝ visit count
                     (the *potentials* the solver is given). Latent size drives choice;
                     usage is the observable outcome.
  3. Example chains— a handful of full activity chains drawn on the map: squares are
                     known anchors (home/work), circles are secondary activities to be
                     placed, lines are trips. Shows the problem the solver actually faces.
  4. Free-leg dist — distribution of the secondary (to-place) leg distances; the target
                     a distance-matching solver must reproduce.

Usage::

    from chainsolvers_eval.synth import generate_world
    from chainsolvers_eval.viz import plot_world
    plot_world(generate_world(n_persons=500), "out/world.png")
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _leg_order(leg_id: str) -> int:
    """Sort key for 'p{pi}-l{k}' so l10 follows l9 (string sort would not)."""
    try:
        return int(leg_id.rsplit("-l", 1)[1])
    except (IndexError, ValueError):
        return 0


def plot_world(world, path: str = "out/world.png", *, n_chains: int = 8,
               seed: int = 0, dpi: int = 130) -> str:
    """Render a :class:`synth.SyntheticWorld` to a 4-panel PNG; returns the path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - optional viz dependency
        raise ImportError(
            "plot_world needs matplotlib; install the viz extra: pip install -e '.[viz]'"
        ) from e

    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    topo = world.topology
    if topo is None:
        raise ValueError("world.topology is None; build the world with generate_world(...).")
    gt = world.ground_truth
    plans = world.plans_df
    coords, sizes, loc_ids = topo.coords, topo.sizes, topo.loc_ids

    # Realized usage per location: count true facility occurrences in the ground truth.
    id_to_idx = {lid: i for i, lid in enumerate(loc_ids)}
    visits = np.zeros(len(loc_ids))
    vc = gt["true_to_identifier"].value_counts()
    for lid, c in vc.items():
        if lid in id_to_idx:
            visits[id_to_idx[lid]] += int(c)

    free_ids = set(gt.loc[gt["to_is_free"], "unique_leg_id"])
    free_dist = plans.loc[plans["unique_leg_id"].isin(free_ids), "distance_meters"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # --- 1. Topology, size ∝ latent attractiveness --------------------------------
    s = 4 + 60 * (sizes / sizes.max())
    ax1.scatter(coords[:, 0], coords[:, 1], s=s, c=sizes, cmap="viridis",
                alpha=0.6, linewidths=0)
    ax1.set_title(f"1. Topology — {len(loc_ids)} facilities\n(marker size ∝ latent attractiveness)")
    ax1.set_aspect("equal"); ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")

    # --- 2. Realized usage (potentials = visit counts) ----------------------------
    used = visits > 0
    sc = ax2.scatter(coords[used, 0], coords[used, 1],
                     s=6 + 50 * visits[used] / max(visits.max(), 1),
                     c=visits[used], cmap="inferno", alpha=0.8, linewidths=0)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, label="visits")
    ax2.set_title(f"2. Realized usage — {int(used.sum())}/{len(loc_ids)} facilities visited\n"
                  "(potentials given to the solver = visit counts)")
    ax2.set_aspect("equal"); ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")

    # --- 3. Example chains --------------------------------------------------------
    ax3.scatter(coords[:, 0], coords[:, 1], s=2, c="lightgrey", alpha=0.5, linewidths=0)
    rng = np.random.default_rng(seed)
    pids = plans["unique_person_id"].unique()
    pick = rng.choice(pids, size=min(n_chains, len(pids)), replace=False)
    gti = gt.set_index("unique_leg_id")
    cmap = plt.get_cmap("tab10")
    for ci, pid in enumerate(pick):
        grp = plans[plans["unique_person_id"] == pid].copy()
        grp = grp.sort_values("unique_leg_id", key=lambda s: s.map(_leg_order))
        first = grp.iloc[0]
        xs = [float(first["from_x"])]; ys = [float(first["from_y"])]
        is_anchor = [True]  # home start
        for _, leg in grp.iterrows():
            row = gti.loc[leg["unique_leg_id"]]
            xs.append(float(row["true_to_x"])); ys.append(float(row["true_to_y"]))
            is_anchor.append(not bool(row["to_is_free"]))
        col = cmap(ci % 10)
        ax3.plot(xs, ys, "-", color=col, alpha=0.7, linewidth=1.2, zorder=2)
        for x, y, anc in zip(xs, ys, is_anchor):
            ax3.scatter([x], [y], marker="s" if anc else "o",
                        s=55 if anc else 35, color=col,
                        edgecolors="black", linewidths=0.5, zorder=3)
    ax3.set_title(f"3. {len(pick)} example chains\n(□ known anchor, ○ secondary activity to place)")
    ax3.set_aspect("equal"); ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")

    # --- 4. Free-leg distance distribution ----------------------------------------
    if free_dist.size:
        ax4.hist(free_dist, bins=40, color="steelblue", alpha=0.85)
        ax4.axvline(np.median(free_dist), color="crimson", linestyle="--",
                    label=f"median {np.median(free_dist):.0f} m")
        ax4.legend()
    ax4.set_title(f"4. Secondary (free) leg distances — n={free_dist.size}")
    ax4.set_xlabel("distance (m)"); ax4.set_ylabel("count")

    m = world.meta
    fig.suptitle(
        f"Synthetic ground-truth world — {m.get('n_persons')} persons, "
        f"{m.get('n_legs')} legs ({m.get('n_free_legs')} to place), "
        f"{len(topo.types)} activity types",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
