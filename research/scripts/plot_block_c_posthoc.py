"""Block C post-hoc figures from the finished prognosis CSVs / raw parquets.

C5  spatial-TV vs boost    -- grid total-variation between each solver's predicted free-leg visit
                             field and the TRUE counterfactual field (regenerated deterministically).
C6  pot-decile-TV vs boost -- the Block-B fit metric (already a CSV column) tracked across the shock.
C7  GT dist-distribution   -- true free-leg distance histogram b=1 vs b=6, extended to osm/two_zone.

Runs over every FINISHED (world, regime) cell it finds on disk; missing cells (e.g. full/two_zone
still solving) are skipped with a note. dp_full oracle rows simply aren't present yet -- that's fine,
the figures redraw cleanly once the appends land. Truth regen uses seed+1 (run seed=0, n=1000), the
same deterministic DGP as block_c.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from block_a_style import apply_paper_style, line, WORLD_NAME
from chainsolvers_eval.worlds import load_world
from chainsolvers_eval.regen import regenerate_world

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out", "block_c")
SEED, NP_ = 0, 1000
ANCHORS = ("home", "work")
WORLDS = ["gauss_hannover", "osm_hannover", "two_zone"]
GRID = 12  # coarse grid for spatial-TV (the realism discriminator; fine grids are sampling-noise)

# (solver, condition) series shown in C5/C6 -- the comparison roster (oracle added later)
SERIES = [("dp_sample", "informed"), ("carla_sample", "informed"),
          ("gravity_independent", "informed"), ("dp_carla", "w1"), ("dp_carla_pot", "w1"),
          ("rda", "dist_only"), ("rda_guided", "dist_only")]
_LBL = {"dp_sample": "dp_sample (gen)", "carla_sample": "carla_sample (gen)",
        "gravity_independent": "gravity_indep (gen)", "dp_carla": "dp_carla (argmin w1)",
        "dp_carla_pot": "dp_carla_pot (argmin w1)", "rda": "rda", "rda_guided": "rda_guided"}


class _Shim:
    __slots__ = ("topology", "meta")

    def __init__(self, topology, meta):
        self.topology, self.meta = topology, meta


def _district_mask(world_name, topo):
    c, box = topo.coords, topo.box
    if world_name == "two_zone":
        cen = box / 2.0
        return np.hypot(c[:, 0] - cen, c[:, 1] - cen) < 0.13 * box
    return (c[:, 0] < 0.4 * box) & (c[:, 1] < 0.4 * box)


def _shock_mask(topo, district_mask, shock_anchors):
    if shock_anchors:
        return district_mask
    anchor = np.zeros(topo.coords.shape[0], dtype=bool)
    for t in ANCHORS:
        if t in topo.type_locs:
            anchor[np.asarray(topo.type_locs[t], dtype=int)] = True
    return district_mask & ~anchor


def _grid_hist(xy, box, edges):
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])
    s = h.sum()
    return h / s if s > 0 else h


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def _truth_field(world, world_name, topo, base_sizes, shock_mask, lv):
    """Free-leg true_to coords for the regenerated counterfactual at boost lv."""
    sizes = base_sizes.copy()
    sizes[shock_mask] *= lv
    w = regenerate_world(_Shim(topo, world.meta), world_name, NP_, sizes=sizes,
                         rng=np.random.default_rng(SEED + 1))
    gt = w.ground_truth
    f = gt[gt["to_is_free"]]
    return f[["true_to_x", "true_to_y"]].to_numpy(float)


# --------------------------------------------------------------------------- #
def plot_c6(df, world, regime):
    """pot-decile-TV vs boost (lower = placed visits match the attractiveness distribution)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for solver, cond in SERIES:
        sub = df[(df.solver == solver) & (df.condition == cond)].sort_values("level")
        if sub.empty or sub["pot_decile_tv"].isna().all():
            continue
        line(ax, sub["level"].to_numpy(float), sub["pot_decile_tv"].to_numpy(float),
             solver, label=_LBL.get(solver, solver))
    ax.set(xlabel="attractiveness boost $b$", ylabel="potential-decile TV (lower = better fit)",
           title=f"C6 potential-decile fit vs shock -- {WORLD_NAME.get(world, world)} ({regime})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"C6_potdecile_attractiveness_{regime}_{world}.{ext}"), dpi=130)
    plt.close(fig)
    print(f"  wrote C6 {world}/{regime}")


def plot_c5(raw, world, regime, topo, world_obj, base_sizes, shock_mask):
    """spatial-TV (predicted vs true field) vs boost."""
    apply_paper_style()
    box = topo.box
    edges = np.linspace(0, box, GRID + 1)
    id2xy = {lid: topo.coords[i] for i, lid in enumerate(topo.loc_ids)}
    levels = sorted(raw["level"].unique())
    truth = {lv: _grid_hist(_truth_field(world_obj, world, topo, base_sizes, shock_mask, lv), box, edges)
             for lv in levels}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for solver, cond in SERIES:
        xs, ys = [], []
        for lv in levels:
            sub = raw[(raw.solver == solver) & (raw.condition == cond) & (raw.level == lv)]
            if sub.empty:
                continue
            xy = np.array([id2xy[i] for i in sub["chosen_loc_id"] if i in id2xy], float)
            if xy.size == 0:
                continue
            xs.append(lv); ys.append(_tv(_grid_hist(xy, box, edges), truth[lv]))
        if xs:
            line(ax, xs, ys, solver, label=_LBL.get(solver, solver))
    ax.set(xlabel="attractiveness boost $b$",
           ylabel=f"spatial TV vs truth ({GRID}x{GRID} grid, lower=better)",
           title=f"C5 spatial reproduction vs shock -- {WORLD_NAME.get(world, world)} ({regime})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"C5_spatialtv_attractiveness_{regime}_{world}.{ext}"), dpi=130)
    plt.close(fig)
    print(f"  wrote C5 {world}/{regime}")


def plot_c7(world):
    """GT free-leg distance distribution b=1 vs b=6, both regimes."""
    apply_paper_style(whitegrid=False)
    w = load_world(os.path.join(ROOT, "data", "worlds", world))
    topo = w.topology
    base = np.asarray(topo.sizes, float).copy()
    dmask = _district_mask(world, topo)

    def freedist(sizes):
        ww = regenerate_world(_Shim(topo, w.meta), world, 4000, sizes=sizes,
                              rng=np.random.default_rng(1))
        g, p = ww.ground_truth, ww.plans_df
        fid = set(g.loc[g["to_is_free"], "unique_leg_id"])
        d = p[p["unique_leg_id"].isin(fid)]["distance_meters"].to_numpy(float)
        return d[np.isfinite(d)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, sa) in zip(axes, [("fixed anchors (secondary only)", False),
                                     ("full shock (incl.\\ homes/works)", True)]):
        smask = _shock_mask(topo, dmask, sa)
        d1 = freedist(base)
        s6 = base.copy(); s6[smask] *= 6.0
        d6 = freedist(s6)
        clip = float(np.percentile(np.concatenate([d1, d6]), 98))
        bins = np.linspace(0, clip, 45)
        ax.hist(d1, bins=bins, density=True, alpha=0.55, color="#4477aa",
                label=f"b=1  (median {np.median(d1):.0f} m)")
        ax.hist(d6, bins=bins, density=True, alpha=0.55, color="#cc6677",
                label=f"b=6  (median {np.median(d6):.0f} m)")
        ax.axvline(float(np.median(d1)), color="#1f5b8a", ls="--", lw=1.2)
        ax.axvline(float(np.median(d6)), color="#a3344a", ls="--", lw=1.2)
        ax.set(title=name, xlabel="free-leg distance (m)", ylabel="density")
        ax.legend()
    fig.suptitle(f"GT free-leg distance: no shock vs b=6 -- {WORLD_NAME.get(world, world)} (N=4000)")
    fig.tight_layout()
    out = os.path.join(OUT, f"C7_GTdistdist_{world}_b1_vs_b6.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote C7 {world}")


def main():
    print("=== C6 pot-decile (CSV) + C5 spatial-TV (regen) over finished cells ===")
    world_cache = {}
    for world in WORLDS:
        for regime, sa in [("fixed", False), ("full", True)]:
            csv = os.path.join(OUT, f"prognosis_attractiveness_{regime}_{world}.csv")
            parq = os.path.join(OUT, f"raw_legs_attractiveness_{regime}_{world}.parquet")
            if not os.path.exists(csv):
                print(f"  skip {world}/{regime} (no CSV -- not finished)")
                continue
            df = pd.read_csv(csv)
            plot_c6(df, world, regime)
            if not os.path.exists(parq):
                print(f"  skip C5 {world}/{regime} (no raw parquet)")
                continue
            if world not in world_cache:
                world_cache[world] = load_world(os.path.join(ROOT, "data", "worlds", world))
            wobj = world_cache[world]
            topo = wobj.topology
            base_sizes = np.asarray(topo.sizes, float).copy()
            shock_mask = _shock_mask(topo, _district_mask(world, topo), sa)
            raw = pd.read_parquet(parq)
            plot_c5(raw, world, regime, topo, wobj, base_sizes, shock_mask)

    print("=== C7 GT dist-distribution (osm, two_zone) ===")
    for world in ["osm_hannover", "two_zone"]:
        plot_c7(world)


if __name__ == "__main__":
    main()
