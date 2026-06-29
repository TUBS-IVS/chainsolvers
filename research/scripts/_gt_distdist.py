"""One-off: ground-truth free-leg distance distributions, no shock (b=1) vs b=6 attractiveness
boost, for both anchor regimes (gauss). Visualizes how little the TRUE distance distribution moves
under an attractiveness shock (the metric corollary). Regenerates truth directly (deterministic)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chainsolvers_eval.worlds import load_world
from chainsolvers_eval.regen import regenerate_world

ROOT = os.path.join(os.path.dirname(__file__), "..")
W = "gauss_hannover"
N = 4000
world = load_world(os.path.join(ROOT, "data", "worlds", W))
topo = world.topology
base = np.asarray(topo.sizes, float).copy()
c, box = topo.coords, topo.box
district = (c[:, 0] < 0.4 * box) & (c[:, 1] < 0.4 * box)
anchor = np.zeros(len(base), bool)
for t in ("home", "work"):
    if t in topo.type_locs:
        anchor[np.asarray(topo.type_locs[t], int)] = True


def freedist(sizes, seed=1):
    w = regenerate_world(world, W, N, sizes=sizes, rng=np.random.default_rng(seed))
    g, p = w.ground_truth, w.plans_df
    fid = set(g.loc[g["to_is_free"], "unique_leg_id"])
    d = p[p["unique_leg_id"].isin(fid)]["distance_meters"].to_numpy(float)
    return d[np.isfinite(d)]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (name, smask) in zip(axes, [("fixed anchors (secondary facilities only)", district & ~anchor),
                                    ("full shock (incl.\\ homes/works)", district)]):
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
fig.suptitle(f"Ground-truth free-leg distance distribution: no shock vs b=6 boost — {W} (N={N})")
fig.tight_layout()
out = os.path.join(ROOT, "out", "block_c", "GT_distdist_gauss_b1_vs_b6.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"fixed: would print medians inline above; see figure")
