"""Block C -- truth-free 'structural capability' figures (fixed-anchor regime).

No ground truth exists in a counterfactual on a known DGP (the DGP is itself a placer, and we
withhold its counterfactual outputs from the solvers), so we rank NOTHING against it. We read each
method's OWN response to an attractiveness lever: does its predicted district share move?

  C1 response   -- gauss: responsive (DP-sample, CARLA-sample, DP-full combined) vs inert (DP-full
                   distance-only = the exact distance optimum, RDA). DGP curve = faint reference only.
  C2 weight     -- OSM (matches the paper's quoted numbers): the DP-full attractiveness-weight
                   sweep; turning the term ON is the whole step, its precise weight barely matters.
  C-crossworld  -- OSM: the pattern reproduces on real geography (paper headline; gauss = C1).

Figure titles are world names only (paper house style: no narrative claims, no LaTeX escapes --
matplotlib renders \\% and --- literally); the finding lives in the caption/body text.
  C-distance    -- the lever moves WHERE activities locate, not HOW FAR (DGP distance distribution,
                   a property of the world).

Colours/markers/names follow block_a_style (house style); the two DP-full objectives share the
DP-full colour and are split by linestyle (same solver -> the objective, not the search, is the point).
Share panels read straight from the prognosis CSVs; the distance panel regenerates the DGP
counterfactual distances (deterministic, seed+1).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from block_a_style import apply_paper_style, line, COL, LABEL, WORLD_NAME

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out", "block_c")
SEED = 0
REF = dict(color="black", ls=(0, (4, 3)), lw=1.4, alpha=0.55, zorder=2)  # unscored DGP reference


def _load(world):
    d = pd.read_csv(os.path.join(OUT, f"prognosis_attractiveness_fixed_{world}.csv"))
    return d, d.base_main.iloc[0] * 100.0


def _curve(d, base, solver, cond):
    s = d[(d.solver == solver) & (d.condition == cond)].sort_values("level")
    return s["level"].to_numpy(float), s["outcome_main"].to_numpy(float) * 100.0 - base


def _resp_panel(ax, world):
    """Responsive-vs-inert response curves for one world, house style."""
    d, base = _load(world)
    x, y = _curve(d, base, "true", "true")
    ax.plot(x, y, label="gravity-DGP reference", **REF)
    # responsive -- carry an estimated attractiveness term (plain house names; roles live in the caption)
    for solver, cond, lab in [("dp_sample", "informed", LABEL["dp_sample"]),
                              ("carla_sample", "informed", LABEL["carla_sample"]),
                              ("dp_full", "w1", f"{LABEL['dp_full']} (combined)")]:
        xx, yy = _curve(d, base, solver, cond)
        if len(xx):
            line(ax, xx, yy, solver, label=lab)
    # inert -- distance-only objective (no attractiveness term)
    xx, yy = _curve(d, base, "dp_full", "w0")           # the EXACT distance optimum
    if len(xx):
        ax.plot(xx, yy, color=COL["dp_full"], ls=(0, (1, 1.5)), lw=1.6, marker="*", ms=7,
                mfc="none", mew=1.2, label=f"{LABEL['dp_full']} (distance-only)")
    xx, yy = _curve(d, base, "rda", "dist_only")
    if len(xx):
        line(ax, xx, yy, "rda")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=1)
    ax.set_xlabel("attractiveness multiplier $b$")
    ax.set_ylabel("change in district share of secondary stops (pp)")
    ax.grid(alpha=0.3)
    return base


def fig_response():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    _resp_panel(ax, "gauss_hannover")
    ax.set_title(WORLD_NAME["gauss_hannover"])
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "block_C1_response")


def fig_crossworld():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    _resp_panel(ax, "osm_hannover")
    ax.set_title(WORLD_NAME["osm_hannover"])
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "block_C_crossworld")


def fig_weight():
    # OSM: the paper quotes this world's numbers (+21.9 pp at weight 0.5, +29.4 at weight 20).
    apply_paper_style()
    d, base = _load("osm_hannover")
    weights = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x, y = _curve(d, base, "true", "true")
    ax.plot(x, y, label="gravity-DGP reference", **REF)
    cols = cm.viridis(np.linspace(0.05, 0.9, len(weights)))
    for w, c in zip(weights, cols):
        xx, yy = _curve(d, base, "dp_full", f"w{w:g}")
        if not len(xx):
            continue
        lab = f"weight {w:g}" + ("  (distance-only)" if w == 0 else "")
        ax.plot(xx, yy, color=c, marker="o", ms=4, lw=1.5, mfc="none", label=lab)
    ax.set(xlabel="attractiveness multiplier $b$",
           ylabel="change in district share of secondary stops (pp)",
           title=WORLD_NAME["osm_hannover"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, "block_C2_weight")


def fig_distance():
    apply_paper_style(whitegrid=False)
    world = "gauss_hannover"
    w = load_world(os.path.join(ROOT, "data", "worlds", world))
    topo = w.topology
    base = np.asarray(topo.sizes, float).copy()
    c, box = topo.coords, topo.box
    district = (c[:, 0] < 0.4 * box) & (c[:, 1] < 0.4 * box)
    anchor = np.zeros(len(base), bool)
    for t in ("home", "work"):
        if t in topo.type_locs:
            anchor[np.asarray(topo.type_locs[t], int)] = True
    smask = district & ~anchor                                   # fixed-anchor shock mask

    def freedist(sizes):
        ww = regenerate_world(_Shim(topo, w.meta), world, 4000, sizes=sizes,
                              rng=np.random.default_rng(SEED + 1))
        g, p = ww.ground_truth, ww.plans_df
        fid = set(g.loc[g["to_is_free"], "unique_leg_id"])
        d = p[p["unique_leg_id"].isin(fid)]["distance_meters"].to_numpy(float)
        return d[np.isfinite(d)]

    d1 = freedist(base)
    s6 = base.copy(); s6[smask] *= 6.0
    d6 = freedist(s6)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    clip = float(np.percentile(np.concatenate([d1, d6]), 98))
    bins = np.linspace(0, clip, 45)
    ax.hist(d1, bins=bins, density=True, alpha=0.55, color="#4477aa",
            label=f"$b=1$  (median {np.median(d1):.0f} m)")
    ax.hist(d6, bins=bins, density=True, alpha=0.55, color="#cc6677",
            label=f"$b=6$  (median {np.median(d6):.0f} m)")
    ax.axvline(float(np.median(d1)), color="#1f5b8a", ls="--", lw=1.2)
    ax.axvline(float(np.median(d6)), color="#a3344a", ls="--", lw=1.2)
    ax.set(xlabel="free-leg distance (m)", ylabel="density",
           title=f"{WORLD_NAME[world]}: free-leg distances, $b=1$ vs $b=6$")
    ax.legend()
    fig.tight_layout()
    _save(fig, "block_C_distance")


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=130)
    plt.close(fig)
    print("wrote", stem)


# imports used only by fig_distance (kept local-light at top for clarity)
from chainsolvers_eval.worlds import load_world          # noqa: E402
from chainsolvers_eval.regen import regenerate_world      # noqa: E402


class _Shim:                                              # noqa: E402
    __slots__ = ("topology", "meta")

    def __init__(self, topology, meta):
        self.topology, self.meta = topology, meta


if __name__ == "__main__":
    fig_response()
    fig_weight()
    fig_crossworld()
    fig_distance()
