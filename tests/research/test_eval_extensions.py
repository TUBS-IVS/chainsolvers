"""Tests for the eval extensions: heterogeneity, distribution-fit metrics, OSM geometry
loader, survey decay calibration, and world-from-topology."""

import os

import numpy as np
import pandas as pd
import pytest

from chainsolvers import run
from chainsolvers.scoring_selection import Scorer
from chainsolvers_eval import survey as S
from chainsolvers_eval.calibration import (fit_location_choice_mixture, fit_mode_kernels,
                                           fit_mode_powerlaw)
from chainsolvers_eval.metrics import (distribution_fit, free_leg_distances, grouped_distribution_fit,
                                       ks_statistic)
from chainsolvers_eval.osm import project_latlon, topology_from_pois
from chainsolvers_eval.synth import (build_topology, generate_world, merge_topologies,
                                     two_zone_world, world_from_topology)


# --- Item 4: heterogeneity --------------------------------------------------------------
def test_heterogeneity_makes_central_core():
    box = 10000.0
    c = box / 2.0
    kw = dict(n_locations=3000, box=box, n_clusters=6)
    homo = build_topology(heterogeneity=0.0, rng=np.random.default_rng(0), **kw)
    inho = build_topology(heterogeneity=0.85, rng=np.random.default_rng(0), **kw)

    def central_frac(t):
        r = np.hypot(t.coords[:, 0] - c, t.coords[:, 1] - c)
        return float((r < box * 0.2).mean())

    assert central_frac(inho) > central_frac(homo) + 0.1, "core should be denser"
    # central facilities are more attractive under heterogeneity
    r = np.hypot(inho.coords[:, 0] - c, inho.coords[:, 1] - c)
    assert inho.sizes[r < box * 0.2].mean() > inho.sizes[r > box * 0.4].mean()


def test_heterogeneity_zero_is_legacy():
    # het=0 must reproduce the old topology exactly (same RNG stream).
    a = build_topology(n_locations=500, box=8000.0, n_clusters=5, rng=np.random.default_rng(3))
    b = build_topology(n_locations=500, box=8000.0, n_clusters=5, heterogeneity=0.0,
                       rng=np.random.default_rng(3))
    assert np.allclose(a.coords, b.coords) and np.allclose(a.sizes, b.sizes)


# --- Item 3: distribution-fit metrics ---------------------------------------------------
def test_ks_and_distribution_fit():
    rng = np.random.default_rng(0)
    a = rng.normal(1000, 150, 4000)
    assert ks_statistic(a, a.copy()) < 1e-9
    assert distribution_fit(a, a)["ks"] < 0.05
    shifted = a + 2000
    f = distribution_fit(a, shifted)
    assert f["ks"] > 0.3 and f["q50_abs_err_m"] > 1000


def test_free_leg_distances_grouped():
    w = generate_world(n_locations=300, n_persons=80, rng=np.random.default_rng(1))
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    pooled = free_leg_distances(rdf, w.ground_truth)
    assert pooled.ndim == 1 and np.isfinite(pooled).all() and pooled.size > 0
    by_purpose = free_leg_distances(rdf, w.ground_truth, by="to_act_type")
    assert isinstance(by_purpose, dict) and len(by_purpose) >= 1
    gdf = grouped_distribution_fit(by_purpose, by_purpose)
    assert (gdf["ks"] < 1e-6).all()  # self-fit is perfect


# --- Item 1: OSM geometry loader --------------------------------------------------------
def test_topology_from_pois_counts_and_types():
    df = pd.DataFrame({
        "type": ["shop", "shop", "leisure", "home", "work"],
        "x": [0.0, 100.0, 200.0, 50.0, 300.0],
        "y": [0.0, 0.0, 100.0, 50.0, 0.0],
        "weight": [5.0, 1.0, 2.0, 1.0, 3.0],
    })
    topo = topology_from_pois(df, weight_col="weight")  # use the supplied capacity column
    assert set(topo.types) == {"shop", "leisure", "home", "work"}
    assert len(topo.type_locs["shop"]) == 2 and len(topo.type_locs["leisure"]) == 1
    assert topo.coords.shape == (5, 2)
    assert topo.coords.min() == 0.0  # shifted to non-negative box
    assert topo.sizes[topo.type_locs["shop"]].max() == 5.0
    # default (no weight_col) draws random lognormal sizes, ignoring the column
    rnd = topology_from_pois(df, rng=np.random.default_rng(0))
    assert not np.allclose(rnd.sizes[topo.type_locs["shop"]], [5.0, 1.0])


def test_project_latlon_origin():
    x, y = project_latlon(np.array([52.37]), np.array([9.73]), 52.37, 9.73)
    assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-6


def test_load_real_osm_snapshot_if_present():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "research", "data", "hannover_pois.csv")
    if not os.path.exists(path):
        pytest.skip("real OSM snapshot not fetched (research/data gitignored)")
    from chainsolvers_eval.osm import load_poi_csv
    topo = load_poi_csv(path)
    assert {"home", "shop", "leisure", "education", "work", "other"} <= set(topo.types)
    assert topo.box > 0
    # attractiveness weights are real (tag-derived), not uniform
    assert topo.sizes[topo.type_locs["shop"]].std() > 0


def test_hannover_osm_world_solves_if_present():
    from chainsolvers_eval.osm import DEFAULT_HANNOVER_POIS, hannover_osm_world
    if not os.path.exists(DEFAULT_HANNOVER_POIS):
        pytest.skip("real OSM snapshot not fetched (research/data gitignored)")
    w = hannover_osm_world(800, rng=np.random.default_rng(0))
    assert w.meta["n_free_legs"] > 0
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    _, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid


def test_hannover_real_homes_if_present():
    from chainsolvers_eval.osm import DEFAULT_HANNOVER_HOMES, DEFAULT_HANNOVER_POIS, hannover_topology
    if not (os.path.exists(DEFAULT_HANNOVER_POIS) and os.path.exists(DEFAULT_HANNOVER_HOMES)):
        pytest.skip("real Hannover snapshots not present (research/data gitignored)")
    topo = hannover_topology(rng=np.random.default_rng(0))
    hs = topo.sizes[topo.type_locs["home"]]
    # home sizes are household counts (real density) -> wide spread, apartment blocks ≫ houses
    assert hs.max() > 10 and hs.std() > 1.5
    # secondary facilities keep random (small-spread) sizes
    assert topo.sizes[topo.type_locs["shop"]].max() < hs.max()


# --- Item 2: survey decay calibration + world_from_topology -----------------------------
def test_transplant_survey_chains_no_gt_solves():
    # Realistic survey regime: WRONG-person donor chains placed on region homes. No resident
    # ground truth, but the instance must be a valid, solvable placement problem.
    w = generate_world(n_locations=600, n_persons=300, distance_noise=0.05,
                       rng=np.random.default_rng(7))
    rng = np.random.default_rng(0)
    sp, _ = S.draw_survey(w.plans_df, w.ground_truth, frac=0.3, rng=rng)
    homes = S.home_coords(w.plans_df)
    tp, free = S.transplant_survey_chains(homes, sp, n_persons=120, rng=rng)

    # free_frame marks only secondary (non-anchor) legs as free, and there is no truth column.
    assert set(free.columns) == {"unique_leg_id", "to_is_free"}
    assert free["to_is_free"].any() and not free["to_is_free"].all()

    # Free legs have unknown (NaN) to-coords on input; anchor legs are known.
    m = tp.set_index("unique_leg_id")
    fid = free.loc[free["to_is_free"], "unique_leg_id"]
    aid = free.loc[~free["to_is_free"], "unique_leg_id"]
    assert m.loc[fid, ["to_x", "to_y"]].isna().all().all()
    assert m.loc[aid, ["to_x", "to_y"]].notna().all().all()

    # Solves, and the placed free distribution is scorable using only the free_frame (no GT).
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=tp)
    assert valid
    d = free_leg_distances(rdf, free)
    assert d.ndim == 1 and np.isfinite(d).all() and d.size > 0


def test_sample_persons_deterministic_keeps_potentials_and_meta():
    # Canonical person sampler: same seed -> byte-identical population (so all solvers/regimes see
    # the same persons); full-population potentials kept; GT aligned to the subset; meta refreshed.
    w = generate_world(n_locations=500, n_persons=200, distance_noise=0.05,
                       rng=np.random.default_rng(11))
    a = S.sample_persons(w, 50, seed=7)
    b = S.sample_persons(w, 50, seed=7)

    # deterministic in seed: identical person ids AND identical rows across two calls (call order
    # cannot perturb the draw — it is not a shared advancing rng)
    pa = list(a.plans_df["unique_person_id"].unique())
    assert pa == list(b.plans_df["unique_person_id"].unique())
    assert len(pa) == 50
    assert a.plans_df.equals(b.plans_df) and a.ground_truth.equals(b.ground_truth)

    # full-population locations_tuple/potentials kept VERBATIM (same object, the dense usage field)
    assert a.locations_tuple is w.locations_tuple

    # ground_truth aligns exactly to the sampled plans' legs
    assert set(a.ground_truth["unique_leg_id"]) == set(a.plans_df["unique_leg_id"])

    # meta n_* refreshed to the subset; unrelated meta preserved
    assert a.meta["n_persons"] == 50
    assert a.meta["n_legs"] == len(a.plans_df)
    assert a.meta["n_free_legs"] == int(a.ground_truth["to_is_free"].sum())
    assert a.meta["n_locations"] == w.meta["n_locations"]

    # a different seed generally yields a different person set
    c = S.sample_persons(w, 50, seed=8)
    assert list(c.plans_df["unique_person_id"].unique()) != pa

    # n >= population -> the whole world is returned (all persons)
    allw = S.sample_persons(w, 10_000, seed=7)
    assert set(allw.plans_df["unique_person_id"]) == set(w.plans_df["unique_person_id"])
    assert allw.meta["n_persons"] == w.meta["n_persons"]

    # the sampled world still solves
    ctx = run.setup(locations_tuple=a.locations_tuple, solver="dp_full", rng_seed=0)
    _, _, valid = run.solve(ctx=ctx, plans_df=a.plans_df)
    assert valid


def test_disturb_anchor_keeps_gt_and_moves_work():
    # Anchor-quality axis: jitter work, keep GT. Work coords move; secondary (free) coords and GT
    # are untouched, and node-sharing stays consistent (leg k.to == leg k+1.from for work).
    w = generate_world(n_locations=500, n_persons=150, rng=np.random.default_rng(4))
    rng = np.random.default_rng(0)
    out = S.disturb_anchor(w.plans_df, anchor="work", noise_m=300.0, rng=rng)

    base = w.plans_df.set_index("unique_leg_id")
    o = out.set_index("unique_leg_id")
    work_to = base["to_act_type"] == "work"
    assert work_to.any()
    # work arrival coords moved; non-anchor (free) arrivals did not (GT-consistent inputs)
    moved = ~np.isclose(o.loc[work_to, "to_x"], base.loc[work_to, "to_x"])
    assert moved.all()
    free_to = ~base["to_act_type"].isin(["home", "work"])
    assert np.allclose(o.loc[free_to, "to_x"].fillna(0), base.loc[free_to, "to_x"].fillna(0))
    # still solvable
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    _, _, valid = run.solve(ctx=ctx, plans_df=out)
    assert valid


def test_demote_anchor_frees_work_and_flips_gt():
    # Anchor-quality axis (extreme): work demoted to a free node, GT flipped so it is scored.
    w = generate_world(n_locations=500, n_persons=150, rng=np.random.default_rng(6))
    pl, g = S.demote_anchor(w.plans_df, w.ground_truth, anchor="work")

    work_ids = set(w.plans_df.loc[w.plans_df["to_act_type"] == "work", "unique_leg_id"])
    assert work_ids
    # truth flipped to free for work legs; their true coords are still present for scoring
    assert g.loc[g["unique_leg_id"].isin(work_ids), "to_is_free"].all()
    assert g.loc[g["unique_leg_id"].isin(work_ids), "true_to_x"].notna().all()
    # work arrival coords are now unknown (NaN) -> the solver must place them
    m = pl.set_index("unique_leg_id")
    assert m.loc[list(work_ids), ["to_x", "to_y"]].isna().all().all()
    # solves, and recovery is now computable over the enlarged free set (incl. work)
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=pl)
    assert valid
    free = g[g["to_is_free"]].merge(rdf[["unique_leg_id", "to_act_identifier"]], on="unique_leg_id")
    assert len(free) > 0 and "true_to_identifier" in free.columns


def test_decay_from_survey_keys():
    w = generate_world(n_locations=400, n_persons=200, rng=np.random.default_rng(2))
    dec = S.decay_from_survey(w.plans_df, w.ground_truth)
    assert dec and all(isinstance(k, tuple) and len(k) == 2 for k in dec)
    assert all(v > 0 for v in dec.values())


def test_merge_topologies():
    a = build_topology(n_locations=200, box=5000.0, n_clusters=3, rng=np.random.default_rng(0))
    b = build_topology(n_locations=300, box=5000.0, n_clusters=4, rng=np.random.default_rng(1))
    m = merge_topologies(a, b, box=5000.0)
    assert m.coords.shape[0] == 500
    for t in set(a.types) | set(b.types):
        assert len(m.type_locs[t]) == len(a.type_locs.get(t, [])) + len(b.type_locs.get(t, []))
    assert m.type_locs["home"].max() < 500


def test_two_zone_urban_rural_contrast():
    # Small fast world; rural ring persons should travel markedly farther (car-dominated) than
    # urban core persons — the whole point of the two-zone calibration.
    w = two_zone_world(n_persons=400, box=14000.0, core_frac_side=0.4, core_density=40.0,
                       ring_density=8.0, rural_pop_share=0.4, rng=np.random.default_rng(0))
    df = w.plans_df
    free = df[df["unique_leg_id"].isin(set(
        w.ground_truth.loc[w.ground_truth["to_is_free"], "unique_leg_id"]))]
    urb = free[free["unique_person_id"].str.startswith("u")]["distance_meters"].median()
    rur = free[free["unique_person_id"].str.startswith("r")]["distance_meters"].median()
    # Rural longer (rural MiD decays, car-dominated). Margin is ~1.2x on this small non-heavy-tail
    # fixture (fixed seed -> deterministic): the calibrated mix gives a sparse work pool, so the
    # long business trips it produces lengthen the *urban* median too, and the 14 km box truncates
    # rural car -- both compress the contrast here. The baked 120 km heavy-tail super-region shows
    # the full contrast (rural car 9.5 vs urban 5.9 km), validated in data/worlds/*/vs_mid_*.png.
    assert rur > 1.2 * urb, f"rural ({rur:.0f}) should be >> urban ({urb:.0f})"
    assert w.meta["n_urban"] + w.meta["n_rural"] == w.meta["n_persons"]
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    _, _, valid = run.solve(ctx=ctx, plans_df=df)
    assert valid


def test_heavy_tail_fattens_car():
    # the local-vs-global sampler must lengthen the car tail (long inter-regional trips).
    kw = dict(n_persons=800, box=50000.0, core_frac_side=0.3, ring_density=6.0)
    base = two_zone_world(rng=np.random.default_rng(0), **kw)
    tail = two_zone_world(rng=np.random.default_rng(0),
                          urban_params={"tail_frac": {"car": 0.3}, "tail_scale": {"car": 60000.0}},
                          rural_params={"tail_frac": {"car": 0.3}, "tail_scale": {"car": 60000.0}}, **kw)

    def car_p95(w):
        c = w.plans_df.loc[w.plans_df["mode"] == "car", "distance_meters"].to_numpy()
        return float(np.percentile(c, 95))

    assert car_p95(tail) > 1.15 * car_p95(base), "tail should extend the car distribution"


def _free_dists(world, **dp_params):
    """dp_sample free-leg straight-line distances under the given solver params."""
    ctx = run.setup(locations_tuple=world.locations_tuple, solver="dp_sample", rng_seed=1,
                    scorer=Scorer(mode="combined", pot_weight=1.0), parameters=dp_params)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=world.plans_df)
    assert valid
    return free_leg_distances(rdf, world.ground_truth)


# --- generative dp_sample: structural tail calibration (Block B) ------------------------
def test_mixture_mle_reports_tail_only_when_present():
    # The structural mixture MLE must not manufacture a tail: on a single-decay (no long-range)
    # DGP it returns ~no tail weight; on a DGP with a planted long-range component it returns a
    # clearly larger weight. This is the empirical answer to "does joint calibration overfit the
    # generator" -- it detects real structure, it doesn't invent it.
    from chainsolvers_eval.synth import generate_world
    common = dict(n_locations=900, n_persons=500, assign_modes=False, gravity_scale=4000.0)

    flat = generate_world(rng=np.random.default_rng(0), **common)
    _, _, w_flat, _ = fit_location_choice_mixture(flat.topology, flat.plans_df, flat.ground_truth)

    heavy = generate_world(rng=np.random.default_rng(0), tail_frac={None: 0.30},
                           tail_scale={None: 30000.0}, **common)
    _, _, w_heavy, ts_heavy = fit_location_choice_mixture(heavy.topology, heavy.plans_df,
                                                          heavy.ground_truth)

    assert w_flat < 0.12, f"spurious tail on a no-tail world: w={w_flat:.3f}"
    assert w_heavy > w_flat + 0.05, f"failed to detect a planted tail: {w_heavy:.3f} vs {w_flat:.3f}"
    assert ts_heavy > 0  # long kernel is the longer of the two (enforced by the parameterization)


def test_heavy_tail_kernel_widens_sample_tail_keeps_body():
    # The dp_sample mixture kernel must lengthen the long-distance tail while leaving the body
    # (median) roughly intact -- the whole point of a body+tail mixture over a single scale.
    from chainsolvers_eval.synth import generate_world
    # roomy box so the tail has room to extend (a tight box truncates it regardless of the kernel).
    w = generate_world(n_locations=800, n_persons=500, box=40000.0, rng=np.random.default_rng(2))
    base = _free_dists(w, default_scale=4000.0, attr_transform="log1p")
    tail = _free_dists(w, default_scale=4000.0, tail_weight=0.20, tail_scale_factor=3.5,
                       attr_transform="log1p")
    assert np.quantile(tail, 0.90) > 1.20 * np.quantile(base, 0.90), "tail should extend"
    assert abs(np.median(tail) - np.median(base)) < 0.40 * np.median(base), "body should hold"


def test_fit_mode_kernels_per_mode_and_feeds_dp_sample():
    # Per-mode structural calibration: returns per-mode body scale + tail shape that plug straight
    # into dp_sample (mode-dict params), and the resulting solve is valid. On a multi-mode world the
    # modes should not collapse to one scale (car travels farther than walk).
    from chainsolvers_eval.synth import generate_world
    w = generate_world(n_locations=700, n_persons=400, rng=np.random.default_rng(5))
    alpha, decay, tw, tf, pooled = fit_mode_kernels(w.topology, w.plans_df, w.ground_truth,
                                                    max_persons=300)
    assert decay and set(decay) == set(tw) == set(tf), "per-mode dicts must share mode keys"
    assert len(decay) >= 2, "a multi-mode world should yield several mode kernels"
    assert max(decay.values()) > 1.3 * min(decay.values()), "mode body scales should differ"
    assert all(v >= 0 for v in tw.values()) and len(pooled) == 3

    # the per-mode maps feed dp_sample directly and produce a valid placement
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=1,
                    scorer=Scorer(mode="combined", pot_weight=alpha),
                    parameters={"decay_scales": decay, "tail_weights": tw,
                                "tail_scale_factors": tf, "default_scale": pooled[0]})
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    d = free_leg_distances(rdf, w.ground_truth)
    assert np.isfinite(d).all() and d.size > 0


def test_powerlaw_kernel_fits_and_solves():
    # Power-law distance kernel (dist_kernel="powerlaw"): scale pinned to the exp MLE (well-identified)
    # with only the exponent k fitted (no s-k ridge divergence -> finite, sane k). The per-mode maps
    # feed dp_sample and produce a valid placement with a heavier-than-exponential tail.
    from chainsolvers_eval.synth import generate_world
    w = generate_world(n_locations=700, n_persons=400, rng=np.random.default_rng(8))
    alpha, decay, shapes, pooled = fit_mode_powerlaw(w.topology, w.plans_df, w.ground_truth,
                                                     max_persons=300)
    assert decay and set(decay) == set(shapes) and len(decay) >= 2
    assert all(np.isfinite(k) and 0.1 < k < 50 for k in shapes.values()), "exponents must be sane (no ridge)"
    s_pool, k_pool = pooled
    assert np.isfinite(s_pool) and np.isfinite(k_pool) and k_pool > 0

    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=1,
                    scorer=Scorer(mode="combined", pot_weight=alpha),
                    parameters={"dist_kernel": "powerlaw", "decay_scales": decay,
                                "dist_shapes": shapes, "default_scale": s_pool, "dist_shape": k_pool})
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    d = free_leg_distances(rdf, w.ground_truth)
    assert np.isfinite(d).all() and d.size > 0


def test_world_from_topology_solves():
    rng = np.random.default_rng(5)
    n = 60
    rows = []
    for t in ["home", "work", "shop", "leisure"]:
        xy = rng.uniform(0, 6000, size=(n, 2))
        rows.append(pd.DataFrame({"type": t, "x": xy[:, 0], "y": xy[:, 1]}))
    topo = topology_from_pois(pd.concat(rows, ignore_index=True))
    w = world_from_topology(topo, 40, templates=[(("home", "shop", "work"), 1.0)],
                            assign_modes=False, gravity_scale=3000.0, rng=rng)
    assert w.meta["n_free_legs"] > 0
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid


def test_save_load_world_round_trip(tmp_path):
    # A baked world must reload byte-faithfully and stay solvable (the bake_worlds plumbing).
    from chainsolvers_eval.worlds import load_world, save_world

    w = two_zone_world(n_persons=300, heavy_tail=True, rng=np.random.default_rng(3))
    d = str(tmp_path / "w")
    save_world(w, d)
    r = load_world(d)

    # locations_tuple (ids/coords/potentials) identical per type
    ids0, co0, po0 = w.locations_tuple
    ids1, co1, po1 = r.locations_tuple
    assert list(ids1.keys()) == list(w.meta["types"])
    for t in w.meta["types"]:
        assert np.array_equal(ids0[t].astype(str), ids1[t].astype(str))
        assert np.allclose(co0[t], co1[t])
        assert np.allclose(po0[t], po1[t])
    # plans / ground_truth survive (same rows), topology rebuilt with real latent sizes
    assert len(r.plans_df) == len(w.plans_df)
    assert len(r.ground_truth) == len(w.ground_truth)
    assert r.anchor_types == w.anchor_types
    assert r.topology.coords.shape == w.topology.coords.shape
    assert not np.isnan(r.topology.sizes).any()
    # the reloaded world still solves
    ctx = run.setup(locations_tuple=r.locations_tuple, solver="carla", rng_seed=0)
    _, _, valid = run.solve(ctx=ctx, plans_df=r.plans_df)
    assert valid
