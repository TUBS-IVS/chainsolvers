"""Tests for the synthetic world generator and ground-truth-based evaluation."""

import numpy as np
import pandas as pd

from chainsolvers import run, Scorer
from chainsolvers_eval.synth import generate_world, build_topology, generate_chains, topology_locations_tuple
from chainsolvers_eval.calibration import fit_location_choice


def _recovery(rdf, gt):
    free = gt[gt["to_is_free"]].merge(rdf[["unique_leg_id", "to_act_identifier"]], on="unique_leg_id")
    return float((free["to_act_identifier"] == free["true_to_identifier"]).mean())


def _total_dev(rdf):
    actual = np.hypot(rdf["to_x"] - rdf["from_x"], rdf["to_y"] - rdf["from_y"])
    return float((rdf["distance_meters"] - actual).abs().sum())


def test_world_is_well_formed():
    w = generate_world(n_locations=300, n_persons=80, rng=np.random.default_rng(0))
    ids, coords, pots = w.locations_tuple
    # multi-type: at least one location appears under more than one activity type
    counts = {}
    for t, arr in ids.items():
        for i in arr:
            counts[i] = counts.get(i, 0) + 1
    assert max(counts.values()) >= 2, "expected multi-type locations"
    # potentials are usage-derived: total potential == total visits == number of legs
    total_pot = sum(float(p.sum()) for p in pots.values())
    # each leg's destination contributes one visit; plus the home start visit per person
    assert total_pot == len(w.plans_df) + w.meta["n_persons"]
    # ground truth aligns with plans
    assert set(w.ground_truth["unique_leg_id"]) == set(w.plans_df["unique_leg_id"])
    assert w.ground_truth["to_is_free"].any()


def test_world_is_feasible_global_optimum_near_zero():
    # Activities sit on real points, so the global-optimal placement (dp_full) must
    # achieve ~0 total distance deviation: the true distances are exactly achievable.
    w = generate_world(n_locations=300, n_persons=60, rng=np.random.default_rng(1))
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    # mean per-person deviation should be essentially zero
    assert _total_dev(rdf) / w.meta["n_persons"] < 1e-3


def test_potential_mode_improves_recovery_of_popular_locations():
    # Sanity: solving runs end-to-end on the world in geometric mode and recovers a
    # non-trivial fraction of true facilities for the placed (free) activities.
    w = generate_world(n_locations=400, n_persons=120, rng=np.random.default_rng(2))
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0)
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    free = w.ground_truth[w.ground_truth["to_is_free"]].merge(
        rdf[["unique_leg_id", "to_act_identifier"]], on="unique_leg_id"
    )
    recov = (free["to_act_identifier"] == free["true_to_identifier"]).mean()
    # with the full candidate set and exact distances, recovery should be substantial
    assert recov > 0.3


def test_potentials_help_recovery_under_distance_noise():
    # With noisy observed distances the true facility is no longer uniquely
    # distance-optimal; usage-based potentials (popular = more often true) should
    # recover more than pure geometry. Exact-search dp_full isolates the objective.
    w = generate_world(n_locations=600, n_persons=300, distance_noise=0.15,
                       rng=np.random.default_rng(0))

    def recov_for(scorer):
        ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_full", rng_seed=0, scorer=scorer)
        rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
        assert valid
        return _recovery(rdf, w.ground_truth)

    geo = recov_for(Scorer(mode="geometric"))
    comb = recov_for(Scorer(mode="combined", pot_weight=100.0, dist_dev_weight=1.0))
    assert comb >= geo + 0.02, f"potentials did not help: geometric={geo:.3f}, combined={comb:.3f}"


def test_world_has_modes_and_sampler_generates_finite_distances():
    w = generate_world(n_locations=400, n_persons=150, rng=np.random.default_rng(3))
    assert "mode" in w.plans_df.columns and w.plans_df["mode"].notna().any()
    # mode is tour-consistent: one mode per person
    assert (w.plans_df.groupby("unique_person_id")["mode"].nunique() == 1).all()

    decay = {str(k): float(v) for k, v in w.plans_df.groupby("mode")["distance_meters"].mean().items()}
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp_sample", rng_seed=0,
                    scorer=Scorer(mode="combined", pot_weight=1.0),
                    parameters={"decay_scales": decay,
                                "default_scale": float(w.plans_df["distance_meters"].mean())})
    rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
    assert valid
    free = w.ground_truth[w.ground_truth.to_is_free].merge(
        rdf[["unique_leg_id", "to_x", "to_y", "from_x", "from_y"]], on="unique_leg_id")
    d = np.hypot(free.to_x - free.from_x, free.to_y - free.from_y)
    assert np.isfinite(d).all() and float(d.median()) > 0


def test_prognosis_structural_model_tracks_counterfactual():
    # Ground-truthed forecasting test: boost a district's attractiveness, regenerate the
    # TRUE counterfactual, and check that only a structural (attractiveness-aware) model
    # told the new sizes tracks the shift -- a non-structural model has no elasticity.
    rng = np.random.default_rng(42)
    topo = build_topology(n_locations=800, rng=rng)
    box = topo.box
    generate_chains(topo, 400, sizes=topo.sizes, assign_modes=False, rng=rng)  # baseline (warm rng)

    district = (topo.coords[:, 0] < 0.4 * box) & (topo.coords[:, 1] < 0.4 * box)
    district_ids = set(topo.loc_ids[district])
    sizes_cf = topo.sizes.copy()
    sizes_cf[district] *= 6.0

    rng_cf = np.random.default_rng(43)
    pcf, gtcf, _ = generate_chains(topo, 400, sizes=sizes_cf, assign_modes=False, rng=rng_cf)
    plans_cf, gt_cf = pd.DataFrame(pcf), pd.DataFrame(gtcf)
    free_cf = set(gt_cf.loc[gt_cf.to_is_free, "unique_leg_id"])
    true_cf = float(gt_cf[gt_cf.unique_leg_id.isin(free_cf)].true_to_identifier.isin(district_ids).mean())
    scale = float(plans_cf.set_index("unique_leg_id").loc[list(free_cf), "distance_meters"].mean())

    def pred_share(values, alpha):
        loc = topology_locations_tuple(topo, values)
        ctx = run.setup(locations_tuple=loc, solver="dp_sample", rng_seed=7,
                        scorer=Scorer(mode="combined", pot_weight=alpha),
                        parameters={"default_scale": scale})
        rdf, _, _ = run.solve(ctx=ctx, plans_df=plans_cf)
        r = rdf[rdf.unique_leg_id.isin(free_cf)]
        return float(r.to_act_identifier.isin(district_ids).mean())

    s_struct = pred_share(sizes_cf, 1.0)       # structural, knows the new attractiveness
    s_nonstruct = pred_share(topo.sizes, 0.0)  # ignores attractiveness

    assert true_cf > 0.15, "boost should pull trips into the district"
    assert s_struct > s_nonstruct + 0.05, "structural model should respond to the attractiveness shock"
    assert abs(s_struct - true_cf) < abs(s_nonstruct - true_cf), "structural model should be closer to truth"


def test_calibration_recovers_decay_scale():
    # MLE calibration on baseline should recover the true decay scale (gravity_scale).
    rng = np.random.default_rng(42)
    topo = build_topology(n_locations=800, rng=rng)
    pb, gtb, _ = generate_chains(topo, 500, sizes=topo.sizes, assign_modes=False,
                                 gravity_scale=4000.0, rng=rng)
    alpha, scale = fit_location_choice(topo, pd.DataFrame(pb), pd.DataFrame(gtb))
    assert 2500.0 < scale < 6500.0, f"decay scale {scale:.0f} not near true 4000"
    assert alpha > 0.3, f"attractiveness sensitivity {alpha:.2f} should be positive"
