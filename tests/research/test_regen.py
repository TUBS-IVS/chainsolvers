"""Counterfactual regeneration on baked worlds (chainsolvers_eval.regen).

Locks the plumbing that re-exposes the chain-generating DGP params (absent from a baked world's
meta.json) so a reloaded world can be regenerated under a shocked attractiveness field. These
tests assert the extraction is single-sourced from the builder constants and that regeneration
reproduces the builders' DGP on the reloaded topology — without touching world creation, the bake
process, or any baked artifact.
"""
import numpy as np
import pandas as pd

from chainsolvers_eval.osm import MID_INFLATE_OSM
from chainsolvers_eval.regen import (CHAIN_PARAM_KEYS, chain_params_for, regenerate_world,
                                     two_zone_bundles)
from chainsolvers_eval.synth import (CITY_PRESETS, city_world, generate_chains, two_zone_world,
                                     world_from_topology)
from chainsolvers_eval.worlds import load_world, save_world


# --- param extraction is single-sourced from the builder constants -----------------------

def test_chain_params_for_gauss_is_preset_chain_subset():
    cp = chain_params_for("gauss_hannover")
    # exactly the chain-side slice of the gauss preset -> cannot drift from city_world
    assert cp == {k: v for k, v in CITY_PRESETS["hannover"].items() if k in CHAIN_PARAM_KEYS}
    assert cp["templates"] is CITY_PRESETS["hannover"]["templates"]
    # topology-construction keys and the topology-mutating alias are NOT chain params
    for dropped in ("n_locations", "density_per_km2", "n_clusters", "spread_scale",
                    "type_prevalence", "type_aliases", "mobile_density_per_km2"):
        assert dropped not in cp


def test_chain_params_for_osm_uses_osm_inflate():
    cp = chain_params_for("osm_hannover")
    assert cp["decay_inflate"] == MID_INFLATE_OSM
    for k in ("templates", "pair_decay", "to_decay", "mode_decay", "mode_split"):
        assert k in cp


def test_chain_params_for_two_zone_refuses():
    import pytest
    with pytest.raises(ValueError):
        chain_params_for("two_zone")


def test_two_zone_bundles_heavy_tail_toggle():
    up, rp = two_zone_bundles(heavy_tail=True)
    assert "tail_frac" in up and "tail_scale" in up and "tail_frac" in rp
    upf, rpf = two_zone_bundles(heavy_tail=False)
    assert "tail_frac" not in upf and "tail_scale" not in upf
    # the body bundle (templates/decay) is present either way
    assert "templates" in upf and "pair_decay" in rpf


# --- regeneration reproduces the builder DGP on the reloaded topology ---------------------

def test_regenerate_gauss_determinism_and_matches_single_call(tmp_path):
    # small real-preset gauss world (override n_locations down so the build is fast)
    w = city_world("hannover", n_persons=150, n_locations=600, rng=np.random.default_rng(0))
    lw = load_world(save_world(w, str(tmp_path / "gauss")))

    base1 = regenerate_world(lw, "gauss_hannover", 150, rng=np.random.default_rng(7))
    base2 = regenerate_world(lw, "gauss_hannover", 150, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(base1.plans_df, base2.plans_df)  # deterministic in the rng

    # regenerate_world == the documented single world_from_topology call with the extracted bundle
    direct = world_from_topology(lw.topology, 150, rng=np.random.default_rng(7),
                                 distance_noise=0.0, **chain_params_for("gauss_hannover"))
    pd.testing.assert_frame_equal(base1.plans_df, direct.plans_df)

    # a shock to the structural sizes moves placement (the counterfactual actually bites)
    shocked = lw.topology.sizes.copy()
    shocked[:60] *= 8.0
    cf = regenerate_world(lw, "gauss_hannover", 150, sizes=shocked, rng=np.random.default_rng(7))
    assert not base1.plans_df.equals(cf.plans_df)
    # potentials are usage-derived: same number of placements, but redistributed by the shock
    assert cf.locations_tuple[2]["work"].sum() == base1.locations_tuple[2]["work"].sum()


def test_regenerate_two_zone_reproduces_two_pool_dgp(tmp_path):
    # tiny non-heavy-tail two-zone fixture (fast), explicit split
    w = two_zone_world(box=4000.0, core_frac_side=0.4, core_density=20.0, ring_density=6.0,
                       n_persons=120, rural_pop_share=0.4, heavy_tail=False,
                       rng=np.random.default_rng(0))
    lw = load_world(save_world(w, str(tmp_path / "two_zone")))

    re1 = regenerate_world(lw, "two_zone", 120, rng=np.random.default_rng(5), heavy_tail=False)
    re2 = regenerate_world(lw, "two_zone", 120, rng=np.random.default_rng(5), heavy_tail=False)
    pd.testing.assert_frame_equal(re1.plans_df, re2.plans_df)  # deterministic

    # split follows the baked urban:rural ratio, both sub-populations present
    assert re1.meta["n_urban"] + re1.meta["n_rural"] == 120
    assert re1.meta["n_urban"] == 72 and re1.meta["n_rural"] == 48
    assert set(re1.plans_df["unique_person_id"].str[0]) == {"u", "r"}

    # dispatch reproduces the manual two-pool generate_chains exactly (same rng, same bundles).
    # Split core/ring homes by the stable construction-order loc id (NOT the positional index, which
    # load_world reorders -> the old `home < n_core` index test silently mis-split the reload).
    rng = np.random.default_rng(5)
    up, rp = two_zone_bundles(heavy_tail=False)
    n_core = int(lw.meta["n_core_fac"])
    locnum = np.array([int(str(s)[3:]) for s in lw.topology.loc_ids])
    home = lw.topology.type_locs["home"]
    core_home, ring_home = home[locnum[home] < n_core], home[locnum[home] >= n_core]
    ru, _, _ = generate_chains(lw.topology, 72, home_pool=core_home, person_prefix="u",
                               distance_noise=0.0, rng=rng, **up)
    rr, _, _ = generate_chains(lw.topology, 48, home_pool=ring_home, person_prefix="r",
                               distance_noise=0.0, rng=rng, **rp)
    pd.testing.assert_frame_equal(re1.plans_df, pd.DataFrame(ru + rr))

    # GROUND-TRUTH check (non-circular) — this is what the old self-referential test missed. In the
    # ORIGINAL (pre-reload) topology the positional index IS construction order, so `home < n_core`
    # there is the TRUE core/ring split. The reloaded recovery must reproduce that exact set of core
    # homes. The positional-index split on the *reloaded* topo (the bug) returns a different set
    # (here 51 vs 44, 7 ring homes mislabelled core) and fails this assertion.
    o_home = w.topology.type_locs["home"]
    true_core_ids = set(w.topology.loc_ids[o_home[o_home < n_core]].tolist())
    recovered_core_ids = set(lw.topology.loc_ids[core_home].tolist())
    assert recovered_core_ids == true_core_ids
    assert 0 < len(core_home) < len(home)

    # the counterfactual shock changes placement
    shocked = lw.topology.sizes.copy()
    shocked[:10] *= 8.0
    cf = regenerate_world(lw, "two_zone", 120, sizes=shocked, rng=np.random.default_rng(5),
                          heavy_tail=False)
    assert not re1.plans_df.equals(cf.plans_df)
