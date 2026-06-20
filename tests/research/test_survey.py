"""Tests for the survey-realistic evaluation primitives."""

import numpy as np
import pandas as pd

from chainsolvers_eval.synth import build_topology, generate_chains
from chainsolvers_eval import survey as sv


def _world(seed=0, n_loc=500, n_persons=400):
    rng = np.random.default_rng(seed)
    topo = build_topology(n_locations=n_loc, rng=rng)
    pr, gr, _ = generate_chains(topo, n_persons, sizes=topo.sizes, assign_modes=True, rng=rng)
    return topo, pd.DataFrame(pr), pd.DataFrame(gr)


def test_draw_survey_samples_persons():
    topo, plans, gt = _world()
    sp, sg = sv.draw_survey(plans, gt, frac=0.1, rng=np.random.default_rng(1))
    n_all = plans["unique_person_id"].nunique()
    n_surv = sp["unique_person_id"].nunique()
    assert 0 < n_surv < n_all
    assert abs(n_surv - 0.1 * n_all) <= 0.05 * n_all + 1
    assert set(sg["unique_leg_id"]) == set(sp["unique_leg_id"])


def test_study_window_and_membership():
    topo, plans, gt = _world()
    lo, hi = sv.study_window(topo, frac_side=0.5)
    pids = sv.persons_in_window(plans, lo, hi)
    assert 0 < len(pids) < plans["unique_person_id"].nunique()
    # every selected person's home really is inside the window
    firsts = plans.sort_values("unique_leg_id").groupby("unique_person_id").head(1)
    sel = firsts[firsts.unique_person_id.isin(pids)]
    assert (sel.from_x.between(lo, hi) & sel.from_y.between(lo, hi)).all()


def test_resample_distances_only_changes_free_legs():
    topo, plans, gt = _world()
    samples = sv.per_mode_distance_samples(plans, gt)
    assert None in samples and len(samples[None]) > 0
    out = sv.resample_distances(plans, gt, samples, np.random.default_rng(2))
    free_ids = set(gt.loc[gt.to_is_free, "unique_leg_id"])
    base = plans.set_index("unique_leg_id")["distance_meters"]
    new = out.set_index("unique_leg_id")["distance_meters"]
    # anchor (non-free) legs are untouched
    anchor = [lid for lid in base.index if lid not in free_ids]
    assert np.allclose(base[anchor].to_numpy(), new[anchor].to_numpy())
    # resampled free-leg distances stay within the observed support
    lo, hi = base.min(), base.max()
    free = [lid for lid in base.index if lid in free_ids]
    assert (new[free] >= lo - 1e-6).all() and (new[free] <= hi + 1e-6).all()
