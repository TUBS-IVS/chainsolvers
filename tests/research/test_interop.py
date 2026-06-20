"""Round-trip test for external-tool interop (export world, re-import assignments)."""

import numpy as np
import pandas as pd

from chainsolvers import run
from chainsolvers_eval.synth import generate_world
from chainsolvers_eval.interop import export_for_external, result_df_from_external


def _total_dev(rdf):
    a = np.hypot(rdf.to_x - rdf.from_x, rdf.to_y - rdf.from_y)
    return float((rdf.distance_meters - a).abs().sum())


def test_export_writes_files(tmp_path):
    import os
    w = generate_world(n_locations=200, n_persons=40, rng=np.random.default_rng(0))
    paths = export_for_external(w.locations_tuple, w.plans_df, str(tmp_path), w.ground_truth)
    for key in ("facilities", "plans", "ground_truth"):
        assert os.path.exists(paths[key])
    fac = pd.read_csv(paths["facilities"])
    assert {"id", "activity", "x", "y", "potential"} <= set(fac.columns) and len(fac) > 0


def test_reimport_assignments_reconstructs_chain(tmp_path):
    # Use our dp solver's output as a stand-in "external tool"; export, then reconstruct
    # the result_df purely from the free-leg assignments and check it matches.
    w = generate_world(n_locations=300, n_persons=60, rng=np.random.default_rng(1))
    ctx = run.setup(locations_tuple=w.locations_tuple, solver="dp", rng_seed=7)
    rdf, _, _ = run.solve(ctx=ctx, plans_df=w.plans_df)

    free = set(w.ground_truth.loc[w.ground_truth.to_is_free, "unique_leg_id"])
    assignments = rdf[rdf.unique_leg_id.isin(free)][
        ["unique_leg_id", "to_x", "to_y", "to_act_identifier"]
    ].copy()

    export_for_external(w.locations_tuple, w.plans_df, str(tmp_path), w.ground_truth)
    recon = result_df_from_external(w.plans_df, assignments)

    m = recon.merge(rdf, on="unique_leg_id", suffixes=("_r", "_o"))
    for c in ("to_x", "to_y", "from_x", "from_y"):
        assert np.allclose(m[f"{c}_r"].to_numpy(), m[f"{c}_o"].to_numpy(), atol=1e-6), c
    assert abs(_total_dev(recon) - _total_dev(rdf)) < 1e-6
