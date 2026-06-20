"""In-process hook for comparing against an external solver on *identical* inputs.

`CallableSolver` wraps a user-supplied per-segment placement function as a solver, so it
runs through `run.setup`/`run.solve` like any other. chainsolvers itself imports nothing
external here — the user's function does any third-party import. That keeps the library
MIT-clean while letting you call e.g. eqasim's relaxation solver in-process on the *same*
endpoints/distances/candidates as our solvers (the clean same-distances per-instance
comparison the file interop can't give).

Because GPL copyleft is about *distribution*, the external-importing glue belongs in a
private / gitignored script (not shipped, not vendored). Example shape:

    # my_private/eqasim_glue.py  (NOT committed to the MIT library)
    from eqasim... import GravityChainSolver, ...           # GPL import, private use only

    def eqasim_place(S, E, distances, act_types, locations, rng):
        # build eqasim's problem from (S, E, distances), relax to continuous points,
        # then discretise to OUR candidate facilities so the candidate set is shared:
        cont = run_eqasim_relaxation(S, E, distances)        # (n-1, 2)
        chosen = []
        for j, t in enumerate(act_types):
            ids, coords, pots = locations.query_closest(t, cont[j], k=1)
            chosen.append((ids[0], coords[0], float(pots[0])))
        return chosen

    # then:
    ctx = run.setup(locations_tuple=..., solver=CallableSolver,
                    parameters={"place_fn": eqasim_place})

The `place_fn` contract:
    place_fn(S, E, distances, act_types, locations, rng) -> list[(id, coord(2,), potential)]
    - S, E         : (2,) float arrays, the fixed segment endpoints
    - distances    : (n,) target leg distances (use them or ignore, as the method dictates)
    - act_types    : list of length n-1, the activity type of each free node
    - locations    : the LocationsIndex (for candidate generation / nearest snapping)
    - returns one (id, coord, potential) per free node, in chain order.
"""

from __future__ import annotations

import numpy as np

from chainsolvers import helpers as h
from chainsolvers.solvers.dp import _build_placed_segment
from chainsolvers_eval.baselines import _BaseBaseline


class CallableSolver(_BaseBaseline):
    """Run an external/user placement function (`parameters={'place_fn': fn}`) as a solver."""

    def _solve_segment(self, segment):
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        place_fn = self.params.get("place_fn")
        if place_fn is None:
            raise ValueError("CallableSolver requires parameters={'place_fn': callable}.")
        S = h.to_point_1d(segment[0].from_location)
        E = h.to_point_1d(segment[-1].to_location)
        distances = np.array([leg.distance for leg in segment], dtype=float)
        act_types = [segment[j].to_act_type for j in range(n - 1)]
        chosen = place_fn(S, E, distances, act_types, self.locations, self.rng)
        if len(chosen) != n - 1:
            raise ValueError(f"place_fn returned {len(chosen)} placements; expected {n - 1}.")
        chosen = [(cid, np.asarray(coord, dtype=float), float(pot)) for (cid, coord, pot) in chosen]
        return _build_placed_segment(segment, chosen)
