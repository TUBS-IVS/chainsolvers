"""Baseline location-assignment algorithms for comparison (eval-only, not core).

These conform to the solver interface (so they run through `run.setup`/`run.solve` via
`setup(solver=<class>)`), but they live in `eval` because they are reference baselines,
not part of the library's offering. They reuse core helpers (no core->eval dependency).

- `RelaxationDiscretization` — independent (MIT-clean) reimplementation of the Hörl & Axhausen
  RDA, written from the paper (NOT a port of eqasim's GPL source): a gravity-chain continuous
  relaxation (points along the S->E line, iteratively nudged to fix leg-length offsets; single
  intermediate via circle-intersection trig) followed by nearest-facility discretization. It
  approximates the published method; for bit-exact eqasim numbers use the vendored glue under
  `_private/` (see `docs/eqasim_comparison.md`). (We feed target distances from the plans,
  matching how all solvers are fed; the stochastic feasible-distance resampling / outer
  assignment loop is handled upstream by the harness.)
- `Nearest` — crude floor: free nodes at cumulative-distance shares on the S->E line, snapped
  to the nearest facility (no relaxation). The "closest location" CARLA argues against.
- `ZoneSample` — attractiveness-weighted random facility of the type, ignoring distance
  (PAM-style facility sampling); a pure-attractiveness floor.
- `GravityIndependent` — sequential gravity sample (size * exp(-d_prev/scale)) per node with
  no joint/backward pass; isolates the value of *chain* coupling vs per-activity placement.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from frozendict import frozendict

from chainsolvers.types import Segment
from chainsolvers import helpers as h
from chainsolvers.solvers.dp import _build_placed_segment, Choice


def _gravity_chain(S, E, distances, rng, max_iter=1000, eps=1.0, alpha=0.1):
    """Independent (MIT-clean) implementation of the RDA gravity relaxation, written from
    Hörl & Axhausen (2023, Algorithm 2) — NOT a port of eqasim's GPL source. It approximates
    the published method; for bit-exact eqasim numbers use the vendored glue (see
    `docs/eqasim_comparison.md`). Places the n-1 free nodes on the S->E line at cumulative-
    distance shares, perturbs them laterally, then runs the spring force model (Eq. 1-2)
    toward the target leg lengths. The single-intermediate case is the circle intersection
    of the two radii with a random mirror side (paper). Returns the n-1 free-node positions."""
    distances = np.asarray(distances, dtype=float)
    n_free = len(distances) - 1
    if rng is None:
        rng = np.random.default_rng()
    direct = E - S
    dd = float(np.linalg.norm(direct))

    if n_free == 1:  # circle intersection of radius d0 around S and d1 around E
        d0, d1 = float(distances[0]), float(distances[1])
        if dd < 1e-9:
            ang = rng.uniform(0, 2 * np.pi)
            return (S + d0 * np.array([np.cos(ang), np.sin(ang)]))[None, :]
        A = 0.5 * (d0 ** 2 - d1 ** 2 + dd ** 2) / dd
        H = np.sqrt(max(0.0, d0 ** 2 - A ** 2))   # 0 when infeasible -> best guess on the line
        direction = direct / dd
        normal = np.array([direction[1], -direction[0]])
        sign = 1.0 if rng.random() < 0.5 else -1.0   # random mirror side (paper)
        return (S + A * direction + sign * H * normal)[None, :]

    # straight-line init at cumulative-distance shares + small lateral jitter
    cum = np.cumsum(distances)
    shares = cum[:-1] / max(float(cum[-1]), 1e-9)
    if dd < 1e-9:
        direct = np.array([max(cum[-1], 1.0), 0.0])
        dd = float(np.linalg.norm(direct))
    pts = S[None, :] + shares[:, None] * direct[None, :]
    pts = pts + rng.normal(0, max(dd, 1.0) * 0.01, pts.shape)

    nodes = np.vstack([S[None, :], pts, E[None, :]])
    for _ in range(max_iter):
        seg = nodes[1:] - nodes[:-1]
        cur = np.linalg.norm(seg, axis=1)
        offset = cur - distances
        if np.all(np.abs(offset) < eps):
            break
        dirs = seg / np.maximum(cur[:, None], 1e-9)
        adj = np.zeros_like(nodes)
        adj[1:] -= 0.5 * alpha * offset[:, None] * dirs    # too-long leg: pull endpoints together
        adj[:-1] += 0.5 * alpha * offset[:, None] * dirs
        adj[0] = 0.0
        adj[-1] = 0.0
        nodes = nodes + adj
    return nodes[1:-1]


def _relax_guided(S, E, distances, node_attractors, rng,
                  alpha=0.3, ff_mean=30.0, ff_ceiling=60.0, max_iter=1000, eps=250.0):
    """Guidance-force relaxation after Langrognet, Côme, Hörl & Oukhellou (2026,
    "Improving the spatial distribution of secondary activities ... through guidance
    forces", Comput. Environ. Urban Syst. 127:102431): independent (MIT-clean)
    implementation from the paper — spring (distance) force + the inverse-cube POI
    attraction `delta/max(||.||,eps)**3` (eps = 125 m), with the position-change
    convergence criterion (Eq. 1-2 / Alg. 1). NOT a port of eqasim's GPL source; for
    bit-exact guidance-RDA numbers use the vendored glue (`docs/eqasim_comparison.md`).

    `node_attractors` is a per-free-node (coords (M,2), weights (M,)). Deliberate
    deviations from the paper: PoIs are weighted by `potentials` (the paper, and eqasim's
    default attractor, use weight 1.0 — density-only), which makes this the attractiveness-
    aware counterpart to CARLA's potential term; and the field is evaluated per node and
    renormalized over the chain's free nodes rather than precomputed/interpolated on a
    250 m grid. Returns the n-1 free-node positions."""
    distances = np.asarray(distances, dtype=float)
    n_free = len(distances) - 1
    direct = E - S
    dd = float(np.linalg.norm(direct))
    cum = np.cumsum(distances)
    shares = cum[:-1] / max(float(cum[-1]), 1e-9)
    if dd < 1e-9:
        direct = np.array([max(float(cum[-1]), 1.0), 0.0]); dd = float(np.linalg.norm(direct))
    pts = S[None, :] + shares[:, None] * direct[None, :]
    if rng is not None:
        pts = pts + rng.normal(0, max(dd, 1.0) * 0.01, pts.shape)
    nodes = np.vstack([S[None, :], pts, E[None, :]])
    prev = nodes.copy()

    for it in range(max_iter):
        seg = nodes[1:] - nodes[:-1]
        cur = np.linalg.norm(seg, axis=1)
        dirs = seg / np.maximum(cur[:, None], 1e-9)
        offset = (cur - distances)[:, None]               # >0 if leg too long
        adj = np.zeros_like(nodes)
        adj[1:] -= 0.5 * alpha * offset * dirs            # pull far node inward
        adj[:-1] += 0.5 * alpha * offset * dirs           # pull near node outward
        adj[0] = 0.0; adj[-1] = 0.0                        # anchors fixed

        # guidance field on each free node (normalized to ~ff_mean, capped at ff_ceiling)
        field = np.zeros((n_free, 2))
        for j in range(n_free):
            ac, aw = node_attractors[j]
            if len(ac) == 0:
                continue
            d = ac - nodes[j + 1]
            nrm = np.linalg.norm(d, axis=1)
            nrm = np.maximum(nrm, 125.0)  # paper's softening max(||.||, eps), eps = 125 m
            field[j] = (aw[:, None] * d / (nrm ** 3)[:, None]).sum(axis=0)
        mags = np.linalg.norm(field, axis=1)
        mean_mag = mags[mags > 0].mean() if np.any(mags > 0) else 0.0
        if mean_mag > 0:
            field = field / mean_mag * ff_mean
            m2 = np.linalg.norm(field, axis=1)
            field *= np.minimum(1.0, ff_ceiling / np.maximum(m2, 1e-9))[:, None]
        adj[1:-1] += field

        nodes = nodes + adj
        if it % 50 == 49:
            if np.all(np.linalg.norm(nodes - prev, axis=1) < eps):
                break
            prev = nodes.copy()
    return nodes[1:-1]


class _BaseBaseline:
    wanted_format: str = "segmented_plans"

    def required_leg_fields(self) -> set[str]:
        return {"unique_leg_id", "distance", "from_location", "to_location", "to_act_type"}

    def __init__(self, locations, scorer=None, selector=None, rng=None,
                 visualizer=None, progress=None, stats=None, **params):
        self.locations = locations
        self.scorer = scorer
        self.rng = rng if rng is not None else np.random.default_rng()
        self.progress = progress
        self.params = params

    def solve(self, *, plans):
        progress_fn = self.progress or (lambda it, **k: it)
        placed = {}
        for pid, segments in progress_fn(plans.items(), desc="baseline placing"):
            placed[pid] = tuple(self._solve_segment(seg) for seg in segments)
        return frozendict(placed)

    def _endpoints(self, segment) -> Tuple[np.ndarray, np.ndarray]:
        return h.to_point_1d(segment[0].from_location), h.to_point_1d(segment[-1].to_location)

    def _snap_nearest(self, act_type, point) -> Choice:
        ids, coords, pots = self.locations.query_closest(act_type, h.to_point_1d(point), k=1)
        return ids[0], np.asarray(coords[0], dtype=float), float(pots[0])

    def _check_single(self, segment):
        leg = segment[0]
        if (leg.from_location is None or leg.to_location is None
                or leg.from_location.size == 0 or leg.to_location.size == 0):
            raise ValueError("Single-leg segment requires known start and end locations.")


class RelaxationDiscretization(_BaseBaseline):
    """Hörl & Axhausen relaxation–discretization (eqasim): gravity relaxation + nearest snap,
    repeated over an assignment loop (stochastic lateral init) keeping the lowest-deviation
    discretized chain — matching eqasim's "keep best" behaviour."""

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        S, E = self._endpoints(segment)
        distances = np.array([leg.distance for leg in segment], dtype=float)
        n_assign = int(self.params.get("assignment_iterations", 20))
        best_chosen, best_dev = None, np.inf
        for _ in range(n_assign):
            cont = _gravity_chain(S, E, distances, self.rng)
            chosen = [self._snap_nearest(segment[j].to_act_type, cont[j]) for j in range(n - 1)]
            coords = [S] + [c[1] for c in chosen] + [E]
            legs = np.linalg.norm(np.diff(np.asarray(coords), axis=0), axis=1)
            dev = float(np.abs(legs - distances).sum())
            if dev < best_dev:
                best_chosen, best_dev = chosen, dev
            if best_dev < 1.0:  # eqasim-style early stop
                break
        return _build_placed_segment(segment, best_chosen)


class RelaxationDiscretizationGuided(RelaxationDiscretization):
    """RDA with guidance forces (Langrognet et al. 2026): the relaxation is biased toward dense
    POI clusters (attractor weights = location potentials), then discretized. The current
    attractiveness-aware state of the art in the RDA line — the fair counterpart to CARLA's
    potential term (attractiveness in a relaxation *force* vs in the choice *objective*)."""

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        S, E = self._endpoints(segment)
        distances = np.array([leg.distance for leg in segment], dtype=float)
        node_attr = [(np.asarray(self.locations.coordinates[segment[j].to_act_type], dtype=float),
                      np.asarray(self.locations.potentials[segment[j].to_act_type], dtype=float))
                     for j in range(n - 1)]
        ff_mean = float(self.params.get("ff_mean", 30.0))
        ff_ceiling = float(self.params.get("ff_ceiling", 60.0))
        n_assign = int(self.params.get("assignment_iterations", 20))
        best_chosen, best_dev = None, np.inf
        for _ in range(n_assign):
            cont = _relax_guided(S, E, distances, node_attr, self.rng,
                                 ff_mean=ff_mean, ff_ceiling=ff_ceiling)
            chosen = [self._snap_nearest(segment[j].to_act_type, cont[j]) for j in range(n - 1)]
            coords = [S] + [c[1] for c in chosen] + [E]
            legs = np.linalg.norm(np.diff(np.asarray(coords), axis=0), axis=1)
            dev = float(np.abs(legs - distances).sum())
            if dev < best_dev:
                best_chosen, best_dev = chosen, dev
        return _build_placed_segment(segment, best_chosen)


class Nearest(_BaseBaseline):
    """Crude floor: free nodes at cumulative-distance shares on the S->E line, snapped nearest."""

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        S, E = self._endpoints(segment)
        distances = np.array([leg.distance for leg in segment], dtype=float)
        cs = np.cumsum(distances)
        shares = cs[:-1] / max(float(cs[-1]), 1e-9)
        pts = S[None, :] + shares[:, None] * (E - S)[None, :]
        chosen = [self._snap_nearest(segment[j].to_act_type, pts[j]) for j in range(n - 1)]
        return _build_placed_segment(segment, chosen)


class ZoneSample(_BaseBaseline):
    """Attractiveness-weighted random facility of the type, ignoring distance (PAM-style)."""

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        chosen = []
        for j in range(n - 1):
            t = segment[j].to_act_type
            ids = self.locations.identifiers[t]
            coords = self.locations.coordinates[t]
            pots = np.asarray(self.locations.potentials[t], dtype=float)
            tot = pots.sum()
            p = pots / tot if np.isfinite(tot) and tot > 0 else np.full(len(ids), 1.0 / len(ids))
            i = int(self.rng.choice(len(ids), p=p))
            chosen.append((ids[i], np.asarray(coords[i], dtype=float), float(pots[i])))
        return _build_placed_segment(segment, chosen)


class GravityIndependent(_BaseBaseline):
    """Sequential gravity sample size*exp(-d_prev/scale) per node — no chain coupling."""

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 1:
            self._check_single(segment)
            return segment
        scale = float(self.params.get("scale", 3000.0))
        S, _ = self._endpoints(segment)
        prev = S
        chosen = []
        for j in range(n - 1):
            t = segment[j].to_act_type
            ids = self.locations.identifiers[t]
            coords = np.asarray(self.locations.coordinates[t], dtype=float)
            sizes = np.asarray(self.locations.potentials[t], dtype=float)
            d = np.hypot(coords[:, 0] - prev[0], coords[:, 1] - prev[1])
            w = sizes * np.exp(-d / scale)
            tot = w.sum()
            p = w / tot if np.isfinite(tot) and tot > 0 else np.full(len(ids), 1.0 / len(ids))
            i = int(self.rng.choice(len(ids), p=p))
            chosen.append((ids[i], coords[i], float(sizes[i])))
            prev = coords[i]
        return _build_placed_segment(segment, chosen)
