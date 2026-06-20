"""Exact solvers for location assignment along chains (DP and MILP oracle).

The per-person problem is, for each segment, to place the free intermediate
activities A_1..A_{n-1} between two fixed endpoints S (segment start) and E
(segment end) so as to maximize

    alpha * sum_j P(T_j)  -  beta * sum_i | d_i^obs - || T_{i-1} - T_i || |

This objective is separable into unary (potential, one node) and pairwise
(distance deviation, two consecutive nodes) terms on a 1-D chain. That makes it
a layered shortest-path problem: build one graph layer of candidate facilities
per free node (plus singleton layers for S and E), weight each inter-layer edge
by the leg's distance deviation minus the destination's potential, and the
minimum-cost S->E path is the *exact global optimum* over the candidate sets.

Solvers here share that layered graph and differ only in how they solve / what
candidates they feed it:

- ``Dp``      : geometric per-node candidate generation (overlapping-ring
               triangle-inequality envelopes) + exact shortest path via SciPy's
               compressed-sparse-graph routines.
- ``CarlaDp`` : same DP, but uses CARLA's circle-intersection generation for the
               single-intermediate (two-leg) case, mirroring CARLA's candidate
               generation so a ``carla`` vs ``carla_dp`` comparison isolates the
               value of CARLA's *search* (recursive branching) vs exact DP.
- ``Milp``    : same candidate generation as ``Dp``, but solves the assignment as
               a min-cost-flow / shortest-path MILP via ``scipy.optimize.milp``
               (HiGHS). On the pure separable chain this returns the identical
               optimum to DP -- it is an *oracle* for validation and the natural
               vehicle for future non-separable side constraints. It does not
               scale like DP; use it on small instances.

Note: candidate generation prunes by distance geometry. That prune is
optimum-preserving for the distance-only objective; once potentials matter
(alpha > 0) it can in principle exclude the potential-optimal facility, so these
solvers are exact *over the generated candidate set* (globally exact only when
that set is an admissible superset). The default scorer is geometric (alpha = 0),
for which the tight ring is sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from frozendict import frozendict
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from chainsolvers.locations import Locations
from chainsolvers.types import Segment
from chainsolvers import helpers as h
from chainsolvers.scoring_selection import ScoreMode


def attr_value(values: np.ndarray, transform: str = "log1p") -> np.ndarray:
    """Attractiveness term entering the MNL utility. ``"log1p"`` (= log(1+x), robust to
    zeros) or ``"log"`` (= log(x), the pure gravity form; strictly-positive inputs)."""
    v = np.asarray(values, dtype=float)
    if transform == "log":
        return np.log(np.maximum(v, 1e-9))
    return np.log1p(v)

Pool = Tuple[np.ndarray, np.ndarray, np.ndarray]   # (ids, coords (k,2), potentials (k,))
Choice = Tuple[Any, np.ndarray, float]             # (id, coord (2,), potential)


@dataclass(slots=True)
class DpConfig:
    min_candidates: int = 50          # minimum candidate facilities to generate per free node
    max_iterations: int = 1000        # ring-expansion iterations before giving up
    use_circle_intersection: bool = False  # CarlaDp sets True (precise two-leg generation)
    refine_passes: int = 0            # iterative neighbour-based refinement passes (0 = one-shot)
    refine_min_candidates: int = 20   # candidates per node generated during refinement


def _alpha_beta_from_scorer(scorer: Any) -> Tuple[float, float]:
    """Map the injected Scorer's mode/weights to the DP objective weights
    (alpha for potentials, beta for distance deviation) so these solvers optimize
    the identical objective as CARLA."""
    mode = getattr(scorer, "mode", ScoreMode.GEOMETRIC)
    pot_weight = float(getattr(scorer, "pot_weight", 1.0))
    dist_dev_weight = float(getattr(scorer, "dist_dev_weight", 1.0))
    if mode is ScoreMode.POTENTIAL:
        return pot_weight, 0.0
    if mode is ScoreMode.COMBINED:
        return pot_weight, dist_dev_weight
    return 0.0, dist_dev_weight  # GEOMETRIC (default)


# ---- shared layered-graph construction -------------------------------------

def _build_layers(S: np.ndarray, E: np.ndarray, pools: List[Pool]):
    """Layers: [S] , pool_1 , ... , pool_{n-1} , [E]."""
    layer_coords = [S.reshape(1, 2)] + [p[1] for p in pools] + [E.reshape(1, 2)]
    layer_pots = [np.zeros(1)] + [np.asarray(p[2], dtype=float) for p in pools] + [np.zeros(1)]
    sizes = [c.shape[0] for c in layer_coords]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    return layer_coords, layer_pots, sizes, offsets, int(offsets[-1])


def _edges(layer_coords, layer_pots, distances, alpha, beta, sizes, offsets):
    """All inter-layer edges as (src, dst, weight). weight = beta*|d - ||a-b||| - alpha*P(b)."""
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    for i in range(1, len(layer_coords)):
        d = float(distances[i - 1])                       # leg i connects layer i-1 -> layer i
        D = cdist(layer_coords[i - 1], layer_coords[i])   # (Ka, Kb) euclidean
        W = beta * np.abs(d - D) - alpha * layer_pots[i][None, :]
        ka, kb = sizes[i - 1], sizes[i]
        rows.append(offsets[i - 1] + np.repeat(np.arange(ka), kb))
        cols.append(offsets[i] + np.tile(np.arange(kb), ka))
        data.append(W.ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


def _path_via_dp(src, dst, w, total) -> List[int]:
    """Exact shortest path (layer 0 -> last layer) via scipy csgraph."""
    # Shift every edge to be strictly positive. Every S->E path uses the same
    # number of edges, so a constant per-edge shift leaves the argmin unchanged;
    # this also dodges the sparse "stored 0 == no edge" pitfall and enables Dijkstra.
    w_pos = w + (1.0 - float(w.min()))
    graph = csr_matrix((w_pos, (src, dst)), shape=(total, total))
    _, predecessors = shortest_path(
        graph, method="D", directed=True, indices=0, return_predecessors=True
    )
    path: List[int] = []
    cur = total - 1
    while cur >= 0:
        path.append(cur)
        if cur == 0:
            break
        cur = int(predecessors[cur])
    path.reverse()
    return path


def _path_via_milp(src, dst, w, total) -> List[int]:
    """Exact path as a unit min-cost flow MILP (scipy.optimize.milp / HiGHS)."""
    from scipy.optimize import milp, LinearConstraint, Bounds  # local import (oracle path)

    m = len(w)
    # Node flow-conservation incidence: out(v) - in(v) = b_v ; b_S=+1, b_E=-1, else 0.
    inc_rows = np.concatenate([src, dst])
    inc_cols = np.concatenate([np.arange(m), np.arange(m)])
    inc_vals = np.concatenate([np.ones(m), -np.ones(m)])
    A = csr_matrix((inc_vals, (inc_rows, inc_cols)), shape=(total, m))
    b = np.zeros(total)
    b[0] = 1.0
    b[total - 1] = -1.0

    res = milp(
        c=w,
        constraints=LinearConstraint(A, b, b),
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")

    active = res.x > 0.5
    succ = {int(s): int(d) for s, d, a in zip(src, dst, active) if a}
    path = [0]
    node = 0
    while node != total - 1:
        if node not in succ:
            raise RuntimeError("MILP solution does not form a complete S->E path.")
        node = succ[node]
        path.append(node)
    return path


def solve_chain(
    S: np.ndarray,
    E: np.ndarray,
    distances: np.ndarray,
    pools: List[Pool],
    alpha: float,
    beta: float,
    method: str = "dp",
) -> List[Choice]:
    """Optimally assign free-node locations on the layered candidate graph.

    Returns a list (length n-1) of (id, coord, potential) in chain order.
    ``method`` is "dp" (scipy shortest path) or "milp" (scipy.optimize.milp oracle).
    Both return the exact optimum over the given candidate ``pools``.
    """
    num_free = len(pools)
    layer_coords, layer_pots, sizes, offsets, total = _build_layers(S, E, pools)
    src, dst, w = _edges(layer_coords, layer_pots, distances, alpha, beta, sizes, offsets)

    if method == "dp":
        path = _path_via_dp(src, dst, w, total)
    elif method == "milp":
        path = _path_via_milp(src, dst, w, total)
    else:
        raise ValueError(f"Unknown method {method!r} (use 'dp' or 'milp').")

    if len(path) != num_free + 2 or path[0] != 0:
        raise RuntimeError("Solver failed to find a complete path through the candidate graph.")

    chosen: List[Choice] = []
    for li in range(1, num_free + 1):
        local = path[li] - int(offsets[li])
        ids, coords, pots = pools[li - 1]
        chosen.append((ids[local], coords[local], float(pots[local])))
    return chosen


def _augment_pool(pool: Pool, choice: Choice) -> Pool:
    """Append a (carried-forward) choice to a candidate pool so the previous
    solution stays reachable -> the next exact solve cannot do worse."""
    ids, coords, pots = pool
    cid, ccoord, cpot = choice
    return (
        np.append(ids, cid),
        np.vstack([coords, np.asarray(ccoord, dtype=float).reshape(1, 2)]),
        np.append(pots, cpot),
    )


def _sample_index(rng, w: np.ndarray) -> int:
    s = float(w.sum())
    p = (w / s) if (np.isfinite(s) and s > 0) else np.full(w.shape[0], 1.0 / w.shape[0])
    return int(rng.choice(w.shape[0], p=p))


def sample_chain(
    S: np.ndarray,
    E: np.ndarray,
    leg_scales: Sequence[float],
    pools: List[Pool],
    alpha: float,
    rng: np.random.Generator,
    transform: str = "log1p",
) -> List[Choice]:
    """Draw a joint assignment from the chain MNL by exact forward-filtering /
    backward-sampling (sum-product). Energy ∝ Σ alpha·log(1+potential) (unary) +
    Σ -dist/scale (pairwise distance decay). Unlike the argmin solvers this does not
    use observed leg distances — it *generates* distances from the decay, so the
    population distance distribution emerges from the model (the MNL / generative view)."""
    num_free = len(pools)
    layer_coords = [S.reshape(1, 2)] + [p[1] for p in pools] + [E.reshape(1, 2)]
    layer_pots = [np.zeros(1)] + [np.asarray(p[2], dtype=float) for p in pools] + [np.zeros(1)]

    # Forward, normalised messages over free layers 1..num_free.
    f: List[np.ndarray] = [np.array([1.0])] + [None] * num_free  # type: ignore
    for j in range(1, num_free + 1):
        scale = max(float(leg_scales[j - 1]), 1e-6)
        Eg = np.exp(-cdist(layer_coords[j - 1], layer_coords[j]) / scale)  # |A| x |B|
        incoming = f[j - 1] @ Eg                                          # |B|
        lp = alpha * attr_value(layer_pots[j], transform)
        node_pot = np.exp(lp - lp.max())                                  # stable, shift cancels
        m = incoming * node_pot
        tot = m.sum()
        f[j] = m / tot if tot > 0 else np.full(m.shape[0], 1.0 / m.shape[0])

    # Backward sampling.
    sel = [0] * (num_free + 1)
    scale = max(float(leg_scales[num_free]), 1e-6)
    edge_E = np.exp(-cdist(layer_coords[num_free], E.reshape(1, 2))[:, 0] / scale)
    sel[num_free] = _sample_index(rng, f[num_free] * edge_E)
    for j in range(num_free - 1, 0, -1):
        scale = max(float(leg_scales[j]), 1e-6)
        nxt = layer_coords[j + 1][sel[j + 1]].reshape(1, 2)
        edge = np.exp(-cdist(layer_coords[j], nxt)[:, 0] / scale)
        sel[j] = _sample_index(rng, f[j] * edge)

    chosen: List[Choice] = []
    for j in range(1, num_free + 1):
        ids, coords, pots = pools[j - 1]
        chosen.append((ids[sel[j]], coords[sel[j]], float(pots[sel[j]])))
    return chosen


def _build_placed_segment(segment: Segment, chosen: List[Choice]) -> Segment:
    """Apply chosen free-node placements to a segment, matching CARLA's output
    convention: free node_j fills leg[j-1].to_location/to_act_identifier and
    leg[j].from_location; fixed endpoints are left untouched."""
    n = len(segment)
    out = []
    for k, leg in enumerate(segment):
        new_from = leg.from_location
        new_to = leg.to_location
        new_id = leg.to_act_identifier
        if k >= 1:                       # leg[k].from = free node_k = chosen[k-1]
            new_from = chosen[k - 1][1]
        if k <= n - 2:                   # leg[k].to = free node_{k+1} = chosen[k]
            new_to = chosen[k][1]
            new_id = chosen[k][0]
        out.append(leg._replace(from_location=new_from, to_location=new_to, to_act_identifier=new_id))
    return tuple(out)


# ---- solvers ----------------------------------------------------------------

class Dp:
    """Exact chain solver with geometric (overlapping-ring) candidate generation,
    solved by shortest path on the layered candidate graph."""

    wanted_format: str = "segmented_plans"
    _use_circle_intersection: bool = False
    _method: str = "dp"
    _refine_passes: int = 0

    def required_leg_fields(self) -> set[str]:
        return {"unique_leg_id", "distance", "from_location", "to_location", "to_act_type"}

    def __init__(
        self,
        locations: Locations,
        scorer: Any,
        selector: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        visualizer: Optional[Any] = None,
        progress: Optional[Any] = None,
        stats: Optional[Any] = None,
        **params: Any,
    ):
        self.locations = locations
        self.scorer = scorer
        self.selector = selector
        self.rng = rng
        self.visualizer = visualizer
        self.progress = progress
        self.stats = stats
        cfg_params = dict(params)
        cfg_params.setdefault("use_circle_intersection", self._use_circle_intersection)
        cfg_params.setdefault("refine_passes", self._refine_passes)
        self.config = DpConfig(**cfg_params)
        self.alpha, self.beta = _alpha_beta_from_scorer(scorer)

    def solve(self, *, plans):
        progress_fn = self.progress or (lambda it, **k: it)
        placed = {}
        for pid, segments in progress_fn(plans.items(), desc="placing persons"):
            placed[pid] = tuple(self._solve_segment(seg) for seg in segments)
        return frozendict(placed)

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 0:
            raise ValueError("No legs in segment.")
        if n == 1:
            leg = segment[0]
            if (leg.from_location is None or leg.to_location is None
                    or leg.from_location.size == 0 or leg.to_location.size == 0):
                raise ValueError("Single-leg segment requires known start and end locations.")
            return segment

        S = h.to_point_1d(segment[0].from_location)
        E = h.to_point_1d(segment[-1].to_location)
        distances = np.array([leg.distance for leg in segment], dtype=float)
        pools = [self._gen_pool(segment, S, E, distances, fj) for fj in range(n - 1)]
        chosen = solve_chain(S, E, distances, pools, self.alpha, self.beta, method=self._method)
        if self.config.refine_passes > 0 and n >= 3:
            chosen = self._refine(segment, S, E, distances, chosen)
        return _build_placed_segment(segment, chosen)

    def _refine(self, segment, S, E, distances, chosen):
        """Iteratively re-bracket each free node by its provisional neighbours and
        re-solve exactly. The previous choice is carried into each node's pool, so
        every pass is monotone (never worse) and the path converges. This closes the
        recall gap that one-shot endpoint-anchored generation leaves on long chains;
        it improves on the one-shot solution but is not a global-optimality guarantee
        (only the full-candidate-set MILP is)."""
        num_free = len(chosen)
        for _ in range(self.config.refine_passes):
            coords_seq = [S] + [c[1] for c in chosen] + [E]   # positions incl. fixed endpoints
            new_pools = []
            for fj in range(num_free):
                j = fj + 1
                Lc = h.to_point_1d(coords_seq[j - 1])
                Rc = h.to_point_1d(coords_seq[j + 1])
                act_type = segment[fj].to_act_type
                pool = self._gen_pool_local(act_type, Lc, Rc, distances[fj], distances[fj + 1])
                new_pools.append(_augment_pool(pool, chosen[fj]))  # carry forward -> monotone
            new_chosen = solve_chain(S, E, distances, new_pools, self.alpha, self.beta, method=self._method)
            if [c[0] for c in new_chosen] == [c[0] for c in chosen]:
                return new_chosen  # converged
            chosen = new_chosen
        return chosen

    def _gen_pool_local(self, act_type, Lc, Rc, dL, dR) -> Pool:
        """Tight candidate pool for a node bracketed by two provisional neighbour
        points (single leg to each): circle-intersection, falling back to rings."""
        try:
            ids, coords, pots = self.locations.get_circle_intersection_candidates(
                Lc, Rc, act_type, float(dL), float(dR), self.config.refine_min_candidates, unsafe=True,
            )
            if ids is not None and ids.size > 0:
                return ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float)
        except RuntimeError:
            pass
        (ids, coords, pots), _ = self.locations.get_overlapping_rings_candidates(
            act_type, Lc, Rc, float(dL), float(dL), float(dR), float(dR),
            min_candidates=self.config.refine_min_candidates,
            max_iterations=self.config.max_iterations,
        )
        return ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float)

    def _gen_pool(self, segment, S, E, distances, fj) -> Pool:
        """Candidate facilities for free node_{fj+1}, bounded by the triangle-inequality
        envelopes of the sub-chains on either side, anchored at the fixed endpoints."""
        n = len(segment)
        act_type = segment[fj].to_act_type
        left = distances[: fj + 1]
        right = distances[fj + 1:]

        # Single intermediate with one leg to each fixed endpoint -> optional precise
        # circle-intersection generation (CarlaDp).
        if self.config.use_circle_intersection and n == 2:
            try:
                ids, coords, pots = self.locations.get_circle_intersection_candidates(
                    S, E, act_type, float(left[0]), float(right[0]),
                    self.config.min_candidates, unsafe=True,
                )
                if ids is not None and ids.size > 0:
                    return ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float)
            except RuntimeError:
                pass  # degenerate / no intersection -> fall back to rings

        r1_min, r1_max = h.get_min_max_distance(left)
        r2_min, r2_max = h.get_min_max_distance(right)
        min_c = min(self.config.min_candidates, self.locations.identifiers[act_type].shape[0])
        (ids, coords, pots), _ = self.locations.get_overlapping_rings_candidates(
            act_type, S, E, r1_max, r1_min, r2_max, r2_min,
            min_candidates=min_c,
            max_iterations=self.config.max_iterations,
        )
        if ids is None or ids.size == 0:
            raise RuntimeError(f"No candidates for free node {fj} (type {act_type!r}).")
        return ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float)


class CarlaDp(Dp):
    """CARLA's geometric candidate generation (incl. circle-intersection for the
    single-intermediate case) feeding the exact DP optimizer. Holding generation
    fixed against the ``carla`` solver isolates the value of exact search vs
    recursive branching."""

    _use_circle_intersection: bool = True
    _method: str = "dp"


class DpRefine(Dp):
    """Exact DP with iterative neighbour-based candidate refinement. Starts from the
    one-shot ``Dp`` solution, then re-brackets each node by its provisional neighbours
    and re-solves until convergence -- closing the candidate-recall gap that endpoint-
    anchored generation leaves on long chains, while remaining monotone (never worse
    than ``Dp``)."""

    _refine_passes: int = 5


class DpFull(Dp):
    """Exact DP over the FULL candidate set per activity type -> the true global
    optimum of the separable objective (no geometric pruning, hence no recall gap).

    Costs O(n * N^2) per chain with N = facilities of the activity type, so it is an
    oracle for moderate catalogs / sampled validation, not a production solver. Every
    other solver's deviation is >= this (it cannot be beaten on the separable objective)."""

    def _gen_pool(self, segment, S, E, distances, fj) -> Pool:
        act_type = segment[fj].to_act_type
        ids = self.locations.identifiers.get(act_type)
        if ids is None or len(ids) == 0:
            raise RuntimeError(f"No facilities of type {act_type!r}.")
        coords = np.asarray(self.locations.coordinates[act_type], dtype=float)
        pots = np.asarray(self.locations.potentials[act_type], dtype=float)
        return ids, coords, pots


class DpSample(Dp):
    """Generative MNL solver: exact forward-backward *sampling* over the chain with
    utility = log-attractiveness + mode-specific distance decay. Produces realistic
    distance distributions (it samples, rather than collapsing onto the argmin) and
    needs no observed per-leg distance — only anchors, type, and mode. This is the
    exact-chain upgrade of the greedy duration-ordered MNL decomposition.

    Params: `decay_scales` (mode -> mean trip length m), `default_scale`,
    `tail_radius_factor` (candidate ball radius = factor * scale). `alpha` (potential
    weight) comes from the injected Scorer's `pot_weight`."""

    def __init__(self, locations, scorer, selector=None, rng=None, visualizer=None,
                 progress=None, stats=None, **params):
        self.decay_scales = dict(params.pop("decay_scales", {}) or {})
        self.default_scale = float(params.pop("default_scale", 3000.0))
        self.tail_radius_factor = float(params.pop("tail_radius_factor", 3.0))
        self.attr_transform = str(params.pop("attr_transform", "log1p"))
        super().__init__(locations, scorer, selector, rng, visualizer, progress, stats, **params)
        self.alpha = float(getattr(scorer, "pot_weight", 1.0))
        if self.rng is None:
            self.rng = np.random.default_rng()

    def required_leg_fields(self) -> set[str]:
        return {"unique_leg_id", "distance", "from_location", "to_location", "to_act_type", "mode"}

    def _scale_for(self, leg) -> float:
        return self.decay_scales.get(leg.mode, self.default_scale)

    def _gen_pool(self, segment, S, E, distances, fj) -> Pool:
        # Decay-scaled buffer around both anchors (no observed distance used).
        act_type = segment[fj].to_act_type
        R = self.tail_radius_factor * self._scale_for(segment[fj])
        min_c = min(self.config.min_candidates, self.locations.identifiers[act_type].shape[0])
        (ids, coords, pots), _ = self.locations.get_overlapping_rings_candidates(
            act_type, S, E, R, 0.0, R, 0.0,
            min_candidates=min_c, max_iterations=self.config.max_iterations,
        )
        if ids is None or ids.size == 0:
            raise RuntimeError(f"No candidates for free node {fj} (type {act_type!r}).")
        return ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float)

    def _solve_segment(self, segment: Segment) -> Segment:
        n = len(segment)
        if n == 0:
            raise ValueError("No legs in segment.")
        if n == 1:
            leg = segment[0]
            if (leg.from_location is None or leg.to_location is None
                    or leg.from_location.size == 0 or leg.to_location.size == 0):
                raise ValueError("Single-leg segment requires known start and end locations.")
            return segment
        S = h.to_point_1d(segment[0].from_location)
        E = h.to_point_1d(segment[-1].to_location)
        leg_scales = [self._scale_for(leg) for leg in segment]
        pools = [self._gen_pool(segment, S, E, None, fj) for fj in range(n - 1)]
        chosen = sample_chain(S, E, leg_scales, pools, self.alpha, self.rng, transform=self.attr_transform)
        return _build_placed_segment(segment, chosen)


class CarlaDpRefine(DpRefine):
    """Production 'best' exact (argmin) solver: CARLA circle-intersection generation for
    the single-intermediate case + iterative neighbour refinement, on the exact DP.
    A clean superset of both `carla_dp` and `dp_refine`."""

    _use_circle_intersection: bool = True
    _refine_passes: int = 5


class Milp(Dp):
    """Exact MILP oracle (scipy.optimize.milp / HiGHS) over the same candidate
    generation as ``Dp``. Equals DP on the pure separable chain; intended for
    validation on small instances and as the home for future non-separable
    constraints. Does not scale like DP."""

    _use_circle_intersection: bool = False
    _method: str = "milp"
