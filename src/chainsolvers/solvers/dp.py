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

Every solver except ``DpFull`` (registry ``dp_full``) is a *pruned* DP: it reuses CARLA's
geometric candidate generation rather than the whole catalog, so it is exact only over the
candidates it generates -- NOT a "pure" DP. Only ``dp_full`` is the unpruned, globally exact
DP. The variants (registry key in parentheses) are:

- ``DpFull``        (``dp_full``)        : exact DP over the FULL per-type catalog -> the
                     true global optimum of the separable objective; an oracle, O(n*N^2),
                     not scalable. The only *pure* DP here.
- ``Dp``            (``dp_rings``)       : exact DP over overlapping-ring (triangle-inequality)
                     envelope candidates per free node.
- ``CarlaDp``       (``dp_carla``)       : exact DP over CARLA's full generation -- ring
                     envelopes plus circle-intersection for the single-intermediate (two-leg)
                     case. Same candidates as ``carla``, so ``carla`` vs ``dp_carla`` isolates
                     CARLA's recursive *search* against exact DP.
- ``DpRefine``      (``dp_rings_refine``)/ ``CarlaDpRefine`` (``dp_carla_refine``)
                     : the above + iterative neighbour-based candidate refinement (monotone).
- ``DpPotential``   (``dp_carla_pot``)   : ``dp_carla_refine`` + potential-aware pooling --
                     each pool is augmented with the top-K facilities by potential, so the
                     pruned DP stays near-exact for the COMBINED (alpha > 0) objective.
- ``Milp``          (``milp``)           : the same candidate generation as ``dp_rings``,
                     solved as a min-cost-flow / shortest-path MILP via ``scipy.optimize.milp``
                     (HiGHS). Returns DP's optimum on the separable chain -- an oracle for
                     validation and the natural home for future non-separable constraints.

Note: CARLA's geometric generation prunes by distance. That prune is optimum-preserving
for the distance-only objective; once potentials matter (alpha > 0) it can exclude the
potential-optimal facility, so the pruned solvers are exact *over the generated candidate
set* only. ``DpPotential`` mitigates this by injecting high-potential facilities into every
pool; ``DpFull`` removes the prune entirely. The default scorer is geometric (alpha = 0),
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
    use_circle_intersection: bool = False  # CarlaDp sets True (precise circle-intersection two-leg generation)
    refine_passes: int = 0            # iterative neighbour-based refinement passes (0 = one-shot)
    refine_min_candidates: int = 20   # candidates per node generated during refinement
    pot_pool_k: int = 0               # if >0, augment every pool with the top-K facilities by potential


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
    tail_scales: Optional[Sequence[float]] = None,
    tail_weight: "float | Sequence[float]" = 0.0,
    kernel: str = "exp",
    dist_shape: Optional[Sequence[float]] = None,
) -> List[Choice]:
    """Draw a joint assignment from the chain MNL by exact forward-filtering /
    backward-sampling (sum-product). Energy ∝ Σ alpha·log(1+potential) (unary) +
    Σ log K(dist) (pairwise distance decay). Unlike the argmin solvers this does not
    use observed leg distances — it *generates* distances from the decay, so the
    population distance distribution emerges from the model (the MNL / generative view).

    Two distance-kernel families (``kernel``):

    * ``"exp"`` (default): ``exp(-d/scale)``, optionally a two-component mixture when
      ``tail_weight > 0``: ``(1-w)·exp(-d/scale) + w·exp(-d/tail_scale)``. A single scale cannot
      match both body and heavy tail (raise it and the median overshoots before the tail fills);
      a small mass ``w`` on a longer ``tail_scale`` fattens the tail while the short scale holds
      the median. ``tail_weight`` may be scalar or per-leg.
    * ``"powerlaw"``: ``(1 + d/scale)^(-k)`` with shape ``k = dist_shape`` (per-leg). A *polynomial*
      (heavy) tail — far candidates survive the both-anchor detour penalty that the exponential kills
      — with the tail thickness set by a single fitted exponent (small ``k`` = heavier), so one knob
      adapts across worlds/modes (≈exponential body for large ``k``). Note: the both-anchor geometry
      still caps the reachable tail; the exponent fits the *available* thickness, it can't exceed it."""
    num_free = len(pools)
    layer_coords = [S.reshape(1, 2)] + [p[1] for p in pools] + [E.reshape(1, 2)]
    layer_pots = [np.zeros(1)] + [np.asarray(p[2], dtype=float) for p in pools] + [np.zeros(1)]

    n_legs = num_free + 1
    tw = ([float(tail_weight)] * n_legs if np.isscalar(tail_weight)
          else [float(x) for x in tail_weight])
    tail_on = tail_scales is not None and any(x > 0.0 for x in tw)

    def kern(D: np.ndarray, leg_idx: int) -> np.ndarray:
        """Distance->weight kernel for leg ``leg_idx``."""
        s = max(float(leg_scales[leg_idx]), 1e-6)
        if kernel == "powerlaw":
            k = max(float(dist_shape[leg_idx]), 1e-6)
            return (1.0 + D / s) ** (-k)
        base = np.exp(-D / s)
        w = tw[leg_idx]
        if not tail_on or w <= 0.0:
            return base
        ts = max(float(tail_scales[leg_idx]), 1e-6)
        return (1.0 - w) * base + w * np.exp(-D / ts)

    # Forward, normalised messages over free layers 1..num_free.
    f: List[np.ndarray] = [np.array([1.0])] + [None] * num_free  # type: ignore
    for j in range(1, num_free + 1):
        Eg = kern(cdist(layer_coords[j - 1], layer_coords[j]), j - 1)     # |A| x |B|
        incoming = f[j - 1] @ Eg                                          # |B|
        lp = alpha * attr_value(layer_pots[j], transform)
        node_pot = np.exp(lp - lp.max())                                  # stable, shift cancels
        m = incoming * node_pot
        tot = m.sum()
        f[j] = m / tot if tot > 0 else np.full(m.shape[0], 1.0 / m.shape[0])

    # Backward sampling.
    sel = [0] * (num_free + 1)
    edge_E = kern(cdist(layer_coords[num_free], E.reshape(1, 2))[:, 0], num_free)
    sel[num_free] = _sample_index(rng, f[num_free] * edge_E)
    for j in range(num_free - 1, 0, -1):
        nxt = layer_coords[j + 1][sel[j + 1]].reshape(1, 2)
        edge = kern(cdist(layer_coords[j], nxt)[:, 0], j)
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
    _pot_pool_k: int = 0

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
        cfg_params.setdefault("pot_pool_k", self._pot_pool_k)
        self.config = DpConfig(**cfg_params)
        self.alpha, self.beta = _alpha_beta_from_scorer(scorer)
        self._pot_top_cache: dict = {}

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

    def _top_by_potential(self, act_type) -> Pool:
        """The top-`pot_pool_k` facilities of a type by potential (cached per type)."""
        cached = self._pot_top_cache.get(act_type)
        if cached is not None:
            return cached
        k = self.config.pot_pool_k
        pots_all = np.asarray(self.locations.potentials[act_type], dtype=float)
        n = pots_all.shape[0]
        idx = np.arange(n) if k >= n else np.argpartition(pots_all, -k)[-k:]
        res = (
            self.locations.identifiers[act_type][idx],
            np.asarray(self.locations.coordinates[act_type], dtype=float)[idx],
            pots_all[idx],
        )
        self._pot_top_cache[act_type] = res
        return res

    def _augment_with_potential(self, act_type, pool: Pool) -> Pool:
        """Union the geometric pool with the top-K facilities by potential (deduped by id).
        No-op when `pot_pool_k == 0`. Lets the pruned DP reach high-potential facilities that
        the distance envelope would otherwise exclude (relevant only for alpha > 0)."""
        if self.config.pot_pool_k <= 0:
            return pool
        ids, coords, pots = pool
        t_ids, t_coords, t_pots = self._top_by_potential(act_type)
        existing = set(np.asarray(ids).tolist())
        keep = [i for i, tid in enumerate(t_ids.tolist()) if tid not in existing]
        if keep:
            keep = np.asarray(keep, dtype=int)
            ids = np.concatenate([ids, t_ids[keep]])
            coords = np.vstack([coords, t_coords[keep]])
            pots = np.concatenate([pots, t_pots[keep]])
        return ids, coords, pots

    def _gen_pool_local(self, act_type, Lc, Rc, dL, dR) -> Pool:
        """Tight candidate pool for a node bracketed by two provisional neighbour
        points (single leg to each): circle-intersection, falling back to rings."""
        pool = None
        try:
            ids, coords, pots = self.locations.get_circle_intersection_candidates(
                Lc, Rc, act_type, float(dL), float(dR), self.config.refine_min_candidates, unsafe=True,
            )
            if ids is not None and ids.size > 0:
                pool = (ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float))
        except RuntimeError:
            pass
        if pool is None:
            (ids, coords, pots), _ = self.locations.get_overlapping_rings_candidates(
                act_type, Lc, Rc, float(dL), float(dL), float(dR), float(dR),
                min_candidates=self.config.refine_min_candidates,
                max_iterations=self.config.max_iterations,
            )
            pool = (ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float))
        return self._augment_with_potential(act_type, pool)

    def _gen_pool(self, segment, S, E, distances, fj) -> Pool:
        """Candidate facilities for free node_{fj+1}, bounded by the triangle-inequality
        envelopes of the sub-chains on either side, anchored at the fixed endpoints."""
        n = len(segment)
        act_type = segment[fj].to_act_type
        left = distances[: fj + 1]
        right = distances[fj + 1:]

        pool = None
        # Single intermediate with one leg to each fixed endpoint -> precise circle-intersection.
        if self.config.use_circle_intersection and n == 2:
            try:
                ids, coords, pots = self.locations.get_circle_intersection_candidates(
                    S, E, act_type, float(left[0]), float(right[0]),
                    self.config.min_candidates, unsafe=True,
                )
                if ids is not None and ids.size > 0:
                    pool = (ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float))
            except RuntimeError:
                pass  # degenerate / no intersection -> fall back to rings

        if pool is None:
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
            pool = (ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float))

        return self._augment_with_potential(act_type, pool)


class CarlaDp(Dp):
    """``dp_carla``: exact DP over **CARLA's** candidate generation — overlapping-ring
    envelopes plus circle-intersection for the single-intermediate (two-leg) case. Holding
    generation fixed against the ``carla`` solver, a ``carla`` vs ``dp_carla`` comparison
    isolates the value of CARLA's *search* (recursive branching) against exact DP."""

    _use_circle_intersection: bool = True
    _method: str = "dp"


class DpRefine(Dp):
    """``dp_rings_refine``: ``dp_rings`` + iterative neighbour-based candidate refinement.
    Starts from the one-shot solution, then re-brackets each node by its provisional
    neighbours and re-solves until convergence -- closing the candidate-recall gap that
    endpoint-anchored generation leaves on long chains, while remaining monotone (never
    worse than ``dp_rings``)."""

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

    To reproduce an observed free-leg distance distribution two things must be right
    and they fix different parts of it:

    * **Decay scale** (`decay_scales` / `default_scale`) controls the *centre* of the
      generated distribution. The default 3000 m is deliberately conservative; feed the
      MLE-calibrated scale (`chainsolvers_eval.calibration.fit_location_choice`, or a
      per-mode mean) so the median stops undershooting.
    * **Local/global candidate mixing** (`global_mix_k`) controls the *tail*. The pool is
      a decay-scaled ball of radius `tail_radius_factor * scale` around the anchors, so
      far facilities are never candidates and long trips are truncated. With
      `global_mix_k > 0` each free-node pool is augmented with that many facilities drawn
      uniformly at random from the full per-type catalog (deduped by id) — the "global"
      arm of a local/global mixture. The MNL distance decay reweights them, so they only
      matter where the local ball stops, restoring the long tail. 0 disables (back-compat).

    To fatten the long-distance tail (which a single decay scale cannot reach without
    overshooting the median), give a small probability mass `tail_weight` to a longer decay
    `tail_scale_factor * scale`; the kernel becomes a body+tail mixture (see `sample_chain`).
    The candidate ball is widened to `tail_scale_factor * scale` when the tail kernel is on,
    so the far facilities the tail kernel now weights are actually in the pool.

    Tail params may be **per-mode**: on a mode-heterogeneous world (car carries a long tail,
    walk/bike do not) a single pooled tail overshoots, so pass `decay_scales`, `tail_weights` and
    `tail_scale_factors` as mode->value dicts (from `chainsolvers_eval.calibration.fit_mode_kernels`);
    the scalar `tail_weight`/`tail_scale_factor`/`default_scale` are the per-mode fallbacks.

    Params: `decay_scales` (mode -> body scale m), `default_scale`, `tail_radius_factor` (candidate
    ball radius = factor * scale), `tail_weight`/`tail_weights` (mixture mass on the long kernel;
    0 = single-scale), `tail_scale_factor`/`tail_scale_factors` (long kernel = factor * scale),
    `global_mix_k` (random global candidates added per pool; 0 = local-only). `alpha` (potential
    weight) comes from the injected Scorer's `pot_weight`."""

    def __init__(self, locations, scorer, selector=None, rng=None, visualizer=None,
                 progress=None, stats=None, **params):
        self.decay_scales = dict(params.pop("decay_scales", {}) or {})
        self.default_scale = float(params.pop("default_scale", 3000.0))
        self.tail_radius_factor = float(params.pop("tail_radius_factor", 3.0))
        self.tail_weight = float(params.pop("tail_weight", 0.0))
        self.tail_scale_factor = float(params.pop("tail_scale_factor", 3.0))
        self.tail_weights = dict(params.pop("tail_weights", {}) or {})
        self.tail_scale_factors = dict(params.pop("tail_scale_factors", {}) or {})
        self.tail_radius_max_reach = float(params.pop("tail_radius_max_reach", 2.5))
        self.dist_kernel = str(params.pop("dist_kernel", "exp"))           # "exp" | "powerlaw"
        self.dist_shape = float(params.pop("dist_shape", 1.0))             # powerlaw exponent k
        self.dist_shapes = dict(params.pop("dist_shapes", {}) or {})       # per-mode k
        self.global_mix_k = int(params.pop("global_mix_k", 0))
        self.attr_transform = str(params.pop("attr_transform", "log1p"))
        super().__init__(locations, scorer, selector, rng, visualizer, progress, stats, **params)
        self.alpha = float(getattr(scorer, "pot_weight", 1.0))
        if self.rng is None:
            self.rng = np.random.default_rng()

    def required_leg_fields(self) -> set[str]:
        return {"unique_leg_id", "distance", "from_location", "to_location", "to_act_type", "mode"}

    def _scale_for(self, leg) -> float:
        return self.decay_scales.get(leg.mode, self.default_scale)

    def _tail_weight_for(self, leg) -> float:
        return float(self.tail_weights.get(leg.mode, self.tail_weight))

    def _tail_factor_for(self, leg) -> float:
        return float(self.tail_scale_factors.get(leg.mode, self.tail_scale_factor))

    def _dist_shape_for(self, leg) -> float:
        return float(self.dist_shapes.get(leg.mode, self.dist_shape))

    def _candidate_radius(self, leg) -> float:
        """Candidate-ball radius. Widened when a heavy tail is on so the far facilities the long
        kernel weights are actually generated as candidates -- but the reach is capped at
        `tail_radius_max_reach` so a near-flat tail doesn't expand the ball to the whole catalog on
        a large world (O(N^2)); far candidates beyond the cap are supplied cheaply by `global_mix_k`.
        The power-law kernel is always heavy-tailed, so it always uses the (capped) widened reach."""
        if self.dist_kernel == "powerlaw":
            reach = self.tail_radius_max_reach
        elif self._tail_weight_for(leg) > 0.0:
            reach = min(self._tail_factor_for(leg), self.tail_radius_max_reach)
        else:
            reach = 1.0
        return self.tail_radius_factor * self._scale_for(leg) * reach

    def _augment_global(self, act_type, pool: Pool) -> Pool:
        """Union the local pool with `global_mix_k` facilities sampled uniformly from the
        full per-type catalog (deduped by id). No-op when `global_mix_k <= 0`. This is the
        global arm of the local/global candidate mixture that restores the long-distance
        tail the decay-scaled ball would otherwise prune."""
        k = self.global_mix_k
        if k <= 0:
            return pool
        ids, coords, pots = pool
        all_ids = self.locations.identifiers[act_type]
        n = all_ids.shape[0]
        pick = self.rng.choice(n, size=min(k, n), replace=False)
        g_ids = all_ids[pick]
        existing = set(np.asarray(ids).tolist())
        keep = [i for i, gid in enumerate(np.asarray(g_ids).tolist()) if gid not in existing]
        if not keep:
            return pool
        keep = np.asarray(keep, dtype=int)
        sel = pick[keep]
        return (
            np.concatenate([ids, all_ids[sel]]),
            np.vstack([coords, np.asarray(self.locations.coordinates[act_type], dtype=float)[sel]]),
            np.concatenate([pots, np.asarray(self.locations.potentials[act_type], dtype=float)[sel]]),
        )

    def _gen_pool(self, segment, S, E, distances, fj) -> Pool:
        # Decay-scaled buffer around both anchors (no observed distance used).
        act_type = segment[fj].to_act_type
        R = self._candidate_radius(segment[fj])
        min_c = min(self.config.min_candidates, self.locations.identifiers[act_type].shape[0])
        (ids, coords, pots), _ = self.locations.get_overlapping_rings_candidates(
            act_type, S, E, R, 0.0, R, 0.0,
            min_candidates=min_c, max_iterations=self.config.max_iterations,
        )
        if ids is None or ids.size == 0:
            raise RuntimeError(f"No candidates for free node {fj} (type {act_type!r}).")
        pool = (ids, np.asarray(coords, dtype=float), np.asarray(pots, dtype=float))
        return self._augment_global(act_type, pool)

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
        tail_scales = [self._scale_for(leg) * self._tail_factor_for(leg) for leg in segment]
        tail_weights = [self._tail_weight_for(leg) for leg in segment]
        dist_shapes = ([self._dist_shape_for(leg) for leg in segment]
                       if self.dist_kernel == "powerlaw" else None)
        pools = [self._gen_pool(segment, S, E, None, fj) for fj in range(n - 1)]
        chosen = sample_chain(S, E, leg_scales, pools, self.alpha, self.rng,
                              transform=self.attr_transform, tail_scales=tail_scales,
                              tail_weight=tail_weights, kernel=self.dist_kernel,
                              dist_shape=dist_shapes)
        return _build_placed_segment(segment, chosen)


class CarlaDpRefine(CarlaDp):
    """``dp_carla_refine``: ``dp_carla`` (CARLA's circle-intersection generation) + iterative
    neighbour refinement -- the most accurate *distance-only* argmin in the pruned family."""

    _refine_passes: int = 5


class DpPotential(CarlaDpRefine):
    """``dp_carla_pot``: ``dp_carla_refine`` **plus** potential-aware pooling -- each free-node
    pool is augmented with the top-`pot_pool_k` facilities by potential, so high-attraction
    locations the distance envelope would prune stay reachable. This keeps the pruned DP
    near-exact for the COMBINED (alpha > 0) objective -- the gap the other pruned solvers
    leave when potentials matter -- without paying ``dp_full``'s O(n*N^2). For the distance-only
    objective (alpha = 0) it matches ``dp_carla_refine``. Set `pot_pool_k` large to approach
    ``dp_full``."""

    _pot_pool_k: int = 64


class Milp(Dp):
    """``milp``: exact MILP oracle (scipy.optimize.milp / HiGHS) over the same candidate
    generation as ``dp_rings``. Equals DP on the pure separable chain; intended for
    validation on small instances and as the home for future non-separable
    constraints. Does not scale like DP."""

    _method: str = "milp"
