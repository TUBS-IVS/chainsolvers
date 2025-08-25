from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List, Sequence, Union, Protocol
import numpy as np
from scipy.spatial import cKDTree
import math
import logging

logger = logging.getLogger(__name__)

import helpers as h


class Locations:
    """
    Spatial index of activity locations split by type.

    Caller provides:
      - identifiers[act_type] -> (n,) array of ids (dtype/object as you use)
      - coordinates[act_type] -> (n,2) float array
      - potentials[act_type]  -> (n,) float array

    Assumes query locations are 1-D float arrays of shape (2,) for single-point methods.
    Batched methods accept (m,2).
    """

    def __init__(
            self,
            identifiers: Dict[str, np.ndarray],
            coordinates: Dict[str, np.ndarray],
            potentials: Dict[str, np.ndarray],
            stats_tracker=None,
    ):
        # basic consistency checks
        keys = set(coordinates.keys())
        if not (set(identifiers.keys()) == set(potentials.keys()) == keys):
            raise ValueError("identifiers/coordinates/potentials must share the same act_type keys")

        for t in keys:
            coords = np.asarray(coordinates[t])
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"coordinates['{t}'] must be shape (n,2)")
            if len(identifiers[t]) != coords.shape[0] or len(potentials[t]) != coords.shape[0]:
                raise ValueError(f"arrays for '{t}' must have matching length n")

        self.identifiers = identifiers
        self.coordinates = coordinates
        self.potentials = potentials
        self.stats_tracker = stats_tracker

        self.trees = {}
        for act_type, coords in self.coordinates.items():
            logger.info(f"Constructing spatial index for {act_type} ...")
            self.trees[act_type] = cKDTree(coords)

    def query_closest(
            self,
            act_type: str,
            locations: np.ndarray,
            k: int = 1,
            unsafe: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors of one or more query points.
        Input: (2,), (1,2), or (m,2).
        Output single→(k,), (k,2), (k,); multi→(m,k), (m,k,2), (m,k).
        Set unsafe=True if already float64 (faster).
        Returns (ids, coords, potentials).
        """
        if unsafe:
            loc = locations
        else:
            loc = np.asarray(locations, dtype=np.float64)

        _, idx = self.trees[act_type].query(loc, k=k)

        if np.ndim(idx) == 0:  # scalar (single point, k=1) → (1,1)
            idx = idx[None, None]
        elif np.ndim(idx) == 1:  # (k,) or (m,) → (1,k) or (m,1)
            if loc.ndim == 1:  # input was (2,) → single point
                idx = idx[None, :]  # (1,k)
            else:  # input was (m,2), k=1
                idx = idx[:, None]  # (m,1)

        ids = self.identifiers[act_type][idx]
        coords = self.coordinates[act_type][idx]
        pots = self.potentials[act_type][idx]

        # --- Collapse if it was a single query ---
        if loc.ndim == 1 or (loc.ndim == 2 and loc.shape[0] == 1):
            return ids[0], coords[0], pots[0]
        return ids, coords, pots

    def query_within_ring(
            self,
            act_type: str,
            location: np.ndarray,
            radius1: float,
            radius2: float,
            unsafe: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return candidates within an annulus around `location`.

        Input: (2,) or (1,2).
        Output (single location): ids (k,), coords (k,2), pots (k,).
        Returns None if no candidates.

        Set unsafe=True if `location` is already (2,) float64 (skips copy/shape fix).
        """
        if unsafe:
            loc = location  # must be (2,) float64 already
        else:
            loc = h.to_point_1d(location)

        tree = self.trees[act_type]
        r_outer, r_inner = (radius1, radius2) if radius1 >= radius2 else (radius2, radius1)
        r2_inner = r_inner * r_inner

        # Outer candidates (query_ball_point → list[int]; convert to ndarray[intp])
        outer = np.asarray(tree.query_ball_point(loc, r_outer), dtype=np.intp)
        if outer.size == 0:
            return None

        # Filter out points inside inner radius using squared distances
        pts = tree.data[outer]  # (n,2) view
        d2 = np.sum((pts - loc) ** 2, axis=1)  # (n,)
        keep = outer[d2 >= r2_inner]  # (k,)
        if keep.size == 0:
            return None

        ids = self.identifiers[act_type][keep]  # (k,)
        coords = self.coordinates[act_type][keep]  # (k,2)
        pots = self.potentials[act_type][keep]  # (k,)

        return ids, coords, pots

    def query_within_two_overlapping_rings(
            self,
            act_type: str,
            location1: np.ndarray,
            location2: np.ndarray,
            r1outer: float,
            r1inner: float,
            r2outer: float,
            r2inner: float,
            unsafe: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Candidates in the overlap of two annuli around `location1` and `location2`.

        Input: location1, location2 as (2,) or (1,2).
        Output (single pair): ids (k,), coords (k,2), pots (k,).
        Returns None if no candidates.

        Set unsafe=True if both locations are already (2,) float64 (skips copy/shape fix).
        """
        if unsafe:
            loc1 = location1  # must be (2,) float64 already
            loc2 = location2
        else:
            loc1 = h.to_point_1d(location1)
            loc2 = h.to_point_1d(location2)

        tree = self.trees[act_type]

        # Choose base = smaller outer radius to reduce candidates
        if r1outer <= r2outer:
            base_loc, other_loc = loc1, loc2
            r_ob, r_ib, r_oo, r_io = r1outer, r1inner, r2outer, r2inner
        else:
            base_loc, other_loc = loc2, loc1
            r_ob, r_ib, r_oo, r_io = r2outer, r2inner, r1outer, r1inner

        # Outer candidates within base outer radius
        cand = np.asarray(tree.query_ball_point(base_loc, r_ob), dtype=np.intp)
        if cand.size == 0:
            return None

        pts = tree.data[cand]  # (n,2) view

        # Precompute squared radii
        r_ib2 = r_ib * r_ib
        r_oo2 = r_oo * r_oo
        r_io2 = r_io * r_io

        # Squared distances to base_loc (component-wise to avoid (n,2) temps)
        dx = pts[:, 0] - base_loc[0]
        dy = pts[:, 1] - base_loc[1]
        d2_base = dx * dx + dy * dy

        # Squared distances to other_loc (reuse dx, dy)
        dx = pts[:, 0] - other_loc[0]
        dy = pts[:, 1] - other_loc[1]
        d2_other = dx * dx + dy * dy

        # Overlap of annuli:
        # base annulus: d2_base in [r_ib^2, r_ob^2]  (outer bound already enforced by query_ball_point)
        # other annulus: d2_other in [r_io^2, r_oo^2]
        mask = (d2_base >= r_ib2) & (d2_other <= r_oo2) & (d2_other >= r_io2)
        if not mask.any():
            return None

        keep = cand[mask]  # (k,)
        ids = self.identifiers[act_type][keep]  # (k,)
        coords = self.coordinates[act_type][keep]  # (k,2)
        pots = self.potentials[act_type][keep]  # (k,)
        return ids, coords, pots

    def sample(self, act_type: str, rng, ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniform random sample from all locations of a given type.
        Output (single sample): id (1,), coord (1,2).

        :param rng: numpy Generator or random.Random
                (must support .integers or .randint)
        """
        n = len(self.coordinates[act_type])
        if hasattr(rng, "integers"):
            i = int(rng.integers(0, n))
        else:
            i = rng.randint(0, n - 1)

        ids = self.identifiers[act_type][i:i + 1]  # (1,)
        coords = self.coordinates[act_type][i:i + 1]  # (1,2)
        pots = self.potentials[act_type][i:i + 1]  # (1,)

        return ids, coords, pots

    # ---- Search helpers with retries ----

    def get_ring_candidates(
            self,
            act_type: str,
            center: np.ndarray,
            radius_outer: float,
            radius_inner: float,
            max_iterations: int = 20,
            min_candidates: int = 10,
            restrict_angle: bool = False,
            direction_point: Optional[np.ndarray] = None,
            angle_range: float = math.pi / 1.5,
            unsafe: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iteratively expand radii until enough candidates are found.

        Input center: (2,) or (1,2).
        Output (single center): ids (k,), coords (k,2), pots (k,).
        Raises if not enough candidates after max_iterations.

        If unsafe=True, `center` (and `direction_point`, if provided) must already be (2,) float64.
        """
        if unsafe:
            c = center  # must be (2,) float64
            dp = direction_point if (restrict_angle and direction_point is not None) else None
        else:
            c = h.to_point_1d(center)
            dp = h.to_point_1d(direction_point) if (restrict_angle and direction_point is not None) else None

        i = 0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Finding candidates for type {act_type} within a ring around {c} "
                f"with radii {radius_outer} (outer) and {radius_inner} (inner)."
            )

        while True:
            cand = self.query_within_ring(act_type, c, radius_outer, radius_inner, unsafe=True)  # already (2,) float64
            if cand is not None:
                ids, coords, pots = cand  # ids:(k,), coords:(k,2), pots:(k,)
                if restrict_angle and ids.size:
                    mask = h.is_within_angle(coords, c, dp, angle_range)
                    ids, coords, pots = ids[mask], coords[mask], pots[mask]

                if ids.size >= min_candidates:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found {ids.size} candidates.")
                    if self.stats_tracker:
                        self.stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                    return ids, coords, pots

            radius_outer, radius_inner = h.spread_radii(radius_outer, radius_inner, iteration=i, first_step=20)
            i += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Iteration {i}. Increasing radii to {radius_outer} (outer) and {radius_inner} (inner).")
            if i > max_iterations:
                raise ValueError(f"Not enough candidates found after {max_iterations} iterations.")

    def ensure_overlap(self, location1, location2, r1a, r1b, r2a, r2b):
        """
        Ensure that two annuli have at least a degenerate overlap by minimally adjusting outer radii.
        Returns (r1a, r1b, r2a, r2b, changed_radii).
        NOTE: Ineffective, needs rework (currently unused).
        """
        D = np.linalg.norm(location2 - location1)
        changed_radii = False
        if D > (r1b + r2b):
            changed_radii = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Locations too far apart. Increasing radii to touch.")
            delta = D - (r1b + r2b)
            r1b += delta / 2.0
            r2b += delta / 2.0
        elif r1b < r2a:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Ring 1 fully inside Ring 2.")
        elif r2b < r1a:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Ring 2 fully inside Ring 1.")
        return r1a, r1b, r2a, r2b, changed_radii

    def get_overlapping_rings_candidates(
            self,
            act_type: str,
            location1: np.ndarray,
            location2: np.ndarray,
            r1outer: float,
            r1inner: float,
            r2outer: float,
            r2inner: float,
            min_candidates: int = 1,
            max_iterations: int = 15,
            unsafe: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
        """
        Iteratively expand two annuli until enough overlap candidates are found.

        Input: location1, location2 as (2,) or (1,2).
        Output (single pair): ids (k,), coords (k,2), pots (k,).
        Returns ((ids, coords, pots), iterations_used).

        Set unsafe=True if both locations are already (2,) float64 (skips copy/shape fix in the called function).
        """
        i = 0
        while True:
            cand = self.query_within_two_overlapping_rings(
                act_type, location1, location2,
                r1outer, r1inner, r2outer, r2inner,
                unsafe=unsafe,
            )
            if cand is not None:
                ids, coords, pots = cand  # ids: (k,), coords: (k,2), pots: (k,)
                m = ids.shape[0]
                if m >= min_candidates:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found {m} candidates after {i} iterations.")
                    if self.stats_tracker:
                        self.stats_tracker.log(f"Find_overlapping_ring_candidates: Iterations for {act_type}", i)
                    return (ids, coords, pots), i

            # Expand both annuli and try again
            r1outer, r1inner = h.spread_radii(r1outer, r1inner, iteration=i, first_step=50)
            r2outer, r2inner = h.spread_radii(r2outer, r2inner, iteration=i, first_step=50)
            i += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Iteration {i}. Increasing radii to "
                    f"[{r1outer:.3f}, {r1inner:.3f}] and [{r2outer:.3f}, {r2inner:.3f}]."
                )
            if i > max_iterations:
                raise RuntimeError(f"Not enough candidates found after {max_iterations} iterations.")

    def get_circle_intersection_candidates(
            self,
            start_coord: np.ndarray,
            end_coord: np.ndarray,
            act_type: str,
            distance_start_to_act: float,
            distance_act_to_end: float,
            num_candidates: int,
            unsafe: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Nearest candidates for up to two circle intersections.

        Input: start_coord, end_coord as (2,) or (1,2).
        Output: ids (L,), coords (L,2), pots (L,), where L = k if one intersection,
                else 2*k (top-k per intersection concatenated; no dedupe/rerank).
        Returns None if no valid intersections and only_return_valid=True.
        Raises RuntimeError if no intersections/fallbacks and only_return_valid=False.

        If `unsafe=True`, this assumes float64 for input points
        """
        p1, p2 = h.get_circle_intersections(
            start_coord, distance_start_to_act, end_coord, distance_act_to_end,
        )

        # no intersections
        if p1 is None and p2 is None:
            raise RuntimeError("No circle intersections or fallbacks produced.")

        # single intersection → direct call; returns (k,), (k,2), (k,)
        if p2 is None or p1 is None:
            pt = p1 if p1 is not None else p2
            ids, coords, pots = self.query_closest(
                act_type, pt, num_candidates, unsafe=unsafe
            )
            return ids, coords, pots

        # two intersections → one batched call, then flatten to (2*k,), (2*k,2), (2*k,)
        qp = np.empty((2, 2), dtype=np.float64)
        qp[0] = p1
        qp[1] = p2

        ids, coords, pots = self.query_closest(
            act_type, qp, num_candidates, unsafe=True
        )  # ids:(2,k), coords:(2,k,2), pots:(2,k)

        # Concatenate per-intersection results (views if contiguous)
        return ids.reshape(-1), coords.reshape(-1, 2), pots.reshape(-1)

    def get_best_circle_intersection_locations(
            self, *,
            act_type: str,
            start_coord: np.ndarray,  # (2,)
            end_coord: np.ndarray,  # (2,)
            dist_start_to_act: float,
            dist_act_to_end: float,
            min_candidates: int = 5,
            sel_candidates: int = 1,
            scorer=None,  # score(potentials:(k,), dist_deviations:(k,)) -> (k,)
            selector=None,  # select(scores:(k,), n:int, strategy:str, coords:ndarray|None, rng:Generator|None) -> (n,)
            selection_strategy: str = "top_n",
            near_eps: float = 1.0,
            rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Choose up to `sel_candidates` best activity locations from circle intersections or annulus fallback.

        Returns ids (m,), coords (m, 2), pots (m,), scores (m,)
        where m = min(sel_candidates, number_of_candidates_found).
        """
        if sel_candidates < 1:
            raise ValueError("sel_candidates must be >= 1")
        if min_candidates < sel_candidates:
            logger.warning(
                f"min_candidates ({min_candidates}) is smaller than sel_candidates ({sel_candidates}). "
                f"This will result in fewer candidates being selected."
            )

        # Candidate generation (ids:(k,), coords:(k,2), pots:(k,))
        dx = start_coord[0] - end_coord[0]
        dy = start_coord[1] - end_coord[1]
        near = (dx * dx + dy * dy) < (near_eps * near_eps)

        if near:
            r_outer, r_inner = h.spread_radii(dist_start_to_act, dist_act_to_end, spread_to=20)
            ids, coords, pots = self.get_ring_candidates(
                act_type, start_coord, r_outer, r_inner, min_candidates=min_candidates, unsafe=True
            )
        else:
            ids, coords, pots = self.get_circle_intersection_candidates(
                start_coord, end_coord, act_type, dist_start_to_act, dist_act_to_end,
                min_candidates, unsafe=True
            )

        if ids is None or ids.size == 0:
            raise RuntimeError("No candidates produced by generator.")

        # Distance deviations (k,)
        ddev = (
                h.get_abs_distance_deviations(coords, start_coord, dist_start_to_act) +
                h.get_abs_distance_deviations(coords, end_coord, dist_act_to_end)
        )

        # Scores (k,)
        scores = scorer.score(potentials=pots, dist_deviations=ddev).astype(float, copy=False)
        sel_idx = selector.select(scores, sel_candidates, selection_strategy, coords=coords, rng=rng)

        # Slice and return
        return ids[sel_idx], coords[sel_idx], pots[sel_idx], scores[sel_idx]

    def get_best_overlap_candidates(
            self,
            *,
            act_type: str,
            start_coord: np.ndarray,  # (2,)
            end_coord: np.ndarray,  # (2,)
            distances_start_to_act: np.ndarray,  # (p,)
            distances_act_to_end: np.ndarray,  # (q,)
            min_candidates: int = 1,
            sel_candidates: int = 1,
            max_iterations: Optional[int] = None,
            selection_strategy: Optional[str] = None,
            scorer=None,  # expects: evaluate(potentials:(k,), dist_deviations:(k,)) -> scores:(k,)
            selector=None,
            rng: np.random.Generator | None = None,
            # expects: select_candidate_indices(scores:(k,), num_candidates:int, strategy:str, coords:(k,2)) -> (idx,(n,)), (sel_scores,(n,))
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Overlapping rings → candidates → score → select.
        Min_candidates: minimum number of candidates to find and score.
        Sel_candidates: number of candidates to select and return.
        Returns (ids (n,), coords (n,2), pots (n,), scores (n,)).
        """

        # Envelope radii: (max, min) for each side (API expects outer, inner)
        r1_min, r1_max = h.get_min_max_distance(distances_start_to_act)
        r2_min, r2_max = h.get_min_max_distance(distances_act_to_end)
        r1outer, r1inner = r1_max, r1_min
        r2outer, r2inner = r2_max, r2_min

        # Rings query (k,), (k,2), (k,)
        (cand_ids, cand_coords, cand_pots), iterations = self.get_overlapping_rings_candidates(
            act_type,
            start_coord, end_coord,
            r1outer, r1inner, r2outer, r2inner,
            min_candidates=min_candidates,
            max_iterations=max_iterations,
        )

        k = 0 if cand_ids is None else cand_ids.size
        if k == 0:
            raise RuntimeError("No overlapping-rings candidates found.")

        # Deviations (only lowest-level side to avoid double counting!!)
        deviations = np.zeros(k, dtype=float)
        if iterations > 0:
            # when we had to expand, penalize only the side that has a single leg length
            if len(distances_start_to_act) == 1:
                deviations += h.get_abs_distance_deviations(cand_coords, start_coord, distances_start_to_act)
            elif len(distances_act_to_end) == 1:
                deviations += h.get_abs_distance_deviations(cand_coords, end_coord, distances_act_to_end)
        else:
            # in the no-expansion case, total deviations should be zero
            if len(distances_start_to_act) == 1:
                deviations += h.get_abs_distance_deviations(cand_coords, start_coord, distances_start_to_act)
            if len(distances_act_to_end) == 1:
                deviations += h.get_abs_distance_deviations(cand_coords, end_coord, distances_act_to_end)
            if np.any(deviations != 0.0):
                raise ValueError("Total deviations should be zero in no-iteration case.")


        scores = scorer.score(potentials=cand_pots, dist_deviations=deviations)
        sel_idx = selector.select(
            scores=scores,
            num_candidates=sel_candidates,
            strategy=selection_strategy,
            coords=cand_coords,  # some strategies may use spatial layout
            rng=rng,
        )
        return cand_ids[sel_idx], cand_coords[sel_idx], cand_pots[sel_idx], scores[sel_idx]
