import math
from typing import List, Tuple, Sequence, Optional
import numpy as np

"""Minimal helper set used by the locator/algorithms."""

# ---- geometry / distances ----
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))
#
# def angle_between(p1: Sequence[float], p2: Sequence[float]) -> float:
#     """Angle (rad) of vector p1->p2 in [-pi, pi]."""
#     dx, dy = float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])
#     return math.atan2(dy, dx)
#
# def is_within_angle(
#     point: np.ndarray,
#     center: np.ndarray,
#     direction_point: np.ndarray,
#     angle_range: float,
# ) -> bool:
#     """
#     Check if 'point' lies within a sector centered at 'center' whose axis
#     points towards 'direction_point' and has full width 'angle_range'.
#     Robust to wrap-around.
#     """
#     # Degenerate axis: accept all to avoid accidental filtering
#     if np.allclose(center, direction_point):
#         return True
#
#     theta_p = angle_between(center, point)
#     theta_axis = angle_between(center, direction_point)
#
#     # Normalize to [0, 2π)
#     def norm(t): return (t + 2 * math.pi) % (2 * math.pi)
#     tp = norm(theta_p)
#     ta = norm(theta_axis)
#
#     half = angle_range / 2.0
#     lo = norm(ta - half)
#     hi = norm(ta + half)
#
#     if lo <= hi:
#         return lo <= tp <= hi
#     # interval wraps across 2π
#     return tp >= lo or tp <= hi

# ---- ring growth / bounds ----
from typing import Tuple


def spread_radii(
    outer: float,
    inner: float,
    *,
    iteration: int = 0,
    first_step: float = 20.0,
    base: float = 1.5,
    spread_to: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Expand an annulus given (outer, inner).

    - Iterative: Each iteration increases the gap by step = first_step * base**iteration
    - spread_to: Ensure the difference outer-inner >= spread_to. If already larger, return unchanged.
    Always enforces outer >= inner >= 0.
    """
    # guard against swapped inputs
    if inner > outer:
        outer, inner = inner, outer

    if spread_to is not None:
        gap = outer - inner
        if gap >= spread_to:
            return outer, inner
        # expand symmetrically
        missing = spread_to - gap
        outer += missing / 2.0
        inner = max(0.0, inner - missing / 2.0)
        return outer, inner

    # default iterative mode
    step = first_step * (base ** iteration)
    new_outer = outer + step
    new_inner = max(0.0, inner - step)
    return new_outer, new_inner



def get_min_max_distance(distances: Sequence[float]) -> Tuple[float, float]:
    """
    Triangle-inequality bounds for radial distance from an endpoint to an anchor
    given a chain of leg lengths.
    min = max( max(li - (sum - li)), 0 )
    max = sum(li)
    """
    if not distances:
        raise ValueError("No distances given.")
    if len(distances) == 1:
        d = float(distances[0])
        return d, d

    arr = np.asarray(distances, dtype=float)
    total = float(arr.sum())
    # largest excess a single leg can have over all the others
    overshoot = float(np.max(arr - (total - arr)))
    min_possible = max(overshoot, 0.0)
    return min_possible, total

def get_abs_distance_deviations(
    candidate_coords: np.ndarray,
    center: np.ndarray,
    target_distance: np.ndarray | float,
) -> np.ndarray:
    """
    | ||x - center|| - target_distance | for each candidate.
    Accepts:
      - candidate_coords: (N,2) or (2,)
      - center: (2,)
      - target_distance: scalar or shape-(N,) array (broadcasted if needed)
    Returns shape-(N,) float array.
    """
    coords = np.atleast_2d(np.asarray(candidate_coords, dtype=float))
    ctr = np.asarray(center, dtype=float).reshape(1, -1)
    dists = np.linalg.norm(coords - ctr, axis=1)
    tgt = np.asarray(target_distance, dtype=float)
    if tgt.ndim == 0:
        tgt = np.full_like(dists, float(tgt))
    elif tgt.shape != dists.shape:
        tgt = np.broadcast_to(tgt, dists.shape)
    return np.abs(dists - tgt)

# ---- activity helpers ----
def get_main_activity_leg(person_legs: list[dict]) -> tuple[Optional[int], Optional[dict]]:
    """
    Return (index, leg) for the first leg with 'is_main_activity' truthy.
    If none found, assert all to_act_type are 'home' and return (None, None).
    """
    for i, leg in enumerate(person_legs):
        if leg.get("is_main_activity"):
            return i, leg

    # Fallback: ensure no non-home legs
    to_types = [leg.get("to_act_type") for leg in person_legs]
    if any(t != "home" for t in to_types):
        raise AssertionError("Person has no main activity but has non-home legs.")
    return None, None

# ---- estimation tree (slack) ----
def build_estimation_tree(distances: List[float]) -> List[List[List[float]]]:
    """
    Build hierarchical pairs of estimates:
      each entry: [real_min, wanted_min, value, wanted_max, real_max]
    Combines adjacent legs bottom-up; last odd leg is carried up unchanged.
    """
    arr = [float(x) for x in distances]
    tree: List[List[List[float]]] = []
    while len(arr) > 1:
        level_vals: List[float] = []
        level_pairs: List[List[float]] = []
        for i in range(0, len(arr) - 1, 2):
            combo = estimate_length_with_slack(arr[i], arr[i + 1])
            level_pairs.append(combo)
            level_vals.append(combo[2])

        if len(arr) % 2 == 1:
            carry = tree[-1][-1] if tree else [arr[-1]] * 5
            level_pairs.append(carry)
            level_vals.append(carry[2])

        tree.append(level_pairs)
        arr = level_vals
    return tree

def estimate_length_with_slack(
    l1: float,
    l2: float,
    *,
    slack_factor: float = 2.0,
    min_slack_lower: float = 0.2,
    min_slack_upper: float = 0.2,
) -> List[float]:
    """
    Heuristic estimate for combined direct length of two legs with slack.
    Guarantees:
      real_min = |l1-l2|, real_max = l1+l2
      wanted_min/max enforce a minimal slack around the shorter leg.
      value is clamped to [wanted_min, wanted_max].
    """
    l1 = float(l1); l2 = float(l2)
    real_max = l1 + l2
    real_min = abs(l1 - l2)
    shorter = min(l1, l2)

    # initial guess: divide sum by slack
    val = real_max / float(slack_factor)

    wanted_min = real_min + shorter * float(min_slack_lower)
    wanted_max = real_max - shorter * float(min_slack_upper)

    # clamp
    val = max(wanted_min, min(val, wanted_max))
    return [real_min, wanted_min, val, wanted_max, real_max]

from typing import Optional, Tuple
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def get_circle_intersections(
    center1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    radius2: float,
    only_return_valid: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find intersection points of two circles.

    Input: center1, center2 as (2,) float-like; radius1, radius2 as floats.
    Output: (p1, p2), each either (2,) float64 or None.
    Behavior:
      - Two proper intersections → (p1, p2)
      - Tangential (internal/external) → (pt, None)
      - Too far / one inside the other:
          * only_return_valid=True  → (None, None)
          * only_return_valid=False → (fallback_point, None)
      - Identical centers (~) → raises RuntimeError (use ring/annulus search upstream)
    """
    # Strict shape check (no reshaping for speed/clarity)
    if not (isinstance(center1, np.ndarray) and center1.ndim == 1 and center1.shape[0] == 2):
        raise ValueError(f"center1 must be shape (2,), got {getattr(center1, 'shape', None)}")
    if not (isinstance(center2, np.ndarray) and center2.ndim == 1 and center2.shape[0] == 2):
        raise ValueError(f"center2 must be shape (2,), got {getattr(center2, 'shape', None)}")

    x1, y1 = float(center1[0]), float(center1[1])
    x2, y2 = float(center2[0]), float(center2[1])
    r1, r2 = float(radius1), float(radius2)

    dx = x2 - x1
    dy = y2 - y1
    d  = math.hypot(dx, dy)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"C1=({x1:.6f},{y1:.6f}), r1={r1:.6f}; "
                     f"C2=({x2:.6f},{y2:.6f}), r2={r2:.6f}; d={d:.6f}")

    # identical centers → undefined intersection set
    if d < 1e-4:
        raise RuntimeError("Identical centers should be handled via ring/annulus search upstream.")

    # no true intersection: too far apart
    if d > (r1 + r2):
        if only_return_valid:
            return None, None
        # Fallback: point along the line between centers at ratio r1:(r1+r2)
        t  = r1 / (r1 + r2)
        px = x1 + t * dx
        py = y1 + t * dy
        return np.array([px, py], dtype=np.float64), None

    # no true intersection: one circle fully inside the other (d < |r1 - r2|)
    if d < abs(r1 - r2):
        if only_return_valid:
            return None, None
        # Fallback: a point along the line between centers, halfway w.r.t. gap
        # Derives to t = 0.5 * (d + r_large + r_small) / d measured from the larger circle.
        if r1 >= r2:
            t  = 0.5 * (d + r1 + r2) / d
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
        else:
            t  = 0.5 * (d + r1 + r2) / d
            px = x2 + t * (x1 - x2)
            py = y2 + t * (y1 - y2)
        return np.array([px, py], dtype=np.float64), None

    # tangential (external or internal)
    if math.isclose(d, r1 + r2, rel_tol=1e-12, abs_tol=1e-12) or \
       math.isclose(d, abs(r1 - r2), rel_tol=1e-12, abs_tol=1e-12):
        a  = (r1*r1 - r2*r2 + d*d) / (2.0 * d)
        x3 = x1 + a * dx / d
        y3 = y1 + a * dy / d
        return np.array([x3, y3], dtype=np.float64), None

    # two proper intersections
    a   = (r1*r1 - r2*r2 + d*d) / (2.0 * d)
    h2  = r1*r1 - a*a
    h   = math.sqrt(h2) if h2 > 0.0 else 0.0  # guard small negatives

    x3  = x1 + a * dx / d
    y3  = y1 + a * dy / d
    rx  = -dy / d
    ry  =  dx / d

    p1  = np.array([x3 + h * rx, y3 + h * ry], dtype=np.float64)
    p2  = np.array([x3 - h * rx, y3 - h * ry], dtype=np.float64)
    return p1, p2




#
# def get_best_circle_intersection_location(
#     self,
#     start_coord: np.ndarray,
#     end_coord: np.ndarray,
#     act_type: str,
#     distance_start_to_act: float,
#     distance_act_to_end: float,
#     num_circle_intersection_candidates: Optional[int] = None,
#     selection_strategy: str = "top_n",
#     max_iterations: int = 15,           # kept for signature parity
#     only_return_valid: bool = False,
# ):
#     # home special case
#     if act_type == s.ACT_HOME:
#         logger.warning("Home activity: returning start location.")
#         return None, start_coord, None, None
#
#     # nearly identical endpoints → annulus fallback
#     if h.euclidean_distance(start_coord, end_coord) < 1e-4:
#         if only_return_valid and abs(distance_act_to_end - distance_start_to_act) > 10:
#             return None, None, None, None
#         r1, r2 = h.spread_radii(distance_start_to_act, distance_act_to_end)
#         cand_ids, cand_xy, cand_pots = self.locations.get_ring_candidates(
#             act_type, start_coord, r1, r2, max_iterations=max_iterations, min_candidates=1
#         )
#     else:
#         cand_ids, cand_xy, cand_pots = self.get_circle_intersection_candidates(
#             start_coord, end_coord, act_type,
#             distance_start_to_act, distance_act_to_end,
#             num_candidates=(num_circle_intersection_candidates or 1),
#             only_return_valid=only_return_valid
#         )
#         if cand_ids is None:
#             if only_return_valid:
#                 return None, None, None, None
#             raise RuntimeError("No candidates from circle intersections.")
#
#     # score by distance deviations to both ends
#     ddev = (
#         h.get_abs_distance_deviations(cand_xy, start_coord, distance_start_to_act)
#         + h.get_abs_distance_deviations(cand_xy, end_coord, distance_act_to_end)
#     )
#     scores = EvaluationFunction.evaluate_candidates(cand_pots, ddev)
#     best_idx = EvaluationFunction.select_candidate_indices(scores, 1, selection_strategy)[0]
#
#     best_id = cand_ids[best_idx][0]
#     best_xy = cand_xy[best_idx][0]
#     best_pot = cand_pots[best_idx][0]
#     best_score = scores[best_idx][0]
#     return best_id, best_xy, best_pot, best_score

# # UNTESTED BELOW -> hlpers for e.g. CARLA. tie locs, eval/selection togwther.
# def get_best_circle_intersection_location(
#     *,
#     target_locs,                  # TargetLocations
#     geom,                         # has find_circle_intersection_candidates(...)
#     scorer, selector,             # pluggable eval/selection
#     act_type: str,
#     start_coord, end_coord,       # (2,)
#     dist_start_to_act: float,
#     dist_act_to_end: float,
#     k: int,                       # 1 or 2
#     selection_strategy: str = "top_n",
#     only_return_valid: bool = False,
# ) -> Tuple[Optional[int], Optional[np.ndarray], Optional[float], Optional[float]]:
#     """
#     Returns (best_id, best_xy(2,), best_pot, best_score) or (None, None, None, None).
#     """
#     ids, xy, pots = geom.find_circle_intersection_candidates(
#         start_coord, end_coord, act_type, dist_start_to_act, dist_act_to_end,
#         num_candidates=k, only_return_valid=only_return_valid
#     )
#     if ids is None:  # empty
#         return (None, None, None, None)
#
#     # xy: (m,k,2), pots: (m,k)
#     flat_xy = xy.reshape(-1, 2)
#     ddev = (
#         h.get_abs_distance_deviations(flat_xy, start_coord, dist_start_to_act) +
#         h.get_abs_distance_deviations(flat_xy, end_coord,   dist_act_to_end)
#     ).reshape(pots.shape)  # (m,k)
#
#     scores = scorer.evaluate_candidates(pots, ddev)             # (m,k)
#     i, j = selector.select_candidate_indices(scores, 1, selection_strategy)[0]
#     return ids[i, j], xy[i, j], pots[i, j], scores[i, j]
#
# def get_best_overlapping_ring_locations(
#     *,
#     target_locs, geom, scorer, selector,
#     act_type: str,
#     loc1, loc2,                         # (2,)
#     r1_outer: float, r1_inner: float,
#     r2_outer: float, r2_inner: float,
#     min_candidates: int = 1,
#     max_iterations: int = 15,
#     selection_strategy: str = "top_n",
# ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], int, Tuple[int, np.ndarray, float, float]]:
#     """
#     Returns ((ids(m,1), xy(m,1,2), pots(m,1)), iterations, best_tuple).
#     best_tuple = (best_id, best_xy(2,), best_pot, best_score)
#     """
#     (ids, xy, pots), iters = target_locs.find_overlapping_rings_candidates(
#         act_type, loc1, loc2, r1_outer, r1_inner, r2_outer, r2_inner,
#         min_candidates=min_candidates, max_iterations=max_iterations
#     )
#
#     # score each row (k=1 → squeeze second dim)
#     m = ids.shape[0]
#     flat_xy = xy.reshape(m, 2)
#     ddev = (
#         h.get_abs_distance_deviations(flat_xy, loc1, r1_outer) +
#         h.get_abs_distance_deviations(flat_xy, loc2, r2_outer)
#     )[:, None]                         # match (m,1)
#
#     scores = scorer.evaluate_candidates(pots, ddev)             # (m,1)
#     (i, j) = selector.select_candidate_indices(scores, 1, selection_strategy)[0]
#     best = (ids[i, j], xy[i, j], pots[i, j], scores[i, j])
#     return (ids, xy, pots), iters, best


def to_point_1d(loc: np.ndarray) -> np.ndarray:
    """
    Ensure a single 2D point is in shape (2,) float64.
    Accepts (2,) or (1,2).
    """
    arr = np.asarray(loc, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (1, 2):
            raise ValueError(f"Expected (2,) or (1,2); got {arr.shape}")
        return arr[0]
    elif arr.shape != (2,):
        raise ValueError(f"Expected (2,) or (1,2); got {arr.shape}")
    return arr


def is_within_angle(
    points: np.ndarray,          # (k,2)
    center: np.ndarray,          # (2,)
    direction_point: np.ndarray, # (2,)
    angle_range: float,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized sector test.
    Returns a boolean mask of shape (k,) indicating which points lie within
    the sector centered at `center`, axis towards `direction_point`,
    full width `angle_range` (radians). Robust to wrap-around and avoids atan2/arccos.
    """
    # Degenerate axis: accept all
    if np.allclose(center, direction_point, atol=atol):
        return np.ones(points.shape[0], dtype=bool)

    v = points - center               # (k,2)
    r = direction_point - center      # (2,)

    # Norms
    v_norm = np.linalg.norm(v, axis=1)            # (k,)
    r_norm = np.linalg.norm(r)                    # scalar

    # Points at the center: accept (no direction)
    mask_nonzero = v_norm > atol
    # Unit vectors where valid
    v_unit = np.zeros_like(v)
    v_unit[mask_nonzero] = v[mask_nonzero] / v_norm[mask_nonzero, None]
    r_unit = r / r_norm

    # cos(angle) via dot product, clamp for numeric safety
    cos_theta = np.clip((v_unit @ r_unit), -1.0, 1.0)  # (k,)

    # Sector test: cos(theta) >= cos(half_range)
    half = angle_range / 2.0
    cos_half = np.cos(half)

    mask = cos_theta >= cos_half
    # Include points exactly at center
    mask |= ~mask_nonzero
    return mask

def even_spatial_downsample(coords: np.ndarray, num_cells_x: int = 20, num_cells_y: int = 20) -> np.ndarray:
    """
    Even spatial downsample by grid; keep ≤1 point per cell (first wins).
    Input: coords (n,2); num_cells_x:int≥1; num_cells_y:int≥1.
    Output: indices (k,), original order; k ≤ min(n, num_cells_x*num_cells_y).
    Returns indices (intp).
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n,2), got {coords.shape}")
    if num_cells_x < 1 or num_cells_y < 1:
        raise ValueError("num_cells_x/num_cells_y must be >= 1")

    x = coords[:, 0].astype(float, copy=False)
    y = coords[:, 1].astype(float, copy=False)

    min_x, max_x = np.nanmin(x), np.nanmax(x)
    min_y, max_y = np.nanmin(y), np.nanmax(y)

    dx = max(max_x - min_x, 1e-12)  # guard against zero span
    dy = max(max_y - min_y, 1e-12)

    step_x = dx / num_cells_x
    step_y = dy / num_cells_y

    cx = np.clip(np.floor((x - min_x) / step_x).astype(np.int64), 0, num_cells_x - 1)
    cy = np.clip(np.floor((y - min_y) / step_y).astype(np.int64), 0, num_cells_y - 1)

    cell_id = cy * num_cells_x + cx  # unique id per cell

    # First index per cell; np.unique returns first occurrence indices (but sorted by cell_id),
    # so sort those indices to preserve original order.
    _, first_idx = np.unique(cell_id, return_index=True)
    keep_indices = np.sort(first_idx)

    return keep_indices


def to_bool(val) -> Optional[bool]:
    """Coerce common truthy/falsy representations to bool."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot convert {val!r} to bool.")
