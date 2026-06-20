from typing import List, Tuple, Sequence, Optional, Dict, Set, TYPE_CHECKING
import numpy as np
import math
import logging

if TYPE_CHECKING:
    from .types import Leg

logger = logging.getLogger(__name__)

"""Minimal helper set used by the locator/algorithms."""

# ---- geometry / distances ----
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


# ---- ring growth / bounds ----


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
    Expects list, tuple, or ndarray (n,).
    Triangle-inequality bounds for radial distance from an endpoint to an anchor
    given a chain of leg lengths.
    min = max( max(li - (sum - li)), 0 )
    max = sum(li)
    """
    if len(distances) == 0:
        raise ValueError("No distances given.")
    if len(distances) == 1:
        d = float(distances[0])
        return d, d

    arr = np.asarray(distances)
    total = float(arr.sum())
    # largest excess a single leg can have over all the others
    overshoot = float(np.max(arr - (total - arr)))
    min_possible = max(overshoot, 0.0)
    return min_possible, total

def get_distance_deviations(
        candidate_coords: np.ndarray,  # (N,2)
        center: np.ndarray,  # (2,)
        target_distance: float,
) -> np.ndarray:
    """
    Return |‖x - center‖ - target_distance| for each candidate (deviations: (N,) array, >= 0).
    """
    assert candidate_coords.ndim == 2 and candidate_coords.shape[1] == 2
    assert center.shape == (2,)
    assert np.isscalar(target_distance)

    dists = np.hypot(candidate_coords[:, 0] - center[0],
                     candidate_coords[:, 1] - center[1])
    return np.abs(dists - target_distance)

def get_abstract_distance_deviations(
    candidate_coords: np.ndarray,  # (N,2)
    center: np.ndarray,            # (2,)
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """
    Deviation from being within [r_min, r_max].
    If d in [r_min, r_max], deviation = 0.
    If d < r_min, deviation = r_min - d.
    If d > r_max, deviation = d - r_max.
    Returns deviations: (N,) array, >= 0
    """
    assert candidate_coords.ndim == 2 and candidate_coords.shape[1] == 2
    assert center.shape == (2,)
    assert np.isscalar(r_min) and np.isscalar(r_max)
    assert r_min <= r_max

    dists = np.hypot(candidate_coords[:, 0] - center[0],
                     candidate_coords[:, 1] - center[1])

    # Vectorized "outside distance"
    dev_below = np.clip(r_min - dists, a_min=0, a_max=None)
    dev_above = np.clip(dists - r_max, a_min=0, a_max=None)
    return dev_below + dev_above


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
        # Fallback: point along the line between centers at ratio r1:(r1+r2).
        # If both radii are zero (both target distances 0), the midpoint minimizes total deviation.
        t  = 0.5 if (r1 + r2) == 0.0 else r1 / (r1 + r2)
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


def assert_point2(p: np.ndarray):
    assert isinstance(p, np.ndarray) and p.dtype == np.float64 and p.shape == (2,), \
        f"Expected (2,) float64, got {getattr(p, 'shape', None)} {getattr(p, 'dtype', None)}"


# ---- Connection detection for household legs ----

def check_distance_match(
    leg_a: 'Leg',
    leg_b: 'Leg',
    tolerance_m: float
) -> bool:
    """
    Check if two legs have similar distances within tolerance.

    Returns False if either leg has None distance.
    Returns True if |leg_a.distance - leg_b.distance| <= tolerance_m.
    """
    if leg_a.distance is None or leg_b.distance is None:
        return False

    return abs(leg_a.distance - leg_b.distance) <= tolerance_m


def check_time_match(
    leg_a: 'Leg',
    leg_b: 'Leg',
    dep_tolerance_s: int,
    arr_tolerance_s: int
) -> bool:
    """
    Check if two legs have similar departure and arrival times.

    Returns False if any required time field is None.
    Checks both: |dep_a - dep_b| <= dep_tolerance_s AND |arr_a - arr_b| <= arr_tolerance_s.
    """
    if (leg_a.dep_time_s is None or leg_b.dep_time_s is None or
        leg_a.arr_time_s is None or leg_b.arr_time_s is None):
        return False

    dep_match = abs(leg_a.dep_time_s - leg_b.dep_time_s) <= dep_tolerance_s
    arr_match = abs(leg_a.arr_time_s - leg_b.arr_time_s) <= arr_tolerance_s

    return dep_match and arr_match


def check_mode_match(
    leg_a: 'Leg',
    leg_b: 'Leg',
    compatible_modes: Dict[str, Set[str]]
) -> bool:
    """
    Check if two legs have compatible travel modes.

    Returns False if either mode is None.
    Checks if mode_a == mode_b OR mode_b in compatible_modes.get(mode_a, set()).
    Compatibility should be symmetric (if a->b then b->a).
    """
    if leg_a.mode is None or leg_b.mode is None:
        return False

    mode_a = leg_a.mode
    mode_b = leg_b.mode

    # Direct match
    if mode_a == mode_b:
        return True

    # Check compatibility dict
    if mode_b in compatible_modes.get(mode_a, set()):
        return True

    # Check reverse (for symmetric compatibility)
    if mode_a in compatible_modes.get(mode_b, set()):
        return True

    return False


def check_activity_match(
    leg_a: 'Leg',
    leg_b: 'Leg',
    compatible_activities: Dict[str, Set[str]]
) -> bool:
    """
    Check if two legs go to compatible activity types.

    Returns False if either to_act_type is None.
    Checks if act_a == act_b OR act_b in compatible_activities.get(act_a, set()).
    Compatibility should be symmetric.
    """
    if leg_a.to_act_type is None or leg_b.to_act_type is None:
        return False

    act_a = leg_a.to_act_type
    act_b = leg_b.to_act_type

    # Direct match
    if act_a == act_b:
        return True

    # Check compatibility dict
    if act_b in compatible_activities.get(act_a, set()):
        return True

    # Check reverse (for symmetric compatibility)
    if act_a in compatible_activities.get(act_b, set()):
        return True

    return False