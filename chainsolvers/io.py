from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Mapping, Tuple, Any, Optional, List
import difflib
import numpy as np
import pandas as pd
from collections import defaultdict
from frozendict import frozendict

from .types import Leg, SegmentedPlans
from .helpers import to_bool

import logging

logger = logging.getLogger(__name__)

# --- Plans --------------------------------------------------------------------
# --- Column spec --------------------------------------------------------------

@dataclass(frozen=True)
class PlanColumns:
    person_id: str = "unique_person_id"
    unique_leg_id: str = "unique_leg_id"
    to_act_type: str = "to_act_type"
    leg_distance_m: str = "distance_meters"
    from_x: str = "from_x"
    from_y: str = "from_y"
    to_x: str = "to_x"
    to_y: str = "to_y"

    # Optional columns (set to None if unused)
    mode: Optional[str] = "mode"
    to_act_is_main: Optional[str] = "to_act_is_main"
    to_act_identifier: Optional[str] = "to_act_identifier"
    to_act_name: Optional[str] = "to_act_name"

    def required(self) -> set[str]:
        return {
            self.person_id, self.unique_leg_id, self.to_act_type, self.leg_distance_m,
            self.from_x, self.from_y, self.to_x, self.to_y
        }

    def optional(self) -> set[str]:
        return {c for c in (self.mode, self.to_act_is_main, self.to_act_identifier, self.to_act_name) if c}

    def all(self) -> set[str]:
        return self.required() | self.optional()


def expected_columns() -> dict[str, list[str]]:
    """Return required/optional column names for docs and error messages."""
    c = PlanColumns()
    return {"required": sorted(c.required()), "optional": sorted(c.optional())}


def _suggest(similar_to: str, universe: Iterable[str], n: int = 3) -> list[str]:
    return difflib.get_close_matches(similar_to, list(universe), n=n, cutoff=0.6)


def validate_input_plans_df(df: pd.DataFrame, strict: bool = False) -> None:
    """
    Validate presence of required columns (and, if strict=True, that all configured optional names exist).
    Raises ValueError with helpful suggestions if something is missing.
    """

    c = PlanColumns()
    present = set(df.columns)

    missing_required = [name for name in c.required() if name not in present]
    missing_optional = [name for name in c.optional() if name not in present]

    msgs: list[str] = []

    if missing_required:
        msgs.append("Missing required columns:")
        for name in missing_required:
            sugg = _suggest(name, present)
            hint = f"  - {name}" + (f" (you did give: {', '.join(sugg)})" if sugg else "")
            msgs.append(hint)

    if strict and missing_optional:
        msgs.append("Missing optional columns (set the respective field to None in PlanColumns if unused):")
        for name in missing_optional:
            sugg = _suggest(name, present)
            hint = f"  - {name}" + (f" (you did give: {', '.join(sugg)})" if sugg else "")
            msgs.append(hint)

    if msgs:
        need = expected_columns()
        msgs.append(f"Expected → required={need['required']}, optional={need['optional']}")
        raise ValueError("\n".join(msgs))

    # --- Helpful summary ------------------------------------------------------
    try:
        n_persons = df[c.person_id].nunique(dropna=True)
        n_rows = len(df)
        logger.info("Input summary: %d legs across %d persons.", n_rows, n_persons)
    except Exception:
        logger.error("Could not compute summary stats.")
        pass

def convert_to_segmented_plans(
    df: pd.DataFrame,
    *,
    forbid_negative_distance: bool = True,
    forbid_missing_distance: bool = True,
) -> SegmentedPlans:
    """
    Convert a long-format trips DataFrame into SegmentedPlans using the default PlanColumns spec.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing at least the required columns defined by `PlanColumns`.
    forbid_negative_distance : bool, default False
        If True, raises on any leg with distance < 0.

    Returns
    -------
    SegmentedPlans
        frozendict mapping person_id -> tuple[Leg, ...]
    """
    c = PlanColumns()

    def safe_xy(x: Any, y: Any) -> Optional[np.ndarray]:
        # Return (2,) float64 if both present and finite; else None.
        if pd.notna(x) and pd.notna(y):
            xv, yv = float(x), float(y)
            if np.isfinite(xv) and np.isfinite(yv):
                return np.array([xv, yv], dtype=np.float64)
        return None

    # Pre-resolve column indices once
    col_index = {name: i for i, name in enumerate(df.columns)}

    def gi(row_tuple: tuple, name: Optional[str], default=None): # Needed bcs we use itertuples not iterrows
        if not name:
            return default
        idx = col_index.get(name)
        if idx is None:
            return default
        try:
            return row_tuple[idx]
        except IndexError:
            return default

    has_mode = bool(c.mode and c.mode in col_index)
    has_main = bool(c.to_act_is_main and c.to_act_is_main in col_index)
    has_ident = bool(c.to_act_identifier and c.to_act_identifier in col_index)

    buckets: dict[str, list[Leg]] = defaultdict(list)

    for row in df.itertuples(index=False, name=None):
        person_id = gi(row, c.person_id)

        to_act_is_main_val = to_bool(gi(row, c.to_act_is_main)) if has_main else None

        dist_raw = gi(row, c.leg_distance_m)

        if dist_raw is None or (isinstance(dist_raw, float) and dist_raw != dist_raw):  # None or NaN
            if forbid_missing_distance:
                raise ValueError(f"Missing distance for person '{person_id}'.")
            dist = 0.0
        else:
            dist = float(dist_raw)

        if forbid_negative_distance and dist < 0:
            raise ValueError(f"Negative distance for person '{person_id}': {dist}")

        leg = Leg(
            unique_leg_id=str(gi(row, c.unique_leg_id)),
            from_location=safe_xy(gi(row, c.from_x), gi(row, c.from_y)),
            to_location=safe_xy(gi(row, c.to_x), gi(row, c.to_y)),
            distance=dist,
            to_act_type=(gi(row, c.to_act_type) or "unknown"),
            to_act_identifier=(gi(row, c.to_act_identifier) if has_ident else None),
            mode=(gi(row, c.mode) if has_mode else None),
            to_act_is_main_act=to_act_is_main_val,
            extras=None
        )

        buckets[str(person_id)].append(leg)

    return frozendict({pid: tuple(legs) for pid, legs in buckets.items()})


def show_expected_columns() -> str:
    """Pretty one-liner for CLI/docs to show users what to provide."""
    e = expected_columns()
    return (
        "Required: " + ", ".join(e["required"]) + "\n"
        "Optional: " + (", ".join(e["optional"]) if e["optional"] else "—")
    )


# --- Locations --------------------------------------------------------------------


@dataclass(frozen=True)
class DictLocationSchema:
    """Field names inside each per-identifier entry."""
    coordinates: str = "coordinates"  # [x, y] or (x, y)
    potential: str = "potential"       # float
    name: str = "name"                # optional; ignored by payload


def build_locations_payload_from_dict(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    missing_potential_default: float = 0.0,
    drop_rows_with_invalid_xy: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convert a nested dict of facilities into the three arrays needed by `Locations`.

    Example
    -------
    data = {
      "home": {
          "h1": {"coordinates": [10, 20], "potential": 3.0},
          "h2": {"coordinates": [15, 25]},  # potential missing → 0.0
      }
    }
    ids, coords, pots = build_locations_payload_from_dict(data)
    ids["home"]

    Returns identifiers, coordinates, potentials : dict[str, np.ndarray]
        ids -> (n,), coords -> (n,2), pots -> (n,)
    """
    schema = DictLocationSchema
    id_out: Dict[str, list] = {}
    coord_out: Dict[str, list] = {}
    pot_out: Dict[str, list] = {}

    def ensure_act(act: str) -> None:
        if act not in id_out:
            id_out[act], coord_out[act], pot_out[act] = [], [], []

    for act_type, entries in data.items():
        if not isinstance(entries, Mapping):
            logger.warning(f"Expected a mapping for act_type '{act_type}', got {type(entries).__name__}. Skipping.")
            continue
        ensure_act(act_type)

        for identifier, entry in entries.items():
            coords = entry.get(schema.coordinates, None)
            pot = entry.get(schema.potential, missing_potential_default)

            # Coordinates validation
            try:
                if coords is None or len(coords) != 2:
                    raise ValueError("coords must be length-2")
                x = float(coords[0]); y = float(coords[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    raise ValueError("coords not finite")
            except Exception:
                if drop_rows_with_invalid_xy:
                    logger.warning(f"Skipping '{identifier}' in '{act_type}' due to invalid coordinates: {coords}")
                    continue
                x, y = np.nan, np.nan

            # Potential to float
            try:
                pot_f = float(pot)
            except Exception:
                pot_f = float(missing_potential_default)

            id_out[act_type].append(identifier)
            coord_out[act_type].append((x, y))
            pot_out[act_type].append(pot_f)

    # Convert to numpy
    identifiers: Dict[str, np.ndarray] = {}
    coordinates: Dict[str, np.ndarray] = {}
    potentials: Dict[str, np.ndarray] = {}

    for act, ids in id_out.items():
        identifiers[act] = np.asarray(ids, dtype=object)
        coordinates[act] = np.asarray(coord_out[act], dtype=float).reshape(-1, 2)
        potentials[act] = np.asarray(pot_out[act], dtype=float)

        n = coordinates[act].shape[0]
        if not (len(identifiers[act]) == n == len(potentials[act])):
            raise RuntimeError(f"Inconsistent lengths for activity '{act}'.")

    return identifiers, coordinates, potentials


@dataclass(frozen=True)
class LocationColumns:
    """
    Column names in the input (Geo)DataFrame.
    """
    id: str = "id"
    activities: str = "activities"
    x: Optional[str] = "x"
    y: Optional[str] = "y"
    potentials: Optional[str] = "potentials"
    name: Optional[str] = "name"

    def required(self) -> set[str]:
        req = {self.id, self.activities}
        if self.x and self.y:
            req |= {self.x, self.y}
        return req


def build_locations_payload_from_df(
    df: pd.DataFrame,
    *,
    cols: LocationColumns = LocationColumns(),
    activity_sep: str = ";",
    missing_potential_default: float = 0.0,
    drop_rows_with_invalid_xy: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Transform a (Geo)DataFrame of facilities into the three dicts needed by `Locations`.
    Returns identifiers, coordinates, potentials : dict[str, np.ndarray]
    ids -> (n,), coords -> (n,2), pots -> (n,)
    """
    df = df.copy()

    # --- Resolve X/Y sources --------------------------------------------------
    have_xy_cols = (cols.x and cols.x in df.columns) and (cols.y and cols.y in df.columns)
    if not have_xy_cols:
        cols_is_multi = isinstance(df.columns, pd.MultiIndex)
        has_geom = ("geometry" in df.columns.get_level_values(0)) if cols_is_multi else ("geometry" in df.columns)
        if has_geom:
            try:
                df["_x"] = df["geometry"].x
                df["_y"] = df["geometry"].y
                x_name, y_name = "_x", "_y"
            except Exception as e:
                raise ValueError(
                    "Could not derive coordinates from 'geometry'. "
                    "Provide explicit x/y columns via LocationColumns."
                ) from e
        else:
            raise ValueError(
                "No coordinate columns found and no 'geometry' column to fall back to. "
                "Provide x_col and y_col in LocationColumns."
            )
    else:
        x_name, y_name = cols.x, cols.y  # type: ignore[assignment]

    # --- Basic validation ------------------------------------------------------
    missing_req = [c for c in cols.required() if c not in df.columns]
    if not have_xy_cols:
        missing_req = [c for c in missing_req if c not in (cols.x, cols.y)]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    # Normalize to string for activity/potential parsing
    act_series = df[cols.activities].fillna("").astype(str)
    if cols.potentials and cols.potentials in df.columns:
        pot_series = df[cols.potentials].fillna("").astype(str)
    else:
        pot_series = pd.Series([""] * len(df), index=df.index)

    # Coordinates
    x = pd.to_numeric(df[x_name], errors="coerce")
    y = pd.to_numeric(df[y_name], errors="coerce")
    finite_mask = np.isfinite(x.values) & np.isfinite(y.values)

    if drop_rows_with_invalid_xy:
        bad = (~finite_mask)
        if bad.any():
            logger.warning(f"Dropping {int(bad.sum())} rows with invalid coordinates.")
        keep_idx = df.index[finite_mask]
        df = df.loc[keep_idx].copy()
        act_series = act_series.loc[keep_idx]
        pot_series = pot_series.loc[keep_idx]
        x = x.loc[keep_idx]; y = y.loc[keep_idx]

    ids = df[cols.id]

    # --- Accumulate per activity type ----------------------------------------
    id_map: Dict[str, list] = {}
    coord_map: Dict[str, list] = {}
    pot_map: Dict[str, list] = {}

    def ensure_keys(k: str) -> None:
        if k not in id_map:
            id_map[k] = []
            coord_map[k] = []
            pot_map[k] = []

    for i in df.index:
        acts = [a.strip() for a in act_series.at[i].split(activity_sep) if a.strip()]
        if not acts:
            continue

        pots_str = [p.strip() for p in pot_series.at[i].split(activity_sep)] if pot_series.at[i].strip() else []
        pots: list[float] = []
        for j, _ in enumerate(acts):
            try:
                pots.append(float(pots_str[j]))
            except Exception:
                pots.append(missing_potential_default)

        xi = float(x.at[i]) if pd.notna(x.at[i]) else np.nan
        yi = float(y.at[i]) if pd.notna(y.at[i]) else np.nan
        coord = (xi, yi)

        for act, pot in zip(acts, pots):
            ensure_keys(act)
            id_map[act].append(ids.at[i])
            coord_map[act].append(coord)
            pot_map[act].append(pot)

    # --- Convert to numpy arrays ---------------------------------------------
    identifiers: Dict[str, np.ndarray] = {}
    coordinates: Dict[str, np.ndarray] = {}
    potentials: Dict[str, np.ndarray] = {}

    for act in coord_map:
        identifiers[act] = np.asarray(id_map[act], dtype=object)
        coordinates[act] = np.asarray(coord_map[act], dtype=float).reshape(-1, 2)
        potentials[act] = np.asarray(pot_map[act], dtype=float)

        n = coordinates[act].shape[0]
        if not (len(identifiers[act]) == n == len(potentials[act])):
            raise RuntimeError(f"Inconsistent lengths for activity '{act}'.")

    return identifiers, coordinates, potentials


def summarize_locations_payload(
    payload: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Return a small summary table: per act_type counts and coordinate bounds."""
    ids, coords, pots = payload
    rows = []
    for act in sorted(coords.keys()):
        c = coords[act]
        p = pots.get(act, np.array([], dtype=float))
        if c.size == 0:
            rows.append({"act_type": act, "n": 0, "min_x": np.nan, "max_x": np.nan,
                         "min_y": np.nan, "max_y": np.nan, "sum_potential": float(np.nansum(p))})
            continue
        rows.append({
            "act_type": act,
            "n": int(c.shape[0]),
            "min_x": float(np.nanmin(c[:, 0])),
            "max_x": float(np.nanmax(c[:, 0])),
            "min_y": float(np.nanmin(c[:, 1])),
            "max_y": float(np.nanmax(c[:, 1])),
            "sum_potential": float(np.nansum(p)),
        })
    return pd.DataFrame(rows)

# ---- Exports ----


def build_name_lookup_from_dict(data: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, str]:
    """Return {identifier -> name} from nested dict input (last non-empty wins)."""
    schema = DictLocationSchema()
    out: Dict[str, str] = {}
    for _act, entries in data.items():
        for ident, entry in entries.items():
            name = entry.get(schema.name)
            if isinstance(name, str) and name.strip():
                out[str(ident)] = name.strip()
    return out


def build_name_lookup_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """Return {identifier -> name} from (Geo)DataFrame if a name column exists."""
    cols = LocationColumns
    if not (cols.name and cols.name in df.columns):
        return {}
    sub = df[[cols.id, cols.name]].dropna(subset=[cols.id]).astype({cols.id: str, cols.name: str})
    return dict(zip(sub[cols.id], sub[cols.name]))  # last occurrence wins

def segmented_plans_to_dataframe(
    plans: SegmentedPlans,
    *,
    include_extras: bool = False,
) -> pd.DataFrame:
    """
    Flatten SegmentedPlans to a DataFrame using `PlanColumns` naming.
    If `include_extras=True`, all keys from `extras` are added as columns.
    """
    rows: list[dict[str, Any]] = []
    cols = PlanColumns()

    for person_id, segments in plans.items():
        for segment in segments:
            for leg in segment:
                g = lambda k, d=None: getattr(leg, k, d)

                fx = fy = tx = ty = np.nan
                fr = g("from_location")
                to = g("to_location")
                if isinstance(fr, np.ndarray) and fr.shape == (2,):
                    fx, fy = float(fr[0]), float(fr[1])
                if isinstance(to, np.ndarray) and to.shape == (2,):
                    tx, ty = float(to[0]), float(to[1])

                row = {
                    cols.person_id: str(person_id),
                    cols.unique_leg_id: str(g("unique_leg_id")),
                    cols.to_act_type: g("to_act_type"),
                    cols.leg_distance_m: float(g("distance", 0.0) or 0.0),
                    cols.from_x: fx,
                    cols.from_y: fy,
                    cols.to_x: tx,
                    cols.to_y: ty,
                }
                if cols.mode:
                    row[cols.mode] = g("mode")
                if cols.to_act_is_main:
                    row[cols.to_act_is_main] = g("to_act_is_main_act")
                if cols.to_act_identifier:
                    row[cols.to_act_identifier] = g("to_act_identifier")

                if include_extras:
                    extras = g("extras") or {}
                    if isinstance(extras, dict):
                        row.update(extras)

                rows.append(row)

    return pd.DataFrame(rows)


def enrich_plans_df_with_names(
    plans_df: pd.DataFrame,
    *,
    name_lookup: Mapping[str, str],
    overwrite: bool = True
) -> pd.DataFrame:
    """
    Add a name column by mapping `plan_cols.to_act_identifier` via `name_lookup`.
    No-op if the identifier column is absent/None.
    """
    plan_cols = PlanColumns()
    id_col = plan_cols.to_act_identifier
    if not id_col or id_col not in plans_df.columns:
        return plans_df.copy()

    df = plans_df.copy()
    if overwrite or plan_cols.to_act_name not in df.columns:
        # map only non-null ids; coerce to str for stable dict lookup
        mapped = df[id_col].where(df[id_col].notna()).astype(str).map(name_lookup)
        df[plan_cols.to_act_name] = mapped
    return df

def enrich_plans_df_with_potentials(df, *, locations) -> "pd.DataFrame":
    """
    Add to_act_potential column by looking up potentials from Locations using
    (to_act_type, to_act_identifier). If required columns are missing or df is
    empty, return df unchanged.

    :param df: plans dataframe (expects 'to_act_type' and 'to_act_identifier')
    :param locations: Locations instance (provides identifiers and potentials)
    :return: new dataframe with 'to_act_potential' (float) added when possible
    """
    # Defensive checks
    if df is None or len(df) == 0:
        return df
    to_type_col = PlanColumns.to_act_type
    to_id_col = PlanColumns.to_act_identifier
    if to_type_col not in df.columns or to_id_col not in df.columns:
        return df

    # Prepare id->potential mapping per activity type
    type_maps = {}
    for act_type, ids in locations.identifiers.items():
        pots = locations.potentials.get(act_type)
        if pots is None:
            continue
        # Build a plain dict for mapping
        type_maps[act_type] = {id_: float(pot) for id_, pot in zip(ids, pots)}

    # Populate new column
    out = df.copy()
    col_name = "to_act_potential"
    out[col_name] = np.nan

    # Assign per-type to avoid custom apply over rows
    for act_type, id_to_pot in type_maps.items():
        mask = out[to_type_col] == act_type
        if mask.any():
            out.loc[mask, col_name] = out.loc[mask, to_id_col].map(id_to_pot)

    return out

def _is_xy(v: Any) -> bool:
    """True iff v is a finite ndarray of shape (2,)."""
    return isinstance(v, np.ndarray) and v.shape == (2,) and np.isfinite(v).all()

def segment_plans(
    plans: SegmentedPlans,
    *,
    drop_open_tail: bool = False,
    require_known_start: bool = False,
) -> SegmentedPlans:
    """
    Segment each person's legs into contiguous segments, closing a segment whenever
    the current leg has a known `to_location` (shape (2,)).

    Parameters
    ----------
    plans : SegmentedPlans
        frozendict[str, tuple[Leg, ...]]
    drop_open_tail : bool, default False
        If True, discard a trailing segment that doesn't end with a known `to_location`.
        If False, keep it as an "open" segment.
    require_known_start : bool, default False
        If True, only start a new segment once a leg has a known `from_location`.
        Legs before that are accumulated into the first segment but won't trigger a new
        segment boundary until a `to_location` appears.

    Returns
    -------
    SegmentedPlans
        frozendict[str, tuple[tuple[Leg, ...], ...]]
    """
    out: Dict[str, Tuple[Tuple[Any, ...], ...]] = {}

    for person_id, legs in plans.items():
        segments: List[Tuple[Any, ...]] = []
        current: List[Any] = []
        started = not require_known_start  # if not required, we can start immediately

        for leg in legs:
            if not started and _is_xy(getattr(leg, "from_location", None)):
                started = True

            if started:
                current.append(leg)

                # Close the segment if this leg provides a concrete end
                if _is_xy(getattr(leg, "to_location", None)):
                    segments.append(tuple(current))
                    current = []
            else:
                # We haven't "started" yet; still accumulate until we get a valid start or end.
                current.append(leg)
                if _is_xy(getattr(leg, "to_location", None)):
                    # We got an end even without a known start; still treat it as a segment.
                    segments.append(tuple(current))
                    current = []
                    # remain not-started until a leg has a known from_location

        # Handle tail
        if current:
            if not drop_open_tail:
                segments.append(tuple(current))
            # else: discard dangling open segment

        out[str(person_id)] = tuple(segments)

    return frozendict(out)

def validate_output_plans_df(
    df: pd.DataFrame,
) -> bool:
    """
    Validate a results DataFrame after placement. Logs findings; returns True/False.
    - Ensures required columns exist.
    - Ensures assigned coordinates (to_x/to_y) are present & finite.
    - Ensures distances are present & nonnegative (if check_distance_nonnegative=True).
    """
    ok = True
    cols = PlanColumns()
    # --- Column presence ------------------------------------------------------
    required = {
        cols.person_id, cols.unique_leg_id, cols.to_act_type, cols.leg_distance_m,
        cols.to_x, cols.to_y
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Validation failed: missing required columns: %s", missing)
        ok = False
    else:
        logger.info("All required columns present: %s", sorted(required))

    if not ok:
        return False  # can't continue checks reliably

    # --- Coerce to numeric views for checks (no mutation) ---------------------
    to_x = pd.to_numeric(df[cols.to_x], errors="coerce")
    to_y = pd.to_numeric(df[cols.to_y], errors="coerce")
    dist = pd.to_numeric(df[cols.leg_distance_m], errors="coerce")

    # --- XY assigned & finite -------------------------------------------------
    bad_xy_mask = ~(np.isfinite(to_x.values) & np.isfinite(to_y.values))
    n_bad_xy = int(bad_xy_mask.sum())
    if n_bad_xy > 0:
        logger.error("Validation failed: %d rows have missing/non-finite assigned coordinates (%s/%s).",
                     n_bad_xy, cols.to_x, cols.to_y)
        ok = False
    else:
        logger.info("All assigned coordinates are present and finite.")

    # --- Distance presence / sign --------------------------------------------
    n_dist_na = int((~np.isfinite(dist.values)).sum())
    if n_dist_na > 0:
        logger.error("Validation failed: %d rows have missing/non-finite %s.", n_dist_na, cols.leg_distance_m)
        ok = False
    n_dist_neg = int((dist.values < 0).sum())
    if n_dist_neg > 0:
        logger.error("Validation failed: %d rows have negative %s.", n_dist_neg, cols.leg_distance_m)
        ok = False
    if n_dist_na == 0 and n_dist_neg == 0:
        logger.info("All distances present and non-negative.")

    # --- Helpful summary ------------------------------------------------------
    try:
        n_persons = df[cols.person_id].nunique(dropna=True)
        n_rows = len(df)
        logger.info("Results summary: %d legs across %d persons.", n_rows, n_persons)
    except Exception:
        logger.error("Could not compute summary stats.")
        pass

    return ok
