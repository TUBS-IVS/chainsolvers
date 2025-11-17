
from typing import Tuple, NamedTuple, Mapping, Any, Union, Sequence
from frozendict import frozendict
import numpy as np

class Leg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray | None
    to_location: np.ndarray | None
    distance: float
    to_act_type: str
    to_act_identifier: str | None = None
    to_act_is_main_act: bool | None = None
    mode: str | None = None
    dep_time_s: int | None = None
    arr_time_s: int | None = None
    conn_leader_id: str | None = None
    conn_to_act: str | None = None
    extras: Mapping[str, Any] | None = None # Only for debugging

Segment = Tuple[Leg, ...]  # A segment of a plan (immutable tuple of legs)
SegmentedPlan = Tuple[Segment, ...]  # A full plan split into segments
SegmentedPlans = frozendict[str, SegmentedPlan]  # Many agents' plans (person_id -> SegmentedPlan)
Households = frozendict[str, SegmentedPlans]

ArrayLike = Union[np.ndarray, Sequence[float]]
