
from typing import Tuple, NamedTuple, Mapping, Any, Union, Sequence
from frozendict import frozendict
import numpy as np

class Leg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str | None = None
    # Currently unused, for future use in added algos.
    mode: str | None = None
    to_act_is_main_act: bool | None = None
    extras: Mapping[str, Any] | None = None

Segment = Tuple[Leg, ...]  # A segment of a plan (immutable tuple of legs)
SegmentedPlan = Tuple[Segment, ...]  # A full plan split into segments
SegmentedPlans = frozendict[str, SegmentedPlan]  # Many agents' plans (person_id -> SegmentedPlan)
Households = frozendict[str, SegmentedPlans]

ArrayLike = Union[np.ndarray, Sequence[float]]
