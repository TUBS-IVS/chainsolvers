from dataclasses import dataclass
from typing import Tuple, NamedTuple, Mapping, Any, Union, Sequence, Optional
from frozendict import frozendict
import numpy as np

class Leg(NamedTuple):
    unique_leg_id: str | None = None
    distance: float | None = None
    from_location: np.ndarray | None = None
    to_location: np.ndarray | None = None
    to_act_type: str | None = None
    to_act_identifier: str | None = None
    to_act_is_main_act: bool | None = None
    mode: str | None = None
    dep_time_s: int | None = None
    arr_time_s: int | None = None
    conn_leader_id: str | None = None
    conn_to_act: str | None = None
    conn_to_mode: str | None = None
    extras: Mapping[str, Any] | None = None  # Only for debugging

Segment = Tuple[Leg, ...]
SegmentedPlan = Tuple[Segment, ...]
SegmentedPlans = frozendict[str, SegmentedPlan]
Households = frozendict[str, SegmentedPlans]

@dataclass(frozen=True)
class PlanColumns:
    person_id: str | None = "unique_person_id"
    unique_leg_id: str | None = "unique_leg_id"
    to_act_type: str | None = "to_act_type"
    leg_distance_m: str | None = "distance_meters"
    from_x: str | None = "from_x"
    from_y: str | None = "from_y"
    to_x: str | None = "to_x"
    to_y: str | None = "to_y"

    mode: str | None = "mode"
    to_act_is_main: str | None = "to_act_is_main"
    to_act_identifier: str | None = "to_act_identifier"
    to_act_name: str | None = "to_act_name"

    household_id: str | None = "unique_household_id"
    dep_time_s: str | None = "dep_time_s"
    arr_time_s: str | None = "arr_time_s"

    # future
    conn_leader_id: str | None = None
    conn_to_act: str | None = None
    conn_to_mode: str | None = None



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
