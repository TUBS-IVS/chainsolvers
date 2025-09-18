import numpy as np
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from frozendict import frozendict

from chainsolvers.locations import Locations
from chainsolvers.types import Segment


@dataclass(slots=True)
class CarlaConfig:
    number_of_branches: int = 30
    candidates_complex_case: int = 100
    candidates_two_leg_case: int = 40
    anchor_strategy: str = "lower_middle"   # {'lower_middle','upper_middle','start','end'}
    selection_strategy_complex_case: str = "top_n_spatial_downsample"
    selection_strategy_two_leg_case: str = "top_n"
    max_iterations_complex_case: int = 1000


class Carla:
    """
    Implementation of the CARLA solver for activity location assignment.

    This solver recursively places intermediate activity locations within trip
    segments of any length > 0. For each Segment, it builds up a solution space and selects the best option,
    as determined by the injected `scorer` (currently some combination of distance deviation and location potential).
    Visualization and statistics tracking are optionally supported.

    Parameters
    ----------
    locations : Locations
        Object providing candidate generation and feasibility checks.
    scorer : Any
        Object or function that assigns scores to candidate locations.
    selector : Any
        Object or function that selects candidate indices based on scores.
    rng : numpy.random.Generator
        Random generator used in candidate sampling and tie-breaking.
    visualizer : Any, optional
        If provided, nodes are added to visualize the recursive solution tree.
    progress : Any, optional
        A progress wrapper (e.g. tqdm); if None, a no-op shim is used.
    stats : Any, optional
        Statistics tracker.
    **params : dict
        Configuration overrides for `CarlaConfig`.

    Raises
    ------
    ValueError
        If a segment is empty, has invalid length, or if an unknown anchor
        strategy is specified.
    RuntimeError
        If no feasible candidates are found in the two-leg or recursive case,
        or if recursion reaches an impossible state.
    """

    needs_segmented_plans = True

    def __init__(
        self,
        locations: Locations,
        scorer: Any,
        selector: Any,
        rng: np.random.Generator,
        visualizer: Optional[Any] = None,
        progress: Optional[Any] = None,
        stats: Optional[Any] = None,
        **params: Any,
    ):
        # allow overrides via loose params
        config = CarlaConfig(**params)

        self.locations = locations
        self.rng = rng
        self.scorer = scorer
        self.selector = selector
        self.config = config
        self.visualizer = visualizer
        self.progress = progress
        self.stats = stats


    def _get_anchor_index(self, num_legs: int) -> int:
        """Anchor index by strategy."""
        if self.config.anchor_strategy == "lower_middle":
            return num_legs // 2 - 1
        if self.config.anchor_strategy == "upper_middle":
            return num_legs // 2
        if self.config.anchor_strategy == "start":
            return 0
        if self.config.anchor_strategy == "end":
            return num_legs - 1
        raise ValueError("Invalid anchor strategy.")


    def solve(
            self,
            *,
            plans,  # SegmentedPlans (frozendict[str, tuple[Segment, ...]])
    ):
        """
        - takes `plans` from the runner (not self.segmented_plans),
        - uses an optional `progress` shim instead of hardcoding tqdm,
        - returns a SegmentedPlans (frozendict) of placed segments.
        """
        progress_fn = self.progress or (lambda it, **k: it)

        if self.visualizer:
            assert len(plans) == 1, "Visualizer can only handle one person-tour at a time."
            person_id, segments = next(iter(plans.items()))
            first_leg = segments[0][0]
            home_location = first_leg.from_location
            root_node = self.visualizer.add_node(None, f"Home for {person_id}", location=home_location)
        else:
            root_node = None

        placed_dict = {}
        for person_id, segments in progress_fn(plans.items(), desc="Processing persons"):
            placed_dict[person_id] = []
            for segment in segments:
                placed_segment, _ = self.solve_segment(segment, root_node)
                if placed_segment is None:
                    raise RuntimeError("Reached impossible state (None segment).")
                placed_dict[person_id].append(placed_segment)

        # Freeze to SegmentedPlans: dict[str, tuple[PlacedSegment, ...]]
        return frozendict({pid: tuple(segs) for pid, segs in placed_dict.items()}) #freeze the inner lists into tuples

    def solve_segment(self, segment: Segment, parent_node=None) -> Tuple[Segment, float]:
        """Recursively solve a segment; returns (placed_segment, total_score)."""

        n = len(segment)
        if n == 0:
            raise ValueError("No legs in segment.")

        # base: one leg
        elif n == 1:
            assert segment[0].from_location.size > 0 and segment[0].to_location.size > 0, \
                "Start and end locations must be known."
            return segment, 0.0  # score computed at parent level

        # base: two legs
        elif n == 2:
            sel_ids, sel_coords, sel_pots, sel_scores = self.locations.get_best_circle_intersection_locations(
                selector=self.selector,
                scorer=self.scorer,
                act_type=segment[0].to_act_type,
                start_coord=segment[0].from_location,
                end_coord=segment[1].to_location,
                dist_start_to_act=segment[0].distance,
                dist_act_to_end=segment[1].distance,
                min_candidates=self.config.candidates_two_leg_case,
                sel_candidates=1,
                selection_strategy=self.config.selection_strategy_two_leg_case,
                rng=self.rng
            )

            if sel_ids.size == 0:
                raise RuntimeError("No feasible circle-intersection candidate for 2-leg case.")

            # take the first (contract keeps shape (1,) for sel_candidates=1)
            best_id = sel_ids.item()
            best_coord = sel_coords[0]  # (2,)
            best_score = sel_scores.item()

            updated_leg1 = segment[0]._replace(to_location=best_coord, to_act_identifier=best_id)
            updated_leg2 = segment[1]._replace(from_location=best_coord)

            if self.visualizer:
                best_pot = sel_pots.item()
                label = f"2-leg node: {best_id}, score: {best_score:.2f}, potential: {best_pot:.2f}"
                self.visualizer.add_node(parent_node, label, location=best_coord,
                                         metadata={"score": best_score, "potential": best_pot})

            return (updated_leg1, updated_leg2), best_score

        # recursive: 3+ legs
        elif n > 2:
            anchor_idx = self._get_anchor_index(len(segment))


            distances = np.array([leg.distance for leg in segment], dtype=float)
            d_start_to_act = distances[:anchor_idx + 1]
            d_act_to_end   = distances[anchor_idx + 1:]

            # generate + select candidates for anchor location
            sel_ids, sel_coords, sel_pots, sel_scores = self.locations.get_best_overlap_candidates(
                selector=self.selector,
                scorer=self.scorer,
                act_type=segment[anchor_idx].to_act_type,
                start_coord=segment[0].from_location,
                end_coord=segment[-1].to_location,
                distances_start_to_act=d_start_to_act,
                distances_act_to_end=d_act_to_end,
                min_candidates=self.config.candidates_complex_case,
                sel_candidates=self.config.number_of_branches,
                selection_strategy=self.config.selection_strategy_complex_case,
                max_iterations=self.config.max_iterations_complex_case,
                rng=self.rng
            )
            if sel_ids.size == 0:
                raise RuntimeError("No valid candidates selected in recursive case.")

            # branch on each candidate
            full_segs = []
            branch_scores = []
            for i in range(sel_ids.size):
                new_coord = sel_coords[i]
                new_id = sel_ids[i]

                if self.visualizer:
                    label = f"Candidate {new_id}: Score {sel_scores[i]:.2f}"
                    child_node = self.visualizer.add_node(parent_node, label, location=new_coord, metadata={"score": sel_scores[i]})
                else:
                    child_node = None

                # update anchor-adjacent legs
                updated_leg1 = segment[anchor_idx]._replace(to_location=new_coord, to_act_identifier=new_id)
                updated_leg2 = segment[anchor_idx + 1]._replace(from_location=new_coord)

                # split problem
                subsegment1 = (*segment[:anchor_idx], updated_leg1)
                subsegment2 = (updated_leg2, *segment[anchor_idx + 2:])

                # recurse
                located_seg1, score1 = self.solve_segment(subsegment1, child_node)
                located_seg2, score2 = self.solve_segment(subsegment2, child_node)
                if located_seg1 is None or located_seg2 is None:
                    raise RuntimeError("Reached impossible state (None subsegment).")

                total_score = float(score1) + float(score2) + float(sel_scores[i])
                branch_scores.append(total_score)
                full_segs.append((*located_seg1, *located_seg2))

            best_idx = int(np.argmax(branch_scores))
            return full_segs[best_idx], branch_scores[best_idx]

        else:
            raise ValueError("Invalid segment length.")
