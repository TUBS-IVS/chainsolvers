"""
Tests for connection detection in households.
"""
import pytest
import numpy as np
import pandas as pd
from chainsolvers import helpers
from chainsolvers.types import Leg
from chainsolvers import io
from frozendict import frozendict


class TestConnectionMatchingHelpers:
    """Test the individual helper functions for connection matching."""

    def test_check_distance_match_exact(self):
        """Test distance matching with exact match."""
        leg_a = Leg(distance=100.0)
        leg_b = Leg(distance=100.0)
        assert helpers.check_distance_match(leg_a, leg_b, tolerance_m=0.0)

    def test_check_distance_match_within_tolerance(self):
        """Test distance matching within tolerance."""
        leg_a = Leg(distance=100.0)
        leg_b = Leg(distance=150.0)
        assert helpers.check_distance_match(leg_a, leg_b, tolerance_m=50.0)
        assert not helpers.check_distance_match(leg_a, leg_b, tolerance_m=49.0)

    def test_check_distance_match_none_values(self):
        """Test distance matching with None values."""
        leg_a = Leg(distance=None)
        leg_b = Leg(distance=100.0)
        assert not helpers.check_distance_match(leg_a, leg_b, tolerance_m=100.0)

    def test_check_time_match_exact(self):
        """Test time matching with exact match."""
        leg_a = Leg(dep_time_s=1000, arr_time_s=2000)
        leg_b = Leg(dep_time_s=1000, arr_time_s=2000)
        assert helpers.check_time_match(leg_a, leg_b, dep_tolerance_s=0, arr_tolerance_s=0)

    def test_check_time_match_within_tolerance(self):
        """Test time matching within tolerance."""
        leg_a = Leg(dep_time_s=1000, arr_time_s=2000)
        leg_b = Leg(dep_time_s=1100, arr_time_s=2050)
        assert helpers.check_time_match(leg_a, leg_b, dep_tolerance_s=100, arr_tolerance_s=50)
        assert not helpers.check_time_match(leg_a, leg_b, dep_tolerance_s=50, arr_tolerance_s=50)

    def test_check_time_match_none_values(self):
        """Test time matching with None values."""
        leg_a = Leg(dep_time_s=None, arr_time_s=2000)
        leg_b = Leg(dep_time_s=1000, arr_time_s=2000)
        assert not helpers.check_time_match(leg_a, leg_b, dep_tolerance_s=100, arr_tolerance_s=100)

    def test_check_mode_match_exact(self):
        """Test mode matching with exact match."""
        leg_a = Leg(mode="car")
        leg_b = Leg(mode="car")
        assert helpers.check_mode_match(leg_a, leg_b, compatible_modes={})

    def test_check_mode_match_compatible(self):
        """Test mode matching with compatible modes."""
        leg_a = Leg(mode="car")
        leg_b = Leg(mode="car_passenger")
        compat = {"car": {"car", "car_passenger"}}
        assert helpers.check_mode_match(leg_a, leg_b, compatible_modes=compat)

    def test_check_mode_match_symmetric_compatibility(self):
        """Test that mode compatibility works symmetrically."""
        leg_a = Leg(mode="car_passenger")
        leg_b = Leg(mode="car")
        compat = {"car": {"car", "car_passenger"}}
        # Should work both ways due to reverse check
        assert helpers.check_mode_match(leg_a, leg_b, compatible_modes=compat)

    def test_check_mode_match_incompatible(self):
        """Test mode matching with incompatible modes."""
        leg_a = Leg(mode="car")
        leg_b = Leg(mode="bike")
        assert not helpers.check_mode_match(leg_a, leg_b, compatible_modes={})

    def test_check_mode_match_none_values(self):
        """Test mode matching with None values."""
        leg_a = Leg(mode=None)
        leg_b = Leg(mode="car")
        assert not helpers.check_mode_match(leg_a, leg_b, compatible_modes={})

    def test_check_activity_match_exact(self):
        """Test activity matching with exact match."""
        leg_a = Leg(to_act_type="work")
        leg_b = Leg(to_act_type="work")
        assert helpers.check_activity_match(leg_a, leg_b, compatible_activities={})

    def test_check_activity_match_compatible(self):
        """Test activity matching with compatible activities."""
        leg_a = Leg(to_act_type="work")
        leg_b = Leg(to_act_type="business")
        compat = {"work": {"work", "business"}}
        assert helpers.check_activity_match(leg_a, leg_b, compatible_activities=compat)

    def test_check_activity_match_symmetric_compatibility(self):
        """Test that activity compatibility works symmetrically."""
        leg_a = Leg(to_act_type="business")
        leg_b = Leg(to_act_type="work")
        compat = {"work": {"work", "business"}}
        # Should work both ways due to reverse check
        assert helpers.check_activity_match(leg_a, leg_b, compatible_activities=compat)

    def test_check_activity_match_incompatible(self):
        """Test activity matching with incompatible activities."""
        leg_a = Leg(to_act_type="work")
        leg_b = Leg(to_act_type="leisure")
        assert not helpers.check_activity_match(leg_a, leg_b, compatible_activities={})

    def test_check_activity_match_none_values(self):
        """Test activity matching with None values."""
        leg_a = Leg(to_act_type=None)
        leg_b = Leg(to_act_type="work")
        assert not helpers.check_activity_match(leg_a, leg_b, compatible_activities={})


class TestFindConnectedLegsInHouseholds:
    """Test the find_connected_legs_in_households function."""

    def test_single_person_household_no_connections(self):
        """Test that single-person households have no connections."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="leg1", distance=100.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(households)

        # Single person household should be unchanged
        assert result["hh1"]["p1"][0].connected_legs is None

    def test_two_person_household_with_matching_legs(self):
        """Test connection detection between two people with matching legs."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1050.0, mode="car",
                        dep_time_s=1050, arr_time_s=2050, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
            compatible_modes={},
            compatible_activities={},
        )

        # Both legs should be connected to each other
        p1_leg = result["hh1"]["p1"][0]
        p2_leg = result["hh1"]["p2"][0]

        assert p1_leg.connected_legs == ("p2_leg1",)
        assert p2_leg.connected_legs == ("p1_leg1",)

    def test_two_person_household_no_match_distance(self):
        """Test that legs don't connect if distance is too different."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=2000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
        )

        # No connection due to distance mismatch
        assert result["hh1"]["p1"][0].connected_legs is None
        assert result["hh1"]["p2"][0].connected_legs is None

    def test_two_person_household_no_match_time(self):
        """Test that legs don't connect if times are too different."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="car",
                        dep_time_s=2000, arr_time_s=3000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
        )

        # No connection due to time mismatch
        assert result["hh1"]["p1"][0].connected_legs is None
        assert result["hh1"]["p2"][0].connected_legs is None

    def test_two_person_household_no_match_mode(self):
        """Test that legs don't connect if modes are incompatible."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="bike",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
        )

        # No connection due to mode mismatch
        assert result["hh1"]["p1"][0].connected_legs is None
        assert result["hh1"]["p2"][0].connected_legs is None

    def test_two_person_household_no_match_activity(self):
        """Test that legs don't connect if activities are incompatible."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="leisure"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
        )

        # No connection due to activity mismatch
        assert result["hh1"]["p1"][0].connected_legs is None
        assert result["hh1"]["p2"][0].connected_legs is None

    def test_multiple_connections_sorted(self):
        """Test that multiple connections are sorted."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p3": (
                    Leg(unique_leg_id="p3_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
        )

        # p1 should be connected to both p2 and p3 (sorted)
        p1_leg = result["hh1"]["p1"][0]
        assert p1_leg.connected_legs == ("p2_leg1", "p3_leg1")

    def test_compatible_modes_used(self):
        """Test that compatible modes configuration is used."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="car_passenger",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
            compatible_modes={"car": {"car", "car_passenger"}},
        )

        # Should be connected due to compatible modes
        assert result["hh1"]["p1"][0].connected_legs == ("p2_leg1",)
        assert result["hh1"]["p2"][0].connected_legs == ("p1_leg1",)

    def test_compatible_activities_used(self):
        """Test that compatible activities configuration is used."""
        households = frozendict({
            "hh1": frozendict({
                "p1": (
                    Leg(unique_leg_id="p1_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="work"),
                ),
                "p2": (
                    Leg(unique_leg_id="p2_leg1", distance=1000.0, mode="car",
                        dep_time_s=1000, arr_time_s=2000, to_act_type="business"),
                )
            })
        })

        result = io.find_connected_legs_in_households(
            households,
            distance_tolerance_m=100.0,
            dep_tolerance_s=100,
            arr_tolerance_s=100,
            compatible_activities={"work": {"work", "business"}},
        )

        # Should be connected due to compatible activities
        assert result["hh1"]["p1"][0].connected_legs == ("p2_leg1",)
        assert result["hh1"]["p2"][0].connected_legs == ("p1_leg1",)
