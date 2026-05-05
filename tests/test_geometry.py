"""Unit tests for geometry.py — left/center/right zone classification with hysteresis."""

from geometry import (
    ZONE_FRONT,
    ZONE_LEFT,
    ZONE_RIGHT,
    bearing_zone,
    zone_with_deadband,
)


class TestBearingZone:
    def test_far_left(self):
        assert bearing_zone(0.0) == ZONE_LEFT
        assert bearing_zone(0.1) == ZONE_LEFT
        assert bearing_zone(0.32) == ZONE_LEFT

    def test_center(self):
        assert bearing_zone(0.33) == ZONE_FRONT
        assert bearing_zone(0.5) == ZONE_FRONT
        assert bearing_zone(0.65) == ZONE_FRONT

    def test_far_right(self):
        assert bearing_zone(0.66) == ZONE_RIGHT
        assert bearing_zone(0.9) == ZONE_RIGHT
        assert bearing_zone(1.0) == ZONE_RIGHT


class TestZoneWithDeadband:
    def test_no_prev_returns_raw_classification(self):
        assert zone_with_deadband(0.5, prev_zone=None, deadband=0.05) == ZONE_FRONT
        assert zone_with_deadband(0.1, prev_zone=None, deadband=0.05) == ZONE_LEFT

    def test_sticky_when_near_boundary(self):
        # cx=0.34 is just past the 0.33 left/front boundary, would normally flip
        # to FRONT — but deadband 0.05 keeps prev_zone LEFT.
        assert zone_with_deadband(0.34, prev_zone=ZONE_LEFT, deadband=0.05) == ZONE_LEFT

    def test_flips_when_clear_of_boundary(self):
        # 0.5 is comfortably past the deadband around 0.33 — must flip.
        assert zone_with_deadband(0.5, prev_zone=ZONE_LEFT, deadband=0.05) == ZONE_FRONT

    def test_zero_deadband_means_no_hysteresis(self):
        # With deadband=0 the function should behave like bearing_zone.
        assert zone_with_deadband(0.34, prev_zone=ZONE_LEFT, deadband=0.0) == ZONE_FRONT
        assert zone_with_deadband(0.67, prev_zone=ZONE_FRONT, deadband=0.0) == ZONE_RIGHT

    def test_no_flip_keeps_prev_zone(self):
        # If raw classification matches prev_zone, return it directly even near boundary.
        assert zone_with_deadband(0.32, prev_zone=ZONE_LEFT, deadband=0.05) == ZONE_LEFT
