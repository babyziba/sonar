"""Unit tests for distance.py — bbox-height and metric-depth distance bucketing."""

import math

from distance import (
    dist_with_deadband,
    distance_bucket,
    metric_bucket,
    metric_with_deadband,
)


class TestBboxHeightBucket:
    def test_very_close(self):
        assert distance_bucket(0.5) == "very close to you"
        assert distance_bucket(0.38) == "very close to you"

    def test_close(self):
        assert distance_bucket(0.3) == "close to you"
        assert distance_bucket(0.24) == "close to you"

    def test_nearby(self):
        assert distance_bucket(0.2) == "nearby"
        assert distance_bucket(0.14) == "nearby"

    def test_in_the_distance(self):
        assert distance_bucket(0.1) == "in the distance"
        assert distance_bucket(0.07) == "in the distance"

    def test_far_away(self):
        assert distance_bucket(0.05) == "far away"
        assert distance_bucket(0.0) == "far away"


class TestBboxDeadband:
    def test_no_prev_returns_raw(self):
        assert dist_with_deadband(0.5, prev_dist=None, deadband=0.02) == "very close to you"

    def test_sticks_near_boundary(self):
        # 0.39 is just past 0.38 boundary; deadband 0.02 should keep prev.
        assert dist_with_deadband(0.39, prev_dist="close to you", deadband=0.02) == "close to you"

    def test_flips_when_clear_of_boundary(self):
        # 0.5 is far past the 0.38 threshold — flip.
        assert dist_with_deadband(0.5, prev_dist="close to you", deadband=0.02) == "very close to you"


class TestMetricBucket:
    def test_very_close(self):
        assert metric_bucket(0.5) == "very close to you"
        assert metric_bucket(0.7) == "very close to you"

    def test_close(self):
        assert metric_bucket(1.0) == "close to you"
        assert metric_bucket(1.5) == "close to you"

    def test_nearby(self):
        assert metric_bucket(2.0) == "nearby"
        assert metric_bucket(3.0) == "nearby"

    def test_in_the_distance(self):
        assert metric_bucket(4.5) == "in the distance"
        assert metric_bucket(6.0) == "in the distance"

    def test_far_away(self):
        assert metric_bucket(10.0) == "far away"
        assert metric_bucket(50.0) == "far away"

    def test_invalid_meters_treated_as_far(self):
        # NaN, zero, or negative depth values should not crash and should
        # return the "far/unknown" label.
        assert metric_bucket(0.0) == "far away"
        assert metric_bucket(-1.0) == "far away"
        assert metric_bucket(float("nan")) == "far away"


class TestMetricDeadband:
    def test_no_prev_returns_raw(self):
        assert metric_with_deadband(1.0, prev_dist=None, deadband_m=0.3) == "close to you"

    def test_sticks_near_boundary(self):
        # 0.6 m is just under the 0.7 boundary; deadband 0.3 keeps prev.
        assert metric_with_deadband(0.6, prev_dist="close to you", deadband_m=0.3) == "close to you"

    def test_flips_when_clear_of_boundary(self):
        # 0.3 m is comfortably below the 0.7 threshold and outside deadband.
        assert metric_with_deadband(0.3, prev_dist="close to you", deadband_m=0.3) == "very close to you"

    def test_invalid_meters_returns_raw_classification(self):
        # When the metric reading is invalid we prefer the raw bucket
        # rather than holding stale state.
        result = metric_with_deadband(float("nan"), prev_dist="close to you", deadband_m=0.3)
        assert result == "far away"
