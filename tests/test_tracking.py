"""Unit tests for tracking.py — frame-to-frame stability counting with hold."""

from tracking import StableTracker


def test_object_becomes_stable_after_min_frames():
    t = StableTracker(min_stable_frames=3, hold_seconds=1.0)
    t.update_frame({"a"}, now=0.0)
    assert "a" not in t.stable_now()
    t.update_frame({"a"}, now=0.1)
    assert "a" not in t.stable_now()
    t.update_frame({"a"}, now=0.2)
    assert "a" in t.stable_now()


def test_disappearance_within_hold_does_not_reset():
    t = StableTracker(min_stable_frames=3, hold_seconds=1.0)
    t.update_frame({"a"}, now=0.0)
    t.update_frame({"a"}, now=0.1)
    t.update_frame({"a"}, now=0.2)
    assert "a" in t.stable_now()
    # Disappear briefly (within hold).
    t.update_frame(set(), now=0.5)
    assert "a" in t.stable_now()


def test_disappearance_past_hold_evicts():
    t = StableTracker(min_stable_frames=2, hold_seconds=0.5)
    t.update_frame({"a"}, now=0.0)
    t.update_frame({"a"}, now=0.1)
    assert "a" in t.stable_now()
    # Gone for longer than hold — should be evicted.
    t.update_frame(set(), now=1.0)
    assert "a" not in t.stable_now()


def test_multiple_keys_tracked_independently():
    t = StableTracker(min_stable_frames=2, hold_seconds=1.0)
    t.update_frame({"a", "b"}, now=0.0)
    t.update_frame({"a"}, now=0.1)
    t.update_frame({"a"}, now=0.2)
    stable = t.stable_now()
    assert "a" in stable
    # b only seen once, not yet stable.
    assert "b" not in stable


def test_stable_count_increments_only_when_present():
    t = StableTracker(min_stable_frames=3, hold_seconds=10.0)
    t.update_frame({"a"}, now=0.0)
    # Hold keeps a alive but does NOT increment the count.
    t.update_frame(set(), now=0.5)
    t.update_frame(set(), now=1.0)
    assert "a" not in t.stable_now()
    # Now seen again; needs two more presence frames to reach 3.
    t.update_frame({"a"}, now=1.5)
    t.update_frame({"a"}, now=2.0)
    assert "a" in t.stable_now()
