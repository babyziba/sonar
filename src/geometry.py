"""Horizontal positioning of detections (left / center / right)."""

from __future__ import annotations

_LEFT_BOUNDARY = 0.33
_RIGHT_BOUNDARY = 0.66

ZONE_LEFT = "to your left"
ZONE_FRONT = "in front of you"
ZONE_RIGHT = "to your right"


def bearing_zone(cx_norm: float) -> str:
    """Map a normalized x-center (0..1) to a verbal zone."""
    if cx_norm < _LEFT_BOUNDARY:
        return ZONE_LEFT
    if cx_norm < _RIGHT_BOUNDARY:
        return ZONE_FRONT
    return ZONE_RIGHT


def zone_with_deadband(cx_norm: float, prev_zone: str | None, deadband: float) -> str:
    """Sticky variant of bearing_zone: keeps prev_zone if within deadband of a boundary."""
    z = bearing_zone(cx_norm)
    if not prev_zone:
        return z
    near_boundary = (
        abs(cx_norm - _LEFT_BOUNDARY) <= deadband
        or abs(cx_norm - _RIGHT_BOUNDARY) <= deadband
    )
    if z != prev_zone and near_boundary:
        return prev_zone
    return z
