"""Distance bucketing.

Two backends share the same verbal labels:
- Bbox-height heuristic (`distance_bucket` / `dist_with_deadband`): cheap
  fallback, no extra model. Inaccurate across object classes with very
  different real-world sizes.
- Metric depth (`metric_bucket` / `metric_with_deadband`): consumes meters
  from Depth Anything v2. Default path when depth estimation is enabled.

Labels are postpositional phrases that complete a sentence like
"a {label} {phrase}" — e.g. "a bed close to you".
"""

from __future__ import annotations

# Bbox-height thresholds (h_norm = bbox_h / frame_h). Closest → farthest.
_BUCKETS: list[tuple[float, str]] = [
    (0.38, "very close to you"),
    (0.24, "close to you"),
    (0.14, "nearby"),
    (0.07, "in the distance"),
]
_FAR = "far away"
_BOUNDARIES = [t for t, _ in _BUCKETS]

# Metric thresholds (meters). First bucket whose ceiling >= meters wins.
_METRIC_BUCKETS: list[tuple[float, str]] = [
    (0.7, "very close to you"),
    (1.5, "close to you"),
    (3.0, "nearby"),
    (6.0, "in the distance"),
]
_METRIC_FAR = "far away"
_METRIC_BOUNDARIES = [m for m, _ in _METRIC_BUCKETS]


def distance_bucket(h_norm: float) -> str:
    """Map normalized bbox height to a verbal distance bucket."""
    for threshold, label in _BUCKETS:
        if h_norm >= threshold:
            return label
    return _FAR


def dist_with_deadband(h_norm: float, prev_dist: str | None, deadband: float) -> str:
    """Sticky variant of distance_bucket: keep prev_dist near any threshold."""
    d = distance_bucket(h_norm)
    if not prev_dist:
        return d
    if d != prev_dist and any(abs(h_norm - b) <= deadband for b in _BOUNDARIES):
        return prev_dist
    return d


def metric_bucket(meters: float) -> str:
    """Bucket a metric depth value (meters) to a verbal label."""
    if not (meters > 0):  # NaN, negative, or zero -> treat as far/unknown
        return _METRIC_FAR
    for max_m, label in _METRIC_BUCKETS:
        if meters <= max_m:
            return label
    return _METRIC_FAR


def metric_with_deadband(meters: float, prev_dist: str | None, deadband_m: float) -> str:
    """Sticky variant of metric_bucket: keep prev_dist near any threshold."""
    d = metric_bucket(meters)
    if not prev_dist:
        return d
    if not (meters > 0):
        return d
    if d != prev_dist and any(abs(meters - b) <= deadband_m for b in _METRIC_BOUNDARIES):
        return prev_dist
    return d
