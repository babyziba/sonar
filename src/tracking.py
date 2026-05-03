"""Frame-to-frame stability tracking for detections."""

from __future__ import annotations

from typing import Hashable, Iterable


class StableTracker:
    """Counts consecutive-frame appearances per key, with a time-based hold
    so that brief disappearances don't reset the count."""

    def __init__(self, min_stable_frames: int = 3, hold_seconds: float = 0.75) -> None:
        self.min_stable = min_stable_frames
        self.hold = hold_seconds
        self.counts: dict[Hashable, int] = {}
        self.last_seen: dict[Hashable, float] = {}

    def update_frame(self, present_keys: Iterable[Hashable], now: float) -> None:
        present_set = set(present_keys)
        for k in present_set:
            self.counts[k] = self.counts.get(k, 0) + 1
            self.last_seen[k] = now
        for k, last in list(self.last_seen.items()):
            if k not in present_set and (now - last) > self.hold:
                self.counts.pop(k, None)
                self.last_seen.pop(k, None)

    def stable_now(self) -> set[Hashable]:
        return {k for k, c in self.counts.items() if c >= self.min_stable}
