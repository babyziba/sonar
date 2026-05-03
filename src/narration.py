"""Phrase composition for the talking-objects scene narration."""

from __future__ import annotations

from dataclasses import dataclass, field


# Map closely-related labels to a shared family for grouping. Used by callers
# that want to dedupe (e.g. "cup" + "wine glass" -> "container").
CLASS_FAMILIES: dict[str, str] = {
    # drinkware / containers
    "bottle": "container",
    "cup": "container",
    "can": "container",
    "wine glass": "container",
    "glass": "container",
    "mug": "container",
    # handheld electronics
    "cell phone": "device",
    "remote": "device",
    "laptop": "device",
    "keyboard": "device",
    "mouse": "device",
    "tv": "device",
}


def label_family(label: str) -> str:
    return CLASS_FAMILIES.get(label.lower(), label.lower())


_IRREGULAR_PLURALS = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "scissors": "scissors",
    "skis": "skis",
}


def pluralize(label: str) -> str:
    lower = label.lower()
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return label[:-1] + "ies"
    if lower.endswith("fe"):
        return label[:-2] + "ves"
    if lower.endswith("f"):
        return label[:-1] + "ves"
    return label + "s"


@dataclass
class SpeakItem:
    label: str
    zone: str
    distance: str
    count: int
    score: float
    track_ids: list[int | None] = field(default_factory=list)

    @property
    def family(self) -> str:
        return label_family(self.label)


# Zone lead-ins. None means "no prefix" — used for the forward zone, where
# direction is implicit (it's what the user is facing).
_ZONE_LEAD_IN: dict[str, str | None] = {
    "in front of you": None,
    "to your left": "On your left,",
    "to your right": "On your right,",
}

_ZONE_PRIORITY = {
    "in front of you": 0,
    "to your left": 1,
    "to your right": 2,
}


def _join_items(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _item_phrase(it: SpeakItem) -> str:
    if it.count > 1:
        return f"{it.count} {pluralize(it.label)} {it.distance}"
    article = "an" if it.label[:1].lower() in "aeiou" else "a"
    return f"{article} {it.label} {it.distance}"


def compose_phrase(
    speak_items: list[SpeakItem],
    max_total_items: int = 3,
    max_items_per_zone: int = 2,
    max_zones: int = 3,
) -> str:
    """Build a single natural-language description of the scene.

    Truncated by importance so phrases stay short enough that TTS doesn't lag
    behind the scene. Order: highest-score items first, capped by per-zone and
    total limits.
    """
    if not speak_items:
        return ""

    top = sorted(speak_items, key=lambda it: it.score, reverse=True)[:max_total_items]

    zones: dict[str, list[SpeakItem]] = {}
    for it in top:
        zones.setdefault(it.zone, []).append(it)
    for z in zones:
        zones[z].sort(key=lambda it: it.score, reverse=True)
        zones[z] = zones[z][:max_items_per_zone]

    zone_order = sorted(
        zones.keys(),
        key=lambda z: (_ZONE_PRIORITY.get(z, 99), -zones[z][0].score),
    )

    sentences: list[str] = []
    for z in zone_order[:max_zones]:
        joined = _join_items([_item_phrase(it) for it in zones[z]])
        lead = _ZONE_LEAD_IN.get(z)
        if lead:
            sentences.append(f"{lead} {joined}.")
        else:
            sentences.append(f"{joined[0].upper()}{joined[1:]}.")

    return " ".join(sentences)
