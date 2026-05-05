"""Unit tests for narration.py — pluralization, families, and phrase composition."""

from narration import (
    SpeakItem,
    compose_phrase,
    label_family,
    pluralize,
)


class TestPluralize:
    def test_regular_s_suffix(self):
        assert pluralize("cup") == "cups"
        assert pluralize("dog") == "dogs"
        assert pluralize("chair") == "chairs"

    def test_sibilant_endings_take_es(self):
        assert pluralize("bus") == "buses"
        assert pluralize("box") == "boxes"
        assert pluralize("buzz") == "buzzes"
        assert pluralize("watch") == "watches"
        assert pluralize("dish") == "dishes"

    def test_consonant_y_becomes_ies(self):
        assert pluralize("cherry") == "cherries"
        assert pluralize("baby") == "babies"

    def test_vowel_y_keeps_y(self):
        assert pluralize("toy") == "toys"
        assert pluralize("key") == "keys"

    def test_fe_becomes_ves(self):
        assert pluralize("knife") == "knives"
        assert pluralize("life") == "lives"

    def test_f_becomes_ves(self):
        assert pluralize("leaf") == "leaves"
        assert pluralize("wolf") == "wolves"

    def test_irregular_plurals(self):
        assert pluralize("person") == "people"
        assert pluralize("mouse") == "mice"
        assert pluralize("sheep") == "sheep"
        assert pluralize("scissors") == "scissors"


class TestLabelFamily:
    def test_known_families(self):
        assert label_family("cup") == "container"
        assert label_family("wine glass") == "container"
        assert label_family("cell phone") == "device"
        assert label_family("laptop") == "device"

    def test_unknown_label_returns_self(self):
        assert label_family("dog") == "dog"
        assert label_family("CHAIR") == "chair"


class TestComposePhrase:
    def test_empty_returns_empty_string(self):
        assert compose_phrase([]) == ""

    def test_single_item_in_front(self):
        items = [SpeakItem("person", "in front of you", "close to you", count=1, score=1.0)]
        out = compose_phrase(items)
        # No "On your..." lead-in for the front zone; sentence is capitalized.
        assert out.startswith("A person close to you")
        assert out.endswith(".")

    def test_left_zone_uses_lead_in(self):
        items = [SpeakItem("chair", "to your left", "nearby", count=1, score=1.0)]
        out = compose_phrase(items)
        assert out.startswith("On your left,")
        assert "chair" in out

    def test_right_zone_uses_lead_in(self):
        items = [SpeakItem("dog", "to your right", "very close to you", count=1, score=1.0)]
        out = compose_phrase(items)
        assert out.startswith("On your right,")

    def test_pluralizes_count(self):
        items = [SpeakItem("person", "in front of you", "nearby", count=3, score=1.0)]
        out = compose_phrase(items)
        assert "3 people" in out

    def test_uses_an_for_vowel_initial(self):
        items = [SpeakItem("orange", "in front of you", "nearby", count=1, score=1.0)]
        out = compose_phrase(items)
        assert "an orange" in out.lower()

    def test_truncates_to_max_total_items(self):
        items = [
            SpeakItem("person", "in front of you", "close to you", count=1, score=5.0),
            SpeakItem("chair", "to your left", "nearby", count=1, score=4.0),
            SpeakItem("cup", "to your right", "nearby", count=1, score=3.0),
            SpeakItem("dog", "in front of you", "nearby", count=1, score=2.0),
            SpeakItem("bottle", "to your left", "nearby", count=1, score=1.0),
        ]
        # Default max_total_items is 3 — bottle and dog (lowest scores) should drop.
        out = compose_phrase(items)
        assert "bottle" not in out
        assert "dog" not in out
        assert "person" in out
        assert "chair" in out
        assert "cup" in out

    def test_front_zone_appears_before_sides(self):
        items = [
            SpeakItem("chair", "to your left", "nearby", count=1, score=10.0),
            SpeakItem("person", "in front of you", "close to you", count=1, score=1.0),
        ]
        out = compose_phrase(items)
        # Front zone has priority 0, so the front sentence comes first
        # even though the left item has a higher score.
        front_idx = out.index("person")
        left_idx = out.index("On your left,")
        assert front_idx < left_idx
