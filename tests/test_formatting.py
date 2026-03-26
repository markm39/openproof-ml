"""Tests for prompt formatting -- the contract between training and inference."""

from openproof_ml.data.formatting import format_prompt, format_training_example, parse_tactic


def test_format_prompt():
    assert format_prompt("a b : Nat\n|- a + b = b + a") == "a b : Nat\n|- a + b = b + a:::"


def test_format_training_example():
    ex = format_training_example("a b : Nat\n|- a + b = b + a", "omega")
    assert ex["prompt"] == "a b : Nat\n|- a + b = b + a:::"
    assert ex["completion"] == "omega"


def test_parse_tactic_basic():
    assert parse_tactic("simp") == "simp"
    assert parse_tactic("  omega  ") == "omega"
    assert parse_tactic("ring\n  -- done") == "ring"
    assert parse_tactic("omega:::") == "omega"


def test_parse_tactic_banned():
    assert parse_tactic("sorry") is None
    assert parse_tactic("admit") is None
    assert parse_tactic("native_decide") is None
    assert parse_tactic("sorry; ring") is None
    assert parse_tactic("") is None


def test_parse_tactic_banned_substrings():
    assert parse_tactic("rcases h with ?_ | ?_") is None
    assert parse_tactic("rcases h with h1 | h2") == "rcases h with h1 | h2"
