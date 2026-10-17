"""Tests for src/fingerprint.py."""

from src.fingerprint import fingerprint


def test_same_parts_same_hash_and_shape():
    a = fingerprint("US text", "India text")
    assert a == fingerprint("US text", "India text")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_one_changed_character_changes_the_hash():
    assert fingerprint("US text", "India text") != fingerprint("US text", "India texT")


def test_part_boundaries_matter_and_none_is_empty():
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
    assert fingerprint(None, "x") == fingerprint("", "x")
