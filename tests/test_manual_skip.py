"""Unit tests for hareskip.manual_skip (pure stdlib, no gradio / Forge).

Covers the two-function contract:
  * parse_manual_steps  — syntax: split, trim, drop empty tokens, dedupe, sort;
    raise ManualSkipError on non-numeric tokens; "" / whitespace -> [].
  * validate_manual_steps — semantics against num_steps: reject step < 1,
    step == 1, and step > num_steps, each with a non-empty message.
"""

from __future__ import annotations

import pytest

from hareskip.manual_skip import (
    ManualSkipError,
    parse_manual_steps,
    validate_manual_steps,
)


# --- parse_manual_steps: normal cases ---------------------------------------


def test_parse_empty_string():
    assert parse_manual_steps("") == []


def test_parse_whitespace_only():
    assert parse_manual_steps("   ") == []


def test_parse_none():
    assert parse_manual_steps(None) == []


def test_parse_trailing_comma():
    assert parse_manual_steps("10, 12,") == [10, 12]


def test_parse_empty_tokens_between_commas():
    assert parse_manual_steps("10, , 12") == [10, 12]


def test_parse_whitespace_around_tokens():
    assert parse_manual_steps("  10 ,   12 ") == [10, 12]


def test_parse_duplicates_deduped():
    assert parse_manual_steps("12, 10, 12, 10") == [10, 12]


def test_parse_sorted_ascending():
    # Output ordering is deterministic (sorted) regardless of input order.
    assert parse_manual_steps("30, 2, 15") == [2, 15, 30]


def test_parse_single_value():
    assert parse_manual_steps("7") == [7]


# --- parse_manual_steps: error cases ----------------------------------------


def test_parse_non_numeric_token_raises():
    with pytest.raises(ManualSkipError):
        parse_manual_steps("10, foo, 12")


def test_parse_float_token_raises():
    # int() rejects "10.5" — floats are not valid step numbers.
    with pytest.raises(ManualSkipError):
        parse_manual_steps("10.5")


def test_parse_error_message_non_empty():
    with pytest.raises(ManualSkipError) as excinfo:
        parse_manual_steps("abc")
    assert str(excinfo.value).strip() != ""


# --- validate_manual_steps: normal cases ------------------------------------


def test_validate_ok_within_range():
    # No exception for valid in-range steps.
    validate_manual_steps([2, 10, 30], 30)


def test_validate_empty_list_ok():
    validate_manual_steps([], 30)


def test_validate_boundary_last_step_ok():
    validate_manual_steps([30], 30)


# --- validate_manual_steps: error cases -------------------------------------


def test_validate_step_zero_raises():
    with pytest.raises(ManualSkipError):
        validate_manual_steps([0], 30)


def test_validate_step_negative_raises():
    with pytest.raises(ManualSkipError):
        validate_manual_steps([-3], 30)


def test_validate_step_one_raises():
    # Step 1 is physically unskippable (no prior residual).
    with pytest.raises(ManualSkipError):
        validate_manual_steps([1], 30)


def test_validate_step_over_num_steps_raises():
    with pytest.raises(ManualSkipError):
        validate_manual_steps([31], 30)


def test_validate_error_messages_non_empty():
    for bad in ([0], [1], [31]):
        with pytest.raises(ManualSkipError) as excinfo:
            validate_manual_steps(bad, 30)
        assert str(excinfo.value).strip() != ""


def test_validate_num_steps_tracks_argument():
    # Same step passes at higher num_steps, fails at lower — no hard-coded cap.
    validate_manual_steps([40], 40)
    with pytest.raises(ManualSkipError):
        validate_manual_steps([40], 20)
