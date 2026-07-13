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
    parse_manual_lines,
    parse_manual_steps,
    validate_manual_lines,
    validate_manual_steps,
)
from hareskip.state import STATE


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


# --- reset_generation lifecycle: Manual Skip list survives ------------------
#
# reset_generation runs per sampling pass (process_before_every_sampling ->
# _begin_generation -> reset_generation), AFTER before_process has validated
# and stored the Manual Skip list. It must NOT wipe manual_skip_parsed, or the
# patcher reads an empty list and Manual Skip silently disables (regression
# fixed 2026-07-13). The raw manual_skip_steps text (a config-style field set
# from UI args) must likewise survive.


def test_reset_generation_preserves_manual_skip_parsed():
    STATE.manual_skip_parsed = [10, 12]
    STATE.reset_generation("test")
    assert STATE.manual_skip_parsed == [10, 12]


def test_reset_generation_preserves_manual_skip_steps():
    STATE.manual_skip_steps = "10, 12"
    STATE.reset_generation("test")
    assert STATE.manual_skip_steps == "10, 12"


# --- parse_manual_lines: normal cases ---------------------------------------
#
# Multiline input: one non-blank line = one job. Line splitting happens first
# (before per-line parsing), and each line is parsed with the single-line
# grammar. Returns (physical_line_no, steps) tuples with 1-based physical line
# numbers (blank lines counted, so the number matches the textbox).


def test_parse_lines_empty_string():
    assert parse_manual_lines("") == []


def test_parse_lines_whitespace_only():
    assert parse_manual_lines("   \n  \n\t") == []


def test_parse_lines_none():
    assert parse_manual_lines(None) == []


def test_parse_lines_single_line():
    assert parse_manual_lines("10, 12") == [(1, [10, 12])]


def test_parse_lines_multiple_lines():
    assert parse_manual_lines("10, 12\n15, 17, 19") == [
        (1, [10, 12]),
        (2, [15, 17, 19]),
    ]


def test_parse_lines_blank_lines_ignored():
    # Blank / whitespace-only lines produce no job but still advance the
    # physical line counter.
    assert parse_manual_lines("10\n\n  \n15") == [(1, [10]), (4, [15])]


def test_parse_lines_trailing_newline_no_extra_job():
    assert parse_manual_lines("10\n12\n") == [(1, [10]), (2, [12])]


def test_parse_lines_no_silent_merge_across_lines():
    # A trailing comma on one line must NOT merge into the next line's list
    # (the trailing-comma silent-merge hazard). Each line stays independent.
    assert parse_manual_lines("10, 12,\n15") == [(1, [10, 12]), (2, [15])]


def test_parse_lines_per_line_dedup_and_sort():
    assert parse_manual_lines("12, 10, 12\n30, 2, 15") == [
        (1, [10, 12]),
        (2, [2, 15, 30]),
    ]


# --- parse_manual_lines: error cases ----------------------------------------


def test_parse_lines_non_numeric_names_line():
    with pytest.raises(ManualSkipError) as excinfo:
        parse_manual_lines("10, 12\nfoo")
    assert "Line 2" in str(excinfo.value)


def test_parse_lines_error_uses_physical_line_number():
    # Blank lines are counted: the offending line is physical line 3.
    with pytest.raises(ManualSkipError) as excinfo:
        parse_manual_lines("10\n\nfoo")
    assert "Line 3" in str(excinfo.value)


# --- validate_manual_lines --------------------------------------------------


def test_validate_lines_all_valid():
    validate_manual_lines([(1, [2, 10]), (2, [30])], 30)


def test_validate_lines_empty_list_ok():
    validate_manual_lines([], 30)


def test_validate_lines_bad_line_named():
    with pytest.raises(ManualSkipError) as excinfo:
        validate_manual_lines([(1, [2, 10]), (5, [31])], 30)
    assert "Line 5" in str(excinfo.value)


def test_validate_lines_boundary_ok():
    validate_manual_lines([(1, [30])], 30)


def test_validate_lines_boundary_over_raises():
    with pytest.raises(ManualSkipError) as excinfo:
        validate_manual_lines([(1, [31])], 30)
    assert "Line 1" in str(excinfo.value)


def test_validate_lines_step_one_detected():
    with pytest.raises(ManualSkipError) as excinfo:
        validate_manual_lines([(1, [2]), (2, [1])], 30)
    assert "Line 2" in str(excinfo.value)
