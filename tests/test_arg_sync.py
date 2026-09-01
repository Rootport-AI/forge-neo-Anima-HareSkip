"""Static UI-argument synchronisation tests (no gradio / Forge required).

Guards the 3-point sync rule documented in ``hareskip/constants.py``:

  1. ``Script.ui()`` return list (order + count),
  2. ``RuntimeState.apply_options`` positional signature (order + count),
  3. ``constants.UI_ARG_ORDER`` / ``EXPECTED_UI_ARG_COUNT``.

These tests parse ``script.py`` as text so they never import gradio.
"""

from __future__ import annotations

import inspect
import os
import re

import hareskip.constants as constants
import hareskip.state as state

from conftest import import_hareskip_script


def _idx(name: str) -> int:
    """Positional index of a UI argument, by name.

    Keeps the normalization tests readable and immune to future appends to
    ``UI_ARG_ORDER`` (which would silently shift hard-coded indices).
    """
    return constants.UI_ARG_ORDER.index(name)


_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hareskip",
    "script.py",
)


def test_expected_count_matches_order_length():
    assert constants.EXPECTED_UI_ARG_COUNT == len(constants.UI_ARG_ORDER)


def test_no_duplicate_arg_names():
    order = constants.UI_ARG_ORDER
    assert len(order) == len(set(order)), "UI_ARG_ORDER has duplicates"


def test_apply_options_signature_matches_order():
    sig = inspect.signature(state.RuntimeState.apply_options)
    params = [name for name in sig.parameters if name != "self"]
    assert params == constants.UI_ARG_ORDER, (
        "apply_options positional params must match UI_ARG_ORDER exactly.\n"
        f"apply_options: {params}\n"
        f"UI_ARG_ORDER: {constants.UI_ARG_ORDER}"
    )


def _extract_ui_return_names() -> list[str]:
    """Parse the ``return [ ... ]`` list at the end of ``ui()`` from source.

    Robust to whitespace and trailing comments: finds the ``ui()`` method,
    takes the final top-level ``return [`` block, and collects bare identifier
    entries. Bracket-depth aware so nested brackets would not confuse it (there
    are none, but this keeps the parse deterministic).
    """
    with open(_SCRIPT_PATH, "r", encoding="utf-8") as handle:
        source = handle.read()

    ui_start = source.index("def ui(")
    # Stop at the next method definition so we only look inside ui().
    next_def = re.search(r"\n    def ", source[ui_start + 1 :])
    ui_end = ui_start + 1 + next_def.start() if next_def else len(source)
    ui_body = source[ui_start:ui_end]

    ret_idx = ui_body.rindex("return [")
    bracket_start = ui_body.index("[", ret_idx)
    depth = 0
    end = None
    for i in range(bracket_start, len(ui_body)):
        ch = ui_body[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "could not find end of ui() return list"

    inner = ui_body[bracket_start + 1 : end]
    names: list[str] = []
    for raw in inner.split(","):
        token = raw.split("#", 1)[0].strip()
        if token:
            names.append(token)
    return names


def _dummy_apply_options_args() -> list:
    """A valid positional arg list for apply_options, in UI_ARG_ORDER order.

    The window / zone / streak / exact-target entries are overridden by the
    normalization tests; sensible defaults are used here.
    """
    return [
        True,   # enabled
        False,  # debug_log_enabled
        "",     # mode
        True,   # print_timing_log
        False,  # verbose_diagnose_log
        False,  # dump_resrefine_residual
        "Custom",  # tea_preset
        0.07,   # tea_threshold
        0.48,   # tea_start_percent
        0.76,   # tea_end_percent
        "Reuse (residual only)",  # resrefine_formula
        0.0,    # resrefine_use_prediction_after_progress
        2,      # resrefine_apply_prediction_from_skip
        0.5,    # resrefine_prediction_strength
        0.25,   # resrefine_taylor2_curve_strength
        0.0,    # resrefine_slope_ema_smoothing
        0.0,    # resrefine_curve_ema_smoothing
        "cuda", # resrefine_cache_device
        state.TEA_PROFILE_DEFAULT,  # tea_coefficient_profile
        0,      # tea_max_skip_streak
        0,      # tea_force_full_interval
        False,  # hareskip_dry_run
        False,  # hareskip_verbose_trace
        False,  # auto_teacache_enabled
        "",     # auto_teacache_csv
        False,  # capture_calibration_pairs
        "HareSkip",  # hareskip_mode
        0.5,    # hareskip_aggressiveness
        0,      # hareskip_skip_seed_offset
        0.05,   # hareskip_window_start
        0.95,   # hareskip_window_end
        -4.0,   # hareskip_zone_low
        0.0,    # hareskip_zone_high
        "",     # manual_skip_steps
        1,      # hareskip_streak_danger
        2,      # hareskip_streak_middle
        3,      # hareskip_streak_safe
        0,      # hareskip_exact_target
        "monotone_saturate_v0.1",  # hareskip_probability_model
    ]


def test_apply_options_normalizes_window_clamp_and_swap():
    args = _dummy_apply_options_args()
    # Reversed and out-of-range window -> clamped to [0, 1] and swapped.
    args[_idx("hareskip_window_start")] = 1.5   # over max
    args[_idx("hareskip_window_end")] = -0.5    # under min
    s = state.RuntimeState()
    s.apply_options(*args)
    # 1.5 -> 1.0, -0.5 -> 0.0, then swapped so start <= end.
    assert s.hareskip_window_start == 0.0
    assert s.hareskip_window_end == 1.0


def test_apply_options_normalizes_zone_clamp_and_swap():
    args = _dummy_apply_options_args()
    # Reversed and out-of-range zone boundaries -> clamped to [-8, 8], swapped.
    args[_idx("hareskip_zone_low")] = 5.0    # hareskip_zone_low
    args[_idx("hareskip_zone_high")] = -9.0  # under min
    s = state.RuntimeState()
    s.apply_options(*args)
    # -9.0 -> -8.0, then swapped so low <= high.
    assert s.hareskip_zone_low == -8.0
    assert s.hareskip_zone_high == 5.0


def test_apply_options_keeps_valid_ranges():
    args = _dummy_apply_options_args()
    args[_idx("hareskip_window_start")], args[_idx("hareskip_window_end")] = 0.1, 0.8
    args[_idx("hareskip_zone_low")], args[_idx("hareskip_zone_high")] = -3.0, 2.0
    s = state.RuntimeState()
    s.apply_options(*args)
    assert (s.hareskip_window_start, s.hareskip_window_end) == (0.1, 0.8)
    assert (s.hareskip_zone_low, s.hareskip_zone_high) == (-3.0, 2.0)


def test_dummy_args_length_matches_expected():
    # Guards the dummy list against future arg-count drift.
    assert len(_dummy_apply_options_args()) == constants.EXPECTED_UI_ARG_COUNT


def test_ui_return_list_matches_order():
    names = _extract_ui_return_names()
    assert all(re.fullmatch(r"[A-Za-z_]\w*", n) for n in names), (
        f"unexpected non-identifier tokens in ui() return list: {names}"
    )
    assert names == constants.UI_ARG_ORDER, (
        "ui() return list must match UI_ARG_ORDER exactly.\n"
        f"ui() return: {names}\n"
        f"UI_ARG_ORDER: {constants.UI_ARG_ORDER}"
    )


# --- HareSkip v0.2 argument normalization -----------------------------------


def test_streak_args_clamped_and_defaulted():
    args = _dummy_apply_options_args()
    args[_idx("hareskip_streak_danger")] = -5    # below min -> 0
    args[_idx("hareskip_streak_middle")] = 99    # above max -> 10
    args[_idx("hareskip_streak_safe")] = "abc"   # non-numeric -> default 3
    s = state.RuntimeState()
    s.apply_options(*args)
    assert s.hareskip_streak_danger == 0
    assert s.hareskip_streak_middle == 10
    assert s.hareskip_streak_safe == 3
    assert s.hareskip_zone_max_streak() == {
        "danger": 0,
        "middle": 10,
        "safe": 3,
    }


def test_streak_defaults_survive_roundtrip():
    s = state.RuntimeState()
    s.apply_options(*_dummy_apply_options_args())
    assert s.hareskip_zone_max_streak() == {"danger": 1, "middle": 2, "safe": 3}


def test_exact_target_normalization():
    for raw, expected in ((-3, 0), ("x", 0), (15, 15), (10 ** 6, 999)):
        args = _dummy_apply_options_args()
        args[_idx("hareskip_exact_target")] = raw
        s = state.RuntimeState()
        s.apply_options(*args)
        assert s.hareskip_exact_target == expected, raw


def test_unknown_probability_model_falls_back_to_default():
    from hareskip.probability_models import (
        DEFAULT_PROBABILITY_MODEL,
        PROBABILITY_MODELS,
    )

    args = _dummy_apply_options_args()
    args[_idx("hareskip_probability_model")] = "no_such_model_v9"
    s = state.RuntimeState()
    s.apply_options(*args)
    assert s.hareskip_probability_model == DEFAULT_PROBABILITY_MODEL

    # Every registered name is preserved verbatim.
    for name in PROBABILITY_MODELS:
        args = _dummy_apply_options_args()
        args[_idx("hareskip_probability_model")] = name
        s = state.RuntimeState()
        s.apply_options(*args)
        assert s.hareskip_probability_model == name


# --- backward-compatible short payloads -------------------------------------


def _reset_script_state(script):
    script.STATE.__dict__.update(state.RuntimeState().__dict__)
    script._ui_arg_count_warned = False


def test_legacy_payload_is_padded_with_defaults():
    script = import_hareskip_script()
    _reset_script_state(script)
    legacy = _dummy_apply_options_args()[: constants.LEGACY_MIN_UI_ARG_COUNT]
    assert len(legacy) == 34
    script._apply_ui_args(legacy)
    # Accepted (not the refresh_settings fallback): the payload's own values
    # landed, and the five appended arguments took their defaults.
    assert script.STATE.hareskip_aggressiveness == 0.5
    assert script.STATE.hareskip_window_start == 0.05
    assert script.STATE.hareskip_zone_max_streak() == {
        "danger": 1,
        "middle": 2,
        "safe": 3,
    }
    assert script.STATE.hareskip_exact_target == 0
    from hareskip.probability_models import DEFAULT_PROBABILITY_MODEL

    assert script.STATE.hareskip_probability_model == DEFAULT_PROBABILITY_MODEL
    # The padding warning is emitted exactly once per process.
    assert script._ui_arg_count_warned is True


def test_payload_shorter_than_legacy_minimum_falls_back(monkeypatch):
    script = import_hareskip_script()
    _reset_script_state(script)
    called = []
    monkeypatch.setattr(
        script.STATE, "refresh_settings", lambda: called.append(True)
    )
    too_short = _dummy_apply_options_args()[: constants.LEGACY_MIN_UI_ARG_COUNT - 1]
    assert len(too_short) == 33
    script._apply_ui_args(too_short)
    assert called == [True]
