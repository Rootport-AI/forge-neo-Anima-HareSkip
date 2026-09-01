"""Forge-independent constants for HareSkip.

Pure stdlib module: no Forge / gradio / torch imports so it can be imported
in unit tests and pure-Python tooling.

3-point sync rule (UI argument plumbing)
========================================
The UI arguments produced by ``hareskip/script.py`` ``ui()`` must stay in
lockstep with three things:

  1. the return list of ``Script.ui()`` (order and count),
  2. the positional signature of ``RuntimeState.apply_options`` in
     ``hareskip/state.py`` (order and count),
  3. ``UI_ARG_ORDER`` / ``EXPECTED_UI_ARG_COUNT`` below.

If any of the three drifts out of sync, generation options are silently
applied to the wrong slots. ``UI_ARG_ORDER`` is the canonical list of
argument names, in positional order, and ``EXPECTED_UI_ARG_COUNT`` is its
length. A static test asserts ``len(UI_ARG_ORDER) == EXPECTED_UI_ARG_COUNT``
and (once wiring lands) that ``apply_options`` has a matching signature.

``script.py`` sets ``_EXPECTED_UI_ARG_COUNT`` from ``EXPECTED_UI_ARG_COUNT``
and ``state.py`` ``apply_options`` mirrors ``UI_ARG_ORDER`` positionally. The
HareSkip stochastic-mode arguments (``hareskip_mode`` /
``hareskip_aggressiveness`` / ``hareskip_skip_seed_offset``) and the
skip-window / zone-boundary scalars (``hareskip_window_start`` /
``hareskip_window_end`` / ``hareskip_zone_low`` / ``hareskip_zone_high``) are
appended after the original 26, and the Manual Skip mode text field
(``manual_skip_steps``) is appended last. The five HareSkip v0.2 arguments
(``hareskip_streak_danger`` / ``middle`` / ``safe``, ``hareskip_exact_target``,
``hareskip_probability_model``) are appended after those, so every earlier
position is untouched (39 total; 2026-09-01: appended at the end, so the
existing 34 positions are unchanged).
"""

# --- Skip-strategy mode identifiers -----------------------------------------

MODE_HARESKIP = "HareSkip"
MODE_TEACACHE = "TeaCache"
MODE_MANUAL = "Manual Skip"

# Exclusive skip-strategy modes offered by the extension (Radio choices).
HARESKIP_MODES = [MODE_HARESKIP, MODE_TEACACHE, MODE_MANUAL]


# --- UI argument synchronisation --------------------------------------------
#
# Canonical positional order of the arguments returned by ``Script.ui()`` and
# consumed by ``RuntimeState.apply_options``. Keep this list, the ``ui()``
# return list, and the ``apply_options`` signature 1:1 in order and count.
#
# The original 26 arguments keep their positions; the three HareSkip
# stochastic-mode arguments and the four skip-window / zone-boundary scalars
# are appended next, the Manual Skip mode text field follows, and the five
# HareSkip v0.2 arguments (zone streak limits, exact target, probability
# model) are appended last (39 total; 2026-09-01: appended at the end, so the
# existing 34 positions are unchanged).
UI_ARG_ORDER = [
    "enabled",
    "debug_log_enabled",
    "mode",
    "print_timing_log",
    "verbose_diagnose_log",
    "dump_resrefine_residual",
    "tea_preset",
    "tea_threshold",
    "tea_start_percent",
    "tea_end_percent",
    "resrefine_formula",
    "resrefine_use_prediction_after_progress",
    "resrefine_apply_prediction_from_skip",
    "resrefine_prediction_strength",
    "resrefine_taylor2_curve_strength",
    "resrefine_slope_ema_smoothing",
    "resrefine_curve_ema_smoothing",
    "resrefine_cache_device",
    "tea_coefficient_profile",
    "tea_max_skip_streak",
    "tea_force_full_interval",
    "hareskip_dry_run",
    "hareskip_verbose_trace",
    "auto_teacache_enabled",
    "auto_teacache_csv",
    "capture_calibration_pairs",
    "hareskip_mode",
    "hareskip_aggressiveness",
    "hareskip_skip_seed_offset",
    "hareskip_window_start",
    "hareskip_window_end",
    "hareskip_zone_low",
    "hareskip_zone_high",
    "manual_skip_steps",
    "hareskip_streak_danger",
    "hareskip_streak_middle",
    "hareskip_streak_safe",
    "hareskip_exact_target",
    "hareskip_probability_model",
]

# Smallest argument count still accepted by ``script._apply_ui_args``: a
# payload from before the 2026-09-01 five-argument append. Anything in
# [LEGACY_MIN_UI_ARG_COUNT, EXPECTED_UI_ARG_COUNT) is padded with
# ``apply_options`` defaults (which reproduce the pre-v0.2 behaviour) and
# warned about once; anything shorter falls back to shared settings.
LEGACY_MIN_UI_ARG_COUNT = 34

EXPECTED_UI_ARG_COUNT = len(UI_ARG_ORDER)
