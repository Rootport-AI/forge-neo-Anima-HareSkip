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

NOTE: ``script.py`` / ``state.py`` do NOT consume these constants yet. This
module just defines them. The UI-restructure commit will (a) rewire the two
files to reference these constants and (b) extend the list to 29 entries by
inserting ``hareskip_mode`` / ``hareskip_aggressiveness`` /
``hareskip_skip_seed_offset``. Until then these values mirror the CURRENT
26-argument state of the repository.
"""

# --- Skip-strategy mode identifiers -----------------------------------------

MODE_HARESKIP = "HareSkip"
MODE_TEACACHE = "TeaCache"

# Exclusive skip-strategy modes offered by the extension (Radio choices).
HARESKIP_MODES = [MODE_HARESKIP, MODE_TEACACHE]


# --- UI argument synchronisation --------------------------------------------
#
# Canonical positional order of the arguments returned by ``Script.ui()`` and
# consumed by ``RuntimeState.apply_options``. Keep this list, the ``ui()``
# return list, and the ``apply_options`` signature 1:1 in order and count.
#
# This currently reflects the 26-argument repo state (pre UI restructure).
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
]

EXPECTED_UI_ARG_COUNT = len(UI_ARG_ORDER)
