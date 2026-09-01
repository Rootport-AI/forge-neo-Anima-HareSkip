"""Frozen reference noise schedule used only for the UI skip-count estimate.

Source of truth: ``experiment-HareSkip/v02-proposal/r2/derivation_r2.json``
key ``["trajectory"]`` — the ``t_now`` values of the ER SDE-Beta schedule at
30 steps with Shift 3.0, as used for the monotone-r2 recalibration. Frozen
2026-09-01; the values below are transcribed to 9 decimals.

DISPLAY ONLY. This table must never be fed into the sampling path: the real
per-run schedule comes from the sampler via ``patcher.py``. It exists so the
UI can answer "roughly how many steps would this setting skip?" before a run
starts, on a fixed, reproducible yardstick. Any number rendered from it is
therefore labelled with ``REFERENCE_SCHEDULE_LABEL`` so the user knows it is
a reference figure, not a prediction for their own step count/shift.

Pure stdlib (no imports at all) — no Forge, gradio, numpy or torch.
"""

from __future__ import annotations

# Human-readable name of the schedule the estimate is computed against.
REFERENCE_SCHEDULE_LABEL = "Beta 30-step (Shift 3)"

# t_now per step, strictly descending, all within (0, 1).
REFERENCE_T_NOW = (
    0.999405273, 0.997992039, 0.993243277, 0.986301303, 0.978079259,
    0.967707634, 0.955337703, 0.941548824, 0.925727427, 0.907582879,
    0.887755156, 0.865482271, 0.840949059, 0.813186824, 0.783667684,
    0.75074929, 0.713836491, 0.674917459, 0.63093853, 0.584249079,
    0.533505082, 0.479591817, 0.420863301, 0.360182405, 0.298076898,
    0.233108103, 0.170212761, 0.113678373, 0.060460649, 0.02071006,
)

REFERENCE_NUM_STEPS = len(REFERENCE_T_NOW)
