"""Calibrated per-sampler ResRefine recommendations for the "Load recommended
settings" button.

Each row maps an exact (sampler, scheduler) pair to the ResRefine values that
were found to help in the calibration campaign. There is no per-step-count
column yet ("skip_band" was considered and dropped as premature — see
below); ``lookup_recommendation`` still takes ``steps`` so its signature
stays stable once such a column is added, and so the button's status text
can show the steps the user is currently running.

Calibration source (第5段階〜検証実験V1, 2026-09-05〜2026-09-14, see
EXPERIMENT-LOG and the HF dataset ``Rootport/HareSkip-calibration``):

- Euler + Beta -> Linear extrapolation (strength 0.30, EMA smoothing 0.10).
  Fixed-point calibration at 15 skips / 30 steps: +22% anchor, p=0.015.
  Confirmed 6/6 improved on unseen V1 prompts.
- Euler + Simple -> Linear extrapolation (strength 0.30, EMA smoothing 0.00).
  Same campaign, average-baseline common point. NOTE: this fixed point's
  significance was borderline (p=0.055) — weaker evidence than Euler+Beta,
  included because it was still the best available default for this pair.

``use_prediction_after_progress=0.0`` and ``apply_prediction_from_skip=2``
are included in every row (not left as "don't touch") because the
calibration campaign fixed both of these ResRefine arguments for the whole
campaign (STAGE5-SCAN1-HANDOFF §1: "other ResRefine arguments held at
defaults"). Since this button claims to apply "recommended settings", it
should reproduce the calibrated condition exactly, not just the formula and
its own strength/smoothing.

No "skip_band" column: each row is a plain dict, so a future column (e.g. a
skip-count band, or a Shift value) can be added by adding a key to the
existing rows plus any new rows that need it — no schema migration. This
was a deliberate YAGNI call; don't pre-add columns "just in case".

How to add a new row:

1. Add a dict to ``RECOMMENDED_SETTINGS`` with the same keys as the existing
   rows (``sampler``, ``scheduler``, ``formula``, ``prediction_strength``,
   ``slope_ema_smoothing``, ``use_prediction_after_progress``,
   ``apply_prediction_from_skip``). Import the formula constant from
   ``hareskip.state`` (``RESREFINE_FORMULA_LINEAR`` /
   ``RESREFINE_FORMULA_TAYLOR2`` / ``RESREFINE_FORMULA_REUSE``) — do not use a
   string literal.
2. Document the calibration source for the new row in this module's
   docstring (campaign name/date range and the relevant statistic).
3. Add a case to ``tests/test_recommended_settings.py`` covering the new
   (sampler, scheduler) lookup.
"""

from __future__ import annotations

from .state import RESREFINE_FORMULA_LINEAR

# formula constants come from hareskip.state (not a string literal here) so a
# typo or a future rename of a RESREFINE_FORMULA_* constant is caught by
# import, not silently mismatched against the dropdown's actual choices.
# state.py is torch-independent and does not import this module, so there is
# no import cycle.
RECOMMENDED_SETTINGS: list[dict] = [
    {
        "sampler": "Euler",
        "scheduler": "Beta",
        "formula": RESREFINE_FORMULA_LINEAR,
        "prediction_strength": 0.30,
        "slope_ema_smoothing": 0.10,
        "use_prediction_after_progress": 0.0,
        "apply_prediction_from_skip": 2,
    },
    {
        "sampler": "Euler",
        "scheduler": "Simple",
        "formula": RESREFINE_FORMULA_LINEAR,
        "prediction_strength": 0.30,
        "slope_ema_smoothing": 0.00,
        "use_prediction_after_progress": 0.0,
        "apply_prediction_from_skip": 2,
    },
]


def lookup_recommendation(sampler: str, scheduler: str, steps: int) -> dict | None:
    """Look up the calibrated row for an exact (sampler, scheduler) pair.

    Matching is exact-string, case-sensitive (so e.g. "Euler a" never
    matches the "Euler" row). Returns ``None`` when there is no calibrated
    entry — callers should fall back to Reuse (residual only), the safe
    default. ``steps`` is not used for matching yet (no per-step-count band
    exists); it is accepted so the signature is stable once one is added and
    so callers can echo the current step count in status text.
    """
    for row in RECOMMENDED_SETTINGS:
        if row["sampler"] == sampler and row["scheduler"] == scheduler:
            return row
    return None
