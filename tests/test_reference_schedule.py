"""Reference schedule table + Monte Carlo calibration anchors (pure stdlib).

``hareskip/reference_schedule.py`` is a frozen, display-only copy of the ER
SDE-Beta 30-step (Shift 3) t_now trajectory. These tests pin its shape, keep
it byte-identical to the copy embedded in ``test_hare_wiring.py``, and assert
the r2 calibration anchors (a=0.2/0.55/0.9 -> 5/10/15 skips) that the whole
recalibration campaign concluded with. If a future model change moves those
anchors, this file is where it shows up.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hareskip.reference_schedule import (  # noqa: E402
    REFERENCE_NUM_STEPS,
    REFERENCE_SCHEDULE_LABEL,
    REFERENCE_T_NOW,
)
from hareskip.skip_pattern import generate_skip_pattern  # noqa: E402

from test_hare_wiring import REFERENCE_T_NOW as EMBEDDED_T_NOW  # noqa: E402


# Same deterministic seed ladder script.py uses for the UI estimate.
_SEED_BASE = 1
_SEED_STRIDE = 7919


def _mean_skip_count(model_name, aggressiveness, draws=256):
    total = 0
    for i in range(draws):
        pattern = generate_skip_pattern(
            REFERENCE_T_NOW,
            aggressiveness,
            _SEED_BASE + i * _SEED_STRIDE,
            probability_model=model_name,
        )
        total += pattern.skip_count
    return total / float(draws)


def test_schedule_shape():
    assert REFERENCE_NUM_STEPS == 30
    assert len(REFERENCE_T_NOW) == 30
    assert all(0.0 < t < 1.0 for t in REFERENCE_T_NOW)


def test_schedule_strictly_descending():
    assert all(
        REFERENCE_T_NOW[i] > REFERENCE_T_NOW[i + 1]
        for i in range(REFERENCE_NUM_STEPS - 1)
    )


def test_schedule_matches_wiring_test_copy():
    # The wiring test embeds its own copy so it needs no module import; the
    # two must never drift.
    assert list(REFERENCE_T_NOW) == list(EMBEDDED_T_NOW)


def test_label_mentions_the_schedule():
    assert REFERENCE_SCHEDULE_LABEL == "Beta 30-step (Shift 3)"


@pytest.mark.parametrize(
    "model_name",
    ["monotone_saturate_v0.1", "sigmoid_band_v0.2"],
)
@pytest.mark.parametrize(
    "aggressiveness, expected",
    [(0.2, 5.0), (0.55, 10.0), (0.9, 15.0)],
)
def test_calibration_anchors(model_name, aggressiveness, expected):
    mean = _mean_skip_count(model_name, aggressiveness)
    assert mean == pytest.approx(expected, abs=0.5)


def test_mean_skip_count_is_deterministic():
    first = _mean_skip_count("monotone_saturate_v0.1", 0.55, draws=64)
    second = _mean_skip_count("monotone_saturate_v0.1", 0.55, draws=64)
    assert first == second
