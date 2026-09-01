"""STATE -> skip_pattern wiring tests (pure stdlib + pytest).

``RuntimeState.hareskip_pattern_kwargs()`` is the single place where UI
settings become ``generate_skip_pattern`` keyword arguments. It is pure and
Forge-independent, so this file exercises the whole wiring without importing
``patcher`` (which needs torch) or ``script`` (which needs gradio).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hareskip.constants as constants
import hareskip.state as state
from hareskip import skip_pattern as sp
from hareskip.probability_models import DEFAULT_PROBABILITY_MODEL

from test_arg_sync import _dummy_apply_options_args, _idx


# The 30 t_now values of the reference schedule (Beta 30-step, Shift 3),
# copied to 9 decimals from experiment-HareSkip/v02-proposal/r2/
# derivation_r2.json ["trajectory"]. Embedded here so the test needs no
# experiment data on disk. Test fixture only — not a generation input.
REFERENCE_T_NOW = [
    0.999405273, 0.997992039, 0.993243277, 0.986301303, 0.978079259,
    0.967707634, 0.955337703, 0.941548824, 0.925727427, 0.907582879,
    0.887755156, 0.865482271, 0.840949059, 0.813186824, 0.783667684,
    0.75074929, 0.713836491, 0.674917459, 0.63093853, 0.584249079,
    0.533505082, 0.479591817, 0.420863301, 0.360182405, 0.298076898,
    0.233108103, 0.170212761, 0.113678373, 0.060460649, 0.02071006,
]


def test_reference_schedule_shape():
    assert len(REFERENCE_T_NOW) == 30
    assert all(0.0 < t < 1.0 for t in REFERENCE_T_NOW)
    # Strictly descending, so z increases with progress.
    assert all(
        REFERENCE_T_NOW[i] > REFERENCE_T_NOW[i + 1] for i in range(29)
    )


def test_pattern_kwargs_defaults():
    s = state.RuntimeState()
    kwargs = s.hareskip_pattern_kwargs()
    assert kwargs["probability_model"] == DEFAULT_PROBABILITY_MODEL
    assert kwargs["skip_window"] == (0.05, 0.95)
    assert kwargs["zone_boundaries"] == (-4.0, 0.0)
    assert kwargs["zone_max_streak"] == {"danger": 1, "middle": 2, "safe": 3}
    # 0 in the state means "disabled", which is None in the generator API.
    assert kwargs["exact_target"] is None


def test_pattern_kwargs_exact_target_zero_is_none():
    s = state.RuntimeState()
    s.hareskip_exact_target = 0
    assert s.hareskip_pattern_kwargs()["exact_target"] is None
    s.hareskip_exact_target = 15
    assert s.hareskip_pattern_kwargs()["exact_target"] == 15


def test_pattern_kwargs_are_valid_generate_skip_pattern_arguments():
    s = state.RuntimeState()
    pat = sp.generate_skip_pattern(
        REFERENCE_T_NOW, 0.55, 12345, **s.hareskip_pattern_kwargs()
    )
    assert pat.num_steps == 30
    assert pat.probability_model == DEFAULT_PROBABILITY_MODEL
    assert pat.skip_window == (0.05, 0.95)


def test_apply_options_roundtrip_into_pattern_kwargs():
    args = _dummy_apply_options_args()
    args[_idx("hareskip_streak_danger")] = 2
    args[_idx("hareskip_streak_middle")] = 4
    args[_idx("hareskip_streak_safe")] = 6
    args[_idx("hareskip_exact_target")] = 15
    args[_idx("hareskip_probability_model")] = "sigmoid_band_v0.1"
    args[_idx("hareskip_window_start")] = 0.1
    args[_idx("hareskip_window_end")] = 0.9
    args[_idx("hareskip_zone_low")] = -3.0
    args[_idx("hareskip_zone_high")] = 1.0
    s = state.RuntimeState()
    s.apply_options(*args)
    assert s.hareskip_pattern_kwargs() == {
        "probability_model": "sigmoid_band_v0.1",
        "skip_window": (0.1, 0.9),
        "zone_boundaries": (-3.0, 1.0),
        "zone_max_streak": {"danger": 2, "middle": 4, "safe": 6},
        "exact_target": 15,
    }


def test_exact_target_reaches_requested_skip_count():
    """Both new knobs reach the generator through hareskip_pattern_kwargs().

    Under the default streak limits (1/2/3) the reachable skip counts at
    a = 0.55 on the reference schedule top out at 13, so the relaxed limits
    are what make 15 attainable — i.e. this asserts the streak wiring as much
    as the exact-target wiring. Impossible targets never raise (see
    test_skip_pattern.test_exact_target_impossible_no_raise); this test is
    about a reachable one.
    """
    args = _dummy_apply_options_args()
    args[_idx("hareskip_exact_target")] = 15
    args[_idx("hareskip_aggressiveness")] = 0.55
    args[_idx("hareskip_streak_danger")] = 2
    args[_idx("hareskip_streak_middle")] = 4
    args[_idx("hareskip_streak_safe")] = 6
    s = state.RuntimeState()
    s.apply_options(*args)
    pat = sp.generate_skip_pattern(
        REFERENCE_T_NOW, 0.55, 12345, **s.hareskip_pattern_kwargs()
    )
    assert pat.skip_count == 15


def test_default_streaks_cannot_reach_15_at_a055():
    # Documents why the test above relaxes the limits: with 1/2/3 the target
    # is out of reach and generate_skip_pattern returns the nearest attempt
    # rather than raising.
    s = state.RuntimeState()
    s.hareskip_exact_target = 15
    pat = sp.generate_skip_pattern(
        REFERENCE_T_NOW, 0.55, 12345, **s.hareskip_pattern_kwargs()
    )
    assert pat.skip_count < 15


def test_arg_order_tail_is_the_v02_append():
    # Guards the "append at the end" decision that keeps the pre-v0.2
    # positions (and the experiment tooling's hard-coded indices) intact.
    assert constants.UI_ARG_ORDER[constants.LEGACY_MIN_UI_ARG_COUNT :] == [
        "hareskip_streak_danger",
        "hareskip_streak_middle",
        "hareskip_streak_safe",
        "hareskip_exact_target",
        "hareskip_probability_model",
    ]
    assert constants.EXPECTED_UI_ARG_COUNT == 39
