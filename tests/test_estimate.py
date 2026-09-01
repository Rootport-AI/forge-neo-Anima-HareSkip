"""UI skip-count estimate rendering (``hareskip/script.py``).

``_format_hareskip_estimate`` is a gradio callback: it must never raise, must
be deterministic (the same settings always show the same number, no flicker),
and must react to every control that feeds ``generate_skip_pattern``. It is
imported through ``conftest.import_hareskip_script()`` because ``script.py``
needs gradio/Forge stubs.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import import_hareskip_script  # noqa: E402

script = import_hareskip_script()


def _count(text):
    """Pull the integer N out of a ``**≈N skips / 30 steps**`` headline."""
    match = re.search(r"≈(\d+) skips", text)
    assert match is not None, f"no skip count in: {text!r}"
    return int(match.group(1))


def test_returns_a_string_with_defaults():
    rendered = script._format_hareskip_estimate()
    assert isinstance(rendered, str)
    assert rendered != "`skip estimate unavailable`"


def test_default_aggressiveness_estimates_ten():
    # a=0.55 -> 10 skips is the r2 calibration anchor.
    assert "≈10 skips" in script._format_hareskip_estimate(0.55)


def test_headline_names_the_reference_schedule():
    rendered = script._format_hareskip_estimate(0.55)
    assert "Beta 30-step (Shift 3)" in rendered
    assert "/ 30 steps" in rendered


def test_second_line_shows_model_parameters():
    rendered = script._format_hareskip_estimate(0.55)
    second = rendered.split("\n\n")[1]
    assert second.startswith("`model=monotone_saturate_v0.1")
    assert "p_cap=" in second
    assert "z_enter=" in second
    assert "tau_enter=" in second
    # The default monotone model has no falling edge.
    assert "z_exit=" not in second


def test_band_model_adds_z_exit():
    rendered = script._format_hareskip_estimate(
        0.55, model_name="sigmoid_band_v0.2"
    )
    assert "model=sigmoid_band_v0.2" in rendered
    assert "z_exit=" in rendered


def test_monotone_non_decreasing_in_aggressiveness():
    counts = [
        _count(script._format_hareskip_estimate(a / 10.0)) for a in range(11)
    ]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), counts
    assert counts[0] < counts[-1]


def test_deterministic_for_identical_arguments():
    first = script._format_hareskip_estimate(0.42, 0.1, 0.9, -3.0, 1.0, 2, 3, 4)
    second = script._format_hareskip_estimate(0.42, 0.1, 0.9, -3.0, 1.0, 2, 3, 4)
    assert first == second


@pytest.mark.parametrize(
    "args",
    [
        (None,),
        ("abc",),
        (float("nan"),),
        (0.55, None, None),
        (0.55, "abc", "def"),
        (0.55, 0.9, 0.1),  # reversed window
        (0.55, 0.05, 0.95, 4.0, -4.0),  # reversed zones
        (0.55, 0.05, 0.95, -4.0, 0.0, -5, -1, "x"),  # negative/garbage streaks
        (0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "bogus"),  # unknown model
        (0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, None, "y"),  # garbage target
        (0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "monotone_saturate_v0.1", -3),
    ],
)
def test_bad_input_never_raises(args):
    rendered = script._format_hareskip_estimate(*args)
    assert isinstance(rendered, str)
    assert rendered != "`skip estimate unavailable`"
    _count(rendered)


def test_unknown_model_falls_back_to_default():
    rendered = script._format_hareskip_estimate(
        0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "bogus"
    )
    assert "model=monotone_saturate_v0.1" in rendered


def test_exact_target_short_circuits_the_monte_carlo(monkeypatch):
    calls = []

    def _spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("generate_skip_pattern must not run")

    monkeypatch.setattr(script, "generate_skip_pattern", _spy)
    rendered = script._format_hareskip_estimate(
        0.3, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "monotone_saturate_v0.1", 12
    )
    assert "≈12 skips" in rendered
    assert calls == []


def test_exact_target_labels_its_basis():
    rendered = script._format_hareskip_estimate(
        0.3, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "monotone_saturate_v0.1", 12
    )
    assert "exact target" in rendered
    assert "シード平均" not in rendered


def test_all_streaks_zero_estimates_no_skips():
    rendered = script._format_hareskip_estimate(
        0.9, 0.05, 0.95, -4.0, 0.0, 0, 0, 0
    )
    assert "≈0 skips" in rendered


def test_degenerate_window_reduces_the_estimate():
    baseline = _count(script._format_hareskip_estimate(0.55))
    narrow = _count(
        script._format_hareskip_estimate(0.55, 0.5, 0.5)
    )
    assert narrow < baseline


def test_model_choice_changes_the_estimate():
    # v0.1 was calibrated on a different scale, so it must not coincide with
    # the r2 default at the same aggressiveness.
    default = _count(script._format_hareskip_estimate(0.55))
    v01 = _count(
        script._format_hareskip_estimate(
            0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "sigmoid_band_v0.1"
        )
    )
    assert v01 != default


def test_estimate_updates_helper_is_callable():
    # gr.update is stubbed to return None in tests; we only assert the
    # 10-argument signature the wiring relies on does not blow up.
    script._hareskip_estimate_updates(
        0.55, 0.05, 0.95, -4.0, 0.0, 1, 2, 3, "monotone_saturate_v0.1", 0
    )


def test_range_slider_helpers_mirror_and_estimate():
    start, end, _update = script._hareskip_window_range_updates(
        (0.2, 0.8), 0.55, -4.0, 0.0, 1, 2, 3, "monotone_saturate_v0.1", 0
    )
    assert (start, end) == (0.2, 0.8)

    low, high, _update = script._hareskip_zone_range_updates(
        (-3.0, 1.0), 0.55, 0.05, 0.95, 1, 2, 3, "monotone_saturate_v0.1", 0
    )
    assert (low, high) == (-3.0, 1.0)
