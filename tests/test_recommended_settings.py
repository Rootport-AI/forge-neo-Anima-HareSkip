"""Tests for the "Load recommended settings" button.

Covers ``hareskip.recommended_settings.lookup_recommendation`` (pure lookup),
``hareskip.script._resolve_recommendation`` (gradio-independent value
resolution), table self-consistency, and a smoke test that actually calls
``Script().ui(False)``/``ui(True)`` under the gradio/Forge stubs from
conftest to make sure the button wiring (including the disabled-button
fallback branch, which is what fires under the stub environment since there
is no real ``modules.script_callbacks``) never raises.
"""

from __future__ import annotations

from conftest import import_hareskip_script

from hareskip.recommended_settings import RECOMMENDED_SETTINGS, lookup_recommendation
from hareskip.state import RESREFINE_FORMULA_LINEAR, RESREFINE_FORMULA_REUSE, RESREFINE_FORMULAS


def test_lookup_euler_beta():
    row = lookup_recommendation("Euler", "Beta", 30)
    assert row is not None
    assert row["formula"] == RESREFINE_FORMULA_LINEAR
    assert row["prediction_strength"] == 0.30
    assert row["slope_ema_smoothing"] == 0.10


def test_lookup_euler_simple():
    row = lookup_recommendation("Euler", "Simple", 30)
    assert row is not None
    assert row["formula"] == RESREFINE_FORMULA_LINEAR
    assert row["prediction_strength"] == 0.30
    assert row["slope_ema_smoothing"] == 0.00


def test_lookup_euler_a_beta_is_none():
    # "Euler a" must not match the "Euler" row.
    assert lookup_recommendation("Euler a", "Beta", 30) is None


def test_lookup_er_sde_beta_is_none():
    assert lookup_recommendation("ER SDE", "Beta", 30) is None


def test_lookup_euler_karras_is_none():
    assert lookup_recommendation("Euler", "Karras", 30) is None


def test_lookup_is_case_sensitive_exact_match_only():
    assert lookup_recommendation("euler", "beta", 30) is None
    assert lookup_recommendation("EULER", "BETA", 30) is None


def test_table_rows_use_known_formulas():
    for row in RECOMMENDED_SETTINGS:
        assert row["formula"] in RESREFINE_FORMULAS


def test_table_rows_values_in_range():
    for row in RECOMMENDED_SETTINGS:
        assert 0.0 <= row["prediction_strength"] <= 1.0
        assert 0.0 <= row["slope_ema_smoothing"] <= 0.99
        assert 0.0 <= row["use_prediction_after_progress"] <= 1.0
        assert 1 <= row["apply_prediction_from_skip"] <= 3


def test_table_rows_no_duplicate_sampler_scheduler_pairs():
    pairs = [(row["sampler"], row["scheduler"]) for row in RECOMMENDED_SETTINGS]
    assert len(pairs) == len(set(pairs))


def test_resolve_recommendation_hit_euler_beta():
    script = import_hareskip_script()
    resolved = script._resolve_recommendation("Euler", "Beta", 30)
    assert resolved["hit"] is True
    assert resolved["formula"] == RESREFINE_FORMULA_LINEAR
    assert resolved["prediction_strength"] == 0.30
    assert resolved["slope_ema_smoothing"] == 0.10
    assert resolved["use_prediction_after_progress"] == 0.0
    assert resolved["apply_prediction_from_skip"] == 2
    # interactive tuple: (use_after, apply_from, strength, taylor2, slope_ema, curve_ema)
    (
        use_after_interactive,
        apply_from_interactive,
        strength_interactive,
        taylor2_interactive,
        slope_ema_interactive,
        curve_ema_interactive,
    ) = resolved["interactive"]
    assert use_after_interactive is True
    assert apply_from_interactive is True
    assert strength_interactive is True
    assert slope_ema_interactive is True
    # Linear (not Taylor2) -> taylor2/curve_ema controls stay non-interactive.
    assert taylor2_interactive is False
    assert curve_ema_interactive is False
    assert "Euler" in resolved["status"]
    assert "Beta" in resolved["status"]
    assert "30" in resolved["status"]


def test_resolve_recommendation_miss_falls_back_to_reuse():
    script = import_hareskip_script()
    resolved = script._resolve_recommendation("ER SDE", "Beta", 30)
    assert resolved["hit"] is False
    assert resolved["formula"] == RESREFINE_FORMULA_REUSE
    # A miss must not clobber whatever value the user had on the sliders —
    # signalled by None ("leave alone"), not a concrete number.
    assert resolved["prediction_strength"] is None
    assert resolved["slope_ema_smoothing"] is None
    assert resolved["use_prediction_after_progress"] is None
    assert resolved["apply_prediction_from_skip"] is None
    # Reuse -> none of the prediction controls are interactive.
    assert all(flag is False for flag in resolved["interactive"])
    assert "ER SDE" in resolved["status"]
    assert "Beta" in resolved["status"]
    assert "Reuse" in resolved["status"]


def test_resolve_recommendation_steps_reflected_in_status():
    script = import_hareskip_script()
    resolved = script._resolve_recommendation("Euler", "Beta", 42)
    assert "42" in resolved["status"]


def test_ui_smoke_txt2img_returns_39_args_and_does_not_raise():
    script = import_hareskip_script()
    result = script.Script().ui(False)
    assert isinstance(result, list)
    assert len(result) == 39


def test_ui_smoke_img2img_returns_39_args_and_does_not_raise():
    script = import_hareskip_script()
    result = script.Script().ui(True)
    assert isinstance(result, list)
    assert len(result) == 39
