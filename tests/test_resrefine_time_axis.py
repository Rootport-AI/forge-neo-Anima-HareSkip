"""ResRefine extrapolates on real flow time, not on step numbers.

2026-09-03: the prediction x-axis moved from the step index to
``tau = 1 - t_now``. These tests pin the new behaviour: on a non-uniform
schedule (the normal case for a Beta scheduler) a residual that is an exact
linear/quadratic function of tau must be predicted exactly, which the old
step-number axis could not do.

torch is not installed in the test environment (package installs are not
permitted here), so the residuals are a tiny stand-in vector type exposing
just the surface the prediction helpers touch: ``.float()``, ``.detach()``,
``.to()``, ``.shape``, and elementwise arithmetic. The helpers under test do
not import torch (only ``_resrefine_validate_prediction`` does, and it is not
exercised here).
"""

from __future__ import annotations

import pytest

from hareskip.resrefine import (
    RESREFINE_MIN_TAU_GAP,
    _ResRefinePredictionFallback,
    _resrefine_ema_info,
    _resrefine_ema_velocity,
    _resrefine_lagrange_prediction,
    _resrefine_predict_linear,
    _resrefine_predict_linear_ema,
    _resrefine_predict_taylor2,
    _resrefine_record_residual,
    _resrefine_tau,
)
from hareskip.state import STATE


class FakeTensor:
    """Minimal stand-in for a torch tensor (see module docstring)."""

    def __init__(self, values):
        self.values = [float(v) for v in values]

    # --- torch-ish surface -------------------------------------------------
    @property
    def shape(self):
        return (len(self.values),)

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return "float32"

    def float(self):
        return self

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    # --- arithmetic --------------------------------------------------------
    def _binary(self, other, op):
        if isinstance(other, FakeTensor):
            assert len(other.values) == len(self.values)
            pairs = zip(self.values, other.values)
        else:
            pairs = ((v, float(other)) for v in self.values)
        return FakeTensor([op(a, b) for a, b in pairs])

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"FakeTensor({self.values})"


@pytest.fixture(autouse=True)
def _reset_resrefine_state():
    """Predictors read tuning knobs off STATE; pin them and restore after."""
    saved = {
        name: getattr(STATE, name)
        for name in (
            "resrefine_prediction_strength",
            "resrefine_taylor2_curve_strength",
            "resrefine_slope_ema_smoothing",
            "resrefine_curve_ema_smoothing",
        )
    }
    STATE.resrefine_prediction_strength = 1.0
    STATE.resrefine_taylor2_curve_strength = 1.0
    STATE.resrefine_slope_ema_smoothing = 0.0
    STATE.resrefine_curve_ema_smoothing = 0.0
    yield
    for name, value in saved.items():
        setattr(STATE, name, value)


def build_slot(t_nows, residual_fn):
    """A slot whose history holds ``residual_fn(tau)`` at each given t_now."""
    slot: dict = {}
    for step_index, t_now in enumerate(t_nows):
        residual = FakeTensor(residual_fn(1.0 - t_now))
        _resrefine_record_residual(slot, step_index, residual, t_now)
        slot["previous_residual"] = residual
    return slot


def approx(tensor):
    return pytest.approx(tensor.values, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# tau conversion
# ---------------------------------------------------------------------------


def test_tau_is_monotone_increasing_while_t_now_decreases():
    # t_now runs downwards over a generation; tau must run upwards.
    t_nows = [0.999, 0.94, 0.61, 0.20, 0.007]
    taus = [_resrefine_tau(t) for t in t_nows]
    assert taus == pytest.approx([1.0 - t for t in t_nows])
    assert all(b > a for a, b in zip(taus, taus[1:]))


def test_tau_rejects_missing_and_non_finite_input():
    assert _resrefine_tau(None) is None
    assert _resrefine_tau(float("nan")) is None
    assert _resrefine_tau(float("inf")) is None
    assert _resrefine_tau("not a number") is None


# ---------------------------------------------------------------------------
# exactness on a NON-UNIFORM schedule
# ---------------------------------------------------------------------------


# Deliberately uneven t_now steps (0.005 / 0.065 gaps, as a Beta schedule
# produces). On the old step-number axis the slope would be measured per step
# and these predictions would miss.
NON_UNIFORM_T_NOW = [0.900, 0.895, 0.830, 0.780]


def test_linear_prediction_is_exact_for_a_linear_function_of_tau():
    slope = [2.0, -3.0]
    intercept = [0.5, 1.5]

    def residual_fn(tau):
        return [intercept[i] + slope[i] * tau for i in range(2)]

    slot = build_slot(NON_UNIFORM_T_NOW[:3], residual_fn)
    target_t_now = NON_UNIFORM_T_NOW[3]
    target_tau = 1.0 - target_t_now

    prediction = _resrefine_predict_linear(
        slot, target_tau, slot["previous_residual"]
    )
    assert prediction.values == approx(FakeTensor(residual_fn(target_tau)))


def test_step_number_axis_would_have_missed_this_layout():
    """Guard the guard: the fixture really does distinguish the two axes."""
    slope = [2.0, -3.0]

    def residual_fn(tau):
        return [slope[i] * tau for i in range(2)]

    slot = build_slot(NON_UNIFORM_T_NOW[:3], residual_fn)
    target_tau = 1.0 - NON_UNIFORM_T_NOW[3]
    on_tau = _resrefine_predict_linear(slot, target_tau, slot["previous_residual"])

    # Same history read on the old step-number axis: nodes 1,2 -> target 3.
    history = [
        {"tau": float(item["step_index"]), "residual": item["residual"]}
        for item in slot["residual_history"]
    ]
    on_steps = _resrefine_lagrange_prediction(history[-2:], 3.0)

    truth = FakeTensor(residual_fn(target_tau))
    assert on_tau.values == approx(truth)
    assert on_steps.values != approx(truth)


def test_taylor2_recovers_curvature_of_a_quadratic_in_tau():
    def residual_fn(tau):
        return [1.0 + 2.0 * tau + 5.0 * tau * tau, -0.5 * tau * tau]

    slot = build_slot(NON_UNIFORM_T_NOW[:3], residual_fn)
    target_tau = 1.0 - NON_UNIFORM_T_NOW[3]

    prediction = _resrefine_predict_taylor2(
        slot, target_tau, slot["previous_residual"]
    )
    # curve_strength == 1.0 -> pure quadratic Lagrange, exact for a quadratic.
    assert prediction.values == approx(FakeTensor(residual_fn(target_tau)))


def test_uniform_spacing_matches_the_old_step_number_axis():
    """The axis change is a no-op when the schedule happens to be uniform."""
    step = 0.05
    t_nows = [0.9 - i * step for i in range(3)]

    def residual_fn(tau):
        return [3.0 * tau, 1.0 - tau]

    slot = build_slot(t_nows, residual_fn)
    target_tau = 1.0 - (0.9 - 3 * step)
    on_tau = _resrefine_predict_linear(slot, target_tau, slot["previous_residual"])

    # Uniform tau spacing means a linear fit through step numbers extrapolates
    # to the same residual, because both axes are affine images of each other.
    history = [
        {"tau": float(item["step_index"]), "residual": item["residual"]}
        for item in slot["residual_history"]
    ]
    on_steps = _resrefine_lagrange_prediction(history[-2:], 3.0)
    assert on_tau.values == approx(on_steps)


# ---------------------------------------------------------------------------
# EMA velocity path
# ---------------------------------------------------------------------------


def test_ema_velocity_uses_delta_tau_on_a_non_uniform_schedule():
    STATE.resrefine_slope_ema_smoothing = 0.0  # EMA == latest observation

    def residual_fn(tau):
        return [4.0 * tau, -1.0 * tau]

    slot = build_slot(NON_UNIFORM_T_NOW[:3], residual_fn)

    # Velocity is d(residual)/d(tau), so the uneven spacing must cancel out.
    assert slot["velocity_ema"].values == pytest.approx([4.0, -1.0])

    target_tau = 1.0 - NON_UNIFORM_T_NOW[3]
    dt_pred, velocity = _resrefine_ema_velocity(
        slot, target_tau, slot["previous_residual"]
    )
    assert dt_pred == pytest.approx(
        target_tau - (1.0 - NON_UNIFORM_T_NOW[2])
    )
    assert velocity.values == pytest.approx([4.0, -1.0])

    prediction, note = _resrefine_predict_linear_ema(
        slot, target_tau, slot["previous_residual"]
    )
    assert note is None
    assert prediction.values == approx(FakeTensor(residual_fn(target_tau)))


def test_velocity_is_computed_although_t_now_decreases():
    """t_now falls 0.98 -> 0.90, yet dt on the tau axis is positive."""

    def residual_fn(tau):
        return [10.0 * tau]

    slot = build_slot([0.98, 0.90], residual_fn)
    assert slot["velocity_ema"] is not None
    assert slot["velocity_ema"].values == pytest.approx([10.0])

    info = _resrefine_ema_info(slot, 1.0 - 0.85)
    assert info["velocity_ready"] is True
    assert info["dt_pred"] > 0.0


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def test_repeated_t_now_leaves_the_velocity_ema_untouched():
    slot: dict = {}
    _resrefine_record_residual(slot, 0, FakeTensor([1.0]), 0.9)
    _resrefine_record_residual(slot, 1, FakeTensor([2.0]), 0.9)
    assert slot.get("velocity_ema") is None


def test_zero_lead_time_falls_back_instead_of_dividing_by_zero():
    def residual_fn(tau):
        return [2.0 * tau]

    slot = build_slot([0.95, 0.90], residual_fn)
    latest_tau = 1.0 - 0.90
    with pytest.raises(_ResRefinePredictionFallback) as excinfo:
        _resrefine_ema_velocity(slot, latest_tau, slot["previous_residual"])
    assert excinfo.value.reason == "duplicate_history_step"


def test_lagrange_rejects_duplicate_nodes_below_the_tau_gap():
    residual = FakeTensor([1.0])
    history = [
        {"tau": 0.10, "residual": residual},
        {"tau": 0.10 + RESREFINE_MIN_TAU_GAP / 10.0, "residual": residual},
    ]
    with pytest.raises(_ResRefinePredictionFallback) as excinfo:
        _resrefine_lagrange_prediction(history, 0.2)
    assert excinfo.value.reason == "duplicate_history_step"


def test_missing_t_now_falls_back_rather_than_guessing_an_axis():
    slot: dict = {}
    _resrefine_record_residual(slot, 0, FakeTensor([1.0]), None)
    previous = FakeTensor([2.0])
    _resrefine_record_residual(slot, 1, previous, None)
    with pytest.raises(_ResRefinePredictionFallback) as excinfo:
        _resrefine_predict_linear(slot, None, previous)
    assert excinfo.value.reason == "missing_time"


def test_short_history_falls_back():
    slot = build_slot([0.9], lambda tau: [tau])
    with pytest.raises(_ResRefinePredictionFallback) as excinfo:
        _resrefine_predict_linear(slot, 1.0 - 0.85, slot["previous_residual"])
    assert excinfo.value.reason == "insufficient_history"

    slot2 = build_slot([0.9, 0.85], lambda tau: [tau])
    with pytest.raises(_ResRefinePredictionFallback) as excinfo:
        _resrefine_predict_taylor2(slot2, 1.0 - 0.80, slot2["previous_residual"])
    assert excinfo.value.reason == "insufficient_history"


def test_history_records_both_axes():
    slot = build_slot([0.9, 0.85], lambda tau: [tau])
    entry = slot["residual_history"][-1]
    # step_index survives for diagnostics; tau is what the predictors read.
    assert entry["step_index"] == 1
    assert entry["tau"] == pytest.approx(1.0 - 0.85)
