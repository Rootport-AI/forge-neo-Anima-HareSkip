"""ResRefine residual extrapolation.

Skipped steps reuse (or extrapolate from) the residual captured on earlier
full computes. All extrapolation happens on an internal monotone time axis
``tau = 1.0 - t_now``.

2026-09-03: the extrapolation x-axis was changed from the step NUMBER to real
flow time ``tau = 1 - t_now`` (user decision). Beta-style schedules take very
non-uniform steps in t_now (~0.005 early, ~0.065 mid), so a slope measured
"per step number" is a biased estimate of the true slope per unit time, and
the bias moves with the schedule. t_now itself DECREASES as denoising
proceeds (0.999 -> 0.007), so every predictor here converts it once to
``tau = 1 - t_now``, which increases monotonically over the generation; that
keeps the existing "dt must be positive" guards meaningful and makes
velocity/acceleration signs consistent across velocity, acceleration,
``dt_pred`` and the Lagrange interpolation alike.
"""

from __future__ import annotations

from typing import Any

from .state import (
    STATE,
    RESREFINE_FORMULA_LINEAR,
    RESREFINE_FORMULA_TAYLOR2,
    RESREFINE_FORMULA_REUSE,
)

RESREFINE_MAX_NORM_RATIO = 3.0

# Minimum separation between two samples on the tau axis before they are
# treated as the same point. tau lives in (0, 1) and the smallest real step of
# a Beta schedule is on the order of 1e-3, so anything below 1e-9 can only be
# the same model call recorded twice (or float noise), never two genuine steps.
RESREFINE_MIN_TAU_GAP = 1e-9


def _resrefine_tau(t_now: Any) -> float | None:
    """Convert flow time ``t_now`` to the monotone internal axis ``tau``.

    ``t_now`` runs downwards (~0.999 -> ~0.007) over a generation, so
    ``tau = 1 - t_now`` runs upwards and can be used directly as the x-axis of
    every extrapolation below. Returns None for missing/non-finite input; the
    callers then fall back to plain reuse rather than predicting on a bad axis.
    """
    import math

    if t_now is None:
        return None
    try:
        value = float(t_now)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return 1.0 - value


def _shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    try:
        return "x".join(str(part) for part in shape)
    except Exception:
        return str(shape)


def _resrefine_residual_for_slot(
    slot: dict[str, Any],
    target_slice: Any,
    step_index: int,
    skip_streak: int,
    late_phase: bool,
    t_now: float | None = None,
) -> tuple[Any, str, str | None, str | None, dict[str, Any]]:
    """Residual to add on a skipped step: an extrapolation, or plain reuse.

    ``t_now`` is this step's flow time; it is converted once to the monotone
    axis ``tau = 1 - t_now`` and every predictor extrapolates against tau (see
    the module docstring). When it is missing the predictors fall back to
    reuse rather than extrapolating on a guessed axis.
    """
    tau = _resrefine_tau(t_now)
    ema_info = _resrefine_ema_info(slot, tau)
    previous = slot.get("previous_residual")
    if previous is None:
        raise RuntimeError("missing previous_residual")
    if getattr(previous, "shape", None) != getattr(target_slice, "shape", None):
        raise RuntimeError(
            f"residual shape mismatch residual={_shape(previous)} target={_shape(target_slice)}"
        )

    formula = STATE.resrefine_formula
    if formula == RESREFINE_FORMULA_REUSE:
        return previous.to(target_slice.device), "fallback", "formula", None, ema_info

    prediction_allowed = late_phase or skip_streak >= STATE.resrefine_apply_prediction_from_skip
    if not prediction_allowed:
        return previous.to(target_slice.device), "fallback", "streak", None, ema_info

    try:
        prediction_note = None
        if formula == RESREFINE_FORMULA_LINEAR:
            if STATE.resrefine_slope_ema_smoothing <= 0.0:
                prediction = _resrefine_predict_linear(slot, tau, previous)
            else:
                prediction, prediction_note = _resrefine_predict_linear_ema(slot, tau, previous)
        elif formula == RESREFINE_FORMULA_TAYLOR2:
            if STATE.resrefine_slope_ema_smoothing <= 0.0:
                prediction = _resrefine_predict_taylor2(slot, tau, previous)
            else:
                prediction, prediction_note = _resrefine_predict_taylor2_ema(slot, tau, previous)
        else:
            return previous.to(target_slice.device), "fallback", "formula", None, ema_info
        prediction = _resrefine_validate_prediction(prediction, previous, target_slice)
        return prediction.to(target_slice.device), "prediction", None, prediction_note, ema_info
    except _ResRefinePredictionFallback as exc:
        return previous.to(target_slice.device), "fallback", exc.reason, None, ema_info
    except Exception:
        return previous.to(target_slice.device), "fallback", "prediction_error", None, ema_info


class _ResRefinePredictionFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _resrefine_record_residual(
    slot: dict[str, Any],
    step_index: int,
    residual: Any,
    t_now: float | None = None,
) -> None:
    """Record a freshly computed residual together with its time coordinate.

    ``step_index`` is kept for diagnostics only; ``tau = 1 - t_now`` is the
    axis every predictor actually uses.
    """
    tau = _resrefine_tau(t_now)
    _resrefine_update_ema(slot, tau, residual)
    history = slot.setdefault("residual_history", [])
    history.append(
        {
            "step_index": int(step_index),
            "tau": tau,
            "residual": residual.detach(),
        }
    )
    if len(history) > 5:
        del history[:-5]


def _resrefine_update_ema(slot: dict[str, Any], tau: float | None, residual: Any) -> None:
    """Update the velocity/acceleration EMAs from the newest residual.

    Both derivatives are per unit of tau (= per unit of real flow time), not
    per step number; tau increases monotonically over a generation, so the
    ``dt > 0`` guards still mark forward progress exactly as they did on the
    old step-number axis.
    """
    history = slot.get("residual_history") or []
    if not history:
        return
    previous_item = history[-1]
    if tau is None:
        return
    try:
        previous_tau = previous_item.get("tau")
        if previous_tau is None:
            return
        t_prev = float(previous_tau)
        t_now = float(tau)
        dt = t_now - t_prev
        if dt <= RESREFINE_MIN_TAU_GAP:
            return
        previous_residual = previous_item["residual"]
        if getattr(previous_residual, "shape", None) != getattr(residual, "shape", None):
            return
        v_obs = (residual.detach().float() - previous_residual.float()) / dt
        v_time = (t_prev + t_now) / 2.0
        previous_velocity = slot.get("previous_velocity")
        previous_velocity_time = slot.get("previous_velocity_time")
        if (
            previous_velocity is not None
            and previous_velocity_time is not None
            and getattr(previous_velocity, "shape", None) == getattr(v_obs, "shape", None)
        ):
            dt_v = v_time - float(previous_velocity_time)
            if dt_v > 0.0:
                a_obs = (v_obs - previous_velocity.to(v_obs.device).float()) / dt_v
                acceleration_ema = slot.get("acceleration_ema")
                beta_a = STATE.resrefine_curve_ema_smoothing
                if acceleration_ema is None or getattr(acceleration_ema, "shape", None) != getattr(a_obs, "shape", None):
                    slot["acceleration_ema"] = a_obs.detach()
                else:
                    slot["acceleration_ema"] = (
                        beta_a * acceleration_ema.to(a_obs.device).float()
                        + (1.0 - beta_a) * a_obs
                    ).detach()

        velocity_ema = slot.get("velocity_ema")
        beta_v = STATE.resrefine_slope_ema_smoothing
        if velocity_ema is None or getattr(velocity_ema, "shape", None) != getattr(v_obs, "shape", None):
            slot["velocity_ema"] = v_obs.detach()
        else:
            slot["velocity_ema"] = (
                beta_v * velocity_ema.to(v_obs.device).float()
                + (1.0 - beta_v) * v_obs
            ).detach()
        slot["previous_velocity"] = v_obs.detach()
        slot["previous_velocity_time"] = v_time
    except Exception:
        return


def _resrefine_predict_linear(slot: dict[str, Any], tau: float | None, previous: Any):
    history = _resrefine_residual_history(slot, 2)
    raw_prediction = _resrefine_lagrange_prediction(history[-2:], tau)
    previous_f32 = previous.float()
    return previous_f32 + STATE.resrefine_prediction_strength * (raw_prediction - previous_f32)


def _resrefine_predict_taylor2(slot: dict[str, Any], tau: float | None, previous: Any):
    history = _resrefine_residual_history(slot, 3)
    linear_prediction = _resrefine_lagrange_prediction(history[-2:], tau)
    quadratic_prediction = _resrefine_lagrange_prediction(history[-3:], tau)
    curve = STATE.resrefine_taylor2_curve_strength
    raw_prediction = (1.0 - curve) * linear_prediction + curve * quadratic_prediction
    previous_f32 = previous.float()
    return previous_f32 + STATE.resrefine_prediction_strength * (raw_prediction - previous_f32)


def _resrefine_predict_linear_ema(
    slot: dict[str, Any],
    tau: float | None,
    previous: Any,
) -> tuple[Any, str | None]:
    dt_pred, velocity_ema = _resrefine_ema_velocity(slot, tau, previous)
    previous_f32 = previous.float()
    raw_prediction = previous_f32 + dt_pred * velocity_ema
    return (
        previous_f32
        + STATE.resrefine_prediction_strength * (raw_prediction - previous_f32),
        None,
    )


def _resrefine_predict_taylor2_ema(
    slot: dict[str, Any],
    tau: float | None,
    previous: Any,
) -> tuple[Any, str | None]:
    dt_pred, velocity_ema = _resrefine_ema_velocity(slot, tau, previous)
    previous_f32 = previous.float()
    linear_prediction = previous_f32 + dt_pred * velocity_ema
    acceleration_ema = slot.get("acceleration_ema")
    if acceleration_ema is None:
        raw_prediction = linear_prediction
        prediction_note = "taylor2_ema_without_acceleration"
    else:
        if getattr(acceleration_ema, "shape", None) != getattr(previous, "shape", None):
            raise _ResRefinePredictionFallback("shape_mismatch")
        curve_term = 0.5 * (dt_pred ** 2) * acceleration_ema.to(previous.device).float()
        raw_prediction = (
            linear_prediction
            + STATE.resrefine_taylor2_curve_strength * curve_term
        )
        prediction_note = None
    return (
        previous_f32
        + STATE.resrefine_prediction_strength * (raw_prediction - previous_f32),
        prediction_note,
    )


def _resrefine_ema_velocity(
    slot: dict[str, Any],
    tau: float | None,
    previous: Any,
) -> tuple[float, Any]:
    """Lead time on the tau axis plus the EMA velocity (per unit tau)."""
    history = slot.get("residual_history") or []
    if not history:
        raise _ResRefinePredictionFallback("insufficient_history")
    latest = history[-1]
    velocity_ema = slot.get("velocity_ema")
    if velocity_ema is None:
        raise _ResRefinePredictionFallback("insufficient_ema_velocity")
    if getattr(velocity_ema, "shape", None) != getattr(previous, "shape", None):
        raise _ResRefinePredictionFallback("shape_mismatch")
    latest_tau = latest.get("tau")
    if tau is None or latest_tau is None:
        raise _ResRefinePredictionFallback("missing_time")
    dt_pred = float(tau) - float(latest_tau)
    if dt_pred <= RESREFINE_MIN_TAU_GAP:
        raise _ResRefinePredictionFallback("duplicate_history_step")
    return dt_pred, velocity_ema.to(previous.device).float()


def _resrefine_ema_info(slot: dict[str, Any], tau: float | None) -> dict[str, Any]:
    history = slot.get("residual_history") or []
    dt_pred = None
    if history and tau is not None:
        try:
            latest_tau = history[-1].get("tau")
            if latest_tau is not None:
                dt_pred = float(tau) - float(latest_tau)
        except Exception:
            dt_pred = None
    return {
        "velocity_ready": slot.get("velocity_ema") is not None,
        "acceleration_ready": slot.get("acceleration_ema") is not None,
        "dt_pred": dt_pred,
    }


def _resrefine_residual_history(slot: dict[str, Any], count: int) -> list[dict[str, Any]]:
    history = slot.get("residual_history") or []
    if len(history) < count:
        raise _ResRefinePredictionFallback("insufficient_history")
    recent = history[-count:]
    reference_shape = getattr(recent[-1].get("residual"), "shape", None)
    for item in recent:
        if getattr(item.get("residual"), "shape", None) != reference_shape:
            raise _ResRefinePredictionFallback("shape_mismatch")
    return recent


def _resrefine_lagrange_prediction(history: list[dict[str, Any]], tau: float | None):
    """Lagrange extrapolation of the residual onto the target tau.

    The nodes are the recorded ``tau`` values (real flow time), not step
    numbers, so the fitted slope/curvature are per unit time and stay correct
    under a non-uniform schedule.
    """
    if not history:
        raise _ResRefinePredictionFallback("insufficient_history")
    if tau is None:
        raise _ResRefinePredictionFallback("missing_time")
    raw_times = [item.get("tau") for item in history]
    if any(value is None for value in raw_times):
        raise _ResRefinePredictionFallback("missing_time")
    times = [float(value) for value in raw_times]
    target = float(tau)
    result = None
    for i, item in enumerate(history):
        weight = 1.0
        for j, other_time in enumerate(times):
            if i == j:
                continue
            denom = times[i] - other_time
            # tau-scale duplicate test; see RESREFINE_MIN_TAU_GAP.
            if abs(denom) < RESREFINE_MIN_TAU_GAP:
                raise _ResRefinePredictionFallback("duplicate_history_step")
            weight *= (target - other_time) / denom
        residual = item["residual"].float()
        weighted = residual * weight
        result = weighted if result is None else result + weighted
    if result is None:
        raise _ResRefinePredictionFallback("insufficient_history")
    return result


def _resrefine_validate_prediction(prediction: Any, previous: Any, target_slice: Any):
    import math
    import torch

    if getattr(prediction, "shape", None) != getattr(previous, "shape", None):
        raise _ResRefinePredictionFallback("shape_mismatch")
    if getattr(prediction, "shape", None) != getattr(target_slice, "shape", None):
        raise _ResRefinePredictionFallback("shape_mismatch")
    if not bool(torch.isfinite(prediction).all().item()):
        raise _ResRefinePredictionFallback("numeric_error")

    prediction_norm = float(torch.linalg.vector_norm(prediction.float()).item())
    previous_norm = float(torch.linalg.vector_norm(previous.float()).item())
    if not math.isfinite(prediction_norm) or not math.isfinite(previous_norm):
        raise _ResRefinePredictionFallback("numeric_error")
    if previous_norm > 0.0 and prediction_norm > previous_norm * RESREFINE_MAX_NORM_RATIO:
        raise _ResRefinePredictionFallback("norm_guard")
    try:
        return prediction.to(device=previous.device, dtype=previous.dtype)
    except Exception as exc:
        raise _ResRefinePredictionFallback("dtype_conversion") from exc
