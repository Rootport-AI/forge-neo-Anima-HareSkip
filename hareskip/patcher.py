from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging import info, warning
from .state import (
    STATE,
    UJICACHE_CACHE_DEVICE_CPU,
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TAYLOR2,
    UJICACHE_FORMULA_TEACACHE,
    UJICACHE_SOURCE_FIRST_BLOCK_SHIFT,
    ujicache_coefficients_for_profile,
)

UJICACHE_MAX_NORM_RATIO = 3.0


@dataclass
class PatchResult:
    ok: bool
    kind: str
    message: str


def apply_patch(kind: str, context: Any = None) -> PatchResult:
    if kind == "hareskip":
        return _apply_hareskip_patch()
    return PatchResult(False, kind, f"unknown patch kind: {kind}")


def remove_patch(kind: str) -> PatchResult:
    patch = STATE.patches.pop(kind, None)
    if patch is None:
        return PatchResult(True, kind, "not patched")
    restore = patch.get("restore") if isinstance(patch, dict) else None
    if callable(restore):
        restore()
    info(f"removed patch kind={kind}")
    return PatchResult(True, kind, "removed")


def remove_all_patches() -> PatchResult:
    ok = True
    messages: list[str] = []
    for kind in list(STATE.patches):
        result = remove_patch(kind)
        ok = ok and result.ok
        messages.append(f"{kind}:{result.message}")
    return PatchResult(ok, "all", ",".join(messages) if messages else "none")


def is_patched(kind: str) -> bool:
    return kind in STATE.patches


def _apply_hareskip_patch() -> PatchResult:
    kind = "hareskip"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    anima_cls = getattr(anima, "Anima", None)
    if anima_cls is None:
        return PatchResult(False, kind, "Anima class not found")

    target_name = "_forward"
    original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        target_name = "forward"
        original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        return PatchResult(False, kind, "Anima._forward/forward not found")

    def hareskip_forward(self, x, timesteps, context, *args, **kwargs):
        if not _should_hareskip_patch():
            return original_forward(self, x, timesteps, context, *args, **kwargs)
        try:
            fps, padding_mask, body_kwargs = _parse_forward_args(target_name, args, kwargs)
            return _ujicache_forward_body(
                self,
                x,
                timesteps,
                context,
                fps=fps,
                padding_mask=padding_mask,
                **body_kwargs,
            )
        except Exception as exc:
            STATE.hareskip_errors += 1
            STATE.hareskip_fallbacks += 1
            STATE.hareskip_unavailable_reason = _short_error(exc)
            if STATE.hareskip_logged_calls < 12:
                STATE.hareskip_logged_calls += 1
                warning(
                    f"hareskip_fallback=reason={_short_error(exc)} "
                    f"route=original_Anima.{target_name}"
                )
            return original_forward(self, x, timesteps, context, *args, **kwargs)

    setattr(anima_cls, target_name, hareskip_forward)

    def restore() -> None:
        setattr(anima_cls, target_name, original_forward)

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied experimental patch kind=ujicache "
        f"target=backend.nn.anima.Anima.{target_name} "
        f"formula={STATE.ujicache_formula} "
        f"threshold={STATE.ujicache_threshold:.4f} "
        f"progress={STATE.ujicache_start_percent:.2f}..{STATE.ujicache_end_percent:.2f} "
        f"use_prediction_after={STATE.ujicache_use_prediction_after_progress:.2f} "
        f"apply_from_skip={STATE.ujicache_apply_prediction_from_skip} "
        f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
        f"taylor2_curve_strength={STATE.ujicache_taylor2_curve_strength:.2f} "
        f"slope_ema_smoothing={STATE.ujicache_slope_ema_smoothing:.2f} "
        f"curve_ema_smoothing={STATE.ujicache_curve_ema_smoothing:.2f} "
        f"cache_device={STATE.ujicache_cache_device} "
        f"source={STATE.ujicache_modulated_source}"
    )
    return PatchResult(True, kind, "applied")


def _parse_forward_args(
    target_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    body_kwargs = dict(kwargs)
    fps = body_kwargs.pop("fps", None)
    padding_mask = body_kwargs.pop("padding_mask", None)
    if target_name == "_forward":
        if len(args) > 0:
            fps = args[0]
        if len(args) > 1:
            padding_mask = args[1]
        if len(args) > 2:
            raise RuntimeError("Anima._forward extra positional arguments are unsupported by UjiCache")
    else:
        if len(args) > 0:
            padding_mask = args[0]
        if len(args) > 1:
            raise RuntimeError("Anima.forward extra positional arguments are unsupported by UjiCache")
    return fps, padding_mask, body_kwargs


def _should_hareskip_patch() -> bool:
    return STATE.active() and STATE.hareskip_enabled


def _ujicache_forward_body(
    model: Any,
    x: Any,
    timesteps: Any,
    context: Any,
    fps: Any = None,
    padding_mask: Any = None,
    **kwargs,
):
    import torch

    transformer_options = kwargs.get("transformer_options", {}) or {}
    cond_or_uncond = _cond_or_uncond(transformer_options.get("cond_or_uncond"))
    if not cond_or_uncond:
        raise RuntimeError("transformer_options.cond_or_uncond is missing")

    STATE.hareskip_model_calls += 1
    _ensure_hareskip_num_blocks(model)

    orig_shape = list(x.shape)
    x = _pad_to_patch_size_5d(
        x,
        (
            int(getattr(model, "patch_temporal", 1)),
            int(getattr(model, "patch_spatial", 1)),
            int(getattr(model, "patch_spatial", 1)),
        ),
    )
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = _prepare_embedded_sequence(
        model,
        x_B_C_T_H_W,
        fps,
        padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)

    t_embedding_B_T_D, adaln_lora_B_T_3D = model.t_embedder[1](
        model.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = model.t_embedding_norm(t_embedding_B_T_D)

    cache_device = _ujicache_cache_device(x_B_T_H_W_D)
    modulated_inp = _ujicache_modulated_input(
        model,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D,
        cache_device,
    )
    cache = _hareskip_state_for_model(model)
    batch_per_slot = _batch_per_slot(x_B_T_H_W_D, cond_or_uncond)
    step_index = max(0, STATE.denoiser_calls - 1)
    progress = _progress(step_index)

    rels: dict[Any, float | None] = {}
    slot_should_calc: dict[Any, bool] = {}
    for slot_index, key in enumerate(cond_or_uncond):
        key = int(key)
        item = _ujicache_slot(cache, key)
        modulated_slice = modulated_inp[
            slot_index * batch_per_slot : (slot_index + 1) * batch_per_slot
        ]
        rels[key] = _ujicache_update_slot(item, modulated_slice)
        slot_should_calc[key] = bool(item["should_calc"])

    force_full_reason = _ujicache_force_full_reason(
        cache,
        step_index,
        progress,
        cond_or_uncond,
    )
    should_calc = force_full_reason is not None or any(slot_should_calc.values())
    if STATE.calibration_capture_active() and not should_calc:
        should_calc = True
        force_full_reason = "calibration_capture"
    if STATE.ujicache_dry_run and not should_calc:
        STATE.ujicache_dry_run_predictions += 1
        should_calc = True
        force_full_reason = "dry_run"

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    if should_calc:
        ori_x = x_B_T_H_W_D.to(cache_device)
        for block in model.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )
        residual = x_B_T_H_W_D.to(cache_device) - ori_x
        _dump_ujicache_residual(
            residual,
            cond_or_uncond,
            batch_per_slot,
            step_index,
            timesteps_B_T,
            cache_device,
        )
        for slot_index, key in enumerate(cond_or_uncond):
            item = _ujicache_slot(cache, int(key))
            start = slot_index * batch_per_slot
            end = (slot_index + 1) * batch_per_slot
            residual_slice = residual[start:end]
            _capture_calibration_pair(
                item,
                int(key),
                step_index,
                rels.get(int(key)),
                residual_slice,
                timesteps_B_T,
                start,
                end,
            )
            item["previous_residual"] = residual_slice
            _ujicache_record_residual(item, step_index, residual_slice)
            item["accumulated_rel_l1_distance"] = 0.0
            item["should_calc"] = True
        cache["skip_streak"] = 0
        STATE.hareskip_full_calcs += 1
        if STATE.hareskip_model_calls == 1:
            STATE.hareskip_first_full_calcs += 1
        if force_full_reason:
            STATE.hareskip_forced_full_calcs += 1
        _hareskip_log_call(
            "full",
            step_index,
            progress,
            rels,
            reason=force_full_reason,
            late_phase=False,
            skip_streak=0,
            slot_actions={},
            slot_reasons={},
            slot_notes={},
            slot_ema_info={},
        )
    else:
        slot_actions, slot_reasons, slot_notes, slot_ema_info, late_phase, skip_streak = (
            _ujicache_apply_residual(
                x_B_T_H_W_D,
                cache,
                cond_or_uncond,
                batch_per_slot,
                step_index,
                progress,
            )
        )
        cache["skip_streak"] = skip_streak
        STATE.hareskip_skips += 1
        if not STATE.hareskip_skipped_steps or STATE.hareskip_skipped_steps[-1] != step_index:
            STATE.hareskip_skipped_steps.append(int(step_index))
        if any(action == "prediction" for action in slot_actions.values()):
            STATE.ujicache_prediction_used += 1
            decision = "prediction"
        else:
            STATE.hareskip_fallback_used += 1
            decision = "fallback"
        for reason in slot_reasons.values():
            if reason:
                STATE.hareskip_fallback_reasons[reason] = (
                    STATE.hareskip_fallback_reasons.get(reason, 0) + 1
                )
        _hareskip_log_call(
            decision,
            step_index,
            progress,
            rels,
            reason=None,
            late_phase=late_phase,
            skip_streak=skip_streak,
            slot_actions=slot_actions,
            slot_reasons=slot_reasons,
            slot_notes=slot_notes,
            slot_ema_info=slot_ema_info,
        )

    x_B_T_H_W_O = model.final_layer(
        x_B_T_H_W_D.to(crossattn_emb.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    return model.unpatchify(x_B_T_H_W_O)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]


def _ensure_hareskip_num_blocks(model: Any) -> None:
    if STATE.hareskip_num_blocks is not None:
        return
    blocks = getattr(model, "blocks", None)
    try:
        STATE.hareskip_num_blocks = len(blocks)
    except Exception:
        STATE.hareskip_num_blocks = _runtime_num_blocks()


def _prepare_embedded_sequence(model: Any, x: Any, fps: Any, padding_mask: Any):
    if fps is not None:
        try:
            return model.prepare_embedded_sequence(
                x,
                fps=fps,
                padding_mask=padding_mask,
            )
        except TypeError:
            pass
    return model.prepare_embedded_sequence(x, padding_mask=padding_mask)


def _pad_to_patch_size_5d(x: Any, patch_size: tuple[int, int, int]):
    if len(getattr(x, "shape", ())) != 5:
        raise RuntimeError(f"expected 5D latent tensor, got shape={_shape(x)}")
    try:
        from backend.utils import pad_to_patch_size
    except Exception:
        pad_to_patch_size = None
    if pad_to_patch_size is not None:
        return pad_to_patch_size(x, patch_size)

    import torch
    import torch.nn.functional as functional

    padding_mode = "reflect" if (torch.jit.is_tracing() or torch.jit.is_scripting()) else "circular"
    pad = ()
    for i in range(x.ndim - 2):
        size = max(1, int(patch_size[i]))
        pad = (0, (size - int(x.shape[i + 2]) % size) % size) + pad
    return functional.pad(x, pad, mode=padding_mode)


def _ujicache_cache_device(x: Any):
    import torch

    if STATE.ujicache_cache_device == UJICACHE_CACHE_DEVICE_CPU:
        return torch.device("cpu")
    return getattr(x, "device", torch.device("cpu"))


def _ujicache_modulated_input(model: Any, t_embedding: Any, adaln_lora: Any, cache_device: Any):
    return _cache_modulated_input(
        model,
        t_embedding,
        adaln_lora,
        cache_device,
        STATE.ujicache_modulated_source,
    )


def _cache_modulated_input(
    model: Any,
    t_embedding: Any,
    adaln_lora: Any,
    cache_device: Any,
    source: str,
):
    if source != UJICACHE_SOURCE_FIRST_BLOCK_SHIFT:
        return t_embedding.to(cache_device)
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise RuntimeError("Anima blocks are unavailable")
    first_block = blocks[0]
    adaln = getattr(first_block, "adaln_modulation_self_attn", None)
    if not callable(adaln):
        raise RuntimeError("first block adaln_modulation_self_attn is unavailable")
    modulated = adaln(t_embedding)
    if adaln_lora is not None and bool(getattr(model, "use_adaln_lora", False)):
        modulated = modulated + adaln_lora
    return modulated.chunk(3, dim=-1)[0].to(cache_device)


def _hareskip_state_for_model(model: Any) -> dict[str, Any]:
    cache = getattr(model, "_hareskip_state", None)
    if not isinstance(cache, dict) or cache.get("generation_index") != STATE.generation_index:
        cache = {
            "generation_index": STATE.generation_index,
            "skip_streak": 0,
            "slots": {},
        }
        setattr(model, "_hareskip_state", cache)
    return cache


def _ujicache_slot(cache: dict[str, Any], key: int) -> dict[str, Any]:
    slots = cache.setdefault("slots", {})
    if key not in slots:
        slots[key] = {
            "should_calc": True,
            "accumulated_rel_l1_distance": 0.0,
            "previous_modulated_input": None,
            "previous_residual": None,
            "residual_history": [],
            "previous_velocity": None,
            "previous_velocity_time": None,
            "velocity_ema": None,
            "acceleration_ema": None,
            "capture_prev_t": None,
        }
    return slots[key]


def _cond_or_uncond(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def _batch_per_slot(x: Any, cond_or_uncond: Any) -> int:
    total = int(x.shape[0])
    slots = len(cond_or_uncond)
    if slots <= 0 or total % slots != 0:
        raise RuntimeError(f"invalid cond_or_uncond={cond_or_uncond} for batch={total}")
    return total // slots


def _ujicache_update_slot(slot: dict[str, Any], modulated_slice: Any) -> float | None:
    return _cache_update_slot(
        slot,
        modulated_slice,
        STATE.ujicache_threshold,
        _ujicache_coefficients(),
        "UjiCache",
    )


def _cache_update_slot(
    slot: dict[str, Any],
    modulated_slice: Any,
    threshold: float,
    coefficients: list[float],
    label: str,
) -> float | None:
    import math

    previous = slot.get("previous_modulated_input")
    rel: float | None = None
    if previous is None:
        slot["should_calc"] = True
    else:
        try:
            denom = previous.abs().mean()
            if float(denom.item()) <= 0.0:
                raise RuntimeError("previous_modulated_input mean is zero")
            rel_tensor = (modulated_slice - previous).abs().mean() / denom
            rel = float(rel_tensor.item())
            if not math.isfinite(rel):
                raise RuntimeError(f"non-finite {label} rel_l1: {rel}")
            estimate = _cache_poly1d(rel, coefficients)
            if not math.isfinite(estimate) or estimate < 0.0:
                raise RuntimeError(f"invalid {label} estimate: {estimate}")
            accumulated = float(slot.get("accumulated_rel_l1_distance", 0.0)) + estimate
            if accumulated < threshold:
                slot["should_calc"] = False
                slot["accumulated_rel_l1_distance"] = accumulated
            else:
                slot["should_calc"] = True
                slot["accumulated_rel_l1_distance"] = 0.0
        except Exception:
            slot["should_calc"] = True
            slot["accumulated_rel_l1_distance"] = 0.0
    slot["previous_modulated_input"] = modulated_slice.detach()
    return rel


def _cache_poly1d(value: float, coefficients: list[float]) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def _ujicache_coefficients() -> list[float]:
    return ujicache_coefficients_for_profile(STATE.ujicache_coefficient_profile)


def _progress(step_index: int) -> float:
    steps = STATE.generation_steps
    if not steps or steps <= 1:
        return 0.0
    return max(0.0, min(1.0, step_index / float(steps - 1)))


def _ujicache_force_full_reason(
    cache: dict[str, Any],
    step_index: int,
    progress: float,
    cond_or_uncond: Any,
) -> str | None:
    if STATE.hareskip_model_calls == 1:
        return "first_call"
    for key in cond_or_uncond:
        if _ujicache_slot(cache, int(key)).get("previous_residual") is None:
            return "missing_residual"
    if progress < STATE.ujicache_start_percent or progress > STATE.ujicache_end_percent:
        return "outside_progress"
    interval = STATE.ujicache_force_full_interval
    if interval > 0 and step_index > 0 and step_index % interval == 0:
        return "force_full_interval"
    max_skip_streak = STATE.ujicache_max_skip_streak
    if max_skip_streak > 0 and int(cache.get("skip_streak", 0)) >= max_skip_streak:
        return "max_skip_streak"
    return None


def _ujicache_apply_residual(
    x: Any,
    cache: dict[str, Any],
    cond_or_uncond: Any,
    batch_per_slot: int,
    step_index: int,
    progress: float,
) -> tuple[
    dict[int, str],
    dict[int, str | None],
    dict[int, str | None],
    dict[int, dict[str, Any]],
    bool,
    int,
]:
    skip_streak = int(cache.get("skip_streak", 0)) + 1
    late_phase = progress > STATE.ujicache_use_prediction_after_progress
    slot_actions: dict[int, str] = {}
    slot_reasons: dict[int, str | None] = {}
    slot_notes: dict[int, str | None] = {}
    slot_ema_info: dict[int, dict[str, Any]] = {}
    for slot_index, key in enumerate(cond_or_uncond):
        key = int(key)
        start = slot_index * batch_per_slot
        end = (slot_index + 1) * batch_per_slot
        target_slice = x[start:end]
        residual, action, reason, note, ema_info = _ujicache_residual_for_slot(
            _ujicache_slot(cache, key),
            target_slice,
            step_index,
            skip_streak,
            late_phase,
        )
        x[start:end] = target_slice + residual.to(x.device)
        slot_actions[key] = action
        slot_reasons[key] = reason
        slot_notes[key] = note
        slot_ema_info[key] = ema_info
    return slot_actions, slot_reasons, slot_notes, slot_ema_info, late_phase, skip_streak


def _ujicache_residual_for_slot(
    slot: dict[str, Any],
    target_slice: Any,
    step_index: int,
    skip_streak: int,
    late_phase: bool,
) -> tuple[Any, str, str | None, str | None, dict[str, Any]]:
    ema_info = _ujicache_ema_info(slot, step_index)
    previous = slot.get("previous_residual")
    if previous is None:
        raise RuntimeError("missing previous_residual")
    if getattr(previous, "shape", None) != getattr(target_slice, "shape", None):
        raise RuntimeError(
            f"residual shape mismatch residual={_shape(previous)} target={_shape(target_slice)}"
        )

    formula = STATE.ujicache_formula
    if formula == UJICACHE_FORMULA_TEACACHE:
        return previous.to(target_slice.device), "fallback", "formula", None, ema_info

    prediction_allowed = late_phase or skip_streak >= STATE.ujicache_apply_prediction_from_skip
    if not prediction_allowed:
        return previous.to(target_slice.device), "fallback", "streak", None, ema_info

    try:
        prediction_note = None
        if formula == UJICACHE_FORMULA_LINEAR:
            if STATE.ujicache_slope_ema_smoothing <= 0.0:
                prediction = _ujicache_predict_linear(slot, step_index, previous)
            else:
                prediction, prediction_note = _ujicache_predict_linear_ema(slot, step_index, previous)
        elif formula == UJICACHE_FORMULA_TAYLOR2:
            if STATE.ujicache_slope_ema_smoothing <= 0.0:
                prediction = _ujicache_predict_taylor2(slot, step_index, previous)
            else:
                prediction, prediction_note = _ujicache_predict_taylor2_ema(slot, step_index, previous)
        else:
            return previous.to(target_slice.device), "fallback", "formula", None, ema_info
        prediction = _ujicache_validate_prediction(prediction, previous, target_slice)
        return prediction.to(target_slice.device), "prediction", None, prediction_note, ema_info
    except _UjiCachePredictionFallback as exc:
        return previous.to(target_slice.device), "fallback", exc.reason, None, ema_info
    except Exception:
        return previous.to(target_slice.device), "fallback", "prediction_error", None, ema_info


class _UjiCachePredictionFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _ujicache_record_residual(slot: dict[str, Any], step_index: int, residual: Any) -> None:
    _ujicache_update_ema(slot, step_index, residual)
    history = slot.setdefault("residual_history", [])
    history.append(
        {
            "step_index": int(step_index),
            "residual": residual.detach(),
        }
    )
    if len(history) > 5:
        del history[:-5]


def _ujicache_update_ema(slot: dict[str, Any], step_index: int, residual: Any) -> None:
    history = slot.get("residual_history") or []
    if not history:
        return
    previous_item = history[-1]
    try:
        t_prev = float(previous_item["step_index"])
        t_now = float(step_index)
        dt = t_now - t_prev
        if dt <= 0.0:
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
                beta_a = STATE.ujicache_curve_ema_smoothing
                if acceleration_ema is None or getattr(acceleration_ema, "shape", None) != getattr(a_obs, "shape", None):
                    slot["acceleration_ema"] = a_obs.detach()
                else:
                    slot["acceleration_ema"] = (
                        beta_a * acceleration_ema.to(a_obs.device).float()
                        + (1.0 - beta_a) * a_obs
                    ).detach()

        velocity_ema = slot.get("velocity_ema")
        beta_v = STATE.ujicache_slope_ema_smoothing
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


def _ujicache_predict_linear(slot: dict[str, Any], step_index: int, previous: Any):
    history = _ujicache_residual_history(slot, 2)
    raw_prediction = _ujicache_lagrange_prediction(history[-2:], step_index)
    previous_f32 = previous.float()
    return previous_f32 + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32)


def _ujicache_predict_taylor2(slot: dict[str, Any], step_index: int, previous: Any):
    history = _ujicache_residual_history(slot, 3)
    linear_prediction = _ujicache_lagrange_prediction(history[-2:], step_index)
    quadratic_prediction = _ujicache_lagrange_prediction(history[-3:], step_index)
    curve = STATE.ujicache_taylor2_curve_strength
    raw_prediction = (1.0 - curve) * linear_prediction + curve * quadratic_prediction
    previous_f32 = previous.float()
    return previous_f32 + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32)


def _ujicache_predict_linear_ema(
    slot: dict[str, Any],
    step_index: int,
    previous: Any,
) -> tuple[Any, str | None]:
    dt_pred, velocity_ema = _ujicache_ema_velocity(slot, step_index, previous)
    previous_f32 = previous.float()
    raw_prediction = previous_f32 + dt_pred * velocity_ema
    return (
        previous_f32
        + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32),
        None,
    )


def _ujicache_predict_taylor2_ema(
    slot: dict[str, Any],
    step_index: int,
    previous: Any,
) -> tuple[Any, str | None]:
    dt_pred, velocity_ema = _ujicache_ema_velocity(slot, step_index, previous)
    previous_f32 = previous.float()
    linear_prediction = previous_f32 + dt_pred * velocity_ema
    acceleration_ema = slot.get("acceleration_ema")
    if acceleration_ema is None:
        raw_prediction = linear_prediction
        prediction_note = "taylor2_ema_without_acceleration"
    else:
        if getattr(acceleration_ema, "shape", None) != getattr(previous, "shape", None):
            raise _UjiCachePredictionFallback("shape_mismatch")
        curve_term = 0.5 * (dt_pred ** 2) * acceleration_ema.to(previous.device).float()
        raw_prediction = (
            linear_prediction
            + STATE.ujicache_taylor2_curve_strength * curve_term
        )
        prediction_note = None
    return (
        previous_f32
        + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32),
        prediction_note,
    )


def _ujicache_ema_velocity(
    slot: dict[str, Any],
    step_index: int,
    previous: Any,
) -> tuple[float, Any]:
    history = slot.get("residual_history") or []
    if not history:
        raise _UjiCachePredictionFallback("insufficient_history")
    latest = history[-1]
    velocity_ema = slot.get("velocity_ema")
    if velocity_ema is None:
        raise _UjiCachePredictionFallback("insufficient_ema_velocity")
    if getattr(velocity_ema, "shape", None) != getattr(previous, "shape", None):
        raise _UjiCachePredictionFallback("shape_mismatch")
    dt_pred = float(step_index) - float(latest["step_index"])
    return dt_pred, velocity_ema.to(previous.device).float()


def _ujicache_ema_info(slot: dict[str, Any], step_index: int) -> dict[str, Any]:
    history = slot.get("residual_history") or []
    dt_pred = None
    if history:
        try:
            dt_pred = float(step_index) - float(history[-1]["step_index"])
        except Exception:
            dt_pred = None
    return {
        "velocity_ready": slot.get("velocity_ema") is not None,
        "acceleration_ready": slot.get("acceleration_ema") is not None,
        "dt_pred": dt_pred,
    }


def _ujicache_residual_history(slot: dict[str, Any], count: int) -> list[dict[str, Any]]:
    history = slot.get("residual_history") or []
    if len(history) < count:
        raise _UjiCachePredictionFallback("insufficient_history")
    recent = history[-count:]
    reference_shape = getattr(recent[-1].get("residual"), "shape", None)
    for item in recent:
        if getattr(item.get("residual"), "shape", None) != reference_shape:
            raise _UjiCachePredictionFallback("shape_mismatch")
    return recent


def _ujicache_lagrange_prediction(history: list[dict[str, Any]], step_index: int):
    if not history:
        raise _UjiCachePredictionFallback("insufficient_history")
    times = [float(item["step_index"]) for item in history]
    target = float(step_index)
    result = None
    for i, item in enumerate(history):
        weight = 1.0
        for j, other_time in enumerate(times):
            if i == j:
                continue
            denom = times[i] - other_time
            if abs(denom) < 1e-6:
                raise _UjiCachePredictionFallback("duplicate_history_step")
            weight *= (target - other_time) / denom
        residual = item["residual"].float()
        weighted = residual * weight
        result = weighted if result is None else result + weighted
    if result is None:
        raise _UjiCachePredictionFallback("insufficient_history")
    return result


def _ujicache_validate_prediction(prediction: Any, previous: Any, target_slice: Any):
    import math
    import torch

    if getattr(prediction, "shape", None) != getattr(previous, "shape", None):
        raise _UjiCachePredictionFallback("shape_mismatch")
    if getattr(prediction, "shape", None) != getattr(target_slice, "shape", None):
        raise _UjiCachePredictionFallback("shape_mismatch")
    if not bool(torch.isfinite(prediction).all().item()):
        raise _UjiCachePredictionFallback("numeric_error")

    prediction_norm = float(torch.linalg.vector_norm(prediction.float()).item())
    previous_norm = float(torch.linalg.vector_norm(previous.float()).item())
    if not math.isfinite(prediction_norm) or not math.isfinite(previous_norm):
        raise _UjiCachePredictionFallback("numeric_error")
    if previous_norm > 0.0 and prediction_norm > previous_norm * UJICACHE_MAX_NORM_RATIO:
        raise _UjiCachePredictionFallback("norm_guard")
    try:
        return prediction.to(device=previous.device, dtype=previous.dtype)
    except Exception as exc:
        raise _UjiCachePredictionFallback("dtype_conversion") from exc


def _capture_calibration_pair(
    slot: dict[str, Any],
    slot_key: int,
    step_index: int,
    rel_l1: float | None,
    residual_slice: Any,
    timesteps: Any,
    start: int,
    end: int,
) -> None:
    if not STATE.calibration_capture_active():
        return
    import math

    from .calibration_capture import capture_pair

    t_now = _capture_timestep_value(timesteps, start, end)
    t_prev = slot.get("capture_prev_t")
    out_rel: float | None = None
    estimate: float | None = None
    try:
        if rel_l1 is not None and math.isfinite(float(rel_l1)):
            candidate = _cache_poly1d(float(rel_l1), _ujicache_coefficients())
            if math.isfinite(candidate):
                estimate = candidate
        previous = slot.get("previous_residual")
        if previous is not None and rel_l1 is not None:
            if getattr(previous, "shape", None) == getattr(residual_slice, "shape", None):
                prev_f32 = previous.detach().float()
                denom = prev_f32.abs().mean()
                if float(denom.item()) > 0.0:
                    out_tensor = (
                        (residual_slice.detach().float() - prev_f32).abs().mean() / denom
                    )
                    value = float(out_tensor.item())
                    if math.isfinite(value):
                        out_rel = value
    except Exception as exc:
        STATE.calibration_capture_errors += 1
        if STATE.hareskip_logged_calls < 12:
            STATE.hareskip_logged_calls += 1
            warning(f"calibration_capture_pair_failed reason={_short_error(exc)}")
    finally:
        slot["capture_prev_t"] = t_now
    capture_pair(slot_key, step_index, rel_l1, out_rel, estimate, t_now, t_prev)


def _capture_timestep_value(timesteps: Any, start: int, end: int) -> float | None:
    try:
        value = timesteps
        if hasattr(value, "ndim") and getattr(value, "ndim", 0) >= 1:
            sliced = value[start:end]
            if hasattr(sliced, "numel") and sliced.numel() > 0:
                value = sliced
        if hasattr(value, "detach"):
            return float(value.detach().flatten()[0].item())
        return float(value)
    except Exception:
        return None


def _dump_ujicache_residual(
    residual: Any,
    cond_or_uncond: list[int],
    batch_per_slot: int,
    step_index: int,
    timestep: Any,
    cache_device: Any,
) -> None:
    if not (
        STATE.tensor_dump_active()
        and STATE.dump_ujicache_residual
        and STATE.hareskip_enabled
    ):
        return
    from .tensor_dump import dump_tensor

    for slot_index, key in enumerate(cond_or_uncond):
        start = slot_index * batch_per_slot
        end = start + batch_per_slot
        local_call_index = STATE.tensor_dump_hareskip_local_call_index
        STATE.tensor_dump_hareskip_local_call_index += 1
        dump_tensor(
            "ujicache_residual",
            residual[start:end],
            logical_step_index=step_index,
            local_call_index=local_call_index,
            call_index=local_call_index,
            slot=int(key),
            decision="full",
            timestep_value=timestep,
            extra={
                "cache_device": str(cache_device),
                "hareskip_model_call": STATE.hareskip_model_calls,
                "formula": STATE.ujicache_formula,
                "slope_ema_smoothing": STATE.ujicache_slope_ema_smoothing,
                "curve_ema_smoothing": STATE.ujicache_curve_ema_smoothing,
            },
        )


def _hareskip_log_call(
    decision: str,
    step_index: int,
    progress: float,
    rels: dict[Any, float | None],
    reason: str | None,
    late_phase: bool,
    skip_streak: int,
    slot_actions: dict[int, str],
    slot_reasons: dict[int, str | None],
    slot_notes: dict[int, str | None],
    slot_ema_info: dict[int, dict[str, Any]],
) -> None:
    if not STATE.ujicache_verbose_trace and STATE.hareskip_logged_calls >= 12:
        return
    STATE.hareskip_logged_calls += 1
    rel_text = ",".join(
        f"{key}:{'None' if value is None else f'{value:.6f}'}"
        for key, value in sorted(rels.items())
    )
    action_text = ",".join(
        f"{key}:{slot_actions[key]}" for key in sorted(slot_actions)
    ) or "None"
    reason_text = ",".join(
        f"{key}:{slot_reasons[key]}" for key in sorted(slot_reasons) if slot_reasons[key]
    ) or (reason or "threshold")
    note_text = ",".join(
        f"{key}:{slot_notes[key]}" for key in sorted(slot_notes) if slot_notes[key]
    ) or "None"
    velocity_ready_text = ",".join(
        f"{key}:{bool(slot_ema_info[key].get('velocity_ready'))}"
        for key in sorted(slot_ema_info)
    ) or "None"
    acceleration_ready_text = ",".join(
        f"{key}:{bool(slot_ema_info[key].get('acceleration_ready'))}"
        for key in sorted(slot_ema_info)
    ) or "None"
    dt_pred_parts = []
    for key in sorted(slot_ema_info):
        value = slot_ema_info[key].get("dt_pred")
        dt_pred_parts.append(
            f"{key}:None" if value is None else f"{key}:{float(value):.3f}"
        )
    dt_pred_text = ",".join(dt_pred_parts) or "None"
    info(
        "hareskip_call="
        f"call={STATE.hareskip_model_calls} step={step_index} "
        f"progress={progress:.3f} late={late_phase} streak={skip_streak} "
        f"decision={decision} reason={reason_text} "
        f"formula={STATE.ujicache_formula} action={action_text} rel_l1={rel_text} "
        f"threshold={STATE.ujicache_threshold:.4f} "
        f"use_prediction_after={STATE.ujicache_use_prediction_after_progress:.2f} "
        f"apply_from_skip={STATE.ujicache_apply_prediction_from_skip} "
        f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
        f"taylor2_curve_strength={STATE.ujicache_taylor2_curve_strength:.2f} "
        f"slope_ema_smoothing={STATE.ujicache_slope_ema_smoothing:.2f} "
        f"curve_ema_smoothing={STATE.ujicache_curve_ema_smoothing:.2f} "
        f"prediction_note={note_text} "
        f"ema_velocity_ready={velocity_ready_text} "
        f"ema_acceleration_ready={acceleration_ready_text} "
        f"dt_pred={dt_pred_text} "
        f"dry_run={STATE.ujicache_dry_run}"
    )


def _runtime_num_blocks() -> int | None:
    blocks = _runtime_blocks()
    if blocks is None:
        return None
    try:
        return len(blocks)
    except Exception:
        return None


def _runtime_blocks() -> Any | None:
    try:
        from modules import shared

        sd_model = getattr(shared, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = getattr(forge_objects, "unet", None)
        model = getattr(unet, "model", None)
        diffusion_model = getattr(model, "diffusion_model", model)
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            return None
        return blocks
    except Exception:
        return None


def _shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    try:
        return "x".join(str(part) for part in shape)
    except Exception:
        return str(shape)


def _short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    if len(text) > 160:
        return text[:157] + "..."
    return text
