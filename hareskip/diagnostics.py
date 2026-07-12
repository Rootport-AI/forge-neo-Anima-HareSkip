from __future__ import annotations

from typing import Any

from . import __version__
from .forge_introspection import (
    attention_info,
    cond_info,
    model_structure_info,
    processing_info,
)
from .logging import info, warning
from .model_detect import ModelDetection
from .constants import MODE_HARESKIP
from .state import MODE_DIAGNOSE, MODE_OFF, STATE
from .timing import timing_summary


def log_generation_start(p: Any) -> None:
    if not STATE.active():
        return

    detection = STATE.model_detection
    proc = processing_info(p)
    info(
        f"version={__version__} enabled={STATE.enabled} debug_mode={STATE.mode} "
        f"status={STATE.status} source={STATE.generation_start_source}"
    )

    if isinstance(detection, ModelDetection):
        info(
            "model_supported="
            f"{detection.supported} confidence={detection.confidence} "
            f"family={detection.family} reason={detection.reason}"
        )
        if STATE.verbose_diagnose_log:
            info(f"model_evidence={detection.evidence}")
        if not detection.supported and _should_warn_unsupported_model():
            key = detection.key or detection.reason
            if key not in STATE.warned_model_keys:
                warning(f"unsupported model: {detection.reason}")
                STATE.warned_model_keys.add(key)
            STATE.status = "unsupported"

    info(
        "sampler="
        f"{proc['sampler']} scheduler={proc['scheduler']} steps={proc['steps']} "
        f"cfg={proc['cfg_scale']} resolution={proc['width']}x{proc['height']}"
    )
    if STATE.generation_steps is None:
        try:
            STATE.generation_steps = int(proc["steps"])
        except Exception:
            STATE.generation_steps = None

    if STATE.mode == MODE_DIAGNOSE and STATE.verbose_diagnose_log:
        log_attention_trace()
        log_model_structure_trace()
    log_experiment_snapshot()
    STATE.generation_logged = True


def log_experiment_snapshot() -> None:
    if STATE.hareskip_enabled:
        info(
            "hare_config="
            f"mode={STATE.hareskip_mode} "
            f"aggressiveness={STATE.hareskip_aggressiveness:.2f} "
            f"skip_seed_offset={STATE.hareskip_skip_seed_offset} "
            f"probability_model={STATE.hareskip_probability_model} "
            f"dry_run={STATE.hareskip_dry_run}"
        )
        info(
            "tea_config="
            f"preset={STATE.tea_preset} "
            f"threshold={STATE.tea_threshold:.4f} "
            f"progress={STATE.tea_start_percent:.2f}..{STATE.tea_end_percent:.2f} "
            f"source={STATE.tea_modulated_source} "
            f"coefficient_profile={STATE.tea_coefficient_profile} "
            f"max_skip_streak={STATE.tea_max_skip_streak} "
            f"force_full_interval={STATE.tea_force_full_interval}"
        )
        info(
            "resrefine_config="
            f"formula={STATE.resrefine_formula} "
            f"use_prediction_after={STATE.resrefine_use_prediction_after_progress:.2f} "
            f"apply_from_skip={STATE.resrefine_apply_prediction_from_skip} "
            f"prediction_strength={STATE.resrefine_prediction_strength:.2f} "
            f"taylor2_curve_strength={STATE.resrefine_taylor2_curve_strength:.2f} "
            f"slope_ema_smoothing={STATE.resrefine_slope_ema_smoothing:.2f} "
            f"curve_ema_smoothing={STATE.resrefine_curve_ema_smoothing:.2f} "
            f"cache_device={STATE.resrefine_cache_device}"
        )
    if STATE.auto_teacache_active:
        info(
            "auto_tea_config="
            f"active=True row_count={STATE.auto_teacache_row_count} "
            f"original_n_iter={STATE.auto_teacache_original_n_iter}"
        )


def log_attention_trace() -> None:
    data = attention_info()
    info(f"attention_backend={data.get('attention_backend')}")
    info(
        "attention_available="
        f"sage={data.get('sage_available')} flash={data.get('flash_available')} "
        f"xformers={data.get('xformers_available')} pytorch={data.get('pytorch_available')} "
        f"anima_attention_path={data.get('anima_attention_path')}"
    )
    if "attention_error" in data:
        warning(f"attention_trace_error={data['attention_error']}")
    if "anima_error" in data:
        warning(f"anima_trace_error={data['anima_error']}")


def log_model_structure_trace() -> None:
    data = model_structure_info(_current_sd_model())
    if not data:
        info("model_structure=unavailable")
        return
    info("model_structure=" + " ".join(f"{key}={_fmt(value)}" for key, value in data.items()))


def log_cond_trace(params: Any) -> None:
    if not STATE.active() or STATE.mode == MODE_OFF:
        return
    if STATE.cond_trace_logged and not STATE.verbose_diagnose_log:
        return
    data = cond_info(params)
    info(
        "cfg="
        f"{data.get('cfg_scale')} uncond_present={not data.get('text_uncond_is_none')} "
        f"sampling_step={data.get('sampling_step')}/{data.get('total_sampling_steps')} "
        f"denoiser_step={data.get('denoiser_step')}/{data.get('denoiser_total_steps')}"
    )
    info(
        "cond_or_uncond="
        f"{data.get('cond_or_uncond')} cond_indices={data.get('cond_indices')} "
        f"uncond_indices={data.get('uncond_indices')} "
        f"stage={data.get('transformer_options_stage')}"
    )
    info(
        "cond_shapes="
        f"x={data.get('x_shape')} sigma={data.get('sigma_shape')} "
        f"text_cond_type={data.get('text_cond_type')} "
        f"text_uncond_type={data.get('text_uncond_type')}"
    )
    STATE.cond_trace_logged = True


def log_timing_summary() -> None:
    if not STATE.active():
        return

    should_print_summary = STATE.hareskip_enabled
    if STATE.print_timing_log:
        data = timing_summary()
        info(
            "denoiser_calls="
            f"{data['denoiser_calls']} avg_step_time={_seconds(data['avg_step_time'])} "
            f"total_sampling_time={_seconds(data['total_sampling_time'])} "
            f"min_step_time={_seconds(data['min_step_time'])} "
            f"max_step_time={_seconds(data['max_step_time'])} status={STATE.status}"
        )
    elif not should_print_summary:
        return

    if STATE.hareskip_enabled:
        active = _is_patch_active("hareskip")
        total_decisions = STATE.hareskip_full_calcs + STATE.hareskip_skips
        skip_rate = (
            STATE.hareskip_skips / total_decisions
            if total_decisions
            else 0.0
        )
        pattern = STATE.hareskip_pattern
        pattern_fields = ""
        if STATE.hareskip_mode == MODE_HARESKIP and pattern is not None:
            pattern_fields = (
                " pattern_skip_count="
                f"{pattern.skip_count} "
                f"pattern_skipped_steps={_fmt_step_ranges(pattern.skipped_steps)} "
                f"pattern_skip_seed={pattern.skip_seed}"
            )
        info(
            "hareskip_summary="
            f"mode={STATE.hareskip_mode} "
            f"model_calls={STATE.hareskip_model_calls} "
            f"full_calcs={STATE.hareskip_full_calcs} "
            f"skips={STATE.hareskip_skips} "
            f"prediction_used={STATE.resrefine_prediction_used} "
            f"fallback_used={STATE.hareskip_fallback_used} "
            f"dry_run_predictions={STATE.hareskip_dry_run_predictions} "
            f"skip_rate={skip_rate:.3f} "
            # STATE.hareskip_skipped_steps is stored 0-based; convert to 1-based
            # for display so it matches pattern_skipped_steps (already 1-based)
            # and Forge's Sampling steps numbering.
            f"skipped_steps={_fmt_step_ranges([s + 1 for s in STATE.hareskip_skipped_steps])}"
            f"{pattern_fields} "
            f"first_full_calcs={STATE.hareskip_first_full_calcs} "
            f"forced_full_calcs={STATE.hareskip_forced_full_calcs} "
            f"fallbacks={STATE.hareskip_fallbacks} "
            f"errors={STATE.hareskip_errors} "
            f"fallback_reasons={_fmt_counts(STATE.hareskip_fallback_reasons)} "
            f"num_blocks={STATE.hareskip_num_blocks} "
            f"active={active} "
            f"formula={STATE.resrefine_formula} "
            f"slope_ema_smoothing={STATE.resrefine_slope_ema_smoothing:.2f} "
            f"curve_ema_smoothing={STATE.resrefine_curve_ema_smoothing:.2f} "
            f"dry_run={STATE.hareskip_dry_run} "
            f"capture_pairs={STATE.calibration_capture_records} "
            f"unavailable_reason={_fmt(STATE.hareskip_unavailable_reason)}"
        )
        # Zero-execution detector: the denoiser ran (sampling happened) but our
        # patched Anima.forward was never invoked. The most likely cause is
        # another extension (e.g. a legacy UjiCache) that also monkey-patches
        # Anima.forward and overwrote our wrapper with its own saved original,
        # silently removing HareSkip from the call chain.
        if (
            STATE.hareskip_enabled
            and STATE.denoiser_calls > 0
            and STATE.hareskip_model_calls == 0
        ):
            warning(
                "hareskip_patch_never_ran "
                f"denoiser_calls={STATE.denoiser_calls} model_calls=0; "
                "the patched Anima.forward was never called — suspected "
                "conflict with another extension patching Anima.forward "
                "(e.g. legacy UjiCache)"
            )


def _should_warn_unsupported_model() -> bool:
    return STATE.hareskip_enabled


def _seconds(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}s"


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def _fmt_counts(value: dict[str, int]) -> str:
    if not value:
        return "None"
    return ",".join(f"{key}:{value[key]}" for key in sorted(value))


def _fmt_step_ranges(steps: list[int]) -> str:
    if not steps:
        return "None"
    ranges: list[str] = []
    start = previous = int(steps[0])
    for raw_step in steps[1:]:
        step = int(raw_step)
        if step == previous + 1:
            previous = step
            continue
        ranges.append(_fmt_step_range(start, previous))
        start = previous = step
    ranges.append(_fmt_step_range(start, previous))
    return ",".join(ranges)


def _fmt_step_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _current_sd_model() -> Any:
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:
        return None


def _is_patch_active(kind: str) -> Any:
    # Report the truthful active state: whether OUR wrapper is still installed
    # (is_patch_installed verifies wrapper identity), not merely whether we
    # registered a patch. This makes active= flip to False when another
    # extension has clobbered our Anima.forward wrapper.
    try:
        from .patcher import is_patch_installed

        return is_patch_installed(kind)
    except Exception:
        return "unknown"
