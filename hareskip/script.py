from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .auto_ujicache import (
    AutoUjiCsvError,
    apply_auto_ujicache_row_to_state,
    parse_auto_ujicache_csv,
)
from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import error, exception, info, warning
from .model_detect import detect_model
from .patcher import apply_patch, remove_patch
from .state import (
    MODE_OFF,
    MODES,
    STATE,
    UJICACHE_CACHE_DEVICE_CUDA,
    UJICACHE_CACHE_DEVICES,
    UJICACHE_COEFFICIENT_PROFILES,
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TAYLOR2,
    UJICACHE_FORMULA_TEACACHE,
    UJICACHE_FORMULAS,
    UJICACHE_PRESET_CUSTOM,
    UJICACHE_PRESETS,
    UJICACHE_PROFILE_DEFAULT,
    ujicache_coefficients_for_profile,
    ujicache_expected_shift_for_profile,
    ujicache_window_for_profile,
)
from .timing import start_sampling

register_callbacks()


class Script(scripts.Script):
    def title(self):
        return "HareSkip"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("HareSkip", open=False, elem_id="hareskip-panel"):
            enabled = gr.Checkbox(
                label="Enable HareSkip",
                value=_default_option("hareskip_enable", False),
                elem_id="hareskip-enable",
            )
            ujicache_preset = gr.Dropdown(
                label="UjiCache preset",
                choices=UJICACHE_PRESETS,
                value=UJICACHE_PRESET_CUSTOM,
                elem_id="ujicache-preset",
            )
            ujicache_threshold = gr.Slider(
                label="Rel L1 threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.005,
                value=0.07,
                elem_id="ujicache-threshold",
            )
            ujicache_coefficient_profile = gr.Dropdown(
                label="Coefficient profile",
                choices=UJICACHE_COEFFICIENT_PROFILES,
                value=UJICACHE_PROFILE_DEFAULT,
                elem_id="ujicache-coefficient-profile",
            )
            p_anima_x = gr.Markdown(
                value=_format_p_anima_x(UJICACHE_PROFILE_DEFAULT),
                elem_id="ujicache-p-anima-x",
            )
            gr.HTML(
                "<style>"
                "#ujicache-p-anima-x code {"
                " background: none;"
                " border: none;"
                " padding: 0;"
                " color: var(--body-text-color-subdued);"
                " }"
                "</style>",
                elem_id="ujicache-p-anima-x-style",
            )
            ujicache_start_percent = gr.Slider(
                label="Start progress",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.48,
                elem_id="ujicache-start-percent",
            )
            ujicache_end_percent = gr.Slider(
                label="End progress",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.76,
                elem_id="ujicache-end-percent",
            )
            ujicache_max_skip_streak = gr.Slider(
                label="Max skip streak (0 = off)",
                minimum=0,
                maximum=64,
                step=1,
                value=0,
                elem_id="ujicache-max-skip-streak",
            )
            ujicache_force_full_interval = gr.Slider(
                label="Force full interval (0 = off)",
                minimum=0,
                maximum=64,
                step=1,
                value=0,
                elem_id="ujicache-force-full-interval",
            )
            ujicache_formula = gr.Dropdown(
                label="Prediction formula",
                choices=UJICACHE_FORMULAS,
                value=UJICACHE_FORMULA_TEACACHE,
                elem_id="ujicache-formula",
            )
            ujicache_use_prediction_after_progress = gr.Slider(
                label="Use prediction after progress",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                interactive=False,
                elem_id="ujicache-use-prediction-after-progress",
            )
            ujicache_apply_prediction_from_skip = gr.Slider(
                label="Apply prediction from skip #",
                minimum=1,
                maximum=3,
                step=1,
                value=2,
                interactive=False,
                elem_id="ujicache-apply-prediction-from-skip",
            )
            ujicache_prediction_strength = gr.Slider(
                label="Prediction strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.50,
                interactive=False,
                elem_id="ujicache-prediction-strength",
            )
            ujicache_taylor2_curve_strength = gr.Slider(
                label="Taylor2 curve strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.25,
                interactive=False,
                elem_id="ujicache-taylor2-curve-strength",
            )
            ujicache_slope_ema_smoothing = gr.Slider(
                label="Slope EMA Smoothing",
                minimum=0.0,
                maximum=0.99,
                step=0.01,
                value=0.0,
                interactive=False,
                elem_id="ujicache-slope-ema-smoothing",
            )
            ujicache_curve_ema_smoothing = gr.Slider(
                label="Curve EMA Smoothing",
                minimum=0.0,
                maximum=0.99,
                step=0.01,
                value=0.0,
                interactive=False,
                elem_id="ujicache-curve-ema-smoothing",
            )
            with gr.Accordion(
                "Auto Uji mode",
                open=False,
                elem_id="ujicache-auto-uji-panel",
            ):
                auto_ujicache_enabled = gr.Checkbox(
                    label="Enable Auto Uji mode",
                    value=False,
                    elem_id="ujicache-auto-uji-enable",
                )
                auto_ujicache_csv = gr.Textbox(
                    label="Auto Uji CSV",
                    lines=6,
                    elem_id="ujicache-auto-uji-csv",
                )
            with gr.Accordion("Debug log mode", open=False, elem_id="hareskip-debug-panel"):
                debug_log_enabled = gr.Checkbox(
                    label="Enable debug log mode",
                    value=_default_option("hareskip_debug_log_enable", False),
                    elem_id="ujicache-debug-enable",
                )
                mode = gr.Dropdown(
                    label="Debug log",
                    choices=MODES,
                    value=_default_mode_option(),
                    elem_id="ujicache-mode",
                )
                dump_ujicache_residual = gr.Checkbox(
                    label="Dump UjiCache residual",
                    value=False,
                    elem_id="ujicache-dump-residual",
                )
                capture_calibration_pairs = gr.Checkbox(
                    label="Capture calibration pairs",
                    value=False,
                    info=(
                        "Force full calculation on every step and write per-step "
                        "(rel_l1, out_rel) pairs plus run conditions to "
                        "calibration_pairs.jsonl in the debug run folder."
                    ),
                    elem_id="ujicache-capture-calibration-pairs",
                )
                gr.HTML(
                    '<div style="border-top: 3px solid var(--block-border-color, #4b5563); margin: 0.85rem 0 0.7rem;"></div>',
                    elem_id="ujicache-debug-dump-divider",
                )
                print_timing_log = gr.Checkbox(
                    label="Print timing log",
                    value=_default_option("hareskip_print_timing_log", True),
                    elem_id="ujicache-print-timing-log",
                )
                verbose_diagnose_log = gr.Checkbox(
                    label="Verbose diagnose log",
                    value=_default_option("hareskip_verbose_diagnose_log", False),
                    elem_id="ujicache-verbose-diagnose-log",
                )
            ujicache_cache_device = gr.Radio(
                label="Cache device",
                choices=UJICACHE_CACHE_DEVICES,
                value=UJICACHE_CACHE_DEVICE_CUDA,
                elem_id="ujicache-cache-device",
            )
            ujicache_dry_run = gr.Checkbox(
                label="Dry run",
                value=False,
                elem_id="ujicache-dry-run",
            )
            ujicache_verbose_trace = gr.Checkbox(
                label="Verbose UjiCache trace",
                value=False,
                elem_id="ujicache-verbose-trace",
            )

            ujicache_formula.change(
                fn=_ujicache_prediction_control_updates,
                inputs=[ujicache_formula, ujicache_slope_ema_smoothing],
                outputs=[
                    ujicache_use_prediction_after_progress,
                    ujicache_apply_prediction_from_skip,
                    ujicache_prediction_strength,
                    ujicache_taylor2_curve_strength,
                    ujicache_slope_ema_smoothing,
                    ujicache_curve_ema_smoothing,
                ],
            )
            ujicache_slope_ema_smoothing.change(
                fn=_ujicache_prediction_control_updates,
                inputs=[ujicache_formula, ujicache_slope_ema_smoothing],
                outputs=[
                    ujicache_use_prediction_after_progress,
                    ujicache_apply_prediction_from_skip,
                    ujicache_prediction_strength,
                    ujicache_taylor2_curve_strength,
                    ujicache_slope_ema_smoothing,
                    ujicache_curve_ema_smoothing,
                ],
            )
            enabled.change(
                fn=_ujicache_enable_updates,
                inputs=[enabled],
                outputs=[auto_ujicache_enabled],
            )
            ujicache_coefficient_profile.change(
                fn=_ujicache_profile_change_updates,
                inputs=[ujicache_coefficient_profile],
                outputs=[ujicache_start_percent, ujicache_end_percent, p_anima_x],
            )

        return [
            enabled,
            debug_log_enabled,
            mode,
            print_timing_log,
            verbose_diagnose_log,
            dump_ujicache_residual,
            ujicache_preset,
            ujicache_threshold,
            ujicache_start_percent,
            ujicache_end_percent,
            ujicache_formula,
            ujicache_use_prediction_after_progress,
            ujicache_apply_prediction_from_skip,
            ujicache_prediction_strength,
            ujicache_taylor2_curve_strength,
            ujicache_slope_ema_smoothing,
            ujicache_curve_ema_smoothing,
            ujicache_cache_device,
            ujicache_coefficient_profile,
            ujicache_max_skip_streak,
            ujicache_force_full_interval,
            ujicache_dry_run,
            ujicache_verbose_trace,
            auto_ujicache_enabled,
            auto_ujicache_csv,
            capture_calibration_pairs,
        ]

    def before_process(self, p, *script_args):
        try:
            _prepare_auto_ujicache_run(p, script_args)
        except AutoUjiCsvError as exc:
            STATE.auto_ujicache_parse_error = str(exc)
            STATE.set_error(f"auto uji csv error: {exc}")
            error(f"auto_uji_csv_error {exc}")
            raise RuntimeError(f"Auto Uji CSV error: {exc}") from exc
        except Exception as exc:
            STATE.set_error(f"auto uji setup failed: {exc}")
            exception("auto uji setup failed")
            raise

    def process(self, p, *script_args):
        try:
            _apply_auto_ujicache_seed_template(p)
        except Exception as exc:
            STATE.set_error(f"auto uji seed setup failed: {exc}")
            exception("auto uji seed setup failed")
            raise

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        try:
            _begin_generation(p, script_args, "process_before_every_sampling")
        except Exception as exc:
            STATE.set_error(f"process_before_every_sampling failed: {exc}")
            exception("process_before_every_sampling failed")

    def postprocess(self, p, processed, *script_args):
        try:
            log_timing_summary()
        except Exception as exc:
            STATE.set_error(f"postprocess failed: {exc}")
            exception("postprocess failed")
        try:
            from .tensor_dump import flush_stats

            flush_stats()
        except Exception as exc:
            STATE.tensor_dump_errors += 1
            warning(f"tensor_dump_flush_failed reason={exc}")
            exception("tensor dump flush failed")
        try:
            from .calibration_capture import log_capture_summary

            log_capture_summary()
        except Exception as exc:
            STATE.calibration_capture_errors += 1
            warning(f"calibration_capture_summary_failed reason={exc}")
        _finish_auto_ujicache_run(p)


_AUTO_HARESKIP_P_ATTRS = (
    "_hareskip_auto_rows",
    "_hareskip_auto_original_n_iter",
    "_hareskip_auto_original_batch_size",
    "_hareskip_auto_seed_template_size",
    "_hareskip_auto_seed_template_ready",
    "_hareskip_auto_logged_row_index",
    "_hareskip_auto_iteration_counter",
)


def _prepare_auto_ujicache_run(p, script_args) -> None:
    _apply_ui_args(script_args)
    _clear_auto_ujicache_p_attrs(p)
    STATE.auto_ujicache_active = False
    STATE.auto_ujicache_row_index = None
    STATE.auto_ujicache_row_name = None
    STATE.auto_ujicache_row_count = 0
    STATE.auto_ujicache_original_n_iter = 1
    STATE.auto_ujicache_parse_error = None

    if not (STATE.enabled and STATE.auto_ujicache_enabled):
        return

    result = parse_auto_ujicache_csv(STATE.auto_ujicache_csv)
    for message in result.warnings:
        warning(f"auto_uji_csv_warning {message}")

    rows = result.rows
    original_n_iter = _positive_int(getattr(p, "n_iter", 1), 1)
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    total_n_iter = len(rows) * original_n_iter

    setattr(p, "_hareskip_auto_rows", rows)
    setattr(p, "_hareskip_auto_original_n_iter", original_n_iter)
    setattr(p, "_hareskip_auto_original_batch_size", batch_size)
    setattr(p, "n_iter", total_n_iter)

    STATE.auto_ujicache_active = True
    STATE.auto_ujicache_row_count = len(rows)
    STATE.auto_ujicache_original_n_iter = original_n_iter

    info(
        "auto_uji_prepare "
        f"rows={len(rows)} original_n_iter={original_n_iter} "
        f"batch_size={batch_size} total_n_iter={total_n_iter}"
    )


def _apply_auto_ujicache_seed_template(p) -> None:
    rows = getattr(p, "_hareskip_auto_rows", None)
    if not rows:
        return

    row_count = len(rows)
    original_n_iter = _positive_int(
        getattr(p, "_hareskip_auto_original_n_iter", 1),
        1,
    )
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    template_size = original_n_iter * batch_size
    setattr(p, "_hareskip_auto_seed_template_size", template_size)

    raw_seed_values = getattr(p, "all_seeds", None)
    if not raw_seed_values and _safe_int(getattr(p, "seed", -1), -1) < 0:
        return

    was_ready = bool(getattr(p, "_hareskip_auto_seed_template_ready", False))
    seed_template = _seed_template(
        raw_seed_values,
        getattr(p, "seed", 0),
        template_size,
    )
    setattr(p, "all_seeds", seed_template * row_count)

    if hasattr(p, "all_subseeds"):
        subseed_template = _seed_template(
            getattr(p, "all_subseeds", None),
            getattr(p, "subseed", 0),
            template_size,
        )
        setattr(p, "all_subseeds", subseed_template * row_count)

    setattr(p, "_hareskip_auto_seed_template_ready", True)
    if not was_ready:
        info(
            "auto_uji_seed_template "
            f"rows={row_count} template_size={template_size} seeds={_seed_label(seed_template)}"
        )


def _apply_auto_ujicache_row_if_needed(p) -> None:
    rows = getattr(p, "_hareskip_auto_rows", None)
    if not rows or not STATE.auto_ujicache_active or not STATE.hareskip_enabled:
        return

    original_n_iter = _positive_int(
        getattr(p, "_hareskip_auto_original_n_iter", 1),
        1,
    )
    iteration = _current_auto_ujicache_iteration(p)
    row_index = max(0, min(len(rows) - 1, iteration // original_n_iter))
    repeat_index = (iteration % original_n_iter) + 1
    row = rows[row_index]

    apply_auto_ujicache_row_to_state(row)
    STATE.auto_ujicache_row_index = row.index
    STATE.auto_ujicache_row_name = row.name

    logged_row_index = getattr(p, "_hareskip_auto_logged_row_index", None)
    if logged_row_index != row_index:
        setattr(p, "_hareskip_auto_logged_row_index", row_index)
        info(
            "auto_uji_row_start "
            f"index={row_index + 1}/{len(rows)} name={row.name} "
            f"repeat={repeat_index}/{original_n_iter} "
            f"threshold={STATE.ujicache_threshold:.4f} "
            f"formula={STATE.ujicache_formula} "
            f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
            f"batch_size={_positive_int(getattr(p, 'batch_size', 1), 1)} "
            f"seeds={_row_seed_label(p, row_index)}"
        )


def _finish_auto_ujicache_run(p) -> None:
    _clear_auto_ujicache_p_attrs(p)
    STATE.auto_ujicache_active = False
    STATE.auto_ujicache_row_index = None
    STATE.auto_ujicache_row_name = None
    STATE.auto_ujicache_row_count = 0
    STATE.auto_ujicache_original_n_iter = 1


def _clear_auto_ujicache_p_attrs(p) -> None:
    for attr in _AUTO_HARESKIP_P_ATTRS:
        try:
            if hasattr(p, attr):
                delattr(p, attr)
        except Exception:
            setattr(p, attr, None)


def _current_auto_ujicache_iteration(p) -> int:
    value = getattr(p, "iteration", None)
    if value is not None:
        try:
            return max(0, int(value))
        except Exception:
            pass
    counter = _positive_int(getattr(p, "_hareskip_auto_iteration_counter", 0), 0)
    setattr(p, "_hareskip_auto_iteration_counter", counter + 1)
    return counter


def _row_seed_label(p, row_index: int) -> str:
    template_size = _positive_int(
        getattr(p, "_hareskip_auto_seed_template_size", 0),
        0,
    )
    if template_size <= 0:
        original_n_iter = _positive_int(
            getattr(p, "_hareskip_auto_original_n_iter", 1),
            1,
        )
        template_size = original_n_iter * _positive_int(getattr(p, "batch_size", 1), 1)
    try:
        seeds = list(getattr(p, "all_seeds", []) or [])
    except Exception:
        seeds = []
    start = row_index * template_size
    return _seed_label(seeds[start : start + template_size])


def _seed_template(values, base_seed, size: int) -> list[int]:
    try:
        template = list(values or [])[:size]
    except Exception:
        template = []
    if len(template) >= size:
        return template

    if template:
        base = _safe_int(template[0], 0)
    else:
        base = _safe_int(base_seed, 0)
        if base < 0:
            base = 0

    while len(template) < size:
        template.append(base + len(template))
    return template


def _seed_label(seeds) -> str:
    values = list(seeds or [])
    if not values:
        return "None"
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}..{values[-1]}"


def _positive_int(value, minimum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, number)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


_EXPECTED_UI_ARG_COUNT = 26
_ui_arg_count_warned = False


def _apply_ui_args(script_args) -> None:
    global _ui_arg_count_warned
    if len(script_args) >= _EXPECTED_UI_ARG_COUNT:
        STATE.apply_options(*script_args[:_EXPECTED_UI_ARG_COUNT])
        return
    if script_args and not _ui_arg_count_warned:
        _ui_arg_count_warned = True
        warning(
            "ui_args_unexpected "
            f"count={len(script_args)} expected={_EXPECTED_UI_ARG_COUNT}; "
            "falling back to shared settings"
        )
    STATE.refresh_settings()


def _begin_generation(p, script_args, source: str) -> None:
    _apply_ui_args(script_args)
    _apply_auto_ujicache_seed_template(p)
    _apply_auto_ujicache_row_if_needed(p)
    if not STATE.active():
        _remove_generation_patches()
        return

    start_sampling(source)
    try:
        STATE.generation_steps = int(getattr(p, "steps", 0) or 0) or None
    except Exception:
        STATE.generation_steps = None

    try:
        from modules import shared

        STATE.model_detection = detect_model(getattr(shared, "sd_model", None))
    except Exception:
        if STATE.model_detection is None:
            raise

    if not getattr(STATE.model_detection, "supported", False) and _requires_supported_model():
        _remove_generation_patches()
        log_generation_start(p)
        return

    if _tensor_dump_will_save():
        try:
            from .tensor_dump import initialize_run_if_needed

            initialize_run_if_needed(p)
        except Exception as exc:
            STATE.tensor_dump_errors += 1
            warning(f"tensor_dump_initialize_failed reason={exc}")
        if STATE.calibration_capture_active():
            try:
                from .calibration_capture import initialize_for_generation

                initialize_for_generation(p)
            except Exception as exc:
                STATE.calibration_capture_errors += 1
                warning(f"calibration_capture_initialize_failed reason={exc}")

    _configure_generation_patches()
    _check_profile_shift_match()
    _apply_infotext_metadata(p)
    log_generation_start(p)


def _apply_infotext_metadata(p) -> None:
    if not STATE.hareskip_enabled:
        return
    try:
        params = getattr(p, "extra_generation_params", None)
        if not isinstance(params, dict):
            params = {}
            setattr(p, "extra_generation_params", params)
        _clear_legacy_ujicache_metadata(params)
        params["Uji enabled"] = True
        params["Uji formula"] = STATE.ujicache_formula
        params["Uji threshold"] = f"{STATE.ujicache_threshold:.4f}"
        params["Uji progress"] = (
            f"{STATE.ujicache_start_percent:.2f}..{STATE.ujicache_end_percent:.2f}"
        )
        params["Uji use_prediction_after_progress"] = (
            f"{STATE.ujicache_use_prediction_after_progress:.2f}"
        )
        params["Uji apply_prediction_from_skip"] = STATE.ujicache_apply_prediction_from_skip
        params["Uji prediction_strength"] = f"{STATE.ujicache_prediction_strength:.2f}"
        params["Uji taylor2_curve_strength"] = (
            f"{STATE.ujicache_taylor2_curve_strength:.2f}"
        )
        params["Uji slope_ema_smoothing"] = f"{STATE.ujicache_slope_ema_smoothing:.2f}"
        params["Uji curve_ema_smoothing"] = f"{STATE.ujicache_curve_ema_smoothing:.2f}"
        params["Uji modulated_source"] = STATE.ujicache_modulated_source
        params["Uji coefficient_profile"] = STATE.ujicache_coefficient_profile
        params["Uji max_skip_streak"] = STATE.ujicache_max_skip_streak
        params["Uji force_full_interval"] = STATE.ujicache_force_full_interval
        shift_value = _model_sampling_shift()
        if shift_value is not None:
            params["Uji shift"] = f"{shift_value:.2f}"
        if STATE.capture_calibration_pairs:
            params["Uji capture_pairs"] = True
        if STATE.auto_ujicache_active and STATE.auto_ujicache_row_index is not None:
            if STATE.auto_ujicache_row_count > 0:
                params["Uji auto_row_index"] = (
                    f"{STATE.auto_ujicache_row_index}/{STATE.auto_ujicache_row_count}"
                )
            else:
                params["Uji auto_row_index"] = STATE.auto_ujicache_row_index
            params["Uji auto_row_name"] = STATE.auto_ujicache_row_name or ""
    except Exception as exc:
        warning(f"hareskip_metadata_failed reason={exc}")


def _clear_legacy_ujicache_metadata(params: dict) -> None:
    for key in (
        "UjiCache enabled",
        "UjiCache formula",
        "UjiCache use_prediction_after_progress",
        "UjiCache apply_prediction_from_skip",
        "UjiCache prediction_strength",
        "UjiCache taylor2_curve_strength",
        "UjiCache slope_ema_smoothing",
        "UjiCache curve_ema_smoothing",
        "Uji auto_row_index",
        "Uji auto_row_name",
    ):
        params.pop(key, None)


def _configure_generation_patches() -> None:
    if STATE.hareskip_enabled:
        result = apply_patch("hareskip")
        if not result.ok:
            STATE.hareskip_unavailable_reason = result.message
            warning(f"hareskip_patch_unavailable reason={result.message}")
    else:
        remove_patch("hareskip")

    if (
        STATE.tensor_dump_active()
        and STATE.dump_ujicache_residual
        and not STATE.hareskip_enabled
        and "ujicache_residual_requires_ujicache" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("ujicache_residual_requires_ujicache")
        warning("tensor_dump_hareskip_residual_inactive reason=hareskip_disabled")

    if (
        STATE.debug_log_enabled
        and STATE.capture_calibration_pairs
        and not STATE.hareskip_enabled
        and "calibration_capture_requires_ujicache" not in STATE.calibration_capture_warned_reasons
    ):
        STATE.calibration_capture_warned_reasons.add("calibration_capture_requires_ujicache")
        warning("calibration_capture_inactive reason=hareskip_disabled")


def _tensor_dump_will_save() -> bool:
    return bool(
        STATE.tensor_dump_active()
        and (STATE.dump_ujicache_residual or STATE.capture_calibration_pairs)
        and STATE.hareskip_enabled
    )


def _remove_generation_patches() -> None:
    remove_patch("hareskip")


def _requires_supported_model() -> bool:
    return STATE.hareskip_enabled


def _model_sampling_shift() -> float | None:
    try:
        from modules import shared

        from .forge_introspection import model_sampling_info

        data = model_sampling_info(getattr(shared, "sd_model", None))
        value = data.get("shift")
        return None if value is None else float(value)
    except Exception:
        return None


def _check_profile_shift_match() -> None:
    """Warn when the selected calibrated preset's Shift does not match the live
    model's effective shift.

    A calibrated preset's coefficients are fitted for a specific Shift; running
    it at a different shift moves the sigma schedule off the fitted domain and
    silently distorts skip decisions. daraskme/Identity carry no Shift and are
    skipped. Warns once per (preset shift, effective shift) pair per session.
    """
    if not STATE.hareskip_enabled:
        return
    expected = ujicache_expected_shift_for_profile(STATE.ujicache_coefficient_profile)
    if expected is None:
        return
    effective = _model_sampling_shift()
    if effective is None:
        return
    if round(effective) == expected:
        return
    key = f"{STATE.ujicache_coefficient_profile}@{effective:.3f}"
    if key in STATE.ujicache_shift_warned_keys:
        return
    STATE.ujicache_shift_warned_keys.add(key)
    warning(
        "hareskip_shift_mismatch "
        f"profile={STATE.ujicache_coefficient_profile} "
        f"expected_shift={expected} effective_shift={effective:.3f}; "
        "coefficients are off their fitted domain — match the model Shift to the preset"
    )


def _default_option(key: str, default):
    try:
        from modules import shared

        return getattr(shared.opts, key, default)
    except Exception:
        return default


def _default_mode_option() -> str:
    mode = str(_default_option("hareskip_debug_mode", MODE_OFF) or MODE_OFF)
    if mode == "Off":
        return MODE_OFF
    return mode if mode in MODES else MODE_OFF


_SUPERSCRIPTS = {0: "", 1: "", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}


def _superscript(power: int) -> str:
    if power <= 1:
        return _SUPERSCRIPTS.get(power, "")
    return "".join(_SUPERSCRIPTS.get(int(digit), "") for digit in str(power))


def _format_p_anima_x(profile: str) -> str:
    """Render the active coefficient polynomial p_Anima(x) in descending powers.

    Read-only reference display, rendered as a single inline-code line in a
    gr.Markdown component (not a text input) so it never looks editable.
    Coefficients are shown at full precision (never rounded): some profiles
    produce very small coefficients that would collapse to 0 if rounded. Signs
    are operator-ized (A·x⁴ − B·x³ + …) for readability.
    """
    coefficients = ujicache_coefficients_for_profile(profile)
    if not coefficients:
        return "`p_Anima(x) = (no coefficients)`"
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        magnitude = abs(float(coefficient))
        variable = f"·x{_superscript(power)}" if power >= 1 else ""
        term = f"{magnitude!r}{variable}"
        if not terms:
            sign = "-" if float(coefficient) < 0 else ""
            terms.append(f"{sign}{term}")
        else:
            operator = "−" if float(coefficient) < 0 else "+"
            terms.append(f"{operator} {term}")
    return "`p_Anima(x) = " + " ".join(terms) + "`"


def _ujicache_profile_change_updates(profile: str):
    """React to a Coefficient profile change: loosely move the Start/End sliders
    to the profile's recommended window and refresh p_Anima(x).

    Window semantics (ujicache_window_for_profile):
      (start, end) -> move the sliders there (calibrated presets and daraskme).
      None         -> leave the sliders untouched (Identity estimate).
    The user can still move the sliders by hand afterwards (loose coupling).
    """
    window = ujicache_window_for_profile(profile)
    if window is None:
        start_update = gr.update()
        end_update = gr.update()
    else:
        start_update = gr.update(value=window[0])
        end_update = gr.update(value=window[1])
    return start_update, end_update, gr.update(value=_format_p_anima_x(profile))


def _ujicache_prediction_control_updates(formula: str, slope_ema_smoothing: float):
    uses_prediction = formula in (UJICACHE_FORMULA_LINEAR, UJICACHE_FORMULA_TAYLOR2)
    uses_taylor = formula == UJICACHE_FORMULA_TAYLOR2
    try:
        slope = float(slope_ema_smoothing)
    except Exception:
        slope = 0.0
    return (
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_taylor),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_taylor and slope > 0.0),
    )


def _ujicache_enable_updates(enabled: bool):
    if enabled:
        return gr.update()
    return False

