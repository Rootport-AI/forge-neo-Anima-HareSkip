from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .auto_teacache import (
    AutoTeaCsvError,
    apply_auto_teacache_row_to_state,
    parse_auto_teacache_csv,
)
from .callbacks import register_callbacks
from .constants import (
    EXPECTED_UI_ARG_COUNT,
    HARESKIP_MODES,
    MODE_HARESKIP,
    MODE_TEACACHE,
)
from .diagnostics import log_generation_start, log_timing_summary
from .logging import error, exception, info, warning
from .model_detect import detect_model
from .patcher import apply_patch, remove_patch
from .probability_models import METHOD_NAME, METHOD_VERSION
from .state import (
    MODE_OFF,
    MODES,
    STATE,
    RESREFINE_CACHE_DEVICE_CUDA,
    RESREFINE_CACHE_DEVICES,
    TEA_COEFFICIENT_PROFILES,
    RESREFINE_FORMULA_LINEAR,
    RESREFINE_FORMULA_TAYLOR2,
    RESREFINE_FORMULA_REUSE,
    RESREFINE_FORMULAS,
    TEA_PRESET_CUSTOM,
    TEA_PRESETS,
    TEA_PROFILE_DEFAULT,
    tea_coefficients_for_profile,
    tea_expected_shift_for_profile,
    tea_window_for_profile,
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
            hareskip_mode = gr.Radio(
                label="Skip mode",
                choices=HARESKIP_MODES,
                value=MODE_HARESKIP,
                elem_id="hareskip-mode",
            )

            # --- HareSkip stochastic mode group (shown first, default) --------
            with gr.Group(visible=True) as hareskip_mode_group:
                hareskip_aggressiveness = gr.Slider(
                    label="Skip aggressiveness",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    elem_id="hare-aggressiveness",
                )
                hareskip_skip_seed_offset = gr.Number(
                    label="Skip seed offset",
                    value=0,
                    precision=0,
                    elem_id="hare-skip-seed-offset",
                )
                hareskip_estimate_md = gr.Markdown(
                    value=_format_hareskip_estimate(0.5),
                    elem_id="hare-estimate",
                )

            # --- TeaCache mode group (hidden by default) ----------------------
            with gr.Group(visible=False) as teacache_mode_group:
                tea_preset = gr.Dropdown(
                    label="Tea preset",
                    choices=TEA_PRESETS,
                    value=TEA_PRESET_CUSTOM,
                    elem_id="tea-preset",
                )
                tea_threshold = gr.Slider(
                    label="Rel L1 threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.005,
                    value=0.07,
                    elem_id="tea-threshold",
                )
                tea_coefficient_profile = gr.Dropdown(
                    label="Coefficient profile",
                    choices=TEA_COEFFICIENT_PROFILES,
                    value=TEA_PROFILE_DEFAULT,
                    elem_id="tea-coefficient-profile",
                )
                p_anima_x = gr.Markdown(
                    value=_format_p_anima_x(TEA_PROFILE_DEFAULT),
                    elem_id="tea-p-anima-x",
                )
                gr.HTML(
                    "<style>"
                    "#tea-p-anima-x code {"
                    " background: none;"
                    " border: none;"
                    " padding: 0;"
                    " color: var(--body-text-color-subdued);"
                    " }"
                    "</style>",
                    elem_id="tea-p-anima-x-style",
                )
                tea_start_percent = gr.Slider(
                    label="Start progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.48,
                    elem_id="tea-start-percent",
                )
                tea_end_percent = gr.Slider(
                    label="End progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.76,
                    elem_id="tea-end-percent",
                )
                tea_max_skip_streak = gr.Slider(
                    label="Max skip streak (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="tea-max-skip-streak",
                )
                tea_force_full_interval = gr.Slider(
                    label="Force full interval (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="tea-force-full-interval",
                )
                with gr.Accordion(
                    "Auto Tea mode",
                    open=False,
                    elem_id="tea-auto-panel",
                ):
                    auto_teacache_enabled = gr.Checkbox(
                        label="Enable Auto Tea mode",
                        value=False,
                        elem_id="tea-auto-enable",
                    )
                    auto_teacache_csv = gr.Textbox(
                        label="Auto Tea CSV",
                        lines=6,
                        elem_id="tea-auto-csv",
                    )

            # --- ResRefine shared section (outside both mode groups) ----------
            with gr.Accordion(
                "ResRefine (residual prediction)",
                open=False,
                elem_id="resrefine-panel",
            ):
                resrefine_formula = gr.Dropdown(
                    label="Prediction formula",
                    choices=RESREFINE_FORMULAS,
                    value=RESREFINE_FORMULA_REUSE,
                    elem_id="resrefine-formula",
                )
                resrefine_use_prediction_after_progress = gr.Slider(
                    label="Use prediction after progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="resrefine-use-prediction-after-progress",
                )
                resrefine_apply_prediction_from_skip = gr.Slider(
                    label="Apply prediction from skip #",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=2,
                    interactive=False,
                    elem_id="resrefine-apply-prediction-from-skip",
                )
                resrefine_prediction_strength = gr.Slider(
                    label="Prediction strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.50,
                    interactive=False,
                    elem_id="resrefine-prediction-strength",
                )
                resrefine_taylor2_curve_strength = gr.Slider(
                    label="Taylor2 curve strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.25,
                    interactive=False,
                    elem_id="resrefine-taylor2-curve-strength",
                )
                resrefine_slope_ema_smoothing = gr.Slider(
                    label="Slope EMA Smoothing",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="resrefine-slope-ema-smoothing",
                )
                resrefine_curve_ema_smoothing = gr.Slider(
                    label="Curve EMA Smoothing",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="resrefine-curve-ema-smoothing",
                )
                resrefine_cache_device = gr.Radio(
                    label="Cache device",
                    choices=RESREFINE_CACHE_DEVICES,
                    value=RESREFINE_CACHE_DEVICE_CUDA,
                    elem_id="resrefine-cache-device",
                )

            with gr.Accordion("Debug log mode", open=False, elem_id="hareskip-debug-panel"):
                debug_log_enabled = gr.Checkbox(
                    label="Enable debug log mode",
                    value=_default_option("hareskip_debug_log_enable", False),
                    elem_id="hareskip-debug-enable",
                )
                mode = gr.Dropdown(
                    label="Debug log",
                    choices=MODES,
                    value=_default_mode_option(),
                    elem_id="hareskip-debug-mode",
                )
                dump_resrefine_residual = gr.Checkbox(
                    label="Dump ResRefine residual",
                    value=False,
                    elem_id="resrefine-dump-residual",
                )
                capture_calibration_pairs = gr.Checkbox(
                    label="Capture calibration pairs",
                    value=False,
                    info=(
                        "Force full calculation on every step and write per-step "
                        "(rel_l1, out_rel) pairs plus run conditions to "
                        "calibration_pairs.jsonl in the debug run folder."
                    ),
                    elem_id="hareskip-capture-calibration-pairs",
                )
                gr.HTML(
                    '<div style="border-top: 3px solid var(--block-border-color, #4b5563); margin: 0.85rem 0 0.7rem;"></div>',
                    elem_id="hareskip-debug-dump-divider",
                )
                print_timing_log = gr.Checkbox(
                    label="Print timing log",
                    value=_default_option("hareskip_print_timing_log", True),
                    elem_id="hareskip-print-timing-log",
                )
                verbose_diagnose_log = gr.Checkbox(
                    label="Verbose diagnose log",
                    value=_default_option("hareskip_verbose_diagnose_log", False),
                    elem_id="hareskip-verbose-diagnose-log",
                )
            hareskip_dry_run = gr.Checkbox(
                label="Dry run",
                value=False,
                elem_id="hareskip-dry-run",
            )
            hareskip_verbose_trace = gr.Checkbox(
                label="Verbose HareSkip trace",
                value=False,
                elem_id="hareskip-verbose-trace",
            )

            resrefine_formula.change(
                fn=_hareskip_prediction_control_updates,
                inputs=[resrefine_formula, resrefine_slope_ema_smoothing],
                outputs=[
                    resrefine_use_prediction_after_progress,
                    resrefine_apply_prediction_from_skip,
                    resrefine_prediction_strength,
                    resrefine_taylor2_curve_strength,
                    resrefine_slope_ema_smoothing,
                    resrefine_curve_ema_smoothing,
                ],
            )
            resrefine_slope_ema_smoothing.change(
                fn=_hareskip_prediction_control_updates,
                inputs=[resrefine_formula, resrefine_slope_ema_smoothing],
                outputs=[
                    resrefine_use_prediction_after_progress,
                    resrefine_apply_prediction_from_skip,
                    resrefine_prediction_strength,
                    resrefine_taylor2_curve_strength,
                    resrefine_slope_ema_smoothing,
                    resrefine_curve_ema_smoothing,
                ],
            )
            enabled.change(
                fn=_hareskip_enable_updates,
                inputs=[enabled],
                outputs=[auto_teacache_enabled],
            )
            tea_coefficient_profile.change(
                fn=_hareskip_profile_change_updates,
                inputs=[tea_coefficient_profile],
                outputs=[tea_start_percent, tea_end_percent, p_anima_x],
            )
            hareskip_mode.change(
                fn=_hareskip_mode_group_updates,
                inputs=[hareskip_mode],
                outputs=[hareskip_mode_group, teacache_mode_group],
            )
            hareskip_aggressiveness.change(
                fn=_hareskip_estimate_updates,
                inputs=[hareskip_aggressiveness],
                outputs=[hareskip_estimate_md],
            )

        return [
            enabled,
            debug_log_enabled,
            mode,
            print_timing_log,
            verbose_diagnose_log,
            dump_resrefine_residual,
            tea_preset,
            tea_threshold,
            tea_start_percent,
            tea_end_percent,
            resrefine_formula,
            resrefine_use_prediction_after_progress,
            resrefine_apply_prediction_from_skip,
            resrefine_prediction_strength,
            resrefine_taylor2_curve_strength,
            resrefine_slope_ema_smoothing,
            resrefine_curve_ema_smoothing,
            resrefine_cache_device,
            tea_coefficient_profile,
            tea_max_skip_streak,
            tea_force_full_interval,
            hareskip_dry_run,
            hareskip_verbose_trace,
            auto_teacache_enabled,
            auto_teacache_csv,
            capture_calibration_pairs,
            hareskip_mode,
            hareskip_aggressiveness,
            hareskip_skip_seed_offset,
        ]

    def before_process(self, p, *script_args):
        try:
            _prepare_auto_teacache_run(p, script_args)
        except AutoTeaCsvError as exc:
            STATE.auto_teacache_parse_error = str(exc)
            STATE.set_error(f"auto tea csv error: {exc}")
            error(f"auto_tea_csv_error {exc}")
            raise RuntimeError(f"Auto Tea CSV error: {exc}") from exc
        except Exception as exc:
            STATE.set_error(f"auto tea setup failed: {exc}")
            exception("auto tea setup failed")
            raise

    def process(self, p, *script_args):
        try:
            _apply_auto_teacache_seed_template(p)
        except Exception as exc:
            STATE.set_error(f"auto tea seed setup failed: {exc}")
            exception("auto tea seed setup failed")
            raise

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        try:
            _begin_generation(p, script_args, "process_before_every_sampling")
        except Exception as exc:
            STATE.set_error(f"process_before_every_sampling failed: {exc}")
            exception("process_before_every_sampling failed")

    def postprocess_image(self, p, pp, *script_args):
        # Runs per-image, after sampling but before the PNG is saved (Forge
        # calls this from process_images_inner ahead of images.save_image).
        # The HareSkip skip pattern is only generated lazily on the first
        # patched model forward call during sampling, so pattern-dependent
        # infotext keys (Hare guard_count/params/skipped_steps/skip_count/
        # skip_seed) cannot be written any earlier than this hook.
        try:
            _apply_hare_pattern_infotext(p)
        except Exception as exc:
            STATE.set_error(f"postprocess_image failed: {exc}")
            exception("postprocess_image failed")

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
        _finish_auto_teacache_run(p)


_AUTO_HARESKIP_P_ATTRS = (
    "_hareskip_auto_rows",
    "_hareskip_auto_original_n_iter",
    "_hareskip_auto_original_batch_size",
    "_hareskip_auto_seed_template_size",
    "_hareskip_auto_seed_template_ready",
    "_hareskip_auto_logged_row_index",
    "_hareskip_auto_iteration_counter",
)


def _prepare_auto_teacache_run(p, script_args) -> None:
    _apply_ui_args(script_args)
    _clear_auto_teacache_p_attrs(p)
    STATE.auto_teacache_active = False
    STATE.auto_teacache_row_index = None
    STATE.auto_teacache_row_name = None
    STATE.auto_teacache_row_count = 0
    STATE.auto_teacache_original_n_iter = 1
    STATE.auto_teacache_parse_error = None

    if not (STATE.enabled and STATE.auto_teacache_enabled):
        return

    result = parse_auto_teacache_csv(STATE.auto_teacache_csv)
    for message in result.warnings:
        warning(f"auto_tea_csv_warning {message}")

    rows = result.rows
    original_n_iter = _positive_int(getattr(p, "n_iter", 1), 1)
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    total_n_iter = len(rows) * original_n_iter

    setattr(p, "_hareskip_auto_rows", rows)
    setattr(p, "_hareskip_auto_original_n_iter", original_n_iter)
    setattr(p, "_hareskip_auto_original_batch_size", batch_size)
    setattr(p, "n_iter", total_n_iter)

    STATE.auto_teacache_active = True
    STATE.auto_teacache_row_count = len(rows)
    STATE.auto_teacache_original_n_iter = original_n_iter

    info(
        "auto_tea_prepare "
        f"rows={len(rows)} original_n_iter={original_n_iter} "
        f"batch_size={batch_size} total_n_iter={total_n_iter}"
    )


def _apply_auto_teacache_seed_template(p) -> None:
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
            "auto_tea_seed_template "
            f"rows={row_count} template_size={template_size} seeds={_seed_label(seed_template)}"
        )


def _apply_auto_teacache_row_if_needed(p) -> None:
    rows = getattr(p, "_hareskip_auto_rows", None)
    if not rows or not STATE.auto_teacache_active or not STATE.hareskip_enabled:
        return

    original_n_iter = _positive_int(
        getattr(p, "_hareskip_auto_original_n_iter", 1),
        1,
    )
    iteration = _current_auto_teacache_iteration(p)
    row_index = max(0, min(len(rows) - 1, iteration // original_n_iter))
    repeat_index = (iteration % original_n_iter) + 1
    row = rows[row_index]

    apply_auto_teacache_row_to_state(row)
    STATE.auto_teacache_row_index = row.index
    STATE.auto_teacache_row_name = row.name

    logged_row_index = getattr(p, "_hareskip_auto_logged_row_index", None)
    if logged_row_index != row_index:
        setattr(p, "_hareskip_auto_logged_row_index", row_index)
        info(
            "auto_tea_row_start "
            f"index={row_index + 1}/{len(rows)} name={row.name} "
            f"repeat={repeat_index}/{original_n_iter} "
            f"threshold={STATE.tea_threshold:.4f} "
            f"formula={STATE.resrefine_formula} "
            f"prediction_strength={STATE.resrefine_prediction_strength:.2f} "
            f"batch_size={_positive_int(getattr(p, 'batch_size', 1), 1)} "
            f"seeds={_row_seed_label(p, row_index)}"
        )


def _finish_auto_teacache_run(p) -> None:
    _clear_auto_teacache_p_attrs(p)
    STATE.auto_teacache_active = False
    STATE.auto_teacache_row_index = None
    STATE.auto_teacache_row_name = None
    STATE.auto_teacache_row_count = 0
    STATE.auto_teacache_original_n_iter = 1


def _clear_auto_teacache_p_attrs(p) -> None:
    for attr in _AUTO_HARESKIP_P_ATTRS:
        try:
            if hasattr(p, attr):
                delattr(p, attr)
        except Exception:
            setattr(p, attr, None)


def _current_auto_teacache_iteration(p) -> int:
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


_EXPECTED_UI_ARG_COUNT = EXPECTED_UI_ARG_COUNT
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
    _apply_auto_teacache_seed_template(p)
    _apply_auto_teacache_row_if_needed(p)
    if not STATE.active():
        _remove_generation_patches()
        return

    start_sampling(source)
    _capture_hareskip_image_seed(p)
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


def _capture_hareskip_image_seed(p) -> None:
    """Record the concrete image seed of the current job for HareSkip.

    The patched Anima forward cannot see the processing object ``p``, so the
    seed that drives the reproducible skip pattern must be captured here at
    generation start. Forge resolves any random (-1) seed into ``p.all_seeds``
    before sampling begins, so ``all_seeds[0]`` is the most reliable concrete
    seed for the current job (mirrors how Auto Tea reads ``p.all_seeds``).
    We prefer it; fall back to ``p.seed`` when the list is absent. A seed that
    is still unresolved (-1/None) leaves ``hareskip_image_seed`` as None, in
    which case pattern generation is skipped and the run degrades to full
    compute — deriving a skip seed from a placeholder 0 would break the
    reproducibility contract.
    """
    try:
        seed_value = None
        all_seeds = getattr(p, "all_seeds", None)
        if all_seeds:
            try:
                candidate = int(all_seeds[0])
                if candidate >= 0:
                    seed_value = candidate
            except Exception:
                seed_value = None
        if seed_value is None:
            try:
                candidate = int(getattr(p, "seed", -1))
                if candidate >= 0:
                    seed_value = candidate
            except Exception:
                seed_value = None
        STATE.hareskip_image_seed = seed_value
    except Exception:
        STATE.hareskip_image_seed = None


def _extra_generation_params(p) -> dict:
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        params = {}
        setattr(p, "extra_generation_params", params)
    return params


def _apply_infotext_metadata(p) -> None:
    """Write mode-independent + mode-selected (non-pattern) infotext keys.

    Called from ``_begin_generation`` (``process_before_every_sampling``),
    i.e. before sampling starts. The HareSkip mode's realized skip pattern
    does not exist yet at this point (it is generated lazily on the first
    patched forward call during sampling) — those keys are written later by
    ``_apply_hare_pattern_infotext`` from the ``postprocess_image`` hook,
    which runs after sampling but before the PNG is saved.
    """
    if not STATE.hareskip_enabled:
        return
    try:
        params = _extra_generation_params(p)
        params["HareSkip enabled"] = True
        params["HareSkip mode"] = STATE.hareskip_mode

        if STATE.hareskip_mode == MODE_TEACACHE:
            params["Tea threshold"] = f"{STATE.tea_threshold:.4f}"
            params["Tea progress"] = (
                f"{STATE.tea_start_percent:.2f}..{STATE.tea_end_percent:.2f}"
            )
            params["Tea coefficient_profile"] = STATE.tea_coefficient_profile
            params["Tea max_skip_streak"] = STATE.tea_max_skip_streak
            params["Tea force_full_interval"] = STATE.tea_force_full_interval
            shift_value = _model_sampling_shift()
            if shift_value is not None:
                params["Tea shift"] = f"{shift_value:.2f}"
            params["Tea modulated_source"] = STATE.tea_modulated_source
            if STATE.capture_calibration_pairs:
                params["Tea capture_pairs"] = True
            if STATE.auto_teacache_active and STATE.auto_teacache_row_index is not None:
                if STATE.auto_teacache_row_count > 0:
                    params["Tea auto_row_index"] = (
                        f"{STATE.auto_teacache_row_index}/{STATE.auto_teacache_row_count}"
                    )
                else:
                    params["Tea auto_row_index"] = STATE.auto_teacache_row_index
                params["Tea auto_row_name"] = STATE.auto_teacache_row_name or ""
        else:
            params["Hare method"] = METHOD_NAME
            params["Hare method_version"] = METHOD_VERSION
            params["Hare probability_model"] = STATE.hareskip_probability_model
            params["Hare aggressiveness"] = f"{STATE.hareskip_aggressiveness:.2f}"
            params["Hare skip_seed_offset"] = STATE.hareskip_skip_seed_offset
            if STATE.hareskip_pattern is None:
                params["Hare pattern"] = "unavailable"

        params["ResRefine formula"] = STATE.resrefine_formula
        if STATE.resrefine_formula != RESREFINE_FORMULA_REUSE:
            params["ResRefine use_prediction_after_progress"] = (
                f"{STATE.resrefine_use_prediction_after_progress:.2f}"
            )
            params["ResRefine apply_prediction_from_skip"] = (
                STATE.resrefine_apply_prediction_from_skip
            )
            params["ResRefine prediction_strength"] = f"{STATE.resrefine_prediction_strength:.2f}"
            if STATE.resrefine_formula == RESREFINE_FORMULA_TAYLOR2:
                params["ResRefine taylor2_curve_strength"] = (
                    f"{STATE.resrefine_taylor2_curve_strength:.2f}"
                )
            params["ResRefine slope_ema_smoothing"] = (
                f"{STATE.resrefine_slope_ema_smoothing:.2f}"
            )
            params["ResRefine curve_ema_smoothing"] = (
                f"{STATE.resrefine_curve_ema_smoothing:.2f}"
            )
    except Exception as exc:
        warning(f"hareskip_metadata_failed reason={exc}")


def _apply_hare_pattern_infotext(p) -> None:
    """Write the realized HareSkip pattern's infotext keys, once it exists.

    HareSkip mode only: the skip pattern is generated lazily on the first
    patched Anima forward call, so it is not available until sampling has
    started. This runs from ``postprocess_image`` (per-image, after
    sampling, before the PNG is saved by Forge) so the pattern keys still
    land in the saved image's metadata.
    """
    if not STATE.hareskip_enabled or STATE.hareskip_mode != MODE_HARESKIP:
        return
    pattern = STATE.hareskip_pattern
    if pattern is None:
        return
    try:
        params = _extra_generation_params(p)
        params.pop("Hare pattern", None)
        params["Hare guard_count"] = pattern.guard_count
        params["Hare params"] = _format_hare_params(pattern.params)
        params["Hare skipped_steps"] = " ".join(str(step) for step in pattern.skipped_steps)
        params["Hare skip_count"] = pattern.skip_count
        params["Hare skip_seed"] = pattern.skip_seed
    except Exception as exc:
        warning(f"hareskip_pattern_metadata_failed reason={exc}")


def _format_hare_params(params: dict) -> str:
    """Compact, infotext-safe rendering of the probability model's params dict.

    Semicolon-joined ``k=v`` pairs with floats rounded to 4 decimals — avoids
    commas/quotes, which would otherwise break Forge's infotext key=value,
    key=value parsing.
    """
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ";".join(parts)


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
        and STATE.dump_resrefine_residual
        and not STATE.hareskip_enabled
        and "resrefine_residual_requires_hareskip" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("resrefine_residual_requires_hareskip")
        warning("tensor_dump_hareskip_residual_inactive reason=hareskip_disabled")

    if (
        STATE.debug_log_enabled
        and STATE.capture_calibration_pairs
        and not STATE.hareskip_enabled
        and "calibration_capture_requires_hareskip" not in STATE.calibration_capture_warned_reasons
    ):
        STATE.calibration_capture_warned_reasons.add("calibration_capture_requires_hareskip")
        warning("calibration_capture_inactive reason=hareskip_disabled")


def _tensor_dump_will_save() -> bool:
    return bool(
        STATE.tensor_dump_active()
        and (STATE.dump_resrefine_residual or STATE.capture_calibration_pairs)
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
    expected = tea_expected_shift_for_profile(STATE.tea_coefficient_profile)
    if expected is None:
        return
    effective = _model_sampling_shift()
    if effective is None:
        return
    if round(effective) == expected:
        return
    key = f"{STATE.tea_coefficient_profile}@{effective:.3f}"
    if key in STATE.tea_shift_warned_keys:
        return
    STATE.tea_shift_warned_keys.add(key)
    warning(
        "hareskip_shift_mismatch "
        f"profile={STATE.tea_coefficient_profile} "
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
    coefficients = tea_coefficients_for_profile(profile)
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


def _hareskip_profile_change_updates(profile: str):
    """React to a Coefficient profile change: loosely move the Start/End sliders
    to the profile's recommended window and refresh p_Anima(x).

    Window semantics (tea_window_for_profile):
      (start, end) -> move the sliders there (calibrated presets and daraskme).
      None         -> leave the sliders untouched (Identity estimate).
    The user can still move the sliders by hand afterwards (loose coupling).
    """
    window = tea_window_for_profile(profile)
    if window is None:
        start_update = gr.update()
        end_update = gr.update()
    else:
        start_update = gr.update(value=window[0])
        end_update = gr.update(value=window[1])
    return start_update, end_update, gr.update(value=_format_p_anima_x(profile))


def _hareskip_prediction_control_updates(formula: str, slope_ema_smoothing: float):
    uses_prediction = formula in (RESREFINE_FORMULA_LINEAR, RESREFINE_FORMULA_TAYLOR2)
    uses_taylor = formula == RESREFINE_FORMULA_TAYLOR2
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


def _hareskip_enable_updates(enabled: bool):
    if enabled:
        return gr.update()
    return False


def _hareskip_mode_group_updates(skip_mode: str):
    """Toggle visibility of the HareSkip and TeaCache mode groups.

    Returns (hareskip_group_update, teacache_group_update) so the two groups
    are mutually exclusive. Unknown values fall back to HareSkip.
    """
    is_tea = skip_mode == MODE_TEACACHE
    return gr.update(visible=not is_tea), gr.update(visible=is_tea)


def _format_hareskip_estimate(aggressiveness) -> str:
    """Static per-aggressiveness summary of the skip-probability band.

    Renders p_cap and the sigmoid band centre from the pure probability model
    (no schedule, so no per-run skip count). Wrapped in a broad try/except so a
    UI callback can never raise; falls back to a neutral hint.
    """
    try:
        from .probability_models import get_model

        model = get_model("sigmoid_band_v0.1")
        params = model.params_from_aggressiveness(float(aggressiveness))
        return (
            "`p_cap="
            f"{params['p_cap']:.2f}"
            f" z_enter={params['z_enter']:.2f}"
            f" z_exit={params['z_exit']:.2f}`"
            " — higher aggressiveness widens/raises the skip band."
        )
    except Exception:
        return "`skip band summary unavailable`"


def _hareskip_estimate_updates(aggressiveness):
    return gr.update(value=_format_hareskip_estimate(aggressiveness))

