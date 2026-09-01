from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

# Forge neo core ships gradio_rangeslider==0.0.8 (with gradio 4.40.0). When the
# dual-thumb RangeSlider is importable the skip window / zone boundaries are
# edited with one two-thumb control each and mirrored into hidden gr.Number
# scalars; when absent the UI degrades to plain gr.Slider pairs. Either way the
# script arg count is identical (four scalar components in the return list).
try:
    from gradio_rangeslider import RangeSlider
except ImportError:
    RangeSlider = None

from .auto_teacache import (
    AutoTeaCsvError,
    apply_auto_teacache_row_to_state,
    parse_auto_teacache_csv,
)
from .callbacks import register_callbacks
from .constants import (
    EXPECTED_UI_ARG_COUNT,
    HARESKIP_MODES,
    LEGACY_MIN_UI_ARG_COUNT,
    MODE_HARESKIP,
    MODE_MANUAL,
    MODE_TEACACHE,
)
from .manual_skip import (
    ManualSkipError,
    parse_manual_lines,
    parse_manual_steps,
    validate_manual_lines,
    validate_manual_steps,
)
from .diagnostics import log_generation_start, log_timing_summary
from .logging import error, exception, info, warning
from .model_detect import detect_model
from .patcher import apply_patch, remove_patch
from .probability_models import (
    DEFAULT_PROBABILITY_MODEL,
    METHOD_NAME,
    METHOD_VERSION,
    PROBABILITY_MODELS,
)
from .reference_schedule import (
    REFERENCE_NUM_STEPS,
    REFERENCE_SCHEDULE_LABEL,
    REFERENCE_T_NOW,
)
from .skip_pattern import generate_skip_pattern
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
    _clamp_float,
    _clamp_int_default,
    _normalize_range,
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

                # Skip window (progress) and Zone boundaries (logSNR proxy).
                # CRITICAL: exactly four scalar components go into the return
                # list (hareskip_window_start/end, hareskip_zone_low/high) so
                # the arg count is identical in both branches.
                #   RangeSlider present -> two visible dual-thumb sliders (NOT
                #     in the return list) drive four hidden gr.Number mirrors.
                #   RangeSlider absent   -> the four components ARE the visible
                #     plain gr.Slider fallbacks (same variable names/positions).
                if RangeSlider is not None:
                    hareskip_window_range = RangeSlider(
                        label="Skip window (progress)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=(0.05, 0.95),
                        elem_id="hare-skip-window",
                    )
                    hareskip_zone_range = RangeSlider(
                        label="Zone boundaries (logSNR proxy)",
                        minimum=-8.0,
                        maximum=8.0,
                        step=0.1,
                        value=(-4.0, 0.0),
                        elem_id="hare-zone-boundaries",
                    )
                    hareskip_window_start = gr.Number(
                        value=0.05, visible=False, elem_id="hare-skip-window-start"
                    )
                    hareskip_window_end = gr.Number(
                        value=0.95, visible=False, elem_id="hare-skip-window-end"
                    )
                    hareskip_zone_low = gr.Number(
                        value=-4.0, visible=False, elem_id="hare-zone-low"
                    )
                    hareskip_zone_high = gr.Number(
                        value=0.0, visible=False, elem_id="hare-zone-high"
                    )
                else:
                    hareskip_window_start = gr.Slider(
                        label="Skip window (progress) start",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.05,
                        elem_id="hare-skip-window-start",
                    )
                    hareskip_window_end = gr.Slider(
                        label="Skip window (progress) end",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.95,
                        elem_id="hare-skip-window-end",
                    )
                    hareskip_zone_low = gr.Slider(
                        label="Zone boundaries (logSNR proxy) low",
                        minimum=-8.0,
                        maximum=8.0,
                        step=0.1,
                        value=-4.0,
                        elem_id="hare-zone-low",
                    )
                    hareskip_zone_high = gr.Slider(
                        label="Zone boundaries (logSNR proxy) high",
                        minimum=-8.0,
                        maximum=8.0,
                        step=0.1,
                        value=0.0,
                        elem_id="hare-zone-high",
                    )

                # Per-zone max consecutive skips (2026-09-01). These were
                # the skip_pattern.ZONE_MAX_STREAK constant; 0 means the zone
                # is always computed in full.
                hareskip_streak_danger = gr.Slider(
                    label="Max skip streak — danger (z < low)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=1,
                    info="0 = never skip in this zone",
                    elem_id="hare-streak-danger",
                )
                hareskip_streak_middle = gr.Slider(
                    label="Max skip streak — middle (low ≤ z < high)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    info="0 = never skip in this zone",
                    elem_id="hare-streak-middle",
                )
                hareskip_streak_safe = gr.Slider(
                    label="Max skip streak — safe (z ≥ high)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=3,
                    info="0 = never skip in this zone",
                    elem_id="hare-streak-safe",
                )

                hareskip_estimate_md = gr.Markdown(
                    value=_format_hareskip_estimate(),
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

            # --- Manual Skip mode group (hidden by default) -------------------
            with gr.Group(visible=False) as manual_mode_group:
                manual_skip_steps = gr.Textbox(
                    label="Manual skip steps",
                    lines=6,
                    placeholder="e.g.\n10, 12\n15, 17, 19",
                    elem_id="manual-skip-steps",
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
                gr.HTML(
                    '<div style="border-top: 3px solid var(--block-border-color, #4b5563); margin: 0.85rem 0 0.7rem;"></div>',
                    elem_id="hareskip-debug-hare-divider",
                )
                hareskip_probability_model = gr.Dropdown(
                    label="HareSkip probability model",
                    choices=sorted(PROBABILITY_MODELS),
                    value=DEFAULT_PROBABILITY_MODEL,
                    elem_id="hare-probability-model",
                )
                hareskip_exact_target = gr.Number(
                    label="Exact skip target (0 = off)",
                    value=0,
                    precision=0,
                    info=(
                        "Re-roll the pattern up to 100 times to hit this exact "
                        "skip count. Unreachable targets fall back to the "
                        "nearest attempt (never errors)."
                    ),
                    elem_id="hare-exact-target",
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
                outputs=[hareskip_mode_group, teacache_mode_group, manual_mode_group],
            )
            # --- Live skip-count estimate ------------------------------
            # Every control that feeds generate_skip_pattern re-renders the
            # estimate through one shared 10-input handler. Sliders use
            # .release (not .change) so dragging does not fire per pixel;
            # Number/Dropdown use .change.
            _hare_estimate_inputs = [
                hareskip_aggressiveness,
                hareskip_window_start,
                hareskip_window_end,
                hareskip_zone_low,
                hareskip_zone_high,
                hareskip_streak_danger,
                hareskip_streak_middle,
                hareskip_streak_safe,
                hareskip_probability_model,
                hareskip_exact_target,
            ]
            for _hare_slider in (
                hareskip_aggressiveness,
                hareskip_streak_danger,
                hareskip_streak_middle,
                hareskip_streak_safe,
            ):
                _hare_slider.release(
                    fn=_hareskip_estimate_updates,
                    inputs=_hare_estimate_inputs,
                    outputs=[hareskip_estimate_md],
                )
            hareskip_probability_model.change(
                fn=_hareskip_estimate_updates,
                inputs=_hare_estimate_inputs,
                outputs=[hareskip_estimate_md],
            )
            hareskip_exact_target.change(
                fn=_hareskip_estimate_updates,
                inputs=_hare_estimate_inputs,
                outputs=[hareskip_estimate_md],
            )
            if RangeSlider is not None:
                # Dual-thumb path: mirror each RangeSlider into its hidden
                # gr.Number pair AND refresh the estimate in the same call,
                # rather than depending on the hidden mirrors re-firing.
                hareskip_window_range.change(
                    fn=_hareskip_window_range_updates,
                    inputs=[
                        hareskip_window_range,
                        hareskip_aggressiveness,
                        hareskip_zone_low,
                        hareskip_zone_high,
                        hareskip_streak_danger,
                        hareskip_streak_middle,
                        hareskip_streak_safe,
                        hareskip_probability_model,
                        hareskip_exact_target,
                    ],
                    outputs=[
                        hareskip_window_start,
                        hareskip_window_end,
                        hareskip_estimate_md,
                    ],
                )
                hareskip_zone_range.change(
                    fn=_hareskip_zone_range_updates,
                    inputs=[
                        hareskip_zone_range,
                        hareskip_aggressiveness,
                        hareskip_window_start,
                        hareskip_window_end,
                        hareskip_streak_danger,
                        hareskip_streak_middle,
                        hareskip_streak_safe,
                        hareskip_probability_model,
                        hareskip_exact_target,
                    ],
                    outputs=[
                        hareskip_zone_low,
                        hareskip_zone_high,
                        hareskip_estimate_md,
                    ],
                )
            else:
                # Fallback path: the window/zone components are plain
                # Sliders, so they carry the generic handler themselves.
                for _hare_slider in (
                    hareskip_window_start,
                    hareskip_window_end,
                    hareskip_zone_low,
                    hareskip_zone_high,
                ):
                    _hare_slider.release(
                        fn=_hareskip_estimate_updates,
                        inputs=_hare_estimate_inputs,
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
            hareskip_window_start,
            hareskip_window_end,
            hareskip_zone_low,
            hareskip_zone_high,
            manual_skip_steps,
            hareskip_streak_danger,
            hareskip_streak_middle,
            hareskip_streak_safe,
            hareskip_exact_target,
            hareskip_probability_model,
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
        # Manual Skip validation runs after UI args are applied (inside
        # _prepare_auto_teacache_run -> _apply_ui_args). Mirrors the Auto Tea
        # error path: parse + validate against the concrete p.steps, and abort
        # generation with a RuntimeError (surfaced in the Forge UI) on any
        # invalid input rather than silently degrading to full compute.
        try:
            _prepare_manual_skip_run(p)
        except ManualSkipError as exc:
            STATE.manual_skip_parsed = None
            STATE.set_error(f"manual skip error: {exc}")
            error(f"manual_skip_error {exc}")
            raise RuntimeError(f"Manual Skip error: {exc}") from exc
        except Exception as exc:
            STATE.set_error(f"manual skip setup failed: {exc}")
            exception("manual skip setup failed")
            raise

    def process(self, p, *script_args):
        try:
            _apply_auto_teacache_seed_template(p)
        except Exception as exc:
            STATE.set_error(f"auto tea seed setup failed: {exc}")
            exception("auto tea seed setup failed")
            raise
        try:
            _apply_manual_skip_seed_template(p)
        except Exception as exc:
            STATE.set_error(f"manual skip seed setup failed: {exc}")
            exception("manual skip seed setup failed")
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
        # infotext keys (Hare skip_window/zone_boundaries/params/skipped_steps/
        # skip_count/skip_seed) cannot be written any earlier than this hook.
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
        _finish_manual_skip_run(p)


_AUTO_HARESKIP_P_ATTRS = (
    "_hareskip_auto_rows",
    "_hareskip_auto_original_n_iter",
    "_hareskip_auto_original_batch_size",
    "_hareskip_auto_seed_template_size",
    "_hareskip_auto_seed_template_ready",
    "_hareskip_auto_logged_row_index",
    "_hareskip_iteration_counter",
)


_MANUAL_HARESKIP_P_ATTRS = (
    "_hareskip_manual_rows",
    "_hareskip_manual_original_n_iter",
    "_hareskip_manual_original_batch_size",
    "_hareskip_manual_seed_template_size",
    "_hareskip_manual_seed_template_ready",
    "_hareskip_manual_logged_row_index",
    "_hareskip_iteration_counter",
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

    if not (
        STATE.enabled
        and STATE.auto_teacache_enabled
        and STATE.hareskip_mode == MODE_TEACACHE
    ):
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


def _prepare_manual_skip_run(p) -> None:
    """Parse + validate the multiline Manual Skip input against the run's p.steps.

    No-op unless HareSkip is enabled and the Manual Skip mode is selected.
    Each non-blank line is one job (§10.2): all lines are parsed and validated
    up front, then the job is expanded by multiplying ``p.n_iter`` by the line
    count (mirroring Auto Tea's row expansion). All lines share one seed
    template. On success stores the first line's validated 1-based list on
    STATE.manual_skip_parsed (per-pass row selection in
    ``_apply_manual_skip_row_if_needed`` then rewrites it for each subsequent
    line); leaves it None / [] otherwise. Raises ManualSkipError on invalid
    input, which before_process turns into a RuntimeError.
    """
    STATE.manual_skip_parsed = None
    _clear_manual_skip_p_attrs(p)
    if not (STATE.enabled and STATE.hareskip_mode == MODE_MANUAL):
        return

    parsed = parse_manual_lines(STATE.manual_skip_steps)
    num_steps = _positive_int(getattr(p, "steps", 0), 0)
    validate_manual_lines(parsed, num_steps)

    rows = [steps for _line_no, steps in parsed]
    if not rows:
        # Empty input / all-blank lines: no expansion, baseline behaviour that
        # matches v1 (skip nothing, single job). manual_skip_parsed stays [].
        STATE.manual_skip_parsed = []
        info(f"manual_skip_prepare rows=0 num_steps={num_steps}")
        return

    # Defensive double-expansion guard (§10.x.4-3): if Auto Tea has already
    # expanded p.n_iter, fail-stop rather than compound both expansions.
    if getattr(p, "_hareskip_auto_rows", None):
        raise RuntimeError(
            "Manual Skip and Auto Tea both tried to expand n_iter for the same "
            "run; refusing to run. Disable Auto Tea mode or switch skip mode."
        )

    original_n_iter = _positive_int(getattr(p, "n_iter", 1), 1)
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    total_n_iter = len(rows) * original_n_iter

    setattr(p, "_hareskip_manual_rows", rows)
    setattr(p, "_hareskip_manual_original_n_iter", original_n_iter)
    setattr(p, "_hareskip_manual_original_batch_size", batch_size)
    setattr(p, "n_iter", total_n_iter)

    STATE.manual_skip_parsed = rows[0]

    info(
        "manual_skip_prepare "
        f"rows={len(rows)} original_n_iter={original_n_iter} "
        f"batch_size={batch_size} total_n_iter={total_n_iter} "
        f"num_steps={num_steps}"
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


def _apply_manual_skip_seed_template(p) -> None:
    """Give every Manual Skip line the SAME seed template (§10.x.3, fixed).

    Mirror of ``_apply_auto_teacache_seed_template``: builds a seed template of
    ``original_n_iter * batch_size`` (batch_size included so line boundaries
    never fall mid-template and shift seeds) and repeats it once per line, so
    each pattern is compared under identical seeds. Not user-configurable.
    """
    rows = getattr(p, "_hareskip_manual_rows", None)
    if not rows:
        return

    row_count = len(rows)
    original_n_iter = _positive_int(
        getattr(p, "_hareskip_manual_original_n_iter", 1),
        1,
    )
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    template_size = original_n_iter * batch_size
    setattr(p, "_hareskip_manual_seed_template_size", template_size)

    raw_seed_values = getattr(p, "all_seeds", None)
    if not raw_seed_values and _safe_int(getattr(p, "seed", -1), -1) < 0:
        return

    was_ready = bool(getattr(p, "_hareskip_manual_seed_template_ready", False))
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

    setattr(p, "_hareskip_manual_seed_template_ready", True)
    if not was_ready:
        info(
            "manual_skip_seed_template "
            f"rows={row_count} template_size={template_size} seeds={_seed_label(seed_template)}"
        )


def _apply_manual_skip_row_if_needed(p) -> None:
    """Assign this pass's Manual Skip line to STATE.manual_skip_parsed.

    Mirror of ``_apply_auto_teacache_row_if_needed``, but called AFTER
    ``start_sampling`` / ``reset_generation`` (§10.x.2): each sampling pass must
    re-select its line because ``reset_generation`` runs per pass and the value
    otherwise carries over from the previous line. Single-line input always
    resolves to ``rows[0]``, so v1 behaviour is preserved exactly.
    """
    rows = getattr(p, "_hareskip_manual_rows", None)
    if not rows or STATE.hareskip_mode != MODE_MANUAL or not STATE.hareskip_enabled:
        return

    original_n_iter = _positive_int(
        getattr(p, "_hareskip_manual_original_n_iter", 1),
        1,
    )
    iteration = _current_generation_iteration(p)
    row_index = max(0, min(len(rows) - 1, iteration // original_n_iter))
    STATE.manual_skip_parsed = rows[row_index]

    logged_row_index = getattr(p, "_hareskip_manual_logged_row_index", None)
    if logged_row_index != row_index:
        setattr(p, "_hareskip_manual_logged_row_index", row_index)
        info(
            "manual_skip_row_start "
            f"index={row_index + 1}/{len(rows)} steps={rows[row_index]}"
        )


def _apply_auto_teacache_row_if_needed(p) -> None:
    rows = getattr(p, "_hareskip_auto_rows", None)
    if not rows or not STATE.auto_teacache_active or not STATE.hareskip_enabled:
        return

    original_n_iter = _positive_int(
        getattr(p, "_hareskip_auto_original_n_iter", 1),
        1,
    )
    iteration = _current_generation_iteration(p)
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


def _finish_manual_skip_run(p) -> None:
    _clear_manual_skip_p_attrs(p)


def _clear_manual_skip_p_attrs(p) -> None:
    for attr in _MANUAL_HARESKIP_P_ATTRS:
        try:
            if hasattr(p, attr):
                delattr(p, attr)
        except Exception:
            setattr(p, attr, None)


def _current_generation_iteration(p) -> int:
    value = getattr(p, "iteration", None)
    if value is not None:
        try:
            return max(0, int(value))
        except Exception:
            pass
    counter = _positive_int(getattr(p, "_hareskip_iteration_counter", 0), 0)
    setattr(p, "_hareskip_iteration_counter", counter + 1)
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
    """Apply a UI-arg payload to STATE, tolerating pre-v0.2 short payloads.

    A payload with at least ``EXPECTED_UI_ARG_COUNT`` entries is applied as
    before. A payload from before the 2026-09-01 five-argument append (length
    in ``[LEGACY_MIN_UI_ARG_COUNT, EXPECTED_UI_ARG_COUNT)``) is accepted with
    the missing trailing arguments left to ``apply_options``' own defaults,
    which reproduce the pre-v0.2 behaviour (streaks 1/2/3, exact target off,
    default probability model); a single warning is emitted. Shorter payloads
    keep the old fallback to shared settings.
    """
    global _ui_arg_count_warned
    if len(script_args) >= _EXPECTED_UI_ARG_COUNT:
        STATE.apply_options(*script_args[:_EXPECTED_UI_ARG_COUNT])
        return
    if len(script_args) >= LEGACY_MIN_UI_ARG_COUNT:
        if not _ui_arg_count_warned:
            _ui_arg_count_warned = True
            warning(
                "hareskip_ui_args_padded "
                f"received={len(script_args)} expected={_EXPECTED_UI_ARG_COUNT}; "
                "missing trailing arguments defaulted (pre-v0.2 payload)"
            )
        STATE.apply_options(*script_args)
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
    _apply_manual_skip_seed_template(p)
    if not STATE.active():
        _remove_generation_patches()
        return

    start_sampling(source)
    # MUST run after start_sampling: start_sampling -> reset_generation runs per
    # pass and manual_skip_parsed must be re-selected for THIS pass's line
    # (§10.x.2).
    _apply_manual_skip_row_if_needed(p)
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
        elif STATE.hareskip_mode == MODE_MANUAL:
            # Manual Skip carries only the shared "HareSkip mode" identity key
            # here; the realized "Manual skipped_steps" key is written later in
            # postprocess_image (after sampling, once skips have happened).
            pass
        else:
            params["Hare method"] = METHOD_NAME
            params["Hare method_version"] = METHOD_VERSION
            params["Hare probability_model"] = STATE.hareskip_probability_model
            params["Hare aggressiveness"] = f"{STATE.hareskip_aggressiveness:.2f}"
            params["Hare skip_seed_offset"] = STATE.hareskip_skip_seed_offset
            params["Hare zone_streaks"] = (
                f"{STATE.hareskip_streak_danger}"
                f"/{STATE.hareskip_streak_middle}"
                f"/{STATE.hareskip_streak_safe}"
            )
            if STATE.hareskip_exact_target > 0:
                params["Hare exact_target"] = STATE.hareskip_exact_target
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
    if not STATE.hareskip_enabled:
        return
    if STATE.hareskip_mode == MODE_MANUAL:
        _apply_manual_skip_infotext(p)
        return
    if STATE.hareskip_mode == MODE_TEACACHE:
        _apply_tea_skip_infotext(p)
        return
    if STATE.hareskip_mode != MODE_HARESKIP:
        return
    pattern = STATE.hareskip_pattern
    if pattern is None:
        return
    try:
        params = _extra_generation_params(p)
        params.pop("Hare pattern", None)
        params["Hare skip_window"] = (
            f"{pattern.skip_window[0]:.2f}-{pattern.skip_window[1]:.2f}"
        )
        params["Hare zone_boundaries"] = (
            f"{pattern.zone_boundaries[0]:.1f}/{pattern.zone_boundaries[1]:.1f}"
        )
        params["Hare params"] = _format_hare_params(pattern.params)
        params["Hare skipped_steps"] = " ".join(str(step) for step in pattern.skipped_steps)
        params["Hare skip_count"] = pattern.skip_count
        params["Hare skip_seed"] = pattern.skip_seed
    except Exception as exc:
        warning(f"hareskip_pattern_metadata_failed reason={exc}")


def _apply_manual_skip_infotext(p) -> None:
    """Write the realized Manual skipped_steps infotext key.

    Manual Skip mode only. Reports the REALIZED skips — the actual per-step skip
    record maintained by the patcher (STATE.hareskip_skipped_steps, 0-based),
    converted to 1-based and space-joined — rather than the user-specified list.
    This is truthful even in the (validation-guarded, near-impossible) case
    where a shared force-full (first_call / missing_residual) overrode a
    requested skip: the key reflects what was actually skipped. Same write
    method as the HareSkip "Hare skipped_steps" key.
    """
    try:
        params = _extra_generation_params(p)
        realized = sorted({int(step) + 1 for step in STATE.hareskip_skipped_steps})
        params["Manual skipped_steps"] = " ".join(str(step) for step in realized)

        rows = getattr(p, "_hareskip_manual_rows", None)
        if rows:
            row_index = getattr(p, "_hareskip_manual_logged_row_index", None)
            if row_index is None:
                row_index = 0
            row_count = len(rows)
        else:
            row_index = 0
            row_count = 1
        info(
            "manual_skip_realized "
            f"index={row_index + 1}/{row_count} steps={realized}"
        )
    except Exception as exc:
        warning(f"manual_skip_metadata_failed reason={exc}")


def _apply_tea_skip_infotext(p) -> None:
    """Write the realized Tea skipped_steps infotext key.

    TeaCache mode only. Mirrors ``_apply_manual_skip_infotext``: the skip
    execution machinery (residual reuse in the patcher's shared forward body)
    is common to all three modes, so TeaCache's threshold-decided skips land
    in the same ``STATE.hareskip_skipped_steps`` (0-based) record as HareSkip
    and Manual Skip. Reports the REALIZED skips — what the patcher actually
    skipped — not the threshold/window configuration (those are already
    written earlier by ``_apply_infotext_metadata`` as ``Tea threshold`` /
    ``Tea progress`` etc.). Same 1-based, space-joined, ascending format as
    ``Hare skipped_steps`` / ``Manual skipped_steps``.
    """
    try:
        params = _extra_generation_params(p)
        realized = sorted({int(step) + 1 for step in STATE.hareskip_skipped_steps})
        params["Tea skipped_steps"] = " ".join(str(step) for step in realized)

        rows = getattr(p, "_hareskip_auto_rows", None)
        if rows:
            row_index = STATE.auto_teacache_row_index
            if row_index is None:
                row_index = 0
            row_count = STATE.auto_teacache_row_count or len(rows)
        else:
            row_index = 0
            row_count = 1
        info(
            "tea_realized "
            f"index={row_index + 1}/{row_count} steps={realized}"
        )
    except Exception as exc:
        warning(f"tea_skip_metadata_failed reason={exc}")


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
    """Toggle visibility of the HareSkip, TeaCache and Manual Skip mode groups.

    Returns (hareskip_group_update, teacache_group_update, manual_group_update)
    so the three groups are mutually exclusive. Unknown values fall back to
    HareSkip (the two non-HareSkip groups hidden).
    """
    is_tea = skip_mode == MODE_TEACACHE
    is_manual = skip_mode == MODE_MANUAL
    is_hareskip = not (is_tea or is_manual)
    return (
        gr.update(visible=is_hareskip),
        gr.update(visible=is_tea),
        gr.update(visible=is_manual),
    )


# Deterministic seed ladder for the UI skip-count estimate: seed_i =
# _HARE_ESTIMATE_SEED_BASE + i * _HARE_ESTIMATE_SEED_STRIDE. Fixed so the same
# settings always render the same number (no flicker between redraws). 32
# draws cost ~1.7 ms, cheap enough for a slider .release handler.
_HARE_ESTIMATE_SEEDS = 32
_HARE_ESTIMATE_SEED_BASE = 1
_HARE_ESTIMATE_SEED_STRIDE = 7919


def _estimate_skip_count(
    aggressiveness,
    window,
    zones,
    streaks,
    model_name,
    exact_target=0,
) -> float:
    """Mean skip count over the reference schedule for the given settings.

    Monte Carlo over ``_HARE_ESTIMATE_SEEDS`` deterministic seeds against
    ``reference_schedule.REFERENCE_T_NOW`` (display only — never the sampling
    path). When ``exact_target`` is enabled the generator re-rolls until it
    hits that count, so the answer is known without sampling and the loop is
    short-circuited. Inputs are expected pre-normalised by the caller.
    """
    if exact_target > 0:
        return float(exact_target)
    total = 0
    for i in range(_HARE_ESTIMATE_SEEDS):
        seed = _HARE_ESTIMATE_SEED_BASE + i * _HARE_ESTIMATE_SEED_STRIDE
        pattern = generate_skip_pattern(
            REFERENCE_T_NOW,
            aggressiveness,
            seed,
            probability_model=model_name,
            skip_window=window,
            zone_boundaries=zones,
            zone_max_streak=streaks,
        )
        total += pattern.skip_count
    return total / float(_HARE_ESTIMATE_SEEDS)


def _format_hareskip_estimate(
    aggressiveness=0.5,
    window_start=0.05,
    window_end=0.95,
    zone_low=-4.0,
    zone_high=0.0,
    streak_danger=1,
    streak_middle=2,
    streak_safe=3,
    model_name=DEFAULT_PROBABILITY_MODEL,
    exact_target=0,
) -> str:
    """Two-line Markdown summary: estimated skip count plus model parameters.

    Raw UI values are normalised with the same helpers ``RuntimeState`` uses,
    so a reversed window, a blank Number or an unknown model name renders a
    sane figure instead of raising. Wrapped in a broad try/except so a UI
    callback can never propagate an exception.
    """
    try:
        from .probability_models import get_model

        aggressiveness = _clamp_float(aggressiveness, 0.0, 1.0)
        window = _normalize_range(
            window_start, window_end, 0.0, 1.0, 0.05, 0.95
        )
        zones = _normalize_range(zone_low, zone_high, -8.0, 8.0, -4.0, 0.0)
        streaks = {
            "danger": _clamp_int_default(streak_danger, 0, 10, 1),
            "middle": _clamp_int_default(streak_middle, 0, 10, 2),
            "safe": _clamp_int_default(streak_safe, 0, 10, 3),
        }
        exact_target = _clamp_int_default(exact_target, 0, 999, 0)
        if model_name not in PROBABILITY_MODELS:
            model_name = DEFAULT_PROBABILITY_MODEL

        mean = _estimate_skip_count(
            aggressiveness, window, zones, streaks, model_name, exact_target
        )
        count = int(round(mean))
        pct = int(round(100.0 * count / REFERENCE_NUM_STEPS))
        basis = (
            "exact target"
            if exact_target > 0
            else f"{_HARE_ESTIMATE_SEEDS}シード平均"
        )
        headline = (
            f"**≈{count} skips / {REFERENCE_NUM_STEPS} steps** "
            f"（約{pct}%時短、"
            f"{REFERENCE_SCHEDULE_LABEL} 基準・{basis}）"
        )

        params = get_model(model_name).params_from_aggressiveness(aggressiveness)
        summary = (
            f"`model={model_name}"
            f" p_cap={params['p_cap']:.2f}"
            f" z_enter={params['z_enter']:.2f}"
            f" tau_enter={params['tau_enter']:.2f}"
        )
        # The default monotone model has no falling edge, hence no z_exit.
        z_exit = params.get("z_exit")
        if z_exit is not None:
            summary += f" z_exit={z_exit:.2f}"
        return headline + "\n\n" + summary + "`"
    except Exception:
        return "`skip estimate unavailable`"


def _hareskip_estimate_updates(
    aggressiveness,
    window_start,
    window_end,
    zone_low,
    zone_high,
    streak_danger,
    streak_middle,
    streak_safe,
    model_name,
    exact_target,
):
    return gr.update(
        value=_format_hareskip_estimate(
            aggressiveness,
            window_start,
            window_end,
            zone_low,
            zone_high,
            streak_danger,
            streak_middle,
            streak_safe,
            model_name,
            exact_target,
        )
    )


def _hareskip_window_range_updates(
    range_value,
    aggressiveness,
    zone_low,
    zone_high,
    streak_danger,
    streak_middle,
    streak_safe,
    model_name,
    exact_target,
):
    """Mirror the dual-thumb window RangeSlider into the hidden Numbers.

    Also returns the refreshed estimate directly rather than relying on the
    hidden gr.Number .change events re-firing (belt and braces: the mirrors
    are not user-visible controls).
    """
    start, end = range_value[0], range_value[1]
    return (
        start,
        end,
        _hareskip_estimate_updates(
            aggressiveness,
            start,
            end,
            zone_low,
            zone_high,
            streak_danger,
            streak_middle,
            streak_safe,
            model_name,
            exact_target,
        ),
    )


def _hareskip_zone_range_updates(
    range_value,
    aggressiveness,
    window_start,
    window_end,
    streak_danger,
    streak_middle,
    streak_safe,
    model_name,
    exact_target,
):
    """Mirror the dual-thumb zone RangeSlider into the hidden Numbers."""
    low, high = range_value[0], range_value[1]
    return (
        low,
        high,
        _hareskip_estimate_updates(
            aggressiveness,
            window_start,
            window_end,
            low,
            high,
            streak_danger,
            streak_middle,
            streak_safe,
            model_name,
            exact_target,
        ),
    )

