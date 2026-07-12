from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from .constants import MODE_HARESKIP, MODE_TEACACHE, MODE_MANUAL, HARESKIP_MODES


MODE_OFF = ""
MODE_DIAGNOSE = "Diagnose only"
MODES = [MODE_OFF, MODE_DIAGNOSE]

TEA_PRESET_CUSTOM = "Custom"
TEA_PRESETS = [TEA_PRESET_CUSTOM]

RESREFINE_FORMULA_REUSE = "Reuse (residual only)"
RESREFINE_FORMULA_LINEAR = "Linear extrapolation"
RESREFINE_FORMULA_TAYLOR2 = "Taylor2 curve"
RESREFINE_FORMULAS = [
    RESREFINE_FORMULA_REUSE,
    RESREFINE_FORMULA_LINEAR,
    RESREFINE_FORMULA_TAYLOR2,
]

RESREFINE_CACHE_DEVICE_CUDA = "cuda"
RESREFINE_CACHE_DEVICE_CPU = "cpu"
RESREFINE_CACHE_DEVICES = [
    RESREFINE_CACHE_DEVICE_CUDA,
    RESREFINE_CACHE_DEVICE_CPU,
]

TEA_SOURCE_FIRST_BLOCK_SHIFT = "first_block_shift"
TEA_SOURCE_TIMESTEP_EMBEDDING = "timestep_embedding"
TEA_MODULATED_SOURCES = [
    TEA_SOURCE_FIRST_BLOCK_SHIFT,
    TEA_SOURCE_TIMESTEP_EMBEDDING,
]

TEA_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT = "Anima 2B 30step first_block_shift"
TEA_PROFILE_IDENTITY = "Identity estimate"

TEA_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT = [
    5954.035087553969,
    -2410.0426539290293,
    349.24023850217395,
    -17.264742642375417,
    0.31229336331906893,
]

# Coefficient profile registry — the single source of truth for both the
# polynomial coefficients (descending powers) and the recommended Start/End
# progress window per profile.
#
# "window" semantics, consumed by tea_window_for_profile():
#   (start, end) -> selecting this profile moves the Start/End sliders there.
#   None         -> leave the Start/End sliders untouched (Identity estimate).
#
# The 24 calibrated presets are 30-steps-only. Their windows are derived from
# each preset's fit step range: start = floor(fit_lo / 29, 0.01),
# end = ceil(fit_hi / 29, 0.01), so the fit endpoints always fall inside the
# window (progress = step_index / (steps - 1), steps = 30). The names embed the
# fit step range for traceability. daraskme's legacy profile resets to the
# original 0.05/0.95 window; Identity carries no window.
TEA_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    TEA_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT: {
        "coefficients": TEA_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
        "window": (0.05, 0.95),
    },
    TEA_PROFILE_IDENTITY: {
        "coefficients": [1.0, 0.0],
        "window": None,
    },
    'ER-SDE-Beta_30steps_Shift1_aggressive(fit2-27step)': {
        "coefficients": [20870041.428504474, -1035486.54180697, 17603.72641341233, -128.73841957973954, 0.5248595690757618],
        "window": (0.06, 0.94),
    },
    'ER-SDE-Beta_30steps_Shift1_balanced(fit6-22step)': {
        "coefficients": [200182946.07957616, -13119216.129569372, 318281.76088353037, -3391.964647805656, 13.495812557199903],
        "window": (0.2, 0.76),
    },
    'ER-SDE-Beta_30steps_Shift1_optimal(fit11-18step)': {
        "coefficients": [-3046316025.6217012, 230984540.5397094, -6555910.571639007, 82538.33363236971, -388.8191463397488],
        "window": (0.37, 0.63),
    },
    'ER-SDE-Beta_30steps_Shift2_aggressive(fit2-27step)': {
        "coefficients": [-2614389.1405815254, 10920.93184366024, 3036.100525221478, -56.09151064031585, 0.3433722071516297],
        "window": (0.06, 0.94),
    },
    'ER-SDE-Beta_30steps_Shift2_balanced(fit7-24step)': {
        "coefficients": [-2817647.979916507, 227511.74600662847, -5838.401698055745, 56.10574150576279, -0.10133103032967061],
        "window": (0.24, 0.83),
    },
    'ER-SDE-Beta_30steps_Shift2_optimal(fit12-21step)': {
        "coefficients": [24396529.365165867, -1681764.8074458963, 43437.05906928717, -500.0356189466142, 2.222156182641241],
        "window": (0.41, 0.73),
    },
    'ER-SDE-Beta_30steps_Shift3_aggressive(fit2-27step)': {
        "coefficients": [633088.7404623311, -70130.49788449529, 2787.42902427481, -41.620894905058606, 0.2581064169845272],
        "window": (0.06, 0.94),
    },
    'ER-SDE-Beta_30steps_Shift3_balanced(fit8-24step)': {
        "coefficients": [3495295.7712097946, -211825.07806675878, 4734.462702493498, -47.98347778390063, 0.2485471655224362],
        "window": (0.27, 0.83),
    },
    'ER-SDE-Beta_30steps_Shift3_optimal(fit14-22step)': {
        "coefficients": [5088727.742530627, -359887.3234944483, 9494.71732083433, -111.57649673251274, 0.5483313501809356],
        "window": (0.48, 0.76),
    },
    'ER-SDE-Simple_30steps_Shift2_aggressive(fit2-27step)': {
        "coefficients": [39102112.026053846, -2318082.0617386415, 51392.344907187115, -502.48805779661876, 1.8619651883977906],
        "window": (0.06, 0.94),
    },
    'ER-SDE-Simple_30steps_Shift2_balanced(fit6-25step)': {
        "coefficients": [23681509.071739938, -1306538.8400072563, 27262.418554592794, -255.7529086014123, 0.9534489360323399],
        "window": (0.2, 0.87),
    },
    'ER-SDE-Simple_30steps_Shift2_optimal(fit10-23step)': {
        "coefficients": [-1407644.652502646, 99616.97324952048, -2004.5239581243582, 12.386900591770305, 0.04077433508697838],
        "window": (0.34, 0.8),
    },
    'ER-SDE-Simple_30steps_Shift3_aggressive(fit2-27step)': {
        "coefficients": [8284353.06193094, -551831.3883056089, 13354.156032515975, -137.43051017862464, 0.5463250234356548],
        "window": (0.06, 0.94),
    },
    'ER-SDE-Simple_30steps_Shift3_balanced(fit7-26step)': {
        "coefficients": [3190536.431767993, -190485.32545561742, 4341.688228815269, -44.87145703832903, 0.22016077772870743],
        "window": (0.24, 0.9),
    },
    'ER-SDE-Simple_30steps_Shift3_optimal(fit12-24step)': {
        "coefficients": [4971804.351014275, -292598.00657492253, 6516.5694706091335, -65.240199323566, 0.290659670461512],
        "window": (0.41, 0.83),
    },
    'Euler-Beta_30steps_Shift1_aggressive(fit2-27step)': {
        "coefficients": [11929119.677434778, -463763.78124246205, 4032.826670335884, 13.787806744866383, -0.05819449599836593],
        "window": (0.06, 0.94),
    },
    'Euler-Beta_30steps_Shift1_balanced(fit6-22step)': {
        "coefficients": [131636494.93557017, -8205633.993442775, 186457.8220189453, -1822.126502201342, 6.4700841745638025],
        "window": (0.2, 0.76),
    },
    'Euler-Beta_30steps_Shift1_optimal(fit10-17step)': {
        "coefficients": [-1980480799.2037754, 151799126.6493625, -4358279.328270996, 55549.07004802356, -265.1561879995355],
        "window": (0.34, 0.59),
    },
    'Euler-Beta_30steps_Shift2_aggressive(fit2-27step)': {
        "coefficients": [-4855098.590777344, 166376.81147842054, -1004.6836785374982, -8.261819559097047, 0.10212622887564968],
        "window": (0.06, 0.94),
    },
    'Euler-Beta_30steps_Shift2_balanced(fit6-23step)': {
        "coefficients": [5961994.036980679, -313824.8564017251, 6039.27168697529, -51.75556189162014, 0.20596269199248698],
        "window": (0.2, 0.8),
    },
    'Euler-Beta_30steps_Shift2_optimal(fit11-19step)': {
        "coefficients": [-6415391.142441988, 449560.9302189869, -11537.67025327111, 127.9371079638685, -0.48462846605011906],
        "window": (0.37, 0.66),
    },
    'Euler-Beta_30steps_Shift3_aggressive(fit2-27step)': {
        "coefficients": [-2144622.6563445744, 120746.33040078155, -1915.8586366851887, 8.242200354949615, 0.03828979084201344],
        "window": (0.06, 0.94),
    },
    'Euler-Beta_30steps_Shift3_balanced(fit8-24step)': {
        "coefficients": [3253531.569517961, -191214.11588000128, 4062.086583316588, -37.40909154427277, 0.1594626949735595],
        "window": (0.27, 0.83),
    },
    'Euler-Beta_30steps_Shift3_optimal(fit13-21step)': {
        "coefficients": [409105.2158414339, -34196.33629136863, 1109.5681200097883, -16.05960638388326, 0.11545517747079355],
        "window": (0.44, 0.73),
    },
}

# Default profile is now a re-calibrated preset (was daraskme's legacy profile).
TEA_PROFILE_DEFAULT = 'ER-SDE-Beta_30steps_Shift3_optimal(fit14-22step)'

TEA_COEFFICIENT_PROFILES = list(TEA_PRESET_REGISTRY)


def tea_coefficients_for_profile(profile: str) -> list[float]:
    """Polynomial coefficients (descending powers) for a coefficient profile.

    Single source of truth shared by the runtime skip decision and the read-only
    p_Anima(x) UI display. Unknown profiles fall back to the default preset.
    """
    entry = TEA_PRESET_REGISTRY.get(profile) or TEA_PRESET_REGISTRY[TEA_PROFILE_DEFAULT]
    return entry["coefficients"]


def tea_window_for_profile(profile: str) -> tuple[float, float] | None:
    """Recommended (start, end) progress window for a profile, or None.

    None means "leave the Start/End sliders untouched" (Identity estimate). A
    tuple means selecting the profile should move the sliders there — the 24
    calibrated presets to their fit range, daraskme back to 0.05/0.95.
    """
    entry = TEA_PRESET_REGISTRY.get(profile)
    if entry is None:
        return None
    return entry["window"]


_PROFILE_SHIFT_PATTERN = re.compile(r"Shift(\d+)")


def tea_expected_shift_for_profile(profile: str) -> int | None:
    """The Shift value a calibrated preset was fitted for, parsed from its name.

    Returns the integer in `..._Shift<n>_...`, or None when the profile carries
    no shift expectation (daraskme legacy, Identity). Used to warn when the live
    model's effective shift does not match the selected preset — a mismatch
    moves the sigma schedule off the coefficients' fitted domain.
    """
    match = _PROFILE_SHIFT_PATTERN.search(profile or "")
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def tea_modulated_source_for_profile(profile: str) -> str:
    """Modulated source is derived from the coefficient profile, not chosen in the UI.

    Identity estimate pairs with timestep_embedding; every other profile uses
    first_block_shift. This is the single source of truth for that mapping.
    """
    if profile == TEA_PROFILE_IDENTITY:
        return TEA_SOURCE_TIMESTEP_EMBEDDING
    return TEA_SOURCE_FIRST_BLOCK_SHIFT


@dataclass
class RuntimeState:
    enabled: bool = False
    debug_log_enabled: bool = False
    mode: str = MODE_OFF
    print_timing_log: bool = True
    verbose_diagnose_log: bool = False
    dump_resrefine_residual: bool = False

    hareskip_enabled: bool = False
    tea_preset: str = TEA_PRESET_CUSTOM
    tea_threshold: float = 0.07
    tea_start_percent: float = 0.48
    tea_end_percent: float = 0.76
    resrefine_formula: str = RESREFINE_FORMULA_REUSE
    resrefine_use_prediction_after_progress: float = 0.0
    resrefine_apply_prediction_from_skip: int = 2
    resrefine_prediction_strength: float = 0.50
    resrefine_taylor2_curve_strength: float = 0.25
    resrefine_slope_ema_smoothing: float = 0.0
    resrefine_curve_ema_smoothing: float = 0.0
    resrefine_cache_device: str = RESREFINE_CACHE_DEVICE_CUDA
    tea_modulated_source: str = TEA_SOURCE_FIRST_BLOCK_SHIFT
    tea_coefficient_profile: str = TEA_PROFILE_DEFAULT
    tea_max_skip_streak: int = 0
    tea_force_full_interval: int = 0
    hareskip_dry_run: bool = False
    hareskip_verbose_trace: bool = False
    capture_calibration_pairs: bool = False

    # HareSkip stochastic-mode configuration and per-generation pattern state.
    # hareskip_mode / hareskip_aggressiveness / hareskip_skip_seed_offset are
    # wired through apply_options from the UI (see constants.UI_ARG_ORDER).
    hareskip_mode: str = MODE_HARESKIP
    hareskip_aggressiveness: float = 0.5
    hareskip_skip_seed_offset: int = 0
    # User-configurable skip window (progress domain) and zone boundaries (z
    # domain), wired through apply_options from the dual-thumb RangeSliders
    # (see constants.UI_ARG_ORDER). Defaults reproduce the former guard rule
    # and the 3-zone boundaries. Stored as separate scalars; the pattern
    # generator receives them as tuples on generation start.
    hareskip_window_start: float = 0.05
    hareskip_window_end: float = 0.95
    hareskip_zone_low: float = -4.0
    hareskip_zone_high: float = 0.0
    hareskip_probability_model: str = "sigmoid_band_v0.1"
    hareskip_image_seed: Optional[int] = None
    # Manual Skip mode: raw comma-separated step text from the UI, and the
    # validated 1-based step list parsed from it at generation start (set in
    # before_process after validate_manual_steps succeeds; None until then and
    # between generations). The patcher reads the parsed list, not the raw text.
    manual_skip_steps: str = ""
    manual_skip_parsed: Optional[list] = None
    # Per-step flow-time (t_now) schedule captured for this generation.
    hareskip_schedule_t_now: Optional[list] = None
    # The generated SkipPattern for this generation (skip_pattern.SkipPattern).
    hareskip_pattern: Any = None
    hareskip_schedule_warned: bool = False

    auto_teacache_enabled: bool = False
    auto_teacache_csv: str = ""
    auto_teacache_active: bool = False
    auto_teacache_row_index: int | None = None
    auto_teacache_row_name: str | None = None
    auto_teacache_row_count: int = 0
    auto_teacache_original_n_iter: int = 1
    auto_teacache_parse_error: str | None = None

    status: str = "disabled"
    error_message: str | None = None
    model_detection: Any | None = None
    warned_model_keys: set[str] = field(default_factory=set)
    generation_index: int = 0
    generation_steps: int | None = None
    generation_start: float | None = None
    generation_start_source: str | None = None
    step_start: float | None = None
    step_durations: list[float] = field(default_factory=list)
    denoiser_calls: int = 0
    cond_trace_logged: bool = False

    hareskip_model_calls: int = 0
    hareskip_full_calcs: int = 0
    hareskip_skips: int = 0
    hareskip_skipped_steps: list[int] = field(default_factory=list)
    resrefine_prediction_used: int = 0
    hareskip_fallback_used: int = 0
    hareskip_dry_run_predictions: int = 0
    hareskip_first_full_calcs: int = 0
    hareskip_forced_full_calcs: int = 0
    hareskip_fallbacks: int = 0
    hareskip_errors: int = 0
    hareskip_logged_calls: int = 0
    hareskip_num_blocks: int | None = None
    hareskip_unavailable_reason: str | None = None
    hareskip_fallback_reasons: dict[str, int] = field(default_factory=dict)

    tensor_dump_run_dir: str | None = None
    tensor_dump_initialized: bool = False
    tensor_dump_records: int = 0
    tensor_dump_errors: int = 0
    tensor_dump_unavailable_reason: str | None = None
    tensor_dump_hareskip_local_call_index: int = 0
    tensor_dump_warned_reasons: set[str] = field(default_factory=set)

    calibration_capture_path: str | None = None
    calibration_capture_header_written: bool = False
    calibration_capture_records: int = 0
    calibration_capture_errors: int = 0
    calibration_capture_warned_reasons: set[str] = field(default_factory=set)

    tea_shift_warned_keys: set[str] = field(default_factory=set)

    generation_logged: bool = False
    patches: dict[str, Any] = field(default_factory=dict)

    def refresh_settings(self) -> None:
        try:
            from modules import shared

            self.enabled = bool(getattr(shared.opts, "hareskip_enable", False))
            self.debug_log_enabled = bool(
                getattr(shared.opts, "hareskip_debug_log_enable", False)
            )
            mode = getattr(shared.opts, "hareskip_debug_mode", MODE_OFF)
            self.mode = _normalize_mode(mode) if self.debug_log_enabled else MODE_OFF
            self.print_timing_log = bool(
                getattr(shared.opts, "hareskip_print_timing_log", True)
            )
            self.verbose_diagnose_log = bool(
                getattr(shared.opts, "hareskip_verbose_diagnose_log", False)
            )
            self.hareskip_enabled = self.enabled
        except Exception as exc:
            self.enabled = False
            self.hareskip_enabled = False
            self.mode = MODE_OFF
            self.status = "error"
            self.error_message = f"failed to read settings: {exc}"

    def apply_options(
        self,
        enabled: bool,
        debug_log_enabled: bool,
        mode: str,
        print_timing_log: bool,
        verbose_diagnose_log: bool,
        dump_resrefine_residual: bool,
        tea_preset: str = TEA_PRESET_CUSTOM,
        tea_threshold: float = 0.07,
        tea_start_percent: float = 0.48,
        tea_end_percent: float = 0.76,
        resrefine_formula: str = RESREFINE_FORMULA_REUSE,
        resrefine_use_prediction_after_progress: float = 0.0,
        resrefine_apply_prediction_from_skip: int = 2,
        resrefine_prediction_strength: float = 0.50,
        resrefine_taylor2_curve_strength: float = 0.25,
        resrefine_slope_ema_smoothing: float = 0.0,
        resrefine_curve_ema_smoothing: float = 0.0,
        resrefine_cache_device: str = RESREFINE_CACHE_DEVICE_CUDA,
        tea_coefficient_profile: str = TEA_PROFILE_DEFAULT,
        tea_max_skip_streak: int = 0,
        tea_force_full_interval: int = 0,
        hareskip_dry_run: bool = False,
        hareskip_verbose_trace: bool = False,
        auto_teacache_enabled: bool = False,
        auto_teacache_csv: str = "",
        capture_calibration_pairs: bool = False,
        hareskip_mode: str = MODE_HARESKIP,
        hareskip_aggressiveness: float = 0.5,
        hareskip_skip_seed_offset: int = 0,
        hareskip_window_start: float = 0.05,
        hareskip_window_end: float = 0.95,
        hareskip_zone_low: float = -4.0,
        hareskip_zone_high: float = 0.0,
        manual_skip_steps: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.debug_log_enabled = bool(debug_log_enabled)
        self.mode = _normalize_mode(mode) if self.debug_log_enabled else MODE_OFF
        self.print_timing_log = bool(print_timing_log)
        self.verbose_diagnose_log = bool(verbose_diagnose_log)
        self.dump_resrefine_residual = bool(dump_resrefine_residual)

        self.hareskip_enabled = self.enabled
        self.tea_preset = (
            tea_preset if tea_preset in TEA_PRESETS else TEA_PRESET_CUSTOM
        )
        self.tea_threshold = _clamp_float(tea_threshold, 0.0, 1.0)
        self.tea_start_percent = _clamp_float(tea_start_percent, 0.0, 1.0)
        self.tea_end_percent = _clamp_float(tea_end_percent, 0.0, 1.0)
        if self.tea_start_percent > self.tea_end_percent:
            self.tea_start_percent, self.tea_end_percent = (
                self.tea_end_percent,
                self.tea_start_percent,
            )
        self.resrefine_formula = (
            resrefine_formula if resrefine_formula in RESREFINE_FORMULAS else RESREFINE_FORMULA_REUSE
        )
        self.resrefine_use_prediction_after_progress = _clamp_float(
            resrefine_use_prediction_after_progress,
            0.0,
            1.0,
        )
        self.resrefine_apply_prediction_from_skip = _clamp_int(
            resrefine_apply_prediction_from_skip,
            1,
            3,
        )
        self.resrefine_prediction_strength = _clamp_float(resrefine_prediction_strength, 0.0, 1.0)
        self.resrefine_taylor2_curve_strength = _clamp_float(resrefine_taylor2_curve_strength, 0.0, 1.0)
        self.resrefine_slope_ema_smoothing = _clamp_float(resrefine_slope_ema_smoothing, 0.0, 0.99)
        self.resrefine_curve_ema_smoothing = _clamp_float(resrefine_curve_ema_smoothing, 0.0, 0.99)
        self.resrefine_cache_device = (
            resrefine_cache_device
            if resrefine_cache_device in RESREFINE_CACHE_DEVICES
            else RESREFINE_CACHE_DEVICE_CUDA
        )
        self.tea_coefficient_profile = (
            tea_coefficient_profile
            if tea_coefficient_profile in TEA_COEFFICIENT_PROFILES
            else TEA_PROFILE_DEFAULT
        )
        self.tea_modulated_source = tea_modulated_source_for_profile(
            self.tea_coefficient_profile
        )
        self.tea_max_skip_streak = _clamp_int(tea_max_skip_streak, 0, 64)
        self.tea_force_full_interval = _clamp_int(tea_force_full_interval, 0, 64)
        self.hareskip_dry_run = bool(hareskip_dry_run)
        self.hareskip_verbose_trace = bool(hareskip_verbose_trace)

        self.auto_teacache_enabled = bool(auto_teacache_enabled) and self.hareskip_enabled
        self.auto_teacache_csv = str(auto_teacache_csv or "")
        self.capture_calibration_pairs = bool(capture_calibration_pairs)

        self.hareskip_mode = (
            hareskip_mode if hareskip_mode in HARESKIP_MODES else MODE_HARESKIP
        )
        self.hareskip_aggressiveness = _clamp_float(hareskip_aggressiveness, 0.0, 1.0)
        try:
            self.hareskip_skip_seed_offset = int(hareskip_skip_seed_offset)
        except Exception:
            self.hareskip_skip_seed_offset = 0

        # Skip window (progress domain): clamp to [0, 1], enforce start <= end.
        self.hareskip_window_start, self.hareskip_window_end = _normalize_range(
            hareskip_window_start, hareskip_window_end, 0.0, 1.0, 0.05, 0.95
        )
        # Zone boundaries (z domain): clamp to [-8, 8], enforce low <= high.
        self.hareskip_zone_low, self.hareskip_zone_high = _normalize_range(
            hareskip_zone_low, hareskip_zone_high, -8.0, 8.0, -4.0, 0.0
        )
        # Manual Skip: keep the raw text (stripped); validation/parsing happens
        # later in before_process against p.steps.
        self.manual_skip_steps = str(manual_skip_steps or "").strip()

    def active(self) -> bool:
        return (
            self.hareskip_enabled
            or (self.debug_log_enabled and self.mode != MODE_OFF)
            or self.tensor_dump_active()
        )

    def tensor_dump_requested(self) -> bool:
        return self.dump_resrefine_residual or self.capture_calibration_pairs

    def calibration_capture_active(self) -> bool:
        return self.enabled and self.debug_log_enabled and self.capture_calibration_pairs

    def tensor_dump_active(self) -> bool:
        return self.enabled and self.debug_log_enabled and self.tensor_dump_requested()

    def reset_generation(self, source: str = "unknown") -> None:
        self.generation_index += 1
        self.generation_steps = None
        self.generation_start = perf_counter()
        self.generation_start_source = source
        self.step_start = None
        self.step_durations.clear()
        self.denoiser_calls = 0
        self.cond_trace_logged = False
        self.hareskip_model_calls = 0
        self.hareskip_full_calcs = 0
        self.hareskip_skips = 0
        self.hareskip_skipped_steps.clear()
        self.resrefine_prediction_used = 0
        self.hareskip_fallback_used = 0
        self.hareskip_dry_run_predictions = 0
        self.hareskip_first_full_calcs = 0
        self.hareskip_forced_full_calcs = 0
        self.hareskip_fallbacks = 0
        self.hareskip_errors = 0
        self.hareskip_logged_calls = 0
        self.hareskip_num_blocks = None
        self.hareskip_unavailable_reason = None
        self.hareskip_fallback_reasons.clear()
        self.hareskip_image_seed = None
        self.hareskip_schedule_t_now = None
        self.hareskip_pattern = None
        self.hareskip_schedule_warned = False
        # NOTE: manual_skip_parsed is deliberately NOT reset here. It is
        # validated once per job in _prepare_manual_skip_run (before_process),
        # which always clears + revalidates + restores it, so no stale value can
        # cross a job boundary. Clearing it here (per-generation, i.e. per
        # sampling pass) would wipe the validated list before the patcher reads
        # it, disabling Manual Skip entirely (bug fixed 2026-07-13). Config-style
        # fields (e.g. skip seed offset) are likewise not reset here.
        self.tensor_dump_run_dir = None
        self.tensor_dump_initialized = False
        self.tensor_dump_records = 0
        self.tensor_dump_errors = 0
        self.tensor_dump_unavailable_reason = None
        self.tensor_dump_hareskip_local_call_index = 0
        self.tensor_dump_warned_reasons.clear()
        self.calibration_capture_path = None
        self.calibration_capture_header_written = False
        self.calibration_capture_records = 0
        self.calibration_capture_errors = 0
        self.calibration_capture_warned_reasons.clear()
        self.generation_logged = False
        self.error_message = None

        if not self.enabled:
            self.status = "disabled"
        elif self.hareskip_enabled:
            self.status = "experimental-hareskip"
        elif self.mode == MODE_DIAGNOSE:
            self.status = "diagnosing"
        else:
            self.status = "disabled"

    def mark_step_start(self) -> None:
        self.step_start = perf_counter()
        self.denoiser_calls += 1

    def mark_step_end(self) -> None:
        if self.step_start is None:
            return
        self.step_durations.append(perf_counter() - self.step_start)
        self.step_start = None

    def total_sampling_time(self) -> float | None:
        if self.generation_start is None:
            return None
        return perf_counter() - self.generation_start

    def avg_step_time(self) -> float | None:
        if not self.step_durations:
            return None
        return sum(self.step_durations) / len(self.step_durations)

    def set_error(self, message: str) -> None:
        self.status = "error"
        self.error_message = message


STATE = RuntimeState()


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _normalize_range(
    lo_value: Any,
    hi_value: Any,
    minimum: float,
    maximum: float,
    default_lo: float,
    default_hi: float,
) -> tuple[float, float]:
    """Clamp a (lo, hi) pair to [minimum, maximum] and enforce lo <= hi.

    Non-numeric inputs fall back to (default_lo, default_hi). If the clamped
    lo exceeds hi they are swapped, so a reversed range is corrected rather
    than rejected. Pure and Forge-independent — unit-testable in isolation.
    """
    try:
        lo = float(lo_value)
    except Exception:
        lo = default_lo
    try:
        hi = float(hi_value)
    except Exception:
        hi = default_hi
    lo = max(minimum, min(maximum, lo))
    hi = max(minimum, min(maximum, hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _normalize_mode(value: Any) -> str:
    text = str(value or MODE_OFF)
    if text == "Off":
        return MODE_OFF
    return text if text in MODES else MODE_OFF
