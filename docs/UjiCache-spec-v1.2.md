# UjiCache Spec v1.4

## Overview

`UjiCache` is a Forge Neo extension for residual reuse experiments in Anima / Cosmos-Predict2 inference.
As an independent extension, it exposes only the UjiCache controls, the `Debug log mode` sub-accordion, and `Auto Uji mode`.

## UI

```text
UjiCache
  Enable UjiCache
  UjiCache preset
  Rel L1 threshold
  Coefficient profile
  p_Anima(x)            (read-only display of the active polynomial)
  Start progress
  End progress
  Max skip streak
  Force full interval
  Prediction formula
  Prediction controls
  Auto Uji mode
    Enable Auto Uji mode
    Auto Uji CSV
  Debug log mode
    Enable debug log mode
    Debug log
    Dump UjiCache residual
    Capture calibration pairs
    Print timing log
    Verbose diagnose log
  Cache device
  Dry run
  Verbose UjiCache trace
```

`Enable UjiCache` is the single UjiCache patch toggle. There is no separate experiment enable checkbox.
`Debug log mode` remains a sub-accordion and does not enable UjiCache automatically.

`Modulated source` is no longer a UI control. It is derived from the selected
`Coefficient profile` (see below): `Identity estimate` uses `timestep_embedding`,
every other profile uses `first_block_shift`. The derived value is still recorded
in the `Uji modulated_source` infotext key.

## Coefficient profile

The `Coefficient profile` dropdown selects the polynomial `p_Anima(x)` used by the
TeaCache-style skip decision (the rescaling of the modulated-input rel L1 into an
estimated output change). Profiles:

- `Anima 2B 30step first_block_shift`: daraskme's ComfyUI-derived legacy coefficients.
- `Identity estimate`: `p(x) = x`.
- 24 re-calibrated presets (`<sampler>-<scheduler>_30steps_Shift<n>_<mode>(fit<lo>-<hi>step)`),
  fitted from Forge Neo `Capture calibration pairs` data. Each preset is **30 steps only**
  and is paired with a fit window; the name embeds the fit step range. Full coefficient
  list and provenance: `docs/PRESET-COEFFICIENTS.md` / `docs/PRESET-COEFFICIENTS.json`.

Coefficients and the recommended window live together in a single registry
(`UJICACHE_PRESET_REGISTRY` in `state.py`). The default profile is
`ER-SDE-Beta_30steps_Shift3_optimal(fit14-22step)`.

### p_Anima(x) display and loose window coupling

`p_Anima(x)` is a read-only Markdown line showing the active polynomial in descending
powers at full precision (never rounded). It is not editable.

Selecting a profile **loosely** drives the `Start progress` / `End progress` sliders:

- A calibrated preset moves the sliders to its fit window (`start = floor(fit_lo / (steps-1), 0.01)`,
  `end = ceil(fit_hi / (steps-1), 0.01)`, with `steps = 30`), so the fit endpoints stay inside the window.
- `Anima 2B 30step first_block_shift` resets the sliders to `0.05 / 0.95`.
- `Identity estimate` leaves the sliders untouched.

After a profile change the user can still move the sliders by hand. The fit windows
assume 30 steps; using a calibrated preset at a different step count moves the window
out of the coefficients' fitted domain (30-steps-only by design).

### Shift must match the preset

Each calibrated preset is fitted for a specific Shift (`..._Shift<n>_...` in the name).
The Shift warps the sigma schedule, which determines the `rel_l1` sequence the
polynomial sees. Running a preset at a different model Shift moves `rel_l1` off the
coefficients' fitted domain and silently distorts the skip decisions (e.g. a `Shift1`
preset run at effective shift 3 inverts the expected threshold sensitivity).

The model's effective shift is `forge_objects.unet.model.predictor.shift`
(`PredictionDiscreteFlow` for Anima), read via
`forge_introspection.model_sampling_info()`. At generation start, UjiCache compares
the preset's expected Shift against the effective shift; on mismatch it logs (once per
session per pair):

```text
[UjiCache] ujicache_shift_mismatch profile=... expected_shift=N effective_shift=M; ...
```

Generation is not blocked — the warning is informational. daraskme legacy and Identity
profiles carry no Shift and are not checked.

## Prediction Formulas

- `TeaCache (residual only)`: reuse the most recent full-calculation residual without prediction.
- `Linear extrapolation`: estimate the next residual from residual history.
- `Taylor2 curve`: mix a second-order curve estimate from residual history.

UjiCache uses TeaCache-like skip decision logic internally, but it does not expose the old standalone TeaCache experiment UI.

## Safety

- `Enable UjiCache` OFF preserves the Forge baseline and removes the `Anima.forward` patch.
- The first model call must always be a full calculation.
- Missing `previous_residual` must fall back to full calculation.
- Residual shape / dtype / device mismatch must fall back safely.
- Unsupported model, disable, unload, and degraded paths must restore monkey patches.
- Exceptions and fallbacks are logged with the `[UjiCache]` prefix.
- Unused Forge Neo kwargs such as `control` must be ignored safely.

## Logs

```text
[UjiCache] ujicache_config=...
[UjiCache] ujicache_call=...
[UjiCache] ujicache_summary=...
```

When `Verbose UjiCache trace` is OFF, per-step detail logs are limited. Summary logs are printed when UjiCache is enabled.

## Metadata

When UjiCache is enabled, PNG infotext receives `Uji ...` keys:

- `Uji enabled`
- `Uji formula`
- `Uji threshold`
- `Uji progress`
- `Uji use_prediction_after_progress`
- `Uji apply_prediction_from_skip`
- `Uji prediction_strength`
- `Uji taylor2_curve_strength`
- `Uji slope_ema_smoothing`
- `Uji curve_ema_smoothing`
- `Uji modulated_source` (derived from the coefficient profile)
- `Uji coefficient_profile`
- `Uji max_skip_streak`
- `Uji force_full_interval`
- `Uji shift` (when the model sampling object exposes shift)
- `Uji capture_pairs` (when `Capture calibration pairs` is on)
- `Uji auto_row_index` / `Uji auto_row_name` (Auto Uji mode only)

## Capture calibration pairs

`Capture calibration pairs` (Debug log mode) forces full calculation on every
model call and writes `calibration_pairs.jsonl` into the debug run folder: one
`type=run` header line with full generation conditions (including Shift and the
active coefficient list) and one `type=pair` line per (model call x cond/uncond
slot) carrying `rel_l1` (x), `out_rel` (y), `estimate`, `t_now`, `t_prev`. The
schema is compatible with `daraskme/comfy_anima_tea_cache` so its
`np.polyfit(rels, outs, deg=4)` step reuses Forge-captured data verbatim.
Requires `Enable UjiCache` and `Enable debug log mode`; off by default and a
no-op when off.

## Implementation Files

- `scripts/ujicache.py`
- `ujicache/script.py`
- `ujicache/state.py`
- `ujicache/patcher.py`
- `ujicache/diagnostics.py`
- `ujicache/auto_ujicache.py`
- `ujicache/tensor_dump.py`
- `ujicache/calibration_capture.py`

## Acceptance

1. Forge Neo shows this extension as `UjiCache`.
2. `Debug log mode` is shown as a sub-accordion.
3. Old PredLab Attention / TeaCache / Spectrum / Sparse / Cond / Low-bit / Compile UI is not shown.
4. `Enable UjiCache` toggles the UjiCache patch.
5. `TeaCache (residual only)` works as nearby residual reuse.
6. The first call and missing residual cases run full calculation.
7. `ujicache_summary` reports model_calls / full_calcs / skips / fallbacks / errors / active.
