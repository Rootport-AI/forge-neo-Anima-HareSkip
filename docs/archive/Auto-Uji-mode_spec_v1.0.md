# Auto Uji Mode Spec v1.0

## Purpose

Auto Uji mode runs multiple UjiCache parameter rows from one CSV in a single Generate action.

## UI

```text
UjiCache
  Enable UjiCache
  Auto Uji mode
    Enable Auto Uji mode
    Auto Uji CSV
```

Auto Uji mode is active only when both `Enable UjiCache` and `Enable Auto Uji mode` are ON.
It must not enable UjiCache automatically.

## CSV

Supported columns:

```csv
name,threshold,formula,prediction_strength,slope_ema_smoothing,curve_ema_smoothing,taylor2_curve_strength,apply_prediction_from_skip,use_prediction_after_progress,max_skip_streak,force_full_interval,coefficient_profile,start_percent,end_percent
teacache_ref,0.07,teacache,0.00,0.00,0.00,0.25,2,0.00,0,0,,,
linear_a,0.21,linear,0.50,0.20,0.00,0.25,2,0.00,0,0,,,
profile_a,0.15,teacache,0.00,0.00,0.00,0.25,2,0.00,0,0,ER-SDE-Beta_30steps_Shift3_optimal(fit14-22step),,
```

Formula aliases:

- `teacache`, `residual`, `residual_only` -> `TeaCache (residual only)`
- `linear`, `linear_extrapolation` -> `Linear extrapolation`
- `taylor2`, `taylor`, `quadratic` -> `Taylor2 curve`

Unspecified values use the current UjiCache UI values.

### `coefficient_profile`

The value must match a registered Coefficient profile name **exactly** (the keys
of `UJICACHE_PRESET_REGISTRY` / the `Coefficient profile` dropdown — daraskme,
Identity, and the 24 calibrated presets). A non-matching value raises
`AutoUjiCsvError` (`reason=unknown_profile`) and aborts the run, like any other
invalid value. Auto Uji does not hardcode the list; it only checks exact string
membership.

Selecting a profile also derives `Modulated source` automatically (Identity ->
`timestep_embedding`, otherwise `first_block_shift`), the same as the UI.

### `start_percent` / `end_percent` (window resolution)

Start and End are resolved **independently**, in this priority:

1. An explicit row value (`start_percent` / `end_percent`) wins.
2. Otherwise, if `coefficient_profile` is set, that profile's fit window applies
   (daraskme -> 0.05/0.95; the calibrated presets -> their fit range; Identity
   carries no window, so the current value is kept).
3. Otherwise the current `STATE` value is left untouched.

So a row may set only `start_percent` and let the profile supply End, or set the
window without a profile (pure window sweep, profile left untouched).

## Runtime

Each CSV row is treated as one UjiCache condition. Forge `batch_size` is preserved, and `n_iter` is expanded to `CSV row count * original n_iter`.

The seed set is shared per CSV row, allowing parameter comparisons under the same seed conditions.

## State

Runtime rows are stored on `p._ujicache_auto_rows`. The current row state is mirrored into `STATE`.

Main state fields:

- `auto_ujicache_enabled`
- `auto_ujicache_csv`
- `auto_ujicache_active`
- `auto_ujicache_row_index`
- `auto_ujicache_row_name`
- `auto_ujicache_row_count`
- `auto_ujicache_original_n_iter`

## Logs

```text
[UjiCache] auto_uji_prepare rows=...
[UjiCache] auto_uji_seed_template rows=...
[UjiCache] auto_uji_row_start index=...
[UjiCache] auto_uji_csv_error ...
```

## Acceptance

1. Auto Uji OFF behaves like a normal UjiCache generation.
2. Auto Uji does not run when `Enable UjiCache` is OFF.
3. Each CSV row applies its UjiCache values to `STATE`.
4. `previous_residual` and `residual_history` are not shared across rows.
5. Auto Uji does not change the core UjiCache prediction logic.
6. A `coefficient_profile` value that does not exactly match a registered profile
   aborts the run with `AutoUjiCsvError` (`reason=unknown_profile`).
7. When a row sets `coefficient_profile` without `start_percent`/`end_percent`,
   the profile's fit window is applied; an explicit start/end value overrides it
   per side.
