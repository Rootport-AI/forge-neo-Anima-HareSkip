# Changelog

## Unreleased

- Added: `Capture calibration pairs` (Debug log mode) — per-step (rel_l1, out_rel) JSONL capture with run conditions incl. Shift; forces full calculation on every model call.
- Added: `Uji shift` / `Uji capture_pairs` infotext keys.
- Added: 24 re-calibrated Coefficient profile presets (ER-SDE / Euler × Beta / Simple × Shift1-3 × aggressive/balanced/optimal), fitted from Forge Neo capture data. Each preset is 30-steps-only and paired with a fit window. See `docs/PRESET-COEFFICIENTS.md`.
- Changed: default Coefficient profile is now `ER-SDE-Beta_30steps_Shift3_optimal(fit14-22step)` (was daraskme's legacy profile); default Start/End progress now 0.48/0.76 to match it. daraskme and Identity profiles are retained.
- Added: selecting a Coefficient profile loosely moves the Start/End sliders to the profile's fit window (calibrated presets) or 0.05/0.95 (daraskme); Identity leaves them untouched. Sliders remain user-adjustable.
- Changed: UI reorder — `Coefficient profile` moved above `Start/End progress`; new read-only `p_Anima(x)` display of the active polynomial.
- Changed: `Modulated source` removed from the UI; it is now derived from the Coefficient profile (Identity → timestep_embedding, otherwise first_block_shift) and still recorded in `Uji modulated_source`.
- Changed: `Debug log mode` now defaults to OFF.
- Added: Auto Uji mode CSV columns `coefficient_profile`, `start_percent`, `end_percent`. `coefficient_profile` requires an exact match against a registered profile (else `AutoUjiCsvError`); selecting one derives Modulated source and applies the profile's fit window unless an explicit start/end overrides it per side. A row may also sweep the window alone without a profile.
- Added: Shift-mismatch warning — at generation start, UjiCache compares a calibrated preset's expected Shift (parsed from its name) against the model's effective shift and logs `ujicache_shift_mismatch` (once per session per pair) when they differ. A mismatch moves the coefficients off their fitted domain and distorts skip decisions; generation is not blocked.

## 0.1.0

- Split UjiCache out as an independent Forge Neo extension.
- Rename the package to `ujicache` and the Forge entrypoint to `scripts/ujicache.py`.
- Replace the top-level panel with `UjiCache`.
- Keep `Debug log mode` as a sub-accordion.
- Remove PredLab-only UI and runtime paths for attention override, standalone TeaCache, Spectrum, 2D sparse attention, cond/uncond optimization, low-bit, compile, and identity patch experiments.
- Keep UjiCache's internal TeaCache-style skip decision helpers where needed by the residual prediction prototype.
- Update logging to use the `[UjiCache]` console prefix.

