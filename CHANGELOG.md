# Changelog

## Unreleased (post-alpha)

- Fixed: **Manual Skip list was silently wiped every sampling pass** — `reset_generation()` unconditionally cleared `manual_skip_parsed` after `before_process` had already validated and stored it, so the patcher always saw an empty list and Manual Skip never actually skipped anything. The reset call was removed; `before_process` already clears + revalidates + restores the list once per job, so no stale value can cross a job boundary. (`6dee42e`)
- Added: **defensive patching against foreign `Anima.forward` patches** — a co-installed extension (e.g. legacy UjiCache) patching the same target could silently overwrite HareSkip's wrapper, or restore its own saved original over it on teardown, removing HareSkip from the call chain undetected (`model_calls=0`). `STATE.patches[kind]` now also tracks wrapper/cls/attr so patch identity can be verified without a Forge import: apply-time self-heal (warns `hareskip_patch_clobbered` and reapplies if the installed attribute is no longer our wrapper) and remove-time guard (warns `hareskip_patch_not_ours` and drops the registry entry without touching the attribute if we're not the one installed). Added `is_patch_installed()` (wrapper-identity check, used for the diagnostics `active=` field) alongside the existing cheap `is_patched()` (registry membership). `log_timing_summary` now warns `hareskip_patch_never_ran` when enabled with denoiser calls but zero model calls. (`28d24ee`)
- Changed: **console step logging unified to 1-based `step=n/total`** at the three remaining 0-based sites (`hareskip_call=`, `hareskip_step=`, `hareskip_summary= skipped_steps=`), via a shared `_fmt_step_display` helper. Internal storage stays 0-based; only display output converts. Manual Skip's normal full-compute steps now report `reason=manual_full` (was incorrectly falling back to the Tea-only `reason=threshold`); `_hareskip_log_call` now puts the explicit reason first and appends per-slot ResRefine reasons after it (e.g. `reason=manual,0:formula,1:formula`) so the mode-specific reason is never swallowed. Diagnostics version line now labels the mode field `debug_mode=` (display only). The applied-patch log now emits Tea-only fields (threshold/progress/source) only in TeaCache mode. (`5c20835`)
- Verified: **Manual Skip mode confirmed on real hardware (2026-07-13)** — skipping steps `15, 17, 19, 21, 23` (1-based) via Manual Skip produced an image **bit-identical** to the legacy UjiCache extension skipping the same steps `14, 16, 18, 20, 22` (0-based), measured via Nz DoppelPix Judge: LPIPS = 0.000000, SSIM = 1.000000, PSNR = inf. Timing matched (73.1s vs 72.5s; ~0.004s per skipped step). This confirms Manual Skip's skip-execution machinery (residual Reuse path) is fully equivalent to UjiCache's. The legacy UjiCache extension has since been disabled on the verification machine.
- Changed: **max-skip-streak model reduced from 4 zones to 3** — the former `final` zone (`z >= 4`, streak 1, "final-polish protection") was removed on 2026-07-10 after stratified re-analysis showed its supporting density dip was a selection-bias artifact. Zones are now `danger` (`z < low` → streak 1) / `middle` (`low <= z < high` → streak 2) / `safe` (`z >= high` → streak 3); `safe` extends to `z >= 0`. End-of-generation moderation is left entirely to the skip-probability taper. See `docs/SPEC-alpha.md` §4.3.
- Changed: **the automatic ~5% first/last guard rule (`guard_count = max(1, round(0.05 * N))`) was replaced by a user-configurable skip window** in the progress domain (default `(0.05, 0.95)`, which reproduces the old guards for 30 steps). WYSIWYG: `(0.0, 1.0)` makes every step eligible — there is no hidden last-step safety net. `SkipPattern` now carries `skip_window` / `zone_boundaries` / `guarded_steps` instead of `guard_count`.
- Added: **zone boundaries are now user-configurable** — the `(low, high)` `z` cutoffs for danger/middle/safe (default `(-4.0, 0.0)`, range `-8.0..+8.0`) are exposed in HareSkip mode.
- Added: **dual-thumb RangeSlider controls** for Skip window (progress) and Zone boundaries (logSNR proxy), using `gradio_rangeslider==0.0.8` (added as the sole `requirements.txt` dependency). Degrades automatically to plain start/end and low/high `gr.Slider`s when RangeSlider is unavailable.
- Added: **Manual Skip mode** — a third exclusive skip-strategy Radio choice (`hareskip/manual_skip.py`). Type a comma-separated list of 1-based step numbers to skip; the list is parsed and validated against the run's `p.steps` before generation, aborting with a `RuntimeError` on non-numeric tokens, out-of-range steps, or step `1` (physically unskippable) rather than silently degrading. Built for the recalibration experiments in `docs/HANDOFF-next-session.md` §4.5; see `docs/ManualSkip-spec.md`.
- Changed: UI mode selector is now `HareSkip` / `TeaCache` / `Manual Skip` (was `HareSkip` / `TeaCache`).
- Changed: UI argument count is now **34** (was 29): added the four skip-window / zone-boundary scalars (`hareskip_window_start` / `hareskip_window_end` / `hareskip_zone_low` / `hareskip_zone_high`) and `manual_skip_steps`.
- Added infotext keys: `Hare skip_window`, `Hare zone_boundaries` (HareSkip mode), and `Manual skipped_steps` (Manual Skip mode). Removed the former `Hare guard_count`.

## 0.1.0 (alpha)

- Forked from UjiCache; renamed the package to `hareskip` and the Forge entrypoint to `scripts/hareskip.py`.
- Renamed namespaces into three concerns: `HareSkip`/`hareskip` (extension identity: panel, settings section, logger `[HareSkip]`, `model._hareskip_state`), `Tea`/`tea` (the legacy TeaCache-style skip decision, formerly `UjiCache`'s only decision path), and `ResRefine`/`resrefine` (residual prediction/reuse, formerly folded into the UjiCache namespace).
- Added a new default skip-decision strategy, **HareSkip stochastic skip-density mode**: skip steps are drawn from a probability density over a trajectory coordinate `z` (logSNR proxy from `t_now`), with zone-based max-skip-streak pruning and mandatory first/last guard steps, instead of a threshold accumulator. TeaCache mode remains available as a Radio-selectable alternative with unchanged numerics.
- Added sigma-schedule capture (via the `on_cfg_denoiser` callback) to obtain the full `t_now` sequence needed to generate a stochastic pattern up front, with a safe degrade to full computation (and a one-time warning) when the schedule or image seed is unavailable.
- Added reproducible skip seeds: `skip_seed = sha256(f"{image_seed}|hareskip|{offset}") mod 2**63`, with a user-facing "skip seed offset" control to re-roll a different pattern for the same image seed.
- Added a UI mode selector (`HareSkip` / `TeaCache` Radio, default `HareSkip`) with a shared `Enable HareSkip` gate and a shared ResRefine section.
- Re-prefixed infotext/metadata keys by concern: `HareSkip ...` (identity), `Hare ...` (HareSkip-mode fields), `Tea ...` (TeaCache-mode fields), `ResRefine ...` (residual prediction fields).
- Renamed Auto Uji mode to **Auto Tea mode** (`auto_teacache.py`).
- Removed `_clear_legacy_ujicache_metadata` and its call site; no longer needed with the new key scheme.

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

