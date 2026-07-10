# Agent Handoff

This repository is a Forge Neo extension named `HareSkip`, forked from `UjiCache`.

Use this file as the short orientation. The canonical design spec for the stochastic skip-density mode is `docs/HareSkip-design.md`.

## Current State

- Package: `hareskip`
- Forge entrypoint: `scripts/hareskip.py`
- Version: `0.1.0`
- UI prefix / setting keys: `hareskip-*` / `hareskip_*` (extension identity), `tea-*` / `tea_*` (TeaCache-mode skip decision), `hare-*` / `hareskip_*` (HareSkip-mode skip decision), `resrefine-*` / `resrefine_*` (shared residual prediction)
- Console prefix: `[HareSkip]` (logger name `HareSkip`, see `hareskip/logging.py`)
- Settings section: `("hareskip", "HareSkip")`

Implemented runtime patch:

- `hareskip` (patches `backend.nn.anima.Anima._forward` / `.forward`)

Retained UI:

- Top-level accordion: `HareSkip` (`elem_id="hareskip-panel"`)
- `Enable HareSkip` checkbox — overall gate, retained from the predecessor extension's equivalent enable checkbox
- Mode selector: `gr.Radio(["HareSkip", "TeaCache"], elem_id="hareskip-mode")`, default `HareSkip`, exclusive
- HareSkip-mode group (`hare-*` controls): aggressiveness slider, skip seed offset, expected-skip estimate
- TeaCache-mode group (`tea-*` controls): preset, threshold, coefficient profile, `p_Anima(x)` display, start/end percent, max skip streak, force full interval, Auto Tea mode sub-accordion
- ResRefine section (`resrefine-*` controls, always visible regardless of mode): formula, prediction strength, Taylor2 curve strength, slope/curve EMA smoothing, use-prediction-after-progress, apply-prediction-from-skip, cache device
- Sub-accordion: `Debug log mode`

Removed from the public extension surface (inherited from the predecessor extension, unchanged):

- Attention backend override
- Standalone TeaCache experiment UI (the historical PredLab one, not this extension's TeaCache mode)
- Spectrum experiment
- 2D sparse attention
- Cond/uncond optimization
- Low-bit and torch.compile experiments
- Identity patch test

Also removed in this fork: the legacy metadata clearer helper and its call site (no longer needed; infotext keys are cleanly re-prefixed instead).

## Important Rules

- Preserve baseline behavior when `Enable HareSkip` is off.
- The first model call must always be full calculation, in both modes.
- Do not allow cache/prediction use when `previous_residual` is missing, in either mode.
- Restore monkey patches on disable, unsupported model, unload, and degraded paths.
- Do not silently fail. Log degraded or fallback reasons with the `[HareSkip]` prefix.
- Forge Neo can pass unused kwargs such as `control` into `Anima.forward`; HareSkip should ignore unused kwargs and consume only the values it needs, especially `transformer_options`.
- `Modulated source` (`hareskip.state.TEA_SOURCE_FIRST_BLOCK_SHIFT` etc.) is not a UI control in TeaCache mode; it is derived from `Coefficient profile`. Do not reintroduce a Modulated source dropdown.
- `TEA_PRESET_REGISTRY` in `state.py` is the single source of truth for both TeaCache coefficient profiles and their recommended Start/End windows. Add presets there; coefficients and the `p_Anima(x)` display follow automatically.
- **TeaCache-mode numerics must not change.** The TeaCache decision path (`_tea_update_slot`, `_tea_force_full_reason`, `_cache_poly1d`, the accumulator/threshold logic) is renamed from the predecessor extension but must stay bit-identical in behavior. Mode dispatch must never alter Tea-mode numerics.
- Naming discipline: use only `skip_probability` / `skip_density` style names for HareSkip's probability concepts. Never introduce `skip_score`, `fatal_score`, or `top_k` — these were explicitly rejected in the design spec because they imply a deterministic top-K schedule, which this method deliberately avoids. Top-K selection must not be implemented anywhere in `skip_pattern.py` or `probability_models.py`.
- Do not confuse `hareskip_mode` (the skip-strategy selector, values `MODE_HARESKIP` / `MODE_TEACACHE`) with `hareskip_debug_mode` (the diagnostics-log mode key). They are separate settings keys.

## 3-Point UI Argument Sync Rule

The UI arguments produced by `hareskip/script.py` `ui()` must stay in lockstep across three places:

1. the return list of `Script.ui()` (order and count),
2. the positional signature of `RuntimeState.apply_options` in `hareskip/state.py` (order and count),
3. `hareskip/constants.py` `UI_ARG_ORDER` (canonical ordered name list) and `EXPECTED_UI_ARG_COUNT` (`len(UI_ARG_ORDER)`).

If any of the three drifts out of sync, generation options are silently applied to the wrong slots — this is the single highest-risk failure mode in this codebase. `script.py` sets `_EXPECTED_UI_ARG_COUNT` from `constants.EXPECTED_UI_ARG_COUNT`. Currently there are **29 arguments** (the original 26 pre-fork arguments keep their positions; `hareskip_mode`, `hareskip_aggressiveness`, and `hareskip_skip_seed_offset` are appended at the end).

`tests/test_arg_sync.py` enforces this statically and without importing gradio or Forge:

- `test_expected_count_matches_order_length` — `EXPECTED_UI_ARG_COUNT == len(UI_ARG_ORDER)`.
- `test_no_duplicate_arg_names` — no name appears twice in `UI_ARG_ORDER`.
- `test_apply_options_signature_matches_order` — `inspect.signature(RuntimeState.apply_options)` parameters (minus `self`) equal `UI_ARG_ORDER` exactly, in order.
- `test_ui_return_list_matches_order` — parses the `return [...]` list at the end of `Script.ui()` from source text and checks it equals `UI_ARG_ORDER` exactly, in order.

When adding/removing a UI control wired to settings, update all three (the `ui()` return list, `apply_options`'s signature, and `UI_ARG_ORDER`) together, then run `pytest tests/test_arg_sync.py` to confirm.

## Mode Dispatch Seam

`hareskip/patcher.py` `_hareskip_forward_body` is the single seam where the two skip-decision strategies diverge; everything else (embed, t_embedder, blocks loop, final_layer, unpatchify, ResRefine residual application) is shared.

- **Shared force-full checks first.** `_shared_force_full_reason` (mirrors `first_call` / `missing_residual` from `_tea_force_full_reason`) is evaluated *before* either mode's own decision logic, in both modes, so a skip can never be applied before a residual exists for that slot.
- **TeaCache path is bit-identical.** When `STATE.hareskip_mode == MODE_TEACACHE`, the original accumulator/threshold/`_tea_force_full_reason` path runs unchanged (renamed only).
- **HareSkip path dispatches through `STATE.hareskip_pattern`.** When `STATE.hareskip_mode == MODE_HARESKIP`, `_hareskip_should_calc(step_index)` consults the per-generation `SkipPattern` (see below) instead of the accumulator. HareSkip mode does not consult Tea-only force-full reasons (`outside_progress` / `force_full_interval` / `max_skip_streak`) — streak and guard logic belong to the pattern itself.
- **Exceptions must never propagate out of `_hareskip_should_calc`.** The whole body is wrapped in try/except and returns `True` (full calculation) on any error, logging at most once per generation (`hareskip_should_calc_failed`). This is required because the outer patched forward (`hareskip_forward`) itself falls back to the original unpatched `Anima.forward` on any exception escaping `_hareskip_forward_body` — HareSkip decision logic must never trigger that outer fallback path; it should degrade internally instead.
- HareSkip mode also skips the modulated-input + rel_l1 computation entirely (it doesn't need it), except when calibration capture is active (a Tea-oriented debug feature that still needs rel_l1 and forces full compute every step in either mode).

## Schedule Acquisition

The stochastic pattern needs `t_now` for every step up front (the max-streak constraint needs the whole picture), but `Anima.forward` cannot see the `p` (`StableDiffusionProcessing`) object. The schedule is captured earlier, via the `on_cfg_denoiser` callback:

- `hareskip/forge_introspection.py` `sampling_schedule_t_now(params)` recovers the full sigma schedule from the `cfg_denoiser` callback params (or the denoiser/sampler/shared-state objects reachable from it), converts it to a plain list of floats, and returns it as `t_now` directly. In Forge Neo's Anima setup the predictor is `PredictionDiscreteFlow` with `multiplier == 1.0`, so `timestep(sigma) == sigma` and each sigma value IS the flow time `t` in `(0, 1]` — no separate sigma-to-t conversion is needed. A trailing boundary `sigma ≈ 0` (which has no corresponding model call) is dropped.
- All recovered values must be finite and strictly inside `(0, 1)`; if any fall outside that range, or if fewer than one usable value is found from any candidate source, the schedule is treated as unavailable (`None`) rather than risking a garbage logSNR proxy.
- `hareskip/patcher.py` `_hareskip_ensure_pattern()` builds the `SkipPattern` once per generation from `STATE.hareskip_schedule_t_now` and `STATE.hareskip_image_seed`, caching it on `STATE.hareskip_pattern`.
- **Degrade to full-calc when unavailable.** If the schedule or image seed is missing, `_hareskip_ensure_pattern()` returns `None`, and `_hareskip_should_calc` treats that as "compute fully," logging `hareskip_schedule_unavailable` once. HareSkip must never crash or skip the wrong steps because a schedule failed to plumb through — it degrades to the safe default (full computation every step) instead.

## Pure Modules

`hareskip/skip_pattern.py`, `hareskip/probability_models.py`, and `hareskip/constants.py` are stdlib-only (`math` / `random` / `hashlib` / `dataclasses` for `skip_pattern.py`; `math` for `probability_models.py`; no imports at all for `constants.py`). None of the three import Forge, torch, or gradio, so they are fully unit-tested without a Forge install — see `tests/test_skip_pattern.py` and `tests/test_probability_models.py`. Keep it that way: if a change to these files needs Forge or torch, it belongs in `patcher.py` instead.

### Registering a new probability model

The design spec's formula (`sigmoid_band_v0.1` in `probability_models.py`) is expected to change substantially as research continues. To add a new one without touching pattern generation, streak constraints, or guards:

1. Define an object (module, class, or instance) exposing two callables:
   - `params_from_aggressiveness(a: float) -> dict` — map the aggressiveness slider to a parameter dict.
   - `skip_probability(z: float, params: dict) -> float` — map trajectory coordinate `z` and the params dict to a probability clamped to `[0, 1]`.
2. Register it: `probability_models.register("my_model_vX", MyModel())`.
3. Pass `probability_model="my_model_vX"` to `generate_skip_pattern(...)` (currently wired from `STATE.hareskip_probability_model`, not yet exposed as a UI control).

Nothing in `skip_pattern.py` needs to change — `generate_skip_pattern` looks the model up by name via `probability_models.get_model`.

## Useful Files

- `hareskip/script.py`: Gradio UI and generation-time patch selection.
- `hareskip/state.py`: settings snapshot, `RuntimeState`, `TEA_PRESET_REGISTRY`, `apply_options`.
- `hareskip/patcher.py`: HareSkip monkey patch implementation, mode dispatch, and restore logic.
- `hareskip/skip_pattern.py`: pure stochastic pattern generation (guards, zones, streak constraint, seed derivation).
- `hareskip/probability_models.py`: pure skip-probability model registry (`sigmoid_band_v0.1` built in).
- `hareskip/resrefine.py`: residual prediction/validation/EMA, extracted from the patcher.
- `hareskip/forge_introspection.py`: sigma-schedule and model-structure introspection helpers.
- `hareskip/diagnostics.py`: console snapshots and summaries.
- `hareskip/calibration_capture.py`: calibration-pair JSONL capture for TeaCache coefficient re-fitting.
- `hareskip/auto_teacache.py`: Auto Tea mode CSV parsing and row application (renamed from the predecessor's equivalent auto mode).
- `docs/HareSkip-design.md`: canonical stochastic skip-density design spec and acceptance criteria.

## Forge Neo Gotcha

Forge Neo can preserve old Gradio component ranges/defaults in `ui-config.json`, keyed by `elem_id`. This extension's `elem_id`s were fully renamed relative to the predecessor extension (`hareskip-*` / `hare-*` / `tea-*` / `resrefine-*`), so a clean install picks up fresh defaults automatically. If upgrading in place over an old `ui-config.json` and a UI change does not appear, check that file and restart Forge Neo.
