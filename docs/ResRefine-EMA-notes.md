# ResRefine EMA Prediction Notes

> Supersedes the UjiCache-era "EMA Prediction spec" (`ujicache/state.py`, `ujicache/script.py`,
> `ujicache/patcher.py`), which described the same mechanism under old names. This note reflects
> the current implementation in `hareskip/resrefine.py`.

## What ResRefine is

ResRefine is the shared residual-prediction layer used on **skipped steps** in both TeaCache mode
and HareSkip mode. When a step's real diffusion residual isn't recomputed, ResRefine decides what
residual to substitute: either the last real residual verbatim, or a predicted residual
extrapolated from recent history. It lives outside both mode groups in the UI ("ResRefine
(residual prediction)" accordion) because both modes route skipped-step residuals through it.

On every **full-calculation step**, the real residual is recorded into a short per-slot history
(`residual_history`, capped at 5 entries) and used to update the EMA velocity/acceleration state.
On every **skip step**, ResRefine reads that state to produce a predicted residual for the slot.

## The three prediction formulas

Controlled by `Prediction formula` (`resrefine_formula`):

- **`Reuse` (`RESREFINE_FORMULA_REUSE`)** — no prediction. Always returns the previous real
  residual unchanged. This is the default and is equivalent to plain TeaCache-style
  residual-only reuse.
- **`Linear extrapolation` (`RESREFINE_FORMULA_LINEAR`)** — extrapolates the residual forward
  using a first-order (velocity) estimate: `r_pred = r_prev + dt * velocity`.
- **`Taylor2 curve` (`RESREFINE_FORMULA_TAYLOR2`)** — adds a second-order (acceleration) term on
  top of the linear estimate: `r_pred = r_prev + dt * velocity + 0.5 * dt^2 * acceleration * Taylor2 curve strength`.

Without EMA smoothing (`Slope EMA Smoothing = 0`), `velocity`/`acceleration` are derived directly
from Lagrange interpolation over the last 2 (linear) or 3 (Taylor2) recorded residuals for the
target `step_index`, rather than from the EMA state described below.

## Slope/Curve EMA smoothing

When `Slope EMA Smoothing` (`resrefine_slope_ema_smoothing`, i.e. `beta_v`) is greater than 0,
ResRefine switches to EMA-smoothed velocity/acceleration instead of raw Lagrange fits:

- On each full step, the observed velocity `v_obs = (residual - previous_residual) / dt` is
  computed from the two most recent recorded residuals, and blended into `velocity_ema`:
  `velocity_ema = beta_v * velocity_ema + (1 - beta_v) * v_obs`.
- If a previous velocity observation is also available, the observed acceleration
  `a_obs = (v_obs - previous_velocity) / dt_v` is computed and blended into `acceleration_ema`
  using `Curve EMA Smoothing` (`resrefine_curve_ema_smoothing`, `beta_a`) the same way.
- `velocity_ema` seeds itself from the first `v_obs` (and likewise for `acceleration_ema`); EMA
  smoothing only "kicks in" (i.e. blends rather than resets) once a prior EMA value of matching
  shape already exists.
- On a skip step, `Linear`/`Taylor2` then predict using `velocity_ema` (and `acceleration_ema`
  for Taylor2) projected forward by `dt_pred = step_index - last_recorded_step_index`. If
  `velocity_ema` isn't ready yet, or acceleration is missing for Taylor2, prediction falls back
  (Taylor2 degrades to its linear term when only acceleration is missing).

Two additional gates limit *when* prediction is attempted at all, regardless of formula:
`Use prediction after progress` (`resrefine_use_prediction_after_progress`) and `Apply prediction
from skip #` (`resrefine_apply_prediction_from_skip`, i.e. don't predict until the `skip_streak`
reaches this count, unless already in the late phase).

## Prediction strength blending

Whatever raw prediction is produced (Lagrange or EMA-based) is blended with the previous real
residual using `Prediction strength` (`resrefine_prediction_strength`):

```text
result = r_prev + prediction_strength * (raw_prediction - r_prev)
```

`prediction_strength = 0` is equivalent to `Reuse`; `1` uses the raw prediction unmodified.
`Taylor2 curve strength` (`resrefine_taylor2_curve_strength`) separately controls how much the
quadratic/acceleration term contributes within the Taylor2 formula itself, before this blend.

## Validation guard and fallback

Every prediction is passed through `_resrefine_validate_prediction` before use:

- Shape must match both the previous residual and the target tensor slice.
- All values must be finite.
- **Norm guard**: if `||prediction|| > ||previous|| * RESREFINE_MAX_NORM_RATIO` (constant
  `RESREFINE_MAX_NORM_RATIO = 3.0` in `hareskip/resrefine.py`), the prediction is rejected as a
  runaway extrapolation.
- The result must be convertible to the previous residual's device/dtype.

ResRefine falls back to the last real residual (verbatim reuse) whenever: there is no
`previous_residual`; the formula is `Reuse`; prediction isn't yet allowed (progress/streak gates);
history is insufficient or has a shape mismatch; EMA velocity isn't ready; the norm guard trips;
or any other numeric/dtype error occurs during prediction.

## UI controls and infotext keys

Accordion: **"ResRefine (residual prediction)"** in `hareskip/script.py`.

| UI label | State field | Infotext key |
|---|---|---|
| Prediction formula | `resrefine_formula` | `ResRefine formula` |
| Use prediction after progress | `resrefine_use_prediction_after_progress` | `ResRefine use_prediction_after_progress` |
| Apply prediction from skip # | `resrefine_apply_prediction_from_skip` | `ResRefine apply_prediction_from_skip` |
| Prediction strength | `resrefine_prediction_strength` | `ResRefine prediction_strength` |
| Taylor2 curve strength | `resrefine_taylor2_curve_strength` | `ResRefine taylor2_curve_strength` |
| Slope EMA Smoothing | `resrefine_slope_ema_smoothing` | `ResRefine slope_ema_smoothing` |
| Curve EMA Smoothing | `resrefine_curve_ema_smoothing` | `ResRefine curve_ema_smoothing` |
| Cache device | `resrefine_cache_device` | — |

The prediction-related infotext keys are only written when `resrefine_formula != Reuse`; the
Taylor2 strength key is only written when `resrefine_formula == Taylor2 curve`. There is also a
debug-only `Dump ResRefine residual` checkbox (`dump_resrefine_residual`) unrelated to prediction
behavior.

## Implementation file

- `hareskip/resrefine.py`
