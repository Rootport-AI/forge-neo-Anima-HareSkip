# UjiCache EMA Prediction 仕様メモ

## 目的

UjiCache の `Linear extrapolation` / `Taylor2 curve` で使う residual 予測に EMA smoothing を導入し、skip step の residual 予測を安定させる。

## UI

関連 control:

- `Slope EMA Smoothing`
- `Curve EMA Smoothing`
- `Prediction strength`
- `Taylor2 curve strength`

`Prediction formula = TeaCache (residual only)` の場合、EMA 予測は使わない。

## 挙動

full calculation step では、実 residual を履歴に追加する。履歴から観測速度 `v_obs` を計算し、`Slope EMA Smoothing` によって `velocity_ema` を更新する。

Taylor2 では、前回速度との差から観測加速度を計算し、`Curve EMA Smoothing` によって `acceleration_ema` を更新する。

skip step では、EMA が十分に準備できていれば以下を使う。

```text
linear:
  r_pred = r_prev + dt * velocity_ema

taylor2:
  r_pred = r_prev + dt * velocity_ema + 0.5 * dt^2 * acceleration_ema * Taylor2 curve strength
```

最後に `Prediction strength` で直近 residual と予測 residual を混ぜる。

## Fallback

以下の場合は予測せず、直近 residual に fallback する。

- `previous_residual` が無い
- 履歴が足りない
- EMA velocity が未準備
- shape mismatch
- non-finite value
- norm guard に引っかかった
- dtype / device 変換に失敗した

## 実装ファイル

- `ujicache/state.py`
- `ujicache/script.py`
- `ujicache/patcher.py`
