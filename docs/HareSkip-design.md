# Stochastic Skip Density Design

> Imported from the UjiCache pre-study archive (`stochastic_skip_density_design_2026-06-30.md`). Canonical design spec for the HareSkip skip-density mode. α implementation: see `hareskip/skip_pattern.py` and `hareskip/probability_models.py`.

> **頭注（2026-07-29）**: 本書は研究アーカイブ（`stochastic_skip_density_design_2026-06-30`）の移入版であり、原典として凍結。実装との差異は `docs/SPEC-alpha.md` §4.3（4→3ゾーン変更）を、設計仮説の実測検証結果（z普遍性の限界・streak項不要・飽和など）は `docs/recalibration-2026-07/REPORT.md` を参照。

作成日: 2026-06-30

この文書は、UjiCache / TeaCache 系の skip step 判定を、固定 step pattern や TeaCache threshold ではなく、`logSNR / t_now` などの軌道座標上の **stochastic skip density** として設計・実装するためのハンドオフである。

実装担当エージェントは、まずこの文書を読む。背景の詳細は `cache_method_research_notes.md`、軌道座標分析の詳細は `trajectory_axis_handoff_2026-06-28.md`、自動生成レポートは `../AnalysisTables/skip_density_gradient/skip_density_gradient_report.md` を参照する。

## 目的

発見したいものは、フル演算からの絵柄変化ができるだけ少ない skip pattern と、その skip pattern を生成する方法である。

実用上の目標は主に次の 2 つ。

- `10 skips`: 30 steps 中 10 step を skip。期待される生成時間はおよそ 2/3。
- `15 skips`: 30 steps 中 15 step を skip。期待される生成時間はおよそ 1/2。

sampler、scheduler、shift は実用上あまり頻繁に変わらない。一方、prompt と seed は頻繁に変わる。そのため、prompt / seed に過適合した固定 pattern ではなく、sampler / scheduler / shift をまたいで意味を持つ軌道座標上の skip density を使う。

## 基本方針

採用する方針:

- skip 判定の主座標は step 番号ではなく `logSNR` または `t_now`。
- `logSNR` は sampler / scheduler / shift をまたいで比較しやすい共通の軌道座標として扱う。
- skip 確率そのものは、序盤・中盤・終盤の硬い境界で段差状にしない。
- skip 確率は `logSNR` 上の滑らかな勾配として定義する。
- 序盤・中盤・終盤という位相分類は、主に `max skip streak` の制約に使う。
- 冒頭と末尾の約 5% は、確率勾配を問わず強制 full computation にする。
- skip pattern は stochastic に生成する。同じ条件でも生成ごとに少し異なる pattern を許す。

採用しない方針:

- step 番号だけで固定 pattern を決める。
- TeaCache threshold の具体 step をそのまま基準にする。
- `skip score` を作り、上位 K step を deterministic に選ぶ。
- prompt / seed ごとのベスト pattern に過適合する。
- 軽量 NN によって pattern を生成する。これは将来案としては残すが、現段階の優先度は低い。

## 重要な観察

既存データから、良好 pattern の skip 位置を `LogSnrProxy` bin に投影した。

入力:

- `../AnalysisTables/best_prompt_seed_skip_patterns/best_by_prompt_seed_metric.csv`
- `../AnalysisTables/best_prompt_seed_skip_patterns/top_consensus_patterns.csv`
- `../AnalysisTables/trajectory_axis_analysis/step_axis_by_condition.csv`

生成物:

- `../AnalysisTables/skip_density_gradient/skip_density_gradient_report.md`
- `../AnalysisTables/skip_density_gradient/best_by_prompt_seed_logsnr_bin_skip_density.csv`
- `../AnalysisTables/skip_density_gradient/consensus_logsnr_bin_skip_density.csv`
- `../AnalysisTables/skip_density_gradient/skip_run_summary_by_zone.csv`

### 10 skips の empirical skip density

prompt / seed / metric 別ベスト 480 行に基づく、eligible step の skip rate:

| LogSnrProxy bin | SkipRate |
| --- | ---: |
| `[-inf,-8)` | 0.000 |
| `[-8,-6)` | 0.000 |
| `[-6,-4)` | 0.011 |
| `[-4,-2)` | 0.426 |
| `[-2,0)` | 0.588 |
| `[0,2)` | 0.580 |
| `[2,4)` | 0.535 |
| `[4,6)` | 0.080 |
| `[6,inf)` | 0.000 |

解釈:

- `z < -6` はほぼ飛ばしていない。
- `z = -4 .. 4` で skip density が高い。
- `z >= 4` 以降は最終仕上げに近づくため、skip density が落ちる。
- 単純な右肩上がりではなく、「上がって、高止まりし、最後に下がる」形。

### 15 skips の empirical skip density

prompt / seed / metric 別ベスト 480 行に基づく、eligible step の skip rate:

| LogSnrProxy bin | SkipRate |
| --- | ---: |
| `[-inf,-8)` | 0.033 |
| `[-8,-6)` | 0.449 |
| `[-6,-4)` | 0.468 |
| `[-4,-2)` | 0.572 |
| `[-2,0)` | 0.694 |
| `[0,2)` | 0.685 |
| `[2,4)` | 0.497 |
| `[4,6)` | 0.550 |
| `[6,inf)` | 0.333 |

解釈:

- 15 skips では skip 数が多いため、10 skips より早い軌道位置まで食い込む。
- ただし `z = -2 .. 2` 付近が最も濃い傾向は維持される。
- 最後の step は強制 full guard にするため、実装時には高 logSNR 側の確率をさらに抑える。

### streak の観察

共通ベスト pattern に基づく rough zone 別の連続 skip:

| TargetSkip | Zone | MaxRunLength | 傾向 |
| ---: | --- | ---: | --- |
| 10 | `middle_minus4_to_0` | 2 | ほぼ 1 連続、少数 2 連続 |
| 10 | `safe_0_to_4` | 2 | 1 連続と 2 連続が混在 |
| 15 | `danger_z_lt_minus4` | 2 | 3 連続は出ない |
| 15 | `middle_minus4_to_0` | 3 | 3 連続が少数出る |
| 15 | `safe_0_to_4` | 3 | 3 連続が自然に出る |
| 15 | `final_z_ge_4` | 1 | 最終付近は連続 skip しない |

この結果は、「skip 確率は滑らかな勾配、位相分類は max skip streak に使う」という設計を支持する。

## 軌道座標

可能なら、scheduler / sampler から物理的な logSNR を取得する。

物理的な logSNR が取得できない場合、現行分析と同じ proxy を使う。

```text
z = LogSnrProxy = 2 * ln((1 - t_now) / t_now)
```

現行 Forge Neo capture では、`t_now` は序盤で大きく、終盤へ進むほど小さくなる。したがって `z` は生成が進むほど大きくなる。

注意:

- `LogSnrProxy` は厳密な物理 logSNR ではない。
- ただし、軌道上の単調座標としては有効だった。
- 実装では `t_now` が 0 または 1 に近い場合を避けるため、必ず clamp する。

推奨:

```text
t = clamp(t_now, eps, 1 - eps)
z = 2 * ln((1 - t) / t)
eps = 1e-6
```

## 確率勾配の数式

推奨する基本式:

```text
p_skip(z; a)
= p_cap(a)
  * sigmoid((z - z_enter(a)) / tau_enter(a))
  * sigmoid((z_exit(a) - z) / tau_exit)
```

定義:

```text
z = logSNR or LogSnrProxy
a = skip aggressiveness slider, 0.0 .. 1.0
sigmoid(x) = 1 / (1 + exp(-x))
```

各項の意味:

- `sigmoid((z - z_enter) / tau_enter)`: 序盤から中盤にかけて skip 確率を滑らかに上げる。
- `sigmoid((z_exit - z) / tau_exit)`: 最終付近で skip 確率を滑らかに下げる。
- `p_cap`: その slider 設定での最大 skip 確率。

初期パラメータ:

```text
p_cap(a)     = 0.40 + 0.40 * a
z_enter(a)   = -1.8 - 5.0 * a^1.35
tau_enter(a) = 0.55 + 0.35 * a
z_exit(a)    = 4.2 + 1.0 * a
tau_exit     = 0.45
```

slider の意味:

| a | 目安 | p_cap | z_enter | tau_enter | z_exit | tau_exit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | conservative | 0.40 | -1.80 | 0.55 | 4.20 | 0.45 |
| 0.5 | balanced, 10 skips 付近 | 0.60 | 約 -3.76 | 0.725 | 4.70 | 0.45 |
| 1.0 | aggressive, 15 skips 付近 | 0.80 | -6.80 | 0.90 | 5.20 | 0.45 |

この式の意図:

- slider を上げると、最大 skip 確率が上がる。
- slider を上げると、skip 可能帯域がより序盤側へ広がる。
- slider を上げても、最終付近は guard / taper によって守る。

## step 間隔補正

sampler / scheduler / shift が違うと、同じ `logSNR` 関数でも離散 step の間隔が変わる。より軌道座標らしく扱うなら、step ごとの軌道幅 `delta_z_i` を入れる。

推奨候補:

```text
lambda_i = alpha(a) * density(z_i) * delta_z_i
p_i = 1 - exp(-lambda_i)
```

ここで `density(z)` は上記の山型関数から `p_cap` を外したものでもよい。

ただし初期実装では、複雑にしすぎず次でよい。

```text
p_i = p_skip(z_i; a)
```

step 間隔補正は、共通密度関数で条件差が大きく残る場合に追加する。

## 強制 full guard

30 steps では、冒頭 2 step と末尾 2 step を強制 full computation にする。

一般化:

```text
guard_count = max(1, round(num_steps * 0.05))
```

30 steps の場合:

```text
guard_count = 2
force full: step 1, 2, 29, 30
```

実装時の注意:

- guard は確率勾配より優先する。
- guard step では `p_skip = 0`。
- guard は stochastic sample の後で戻すのではなく、最初から sampling 対象外にする。

## rough zone と max skip streak

skip 確率は硬い zone 境界で段差にしない。zone は `max skip streak` の制御に使う。

初期 zone:

```text
danger: z < -4
middle: -4 <= z < 0
safe:   0 <= z < 4
final:  z >= 4
```

初期 max skip streak:

| Zone | max_skip_streak | 備考 |
| --- | ---: | --- |
| danger | 1 | 15 skips では 2 を試す余地あり。ただし初期値は 1。 |
| middle | 2 | 10 skips では自然。15 skips では 3 を試す余地あり。 |
| safe | 3 | 15 skips の良好 pattern では 3 連続が自然に出る。 |
| final | 1 | 末尾 guard と併用。最終仕上げ保護。 |

streak 判定時、連続 run が zone をまたぐ場合は、より危険な zone の制約を優先する。

例:

```text
run steps: 10,11,12
zones: danger, middle, middle
allowed max streak = min(1,2,2) = 1
```

この場合は run を壊し、少なくとも一部 step を full に戻す。

## stochastic sampling algorithm

最小実装:

```text
for each step i:
    if i in first_guard or last_guard:
        skip[i] = false
        continue

    z_i = get_logsnr_or_proxy(i)
    p_i = p_skip(z_i; aggressiveness)
    skip[i] = random_uniform(0, 1) < p_i

apply_zone_max_skip_streak(skip, z)
```

streak 制約の適用方法:

1. skip run を左から走査する。
2. run 内の各 step の zone から、その run に許される最大 streak を求める。
3. run length が上限以下なら何もしない。
4. run length が上限を超える場合、run 内の一部 step を full に戻す。

full に戻す step の選び方:

- 初期実装では、run の中で `p_i` が最も低い step から full に戻す。
- 同率なら、より低い `z`、つまりより早い軌道位置を full に戻す。
- これにより、危険寄りの step が優先的に保護される。

疑似コード:

```text
while run_length > max_allowed:
    j = argmin_over_run(p_j, tie_breaker = lower_z_first)
    skip[j] = false
    recompute runs
```

注意:

- これは `skip score` ではない。
- すでに stochastic に選ばれた run を安全制約で整えるだけである。
- top-K selection のように全 step を順位付けして選ぶ方式にはしない。

## slider と target skip count

UI では TeaCache threshold のように、ユーザーが「たくさん飛ばすか、少しだけ飛ばすか」を slider で決める。

推奨 UI:

```text
Skip Density: 0.00 .. 1.00
default: 0.50
```

意味:

- `0.00`: conservative。skip は少なめ。主に安全帯だけ。
- `0.50`: balanced。30 steps では 10 skips 付近を期待。
- `1.00`: aggressive。30 steps では 15 skips 付近を期待。

重要:

- slider は「skip 数そのもの」ではない。
- slider は `p_skip(z)` の密度と帯域を動かす。
- 同じ slider でも、乱数、scheduler、streak 制約、guard により実 skip 数は揺れる。

実装では、UI に expected skip count を表示するとよい。

```text
expected_skips = sum_i p_i over non-guard steps
```

より正確には、streak 制約後の期待値をモンテカルロで推定してもよい。

```text
simulate 100 patterns
show mean skips and 5-95 percentile
```

## exact target skip mode

研究実験では `10 skips` と `15 skips` を比較したい。そのため、UI slider とは別に exact target skip mode を持つと便利。

ただし、通常実装では off でよい。

exact target skip mode の案:

1. slider から `p_i` を計算する。
2. stochastic に pattern を生成する。
3. streak / guard を適用する。
4. skip 数が target に近ければ採用する。
5. 離れすぎたら再サンプルする。

推奨:

```text
max_resample = 100
accept if skip_count == target
fallback: abs(skip_count - target) が最小の pattern を使う
```

研究用には exact target skip mode を使う。ユーザー向け extension では expected skip count 表示だけで十分かもしれない。

## 条件ごとの扱い

`logSNR` は共通の軌道座標なので、確率勾配の形は条件ごとに変えないことを初期方針にする。

共通にする:

- `p_skip(z; a)` の関数形
- `p_cap(a)`、`z_enter(a)`、`tau_enter(a)`、`z_exit(a)` の初期式
- rough zone の初期境界
- first / last 5% guard

条件ごとに変わる:

- 各 step の `z_i`
- step 間隔
- stochastic sample の結果
- 実 skip 数
- 必要なら exact target mode の resampling 成否

条件ごとに変える可能性があるが、初期実装では固定でよい:

- `max_skip_streak`
- `tau_exit`
- `z_exit`
- step 間隔補正の有無

## 実装に必要な入力

各 denoising step で必要:

- step index
- total steps
- `t_now` または scheduler が持つ sigma / logSNR
- optional: sampler / scheduler / shift metadata

最低限:

```text
step_index
num_steps
t_now
random_seed_for_skip_sampler
aggressiveness
```

skip sampler 用の乱数 seed は、画像生成 seed と同じにするか、別 seed にするかを設計で決める。

推奨:

- 初期実装では、画像 seed から deterministic に派生させる。
- 例: `skip_seed = hash(image_seed, "stochastic_skip_density", user_skip_seed_offset)`
- これにより再現性を保ちながら、必要なら offset で別 pattern を引ける。

## 出力すべきログ

後で検証できるように、PNG infotext や JSON metadata に次を残す。

必須:

- method name: `StochasticSkipDensity`
- version
- aggressiveness slider value
- guard count
- `p_cap`, `z_enter`, `tau_enter`, `z_exit`, `tau_exit`
- zone boundaries
- max skip streak per zone
- actual skipped steps
- actual skip count
- skip sampler seed

推奨:

- step ごとの `z_i`
- step ごとの `p_i`
- guard 適用前の sampled skip pattern
- streak 制約後の final skip pattern
- expected skip count before streak constraint

metadata 例:

```json
{
  "skip_method": "StochasticSkipDensity",
  "skip_method_version": "0.1",
  "aggressiveness": 0.5,
  "guard_count": 2,
  "params": {
    "p_cap": 0.6,
    "z_enter": -3.758,
    "tau_enter": 0.725,
    "z_exit": 4.7,
    "tau_exit": 0.45
  },
  "zone_boundaries": {
    "danger_lt": -4,
    "middle_lt": 0,
    "safe_lt": 4
  },
  "max_skip_streak": {
    "danger": 1,
    "middle": 2,
    "safe": 3,
    "final": 1
  },
  "skipped_steps": [8, 10, 12, 13, 15, 16, 18, 19, 21, 23],
  "skip_count": 10,
  "skip_seed": 123456789
}
```

## 検証方法

評価指標:

- LPIPS-VGG
- SSIM 11p

比較対象:

- full computation
- existing TeaCache / UjiCache threshold patterns
- Daraskme Comfy patterns
- best consensus patterns found in existing candidate set
- stochastic skip density patterns

最低限の実験:

1. 各 ConditionKey で aggressiveness を `0.3`, `0.5`, `0.7`, `1.0` などに振る。
2. 各設定で複数 skip seed を生成する。
3. 5 prompts x 3 seeds で画像生成する。
4. Nz DoppelPix Judge で LPIPS-VGG / SSIM 11p を計測する。
5. 平均、中央値、最悪値、破綻率を比較する。

重要:

- この方法は「一発の最良 pattern」だけを狙うものではない。
- 実務上、画像生成は多数試行して選ぶ。
- したがって、平均的に良いこと、破綻率が低いこと、複数 sample の上位候補が既存手法を上回ることを重視する。

論文上の主張候補:

```text
Existing acceleration methods usually instantiate temporal non-uniformity as
deterministic schedules, threshold-based reuse, offline searched schedules, or
learned per-sample policies. In contrast, this method models skip placement as
a training-free stochastic density over a trajectory coordinate such as logSNR.
This matches the practical use of image generation, where users often sample
multiple candidates rather than relying on a single deterministic output.
```

## 実装上の注意点

### stochastic だが再現可能にする

同じ image seed、prompt、skip seed、slider、sampler、scheduler、shift なら、同じ skip pattern が再現されるようにする。

### `skip score` と呼ばない

このプロジェクトでは `skip score` という概念は採用しない。overfitting リスクが高く、trajectory-coordinate skip density という方針から外れるため。

実装内でどうしても変数名が必要なら、次を使う。

- `skip_probability`
- `skip_density`
- `density_weight`
- `stochastic_skip_probability`

避ける名前:

- `skip_score`
- `fatal_score`
- `top_k_skip_score`

### top-K selection にしない

`p_i` が高い step を上位 K 個選ぶ方式にしない。これは deterministic schedule に戻ってしまう。

研究用 exact target skip mode でも、基本は stochastic sample と resampling で target skip 数に合わせる。

### first / last guard は必須

30 steps では最初と最後の 2 steps を必ず full にする。これは経験的に重要。

### final taper を忘れない

coarse-to-fine からは終盤ほど skip できそうに見えるが、実データでは最後付近の skip density は落ちる。線、塗り、締まり、ぼけに影響するため、式には必ず final taper を入れる。

### 15 skips は危険帯域に食い込む

15 skips では、良好 pattern でも `z < -6` 付近まで skip が入る。したがって 15 skips は、10 skips より破綻率が上がる可能性が高い。`danger` zone の max streak と stochastic なばらつきが重要。

## 初期実装の疑似コード

```python
import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def logsnr_proxy_from_t_now(t_now, eps=1e-6):
    t = clamp(t_now, eps, 1.0 - eps)
    return 2.0 * math.log((1.0 - t) / t)

def params_from_aggressiveness(a):
    a = clamp(a, 0.0, 1.0)
    return {
        "p_cap": 0.40 + 0.40 * a,
        "z_enter": -1.8 - 5.0 * (a ** 1.35),
        "tau_enter": 0.55 + 0.35 * a,
        "z_exit": 4.2 + 1.0 * a,
        "tau_exit": 0.45,
    }

def skip_probability(z, params):
    p = (
        params["p_cap"]
        * sigmoid((z - params["z_enter"]) / params["tau_enter"])
        * sigmoid((params["z_exit"] - z) / params["tau_exit"])
    )
    return clamp(p, 0.0, 1.0)

def zone_from_z(z):
    if z < -4.0:
        return "danger"
    if z < 0.0:
        return "middle"
    if z < 4.0:
        return "safe"
    return "final"

MAX_STREAK = {
    "danger": 1,
    "middle": 2,
    "safe": 3,
    "final": 1,
}

def generate_skip_pattern(t_now_by_step, aggressiveness, rng):
    num_steps = len(t_now_by_step)
    guard_count = max(1, round(num_steps * 0.05))
    params = params_from_aggressiveness(aggressiveness)

    z_by_step = []
    p_by_step = []
    skip = []

    for idx, t_now in enumerate(t_now_by_step):
        step_no = idx + 1
        is_guard = step_no <= guard_count or step_no > num_steps - guard_count
        z = logsnr_proxy_from_t_now(t_now)
        p = 0.0 if is_guard else skip_probability(z, params)

        z_by_step.append(z)
        p_by_step.append(p)
        skip.append(False if is_guard else (rng.random() < p))

    apply_max_streak_constraint(skip, z_by_step, p_by_step)
    return skip, z_by_step, p_by_step, params

def apply_max_streak_constraint(skip, z_by_step, p_by_step):
    # Implementation detail:
    # scan skip runs, compute allowed max streak from the most conservative
    # zone inside the run, and turn the lowest-probability / lowest-z steps
    # back to full until the run satisfies the constraint.
    pass
```

## 次の実験タスク

1. 上記式で candidate pattern generator を実装する。
2. まず画像生成なしで、既存 `step_axis_by_condition.csv` に対して pattern を多数生成し、expected skip count と zone/streak 分布を見る。
3. `a = 0.5` が 10 skips 付近、`a = 1.0` が 15 skips 付近になるか確認する。
4. ずれる場合は、`p_cap(a)` と `z_enter(a)` を最小限調整する。
5. exact target skip mode を使って 10 skips / 15 skips の候補を各条件で複数生成する。
6. 実画像生成と Nz DoppelPix Judge に回す。
7. 既存の consensus best pattern、TeaCache / UjiCache / Daraskme pattern と比較する。

## 現時点の結論

現データは、次の設計を支持している。

```text
skip probability:
    smooth logSNR density, not hard three-stage steps

phase / zone:
    used for max skip streak and final protection

first / last guard:
    mandatory full computation, about 5% each side

slider:
    controls density strength and how far the skip-eligible band expands

selection:
    stochastic sampling, not top-K score selection
```

この方針は、coarse-to-fine の理論、既存 best pattern の empirical skip density、実用上の「多数生成して選ぶ」ワークフローを同時に説明できる。
