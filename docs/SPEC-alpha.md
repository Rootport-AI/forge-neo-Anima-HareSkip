# HareSkip α版 統合仕様書（要件定義・設計・仕様）

- 作成日: 2026-07-10（α+1・Manual Skip mode 追加後に更新）
- HEAD コミット: `a67f5594602083a50d23c3966542290f0ab78e43`
- 対象: HareSkip 0.1.0（α版＋α+1、Forge Neo 拡張、Anima base 向けステップスキップ推論）
- 設計正典: [`docs/HareSkip-design.md`](HareSkip-design.md)（本書はこれを参照し、内容を重複させない）

本書は要件定義・アーキテクチャ設計・仕様を 1 本にまとめた統合資料である。すべての数値・キー名・関数名はリポジトリの実コード（`hareskip/*.py`, `tests/*`, HEAD=`a67f559`）で確認した値を記載する。計画書と実装が食い違う箇所は実装を正とし、§5 に明記する。

---

## 1. 背景と目的

### 1.1 UjiCache 研究の結論

前身拡張 UjiCache（TeaCache 方式: rel_l1 accumulator + 較正多項式 `p_Anima(x)`）に対する研究で、次が判明した。

- **係数精度は支配的でない**: 出力の相対変化 `out_rel` と画質危険度の相関は `r = −0.06`。多項式係数をどれだけ高精度化しても「どのステップを飛ばすか」が悪ければ画質で負ける。
- **「どのステップを飛ばすか」が支配的**: skip step 選択そのものが画質を決める。指標 `SkipMinLogSnr`（skip 位置の最小 logSNR）は LPIPS と `r = −0.94` と強く相関する。
- **致命的ステップは序盤に集中**: 特に step2 / step4 がサンプラ・スケジューラ・shift をまたぐクロス条件で致命的になりやすい。
- **安全帯は軌道座標上の帯域**: 安全に飛ばせる領域は step 番号ではなく `logSNR` / `t_now` 軌道座標上の帯域として現れる。sampler / scheduler / shift は実用上変わりにくく、prompt / seed は頻繁に変わるため、固定 pattern ではなく軌道座標上の密度が汎化する。

研究アーカイブ（本リポジトリにコピーしない、パス参照のみ）:

```
S:\30_OriginalApps\16_HareSkip\ER SDE-Beta 30steps Shift3
```

### 1.2 HareSkip の中心思想

上記結論から、HareSkip（High-Adaptive Regime-based Extrapolation Skip）は以下を採用する。

- **確率的スキップ密度**: 軌道座標 `z`（logSNR proxy）上の滑らかな確率密度 `p_skip(z; a)` から skip step を抽選する。硬い段差状の位相境界で決めない。
- **ゾーン依存 max skip streak**: 序盤・中盤・終盤の位相分類は確率ではなく「連続スキップ長の上限」に使う。danger / middle / safe の 3 ゾーンで許容 streak を変える（旧 final ゾーンは 2026-07-10 に廃止、§4.3 参照）。生成終盤の減速は skip-probability taper が担う。
- **ガチャ前提の分布平均での優位性**: pattern は stochastic に生成され、同じ条件でも生成ごとに少し異なる。単発ベストではなく「分布として平均的に安全」を狙う。`skip seed offset` により同一画像 seed のまま pattern を引き直せる。

top-K の deterministic 選択、prompt/seed 過適合、軽量 NN 生成は不採用（設計書 §基本方針を参照）。

---

## 2. 要件定義

### 2.1 機能要件

| ID | 要件 |
| --- | --- |
| F-1 | HareSkip / TeaCache / Manual Skip の 3 モードを**排他**で提供（Radio 選択）。**デフォルト = HareSkip**。Manual Skip は明示指定したステップのみを飛ばす実験用モード（`docs/ManualSkip-spec.md`）。 |
| F-2 | 拡張全体の `Enable HareSkip` チェックボックスを温存（全モード共通のゲート）。OFF 時はベースライン挙動。 |
| F-3 | ResRefine（residual 予測/再利用）は**モード外の共通セクション**。両モードで共有。 |
| F-4 | 確率性は**再現可能**。同一 image seed + 同一 offset は同一 pattern を再生する。 |
| F-5 | `skip seed offset`（整数 UI）で同一 image seed のまま別 pattern を引き直せる（ガチャ）。 |
| F-6 | 確率式は**差し替え可能**。`probability_models` レジストリに新モデルを登録するだけで pattern/streak/guard 機構に触れずに交換できる。 |
| F-7 | TeaCache モードの数値挙動は旧 UjiCache と**ビット同一**（リネームのみ）。 |
| F-8 | **後方互換なし**。UjiCache 命名・レガシーメタデータ（`_clear_legacy_ujicache_metadata`）は削除。 |
| F-9 | exact-target skip mode は純粋モジュール API とテストにのみ存在し、**UI に露出しない**。 |

### 2.2 非機能要件

| ID | 要件 |
| --- | --- |
| NF-1 | Forge neo 本体を触らない。拡張だけで完結する monkey patch とコールバック。 |
| NF-2 | 失敗時は**フル演算に劣化**する。クラッシュ・誤スキップは禁止。判定ロジックの例外は握って `True`（フル演算）を返す。 |
| NF-3 | 純粋モジュール（`constants.py` / `skip_pattern.py` / `probability_models.py`）は Forge/torch/gradio 非依存で、stdlib のみで単体テスト可能。 |
| NF-4 | 命名規律: HareSkip の確率概念は `skip_probability` / `skip_density` のみ。`skip_score` / `fatal_score` / `top_k` は禁止（deterministic top-K を含意するため設計で棄却済み）。 |

---

## 3. アーキテクチャ設計

### 3.1 3 名前空間と境界

| 名前空間 | 責務 | 主なキー/接頭辞 |
| --- | --- | --- |
| **HareSkip / hareskip** | 拡張アイデンティティ（パネル、設定セクション、ロガー `[HareSkip]`、`model._hareskip_state`） | `hareskip-*` / `hareskip_*`、infotext `HareSkip ...` |
| **Tea / tea** | 旧 TeaCache 方式のスキップ**判定**（accumulator + 多項式） | `tea-*` / `tea_*`、infotext `Tea ...` |
| **ResRefine / resrefine** | residual の予測/検証/EMA（スキップ**加工**） | `resrefine-*` / `resrefine_*`、infotext `ResRefine ...` |

判定側と加工側は **slot dict 契約**（`_hareskip_slot`）のみで連結する。判定側 write: `previous_modulated_input` / `accumulated_rel_l1_distance` / `should_calc`。加工側 read: `previous_residual` / `residual_history` / `velocity_ema` / `acceleration_ema`。これが 3 名前空間分割の自然な継ぎ目。

### 3.2 forward パッチとモードディスパッチ

- `hareskip/patcher.py` が `backend.nn.anima.Anima._forward` / `.forward` を monkey patch。DiT forward 全体（embed → t_embedder → blocks ループ → final_layer → unpatchify）を再実装し、blocks ループをスキップ可能化する。
- 唯一の分岐点は `_hareskip_forward_body` のステップ毎判定ブロック。それ以外（embed, ResRefine residual 適用など）は共通。

```python
# patcher.py（要旨）
hareskip_mode = STATE.hareskip_mode == MODE_HARESKIP
compute_rel_l1 = (not hareskip_mode) or STATE.calibration_capture_active()
...
if hareskip_mode:
    shared_force = _shared_force_full_reason(cache, cond_or_uncond)  # first_call / missing_residual
    should_calc = (shared_force is not None) or _hareskip_should_calc(step_index)
else:  # MODE_TEACACHE
    # 旧 accumulator → threshold → _tea_force_full_reason 経路をビット同一で維持
```

- `_shared_force_full_reason`（両モード共通、パターン参照より**先に**評価）: `hareskip_model_calls == 1` → `first_call`、いずれかの slot に `previous_residual` が無ければ → `missing_residual`。これで「residual が存在する前にスキップしない」保証を両モードが継承する。
- HareSkip モードでは Tea 専用の force-full 理由（`outside_progress` / `force_full_interval` / `max_skip_streak`）は**参照しない**。streak とガードは pattern 側が所有する。
- HareSkip モードでは modulated input + rel_l1 計算をスキップして計算コストを節約（calibration capture 有効時を除く。`compute_rel_l1` で制御）。
- **例外封じ込め**: `_hareskip_should_calc` は全体を try/except で包み、いかなる例外でも `True`（フル演算）を返す。外側の patched forward は例外時に元の未パッチ forward へフォールバックするため、判定ロジックから例外を漏らしてはならない（NF-2）。

### 3.3 スケジュール捕捉とパターン保持

- パターン生成には全ステップの `t_now` が事前に必要（max-streak 制約に全体像が要る）。`Anima.forward` は `p`（`StableDiffusionProcessing`）を見られないため、`on_cfg_denoiser` コールバックで sigma スケジュールを捕捉する。
- `forge_introspection.sampling_schedule_t_now(params)` が cfg_denoiser params 到達可能なオブジェクトから sigma スケジュールを復元。Forge Neo の Anima 構成では predictor が `PredictionDiscreteFlow`（`multiplier == 1.0`）で `timestep(sigma) == sigma`、各 sigma がそのまま flow time `t ∈ (0, 1]` となるため sigma→t 変換は不要。末尾の `sigma ≈ 0`（対応する model call が無い）は除去。
- 復元値がすべて有限かつ厳密に `(0, 1)` 内でなければ `None` 扱い（誤った logSNR proxy を避ける）。
- `_hareskip_ensure_pattern()` が生成開始時に `STATE.hareskip_schedule_t_now` と `STATE.hareskip_image_seed` から `SkipPattern` を一括生成し `STATE.hareskip_pattern` にキャッシュ。
- **取得失敗時は劣化**: スケジュール or image seed 欠落なら `None` を返し、`_hareskip_should_calc` はフル演算扱い。`hareskip_schedule_unavailable`（`reason=no_schedule_or_seed`）を 1 回だけ警告。

### 3.4 34 引数 3 点同期

UI 引数は 3 箇所で 1:1 に一致させる（AGENTS.md「3-Point UI Argument Sync Rule」）。

1. `Script.ui()` の `return [...]`（順序・数）
2. `RuntimeState.apply_options` の位置引数シグネチャ（順序・数）
3. `constants.UI_ARG_ORDER`（正典の順序付き名前リスト）と `EXPECTED_UI_ARG_COUNT`（= `len(UI_ARG_ORDER)`）

現在**34 引数**。元 26 引数の位置は不変で、末尾に `hareskip_mode` / `hareskip_aggressiveness` / `hareskip_skip_seed_offset`（α版）＋ `hareskip_window_start` / `hareskip_window_end` / `hareskip_zone_low` / `hareskip_zone_high`（α+1、skip window / zone boundaries のスカラーミラー）＋ `manual_skip_steps`（Manual Skip mode の生テキスト、末尾追加）を追加。`tests/test_arg_sync.py` が gradio/Forge を import せず静的に検証する。

**Skip window / Zone boundaries の UI 配線（α+1）**: `gradio_rangeslider.RangeSlider`（Forge neo core が `==0.0.8` を同梱）が import 可能なら 2 本の dual-thumb RangeSlider（`hare-skip-window`, `hare-zone-boundaries`）を可視入力ウィジェットとして置き（**return list には入れない**）、その `.change`（`lambda t: (t[0], t[1])`）で `visible=False` の 4 スカラー `gr.Number` ミラーを更新する。RangeSlider が無い環境では、この 4 コンポーネント**自体**を可視の `gr.Slider`（start/end/low/high）フォールバックとして生成する（同一変数名・同一位置）。どちらの経路でも return list に入るのは 4 スカラーのみで、これらの位置は不変（末尾の `manual_skip_steps` は 34 番目として更に後ろに追加）。

---

## 4. HareSkip モード仕様

### 4.1 軌道座標 z

```
t = clamp(t_now, 1e-6, 1 - 1e-6)
z = 2 * ln((1 - t) / t)
```

生成が進む（`t_now` 降順）ほど z は増加する。実装は `skip_pattern.logsnr_proxy_from_t_now`。

### 4.2 p_skip 式と aggressiveness 写像（`sigmoid_band_v0.1`）

```
p_skip(z; a) = p_cap
             * sigmoid((z - z_enter) / tau_enter)
             * sigmoid((z_exit - z) / tau_exit)
```

`a ∈ [0, 1]`（スライダー、既定 0.5）を `[0, 1]` にクランプした上で:

| パラメータ | 式 | a=0.0 | a=0.5 | a=1.0 |
| --- | --- | --- | --- | --- |
| `p_cap` | `0.40 + 0.40·a` | 0.40 | 0.60 | 0.80 |
| `z_enter` | `−1.8 − 5.0·a^1.35` | −1.80 | −3.75 | −6.80 |
| `tau_enter` | `0.55 + 0.35·a` | 0.55 | 0.725 | 0.90 |
| `z_exit` | `4.2 + 1.0·a` | 4.20 | 4.70 | 5.20 |
| `tau_exit` | `0.45`（定数） | 0.45 | 0.45 | 0.45 |

実装は `probability_models.SigmoidBandV0_1`。返り値は `[0, 1]` にクランプ。`METHOD_NAME = "HareSkipStochasticDensity"`、`METHOD_VERSION = "0.1"`。

### 4.3 ゾーン境界と max skip streak（3 ゾーンモデル）

ゾーン境界は**ユーザー設定可能**（α+1）。UI は 1 本の dual-thumb コントロール（`(low, high)`, UI レンジ −8.0..+8.0 step 0.1, 既定 `(−4.0, 0.0)`）。既定境界での分類:

| ゾーン | z 範囲（既定境界） | max streak |
| --- | --- | --- |
| danger | `z < low`（既定 −4） | 1 |
| middle | `low ≤ z < high`（既定 −4..0） | 2 |
| safe | `z ≥ high`（既定 0） | 3 |

実装は `skip_pattern.zone_from_z(z, boundaries=(low, high))` と `ZONE_MAX_STREAK`。`apply_max_streak_constraint(skip, z, p, zone_boundaries)` が境界を受け取る。**上限は各ステップが自分のゾーンのものを使う**（run 単位の `min` ではない。2026-08-03 変更、§4.5 参照）。境界の正規化（`[−8, 8]` クランプと `low ≤ high` の swap）は `apply_options` の責務。

> **2026-07-10 更新 — final ゾーン廃止**: α版実装は当初 `docs/HareSkip-design.md` §rough zone と max skip streak に従い danger/middle/safe/final の 4 ゾーン（`final: z ≥ 4`, max streak 1、「最終仕上げ保護」）を実装していた。ユーザーの本来の意図は序盤・中盤・終盤の **3 phase** streak モデルであり、生成終盤の減速は `probability_models.py` の skip-probability taper（`z_exit` / `tau_exit` による確率の絞り込み）**のみ**で担う設計だった。2026-07-10 の層別再分析で、アーカイブデータに見えた「final 帯域での密度低下」が選択バイアスであったと判明した: 較正プール（`ER SDE-Beta 30steps Shift3`）は daraskme 74% / Uji 26% の構成で、z ≥ 4 に到達する係数は daraskme 側にしか出現しない。daraskme に層別した上で密度を見ると final 帯域での低下は消える。また、条件を揃えた step-effect 比較では、step25 の skip はどの条件でも LPIPS を改善しており、「終盤は危険」という根拠にならない。したがって final ゾーン（と streak=1 の追加制約）には定量的な裏付けが無いと判断し、danger/middle/safe の 3 ゾーンに統合した（`safe` は `z ≥ 0` に拡張、旧 `z ≥ 4` の境界は消滅）。skip-probability taper は温存するが、**その較正方法自体は現行データでは正当化できず、再較正が必要**（`docs/archive/HANDOFF-next-session.md` §4 参照）。これは `docs/HareSkip-design.md` §rough zone と max skip streak からの**意図的な逸脱**であり、設計正典は書き換えず本書に差異として記録する。

### 4.4 Skip window（旧ガード式を置換、α+1）

自動ガード規則（旧 `guard_count = max(1, round(num_steps * 0.05))`）を廃止し、**ユーザー設定可能な skip window**（progress 領域）に置き換えた。UI は 1 本の dual-thumb コントロール（`(window_start, window_end)`, レンジ 0.0..1.0, 既定 `(0.05, 0.95)`）。

```
progress_i = idx / (num_steps − 1)   # 0-based, num_steps ≥ 2
eligible = window_start ≤ progress_i ≤ window_end
```

window 外のステップは強制フル（`p = 0.0`, `skip = False`）。既定 `(0.05, 0.95)` は 30 steps で旧ガード（index 0,1,28,29 がフル・`p = 0`）を厳密に再現する（`idx 1` の progress = 0.0345 < 0.05 で除外、`idx 28` の progress = 0.9655 > 0.95 で除外）。実装は `skip_pattern.generate_skip_pattern(..., skip_window=(...))` と内部の `progress_for_step` / `_draw_pattern`。`SkipPattern` は `guard_count` に代えて `skip_window` / `zone_boundaries` / `guarded_steps`（window 除外ステップ数、ログ用）を持つ。window の正規化（`[0, 1]` クランプと `start ≤ end` の swap）は `apply_options` の責務。

> **2026-07-10 WYSIWYG 決定（ユーザー決定）— 隠れた末尾セーフティネット無し**: skip window は「見たまま」の挙動を持つ。UI で `(0.0, 1.0)` を設定すれば**最終ステップを含む全ステップが skip 対象**になる。旧実装のような自動末尾ガードや隠れた「最終ステップ保護」は**存在しない**——これは意図的な UX 決定である。（実運用で第 1 ステップがフルになるのは patcher の共有 `first_call` / `missing_residual` 強制フル経路によるものであり、これは推論上の必然であって UI ガードではない。この経路は skip window とは独立で、除去してはならない。）

### 4.5 streak 刈り込みアルゴリズム（`apply_max_streak_constraint`）

決定論的（RNG 不使用）。左→右の 1 パスで、`skip` を in place に書き換える。

1. 連続スキップ数のカウンタを 0 で始める。
2. ステップ `k` を先頭から走査する。`skip[k]` が False なら（元からフルなので）カウンタを 0 に戻して次へ。
3. `skip[k]` が True なら、**そのステップ自身のゾーン**の上限 `allowed = ZONE_MAX_STREAK[zone_from_z(z[k], boundaries)]` を取る。`counter >= allowed` ならそのステップをフル化（`skip[k] = False`）してカウンタを 0 に戻す。そうでなければスキップを維持してカウンタ +1。
4. `p_by_step` はシグネチャ互換のため受け取るが、ステップ単位の規則にはタイブレークが不要なので未使用。

> **2026-08-03 更新 — 塊単位 min 適用からステップ単位適用へ（ユーザー承認済み）**: 旧実装は「連続スキップの塊（run）ごとに、塊内ゾーンの `min` を許容 streak とし、`(p, z, index)` 最小のステップを 1 つずつフル化して再走査する」貪欲法だった。この規則は高確率域で破綻する: p が高いと全ステップが 1 つの run に融合し、その run が danger ステップを 1 つでも含むと許容が 1 に落ちて**スケジュール全域に danger 上限が適用される**。合成 30-step 降順スケジュールで全ステップ提案（p = 1.0 相当）した場合、刈り込み後のスキップ数は **3 本**まで崩壊した（ステップ単位で最適に刈れば 21 本）。aggressiveness を上げるほどスキップ数がむしろ減る非単調性を生み、v0.1 時代からの「a = 1.0 で目標 15 に届かない」問題の真因だった。新規則では各ステップが自分のゾーンの上限だけを見るため、danger ステップは自分の前後を短く切るだけで済み、崩壊が消える（同条件で 3 → 21 本、平坦 p 0.1〜1.0 の掃引で平均スキップ数が単調非減少になることを回帰テストで固定）。決定論・RNG 不使用・in place・関数シグネチャは従来どおり。影響範囲は HareSkip モードと exact-target の実現パターン（Manual Skip 経路は刈り込みを通らないため既存実験データの妥当性に影響なし）。`p_cap(a)` の再較正が必要。

### 4.6 skip_seed 導出

```
skip_seed = int(sha256(f"{image_seed}|hareskip|{offset}").hexdigest(), 16) mod 2^63
```

組み込み `hash()` は使わない（プロセス毎ソルトで再現不能になるため）。実装は `skip_pattern.derive_skip_seed`。`generate_skip_pattern` は各再抽選で `random.Random(skip_seed + attempt)` を使い決定論を保つ。`offset` は「シード値」ではなく派生計算のオフセットであり、`offset=0` は「その画像シードに対する標準パターン」を意味する（ランダムでもシード 0 でもない）。パターンのランダム性の源は画像シードで、offset は画像シードを固定したままパターンだけ引き直す用途に使う。

### 4.7 infotext キー一覧（`script.py` で確認）

すべて `postprocess_image` / 生成開始フックで書き込む。pattern 実現値（`skip_window` / `zone_boundaries` 以下のキー）は pattern が最初の forward で遅延生成されるため `postprocess_image` で書き込む設計。

| モード/層 | キー |
| --- | --- |
| 共通（identity） | `HareSkip enabled`, `HareSkip mode` |
| HareSkip モードのみ | `Hare method`, `Hare method_version`, `Hare probability_model`, `Hare aggressiveness`, `Hare skip_seed_offset`（生成開始時）／`Hare skip_window`（例 `0.05-0.95`）, `Hare zone_boundaries`（例 `-4.0/0.0`）, `Hare params`, `Hare skipped_steps`, `Hare skip_count`, `Hare skip_seed`（`postprocess_image`）／スケジュール失敗時のみ `Hare pattern = "unavailable"` |
| TeaCache モードのみ | `Tea threshold`, `Tea progress`, `Tea coefficient_profile`, `Tea max_skip_streak`, `Tea force_full_interval`, `Tea shift`, `Tea modulated_source`, `Tea capture_pairs`, `Tea auto_row_index`, `Tea auto_row_name` |
| Manual Skip モードのみ | `Manual skipped_steps`（実現値。1始まりステップ番号をスペース区切り、`postprocess_image`） |
| 共通（両モード） | `ResRefine formula`（＋非 Reuse 時 `ResRefine use_prediction_after_progress`, `ResRefine apply_prediction_from_skip`, `ResRefine prediction_strength`, `ResRefine slope_ema_smoothing`, `ResRefine curve_ema_smoothing`／Taylor2 時 `ResRefine taylor2_curve_strength`） |

**ステップ番号は 1 始まりで統一**: ログ・infotext のユーザー可視ステップ表記はすべて 1 始まりで Forge の `Sampling steps` 数と一致する（例 `step=10/30`、`Manual skipped_steps = 10 12`）。内部インデックスは 0 始まりで格納し、出力時にのみ `+1` 変換する（表示を直すために格納表現を変えない）。2026-07-13（`5c20835`）に `hareskip_call=` / `hareskip_step=` / `hareskip_summary=` の残る0始まり表記箇所を共通ヘルパ `_fmt_step_display`（`STATE.generation_steps` 既知なら `{n+1}/{total}`、未知なら `{n+1}`）で統一。同コミットで decision reason も整理: Manual Skip の通常フル演算は `reason=manual_full`（従来 Tea 専用語彙の `threshold` に誤フォールバックしていた）、`_hareskip_log_call` は明示 reason 引数を先頭に置き ResRefine スロット理由をその後に付す（例 `reason=manual,0:formula,1:formula`）。

verbose_trace 有効時はステップ毎に `hareskip_step=` で z_i / p_i / skip をコンソール出力。パターン一括生成時は `hareskip_pattern=` サマリを 1 行出力。

**防御的パッチング（2026-07-13, `28d24ee`）**: UjiCache 等、同一 `Anima.forward` を patch する他拡張と共存した場合、そちらの patch がこちらの wrapper を上書き・teardown 時に上書きしたまま自身の保存 original を setattr する事故が起こり得る（`model_calls=0` として観測される）。対策として `STATE.patches[kind]` に wrapper/cls/attr を保持し、apply 時に整合性チェック（wrapper が入れ替わっていれば `hareskip_patch_clobbered` を警告し自己修復・再適用）、remove 時のガード（自分の wrapper が入っていなければ `hareskip_patch_not_ours` を警告しレジストリのみ破棄、他拡張の patch には触れない）を実装。`is_patch_installed()`（wrapper 同一性チェック）を追加し診断表示の `active=` フィールドに使用（`is_patched()` はレジストリ所属のみを見る軽量版として存置）。`log_timing_summary` に zero-execution 検出（`hareskip_enabled` かつ `denoiser_calls>0` だが `model_calls==0` の場合に `hareskip_patch_never_ran` を警告）を追加。

---

## 5. 既知の制約・α版の限界

1. **aggressiveness → 実スキップ数の較正未実施（要注意・計画との差異あり）**: 純粋モジュールを合成 30-step 降順スケジュール（`t_now` 0.999→0.003 線形、`tests/test_skip_pattern.py` の `_t_now_schedule`）で実測すると、streak 刈り込み後のスキップ数は **seed 依存**で以下（`derive_skip_seed(0,0)` 使用時）:
   - a=0.0 → 4 skips（`expected_skips_before_streak ≈ 6.8`）
   - a=0.5 → 10 skips（`expected ≈ 13.2`）
   - a=1.0 → 13 skips（`expected ≈ 20.1`）

   （2026-08-03 §4.5 の刈り込み修正後の実測値。修正前は同 seed で 3 / 8 / 10 skips だった。50 seed 平均では a=0.0 → 6.7、a=0.5 前後 → 11〜13、a=1.0 → 15.8 で、a に対して単調増加する。修正前は a=0.8 の 10.6 をピークに a=1.0 で 9.5 へ**下降**していた。）**計画書（`ok-cuddly-hamster.md`）の「合成 30step・a=0.5 で streak 刈り込み後 5skip」という記述は現行実装と一致しない**（§ 末尾の差異報告を参照）。設計目標（a=0.5 で約 10 skips、a=1.0 で約 15 skips）に対し、a=1.0 が目標を下回る傾向があり、`p_cap(a)` / `z_enter(a)` の実機較正が必要。

2. **z 符号・スケジュール捕捉パスの実機未検証**: sigma→t 写像と z の増減方向は検証マシンで未確認。`sampling_schedule_t_now` の属性到達パスが実機で正しく解決されるかは要検証。失敗時はフル演算に劣化する設計なので安全側。

3. **prompt 依存の致命ステップは検出不能**: 本手法は軌道座標のみに依存し、prompt に依存する致命ステップは検出できない。streak 制約とガードで緩和するのみ。

4. **exact-target mode は API のみ**: `generate_skip_pattern(exact_target=...)` はテストと API に存在するが UI 露出なし（ユーザー決定）。到達不能な target でも例外を投げず最近傍を返す。

5. **calibration_capture.py の JSONL ヘッダのネスト未整理（軽微）**: `_build_header` の出力構造がネスト整理されていない。TeaCache デバッグ機能であり動作に支障はない。

---

## 6. 将来課題

設計書 [`docs/HareSkip-design.md`](HareSkip-design.md) の未決事項を参照:

- step 間隔補正 `Δz`（不均一なステップ間隔への補正）。
- ゾーン境界 / streak の条件依存化（sampler / scheduler / shift ごとの調整）。**2026-07-29 更新 — Shift依存化が必要と実証された**: Shift転移スポットチェック（ER SDE-Beta、z軸オーバーレイ分析、出典 `docs/recalibration-2026-07/REPORT.md` および `experiment-HareSkip/EXPERIMENT-LOG.md` 2026-07-29エントリ）により、中心仮説「損傷は z（logSNR）だけの関数」は素直には成立しないと判明した。Shift変換によるΔz（=2·ln(shift比)、Shift 3→1 で +2.1972）は全ステップ一様なオフセットとして解析的に説明できるが、それを補正してもなお**損傷カーブの急峻さ自体がShiftで変化する**: z<0（danger側）ではShift=1の損傷がShift=3カーブの中央値の1.3〜1.7倍、z>0（safe側）では0.3〜0.6倍にとどまる。skip軸（ステップ位置そのもの）で比較しても重ならないため、「z整列＋横ずらし」では吸収できない。実用方向の含意: **低Shiftで運用するほど序盤（danger側）の保護を強める方向の較正が安全**。ゾーン境界の単純な平行移動では不十分な可能性があり、Shift別のゾーン境界較正が今後の課題として残る。
- `p_cap` / `z_enter` の較正手順の確立（実機での a → skip 数較正）。
- 確率モデルの対抗仮説 `monotone_saturate`（単調飽和型）の A/B 比較（レジストリ登録で機構変更不要）。
- **streak項について（2026-07-29 更新 — 予測式には不要と確定）**: 第3段階 out-of-sample 検証（凍結事前予測表と実測の照合、出典 `docs/recalibration-2026-07/REPORT.md` および `experiment-HareSkip/EXPERIMENT-LOG.md` 2026-07-29エントリ）で、streak項を加えた対照式はプールでΔρ=−0.049（事前登録基準の+0.02未満を満たさず）となり、**streak項は品質順位の予測式には不要と結論**した。第2段階 in-sample でEuler-Betaにのみ見えていた+0.046の改善は、out-of-sampleでは再現せず符号反転しており過学習と判明。ただし、ゾーン別 max skip streak 制約（§4.3・§4.5 の実装済みガード機構）は予測式とは別の役割（実行時の連続スキップ長そのものの制約）を持ち、第2段階の実測（アンカー幅の5〜15%の微小だが実在するペナルティ）に基づき維持する。
- **probability_models 再較正について（2026-07-29 更新 — 較正に必要な実測データが揃った）**: 単発29位置×45条件（第1段階）、相互作用28パターン×45条件（第2段階）、out-of-sample 450点（第3段階）、±1ステップアンカー90点、いずれも固定コミット`b6164214`製の実測データが揃った（出典 `docs/recalibration-2026-07/REPORT.md` および `experiment-HareSkip/EXPERIMENT-LOG.md` 2026-07-27〜29の各エントリ）。予測式（単発損傷の和＋飽和リンク、調整項2個）は事前登録基準（プールSpearman ρ≥0.8）に対しρ=0.930で合格した。実装規律は従来どおり: `sigmoid_band_v0.1` は書き換えず、較正結果を反映する場合は新バージョン（例 `sigmoid_band_v0.2`）を `probability_models.py` レジストリに登録して切り替える（下記末尾の記述を参照）。
- 再キャリブレーション実験計画は `docs/archive/HANDOFF-next-session.md` §4.5 参照（実験計画の正典は移管済み。§4.5 冒頭の移管注記のとおり `experiment-HareSkip/EXPERIMENT-PLAN.md` を参照。実験の総括は `docs/recalibration-2026-07/REPORT.md`）。
- **Skip seed offset の Forge 準拠化**（2026-07-12 決定、未実装）: `offset=-1` をランダム扱いにする、サイコロ／リユースボタンの追加、`infotext_fields` 登録による PNG Info タブからの HareSkip 設定一式復元。詳細は `docs/skip-seed-offset-plan.md` 参照。
- **Manual Skip 複数行化**（2026-07-13 設計確定、未実装、次セッション最優先）: Manual skip steps テキストボックスを multiline 化し、空でない各行を1ジョブとして Auto Tea mode の展開機構で順次生成する。詳細は `docs/ManualSkip-spec.md` §10、`docs/archive/HANDOFF-next-session.md` §1.5 参照。

較正の実運用は §5-1 と `docs/archive/HANDOFF-next-session.md` の「予想される修正ポイント」に従い、`probability_models.py` に新バージョン（例 `sigmoid_band_v0.2`）を登録して切り替える方針（v0.1 は書き換えない）。
