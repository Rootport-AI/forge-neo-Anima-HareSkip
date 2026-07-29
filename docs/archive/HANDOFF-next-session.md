# HareSkip 次セッション向けハンドアウト

> **頭注（2026-07-29）**: §4.6は `docs/skip-seed-offset-plan.md` へ切り出し済み（そちらが現役の正典）。本書全体は2026-07-13時点の歴史的スナップショットとして凍結。実験計画の正典は `experiment-HareSkip/EXPERIMENT-PLAN.md` へ移管済み（§4.5の注記どおり）。実験の総括は `docs/recalibration-2026-07/REPORT.md`。

- 作成日: 2026-07-10（α+1・Manual Skip mode 追加後に更新）／2026-07-13 更新（3件のバグ修正・Manual Skip 実機検証＋等価性証明・1始まりログ規約の反映）
- HEAD コミット: `5c20835ac0caa9120948015f312ecbd90e9b7e6d`（branch `main`, clean）

このファイルは、次の Claude セッション（またはユーザー）が再調査ゼロで作業を引き継ぐためのハンドオフである。設計の正典は [`docs/HareSkip-design.md`](HareSkip-design.md)、要件・設計・仕様の統合資料は [`docs/SPEC-alpha.md`](SPEC-alpha.md)。

---

## 1. 現在地

α版＋α+1（3 ゾーン化・RangeSlider・Manual Skip mode）は実装完了。**Manual Skip mode は実機検証済み**（2026-07-13、§3-9 参照）。HareSkip 確率モードの実機検証（z 符号、a→skip 数較正等）はまだ未実施。

コミット（`main` 直上、抜粋・新しい順、すべて `Co-Authored-By: Claude Fable 5`）:

| hash | 内容 |
| --- | --- |
| `5c20835` | fix: ステップログを1始まりに統一・decision reason を明確化（= HEAD） |
| `28d24ee` | fix: 他拡張による Anima.forward パッチ衝突への防御的パッチング |
| `6dee42e` | fix: Manual Skip リストが generation reset で消える不具合を修正 |
| `a67f559` | feat: Manual Skip mode 追加（明示ステップ指定・独立モジュール） |
| `54cf89b` | docs: Manual Skip mode 仕様書追加（設計確定） |
| `03e0acc` | docs: 2026-07-10 ブレインストームの再較正実験計画を追加 |
| `aaa2aea` | feat: skip window / zone の dual-thumb RangeSlider コントロール追加 |
| `84bcfb4` | feat: skip_pattern の skip window / zone boundaries をパラメータ化 |
| `074462e` | feat: max-skip-streak を 3 ゾーンモデルに（final ゾーン廃止） |
| `b337e36` | docs: UjiCache 較正履歴をアーカイブ、ResRefine EMA notes 書き直し |
| `f2641b3` | docs: α版統合仕様書と次セッションハンドオフ追加 |
| `5520a92` | docs: README/AGENTS/CHANGELOG/NOTICE を HareSkip 用に書き直し |

- **テスト**: `pytest tests/ -q` → **89 件緑**（`test_arg_sync.py` / `test_probability_models.py` / `test_skip_pattern.py` / `test_manual_skip.py`）。gradio/Forge を import せず実行可能。
- **済み**: 純粋モジュール実装・単体テスト、3 モードディスパッチ（HareSkip / TeaCache / Manual Skip）、スケジュール捕捉ヘルパ、UI 再構成、**34 引数 3 点同期**（α+1 で skip window / zone boundaries の 4 スカラー、Manual Skip mode で `manual_skip_steps` を末尾追加）、dual-thumb RangeSlider コントロール＋プレーン Slider フォールバック、3 ゾーン max-skip-streak（final ゾーン廃止）、ユーザー設定可能 skip window（旧自動ガード置換）、Manual Skip mode（`hareskip/manual_skip.py`）、infotext 分割、docs 一式。**Manual Skip mode 実機検証＋ UjiCache 等価性証明済み**（2026-07-13、§3-9）。他拡張による `Anima.forward` パッチ衝突に対する自己修復・警告（`hareskip_patch_clobbered` / `hareskip_patch_not_ours` / `hareskip_patch_never_ran`, `is_patch_installed()`）、コンソールログの1始まりステップ表記統一（`step=n/total`）。
- **未**: HareSkip 確率モードの実機検証（拡張ロード、スキップ発火、z 符号、a→skip 数較正、再現性、PNG メタデータ、TeaCache ビット同一性）。§3 のチェックリストが対象（Manual Skip 項目は完了済み）。
- 本ドキュメント自体のコミット（docs のみ）が HEAD の場合があるため、コード最新＝`5c20835`、docs 追記＝`796c15f` 以降を参照すること。

---

## 1.5 次セッション最優先作業

1. **【最優先・実装着手可】Manual Skip 複数行化の実装**: 2026-07-13 ユーザー承認済みの確定設計。設計は完全に決まっており、追加のブレインストームは不要 — そのまま実装フェーズに入れる。詳細仕様は `docs/ManualSkip-spec.md` §10「複数行化（v2、2026-07-13 設計確定・未実装）」参照。要点: テキストボックスを multiline 化し、空でない各行 = 1ジョブ（現行1行文法をそのまま流用、ヘッダ・範囲記法は導入しない）、空行は無視、全行を `before_process` で事前検証（不正行は行番号を名指ししてジョブ全体を中止）、Auto Tea mode の展開機構（`p.n_iter` を行数倍化）を再利用、全行が同一 seed テンプレートを共有、単一行入力は完全後方互換。
2. HareSkip 確率モードの実機検証（§3 のチェックリスト項目 1〜8）— 引き続き優先度が高いが、上記 Manual Skip 複数行化より後で良い。

---

## 2. リポジトリ地図

純粋モジュール群（`constants.py` / `skip_pattern.py` / `probability_models.py`）は **Forge/torch/gradio 非依存で pytest 可能**。それ以外は Forge 実行時に依存。

| ファイル | 役割 |
| --- | --- |
| `scripts/hareskip.py` | Forge エントリ（3 行 re-export）。 |
| `requirements.txt` | 唯一の依存 `gradio_rangeslider==0.0.8`（dual-thumb RangeSlider 用。Forge Neo core 同梱、不在時はプレーン Slider に劣化）。 |
| `hareskip/__init__.py` | パッケージ初期化。 |
| `hareskip/constants.py` | **純粋**。モード ID（HareSkip / TeaCache / Manual Skip の 3 種）、`UI_ARG_ORDER`（34）、`EXPECTED_UI_ARG_COUNT`。 |
| `hareskip/skip_pattern.py` | **純粋**。stochastic pattern 生成（z, ユーザー設定可能ゾーン境界〔3 ゾーン〕, skip window, streak 刈り込み, seed 導出, exact-target）。 |
| `hareskip/probability_models.py` | **純粋**。`p_skip` レジストリ。`sigmoid_band_v0.1` 組み込み。 |
| `hareskip/manual_skip.py` | **純粋**。Manual Skip mode のステップ列パース/検証（`parse_manual_steps` / `validate_manual_steps` / `ManualSkipError`）。 |
| `hareskip/script.py` | Gradio UI（`ui()`）と生成時フック、infotext 書き込み。 |
| `hareskip/state.py` | 設定スナップショット、`RuntimeState`、`TEA_PRESET_REGISTRY`、`apply_options`。 |
| `hareskip/patcher.py` | `Anima._forward` monkey patch、モードディスパッチ、`_hareskip_should_calc`、`_hareskip_ensure_pattern`、restore。 |
| `hareskip/resrefine.py` | residual 予測/検証/EMA（patcher から抽出）。 |
| `hareskip/forge_introspection.py` | sigma スケジュール捕捉（`sampling_schedule_t_now`）、モデル構造 introspection。 |
| `hareskip/diagnostics.py` | コンソールスナップショット/サマリ。 |
| `hareskip/calibration_capture.py` | TeaCache 係数再較正用 JSONL 捕捉（Tea 専用デバッグ）。 |
| `hareskip/auto_teacache.py` | Auto Tea mode CSV 解析/適用。 |
| `hareskip/callbacks.py` `settings.py` `model_detect.py` `tensor_dump.py` `timing.py` `logging.py` | コールバック配線 / 設定 / モデル判定 / テンソルダンプ / 計時 / ロガー（`[HareSkip]`）。 |
| `tests/test_arg_sync.py` | 3 点同期の静的検証（34 引数）。 |
| `tests/test_probability_models.py` `tests/test_skip_pattern.py` `tests/test_manual_skip.py` | 純粋モジュールの単体テスト。 |
| `docs/HareSkip-design.md` | 設計正典（stochastic skip density）。 |
| `docs/SPEC-alpha.md` | 要件・設計・仕様の統合資料。 |
| `docs/ManualSkip-spec.md` | Manual Skip mode 仕様書（実装済み）。 |

---

## 3. 検証マシンでの確認チェックリスト（優先順）

Forge Neo + Anima + GPU の検証マシンで、git pull → 正規手順で拡張導入（**導入はユーザー手動**）後に上から順に確認する。

1. **拡張ロードとログ**: 起動時に `[HareSkip]` ログ、生成時に `hareskip_pattern=` サマリが出るか。
   - 出なければ `hareskip_schedule_unavailable`（`reason=no_schedule_or_seed`）を確認。→ `forge_introspection.sampling_schedule_t_now` の属性到達パス修正が必要。試行順は `_candidate_sigma_sources`: **(a)** `params.sigmas` / `sigma_schedule` / `all_sigmas`、**(b)** `params.denoiser` とその `.sampler` の同名属性、**(c)** `modules.shared.state` の `sigmas` / `sigma_schedule`。実機で実際にスケジュールを持つ属性を特定し追加する。
2. **z 符号の実測**: verbose trace（`hareskip_verbose_trace`）を ON にし `hareskip_step= step=.. z=.. p=.. skip=..` ログで z_i / p_i を確認。**生成後半で z が増加**するはず（`t_now` 降順 → z 増加）。逆なら sigma→t 写像か proxy 方向の再確認。
3. **a=0.5 / 1.0 の実スキップ数**: `Hare skip_count` infotext を確認。設計目標は a=0.5 で約 10 skips、a=1.0 で約 15 skips（`docs/SPEC-alpha.md` §5-1: 合成スケジュールでは a=0.5→8〜10, a=1.0→10 と目標未達傾向）。
4. **再現性**: 同一 image seed + 同一 offset で同一 `Hare skipped_steps` / `Hare skip_seed` になるか。
5. **offset 変更で別 pattern**: offset を変えると別 `skipped_steps` になるか（同一 image seed のまま引き直し）。
6. **TeaCache モードが旧 UjiCache と同一挙動**: TeaCache モードに切替え、同条件で旧 UjiCache とビット同一の skip 挙動・画質か。
7. **PNG infotext**: 保存画像メタデータに `Hare skipped_steps` / `Hare skip_count` / `Hare skip_seed` / `Hare skip_window` / `Hare zone_boundaries` / `Hare params` 等が入るか（`postprocess_image` 書き込み）。
8. **RangeSlider レンダリング**: HareSkip グループに dual-thumb の「Skip window (progress)」（`hare-skip-window`）と「Zone boundaries (logSNR proxy)」（`hare-zone-boundaries`）が表示され、両端つまみで値を動かすと `Hare skip_window` / `Hare zone_boundaries` infotext に反映されるか（RangeSlider → 隠し `gr.Number` ミラー → `apply_options` の経路）。Forge neo core は `gradio_rangeslider==0.0.8` を同梱するため通常この経路で描画される。フォールバック（RangeSlider 不在時のプレーン Slider 4 本）は**わざわざ検証用にアンインストールする必要は無い**——両経路はコードに実装済みで、引数数は 34 で不変（`tests/test_arg_sync.py` が静的に保証）。RangeSlider 版の window / zones を動かしたとき infotext が変化すれば配線は正しい。
9. **Manual Skip mode 動作確認 — ✅ DONE（2026-07-13）**: Skip mode Radio を「Manual Skip」に切替え、「Manual skip steps」テキストボックスに `15, 17, 19, 21, 23`（1始まり）を指定して生成し、旧 UjiCache 拡張が同じステップを0始まり（`14, 16, 18, 20, 22`）でスキップした過去の生成と比較。**結果: ビット同一**（Nz DoppelPix Judge: LPIPS=0.000000, SSIM=1.000000, PSNR=inf。所要時間も整合: 73.1s vs 72.5s、スキップ各ステップ約0.004s）。Manual Skip のスキップ実行機構（residual Reuse 経路）が UjiCache のそれと完全等価であることを実測で証明。検証マシン上の旧 UjiCache 拡張は本検証後に無効化済み。詳細は `docs/ManualSkip-spec.md` §9。
   - (a)〜(d) の個別動作確認項目（ログの `step=n/total` 1始まり表記・`reason=manual`/`manual_full`・infotext・エラー中止・空欄baseline）も本検証で通過を確認済み。ログ表記は `5c20835` で1始まりに統一されており、上記結果はその修正後の状態。

---

## 4. 予想される修正ポイント

- **スケジュール捕捉失敗時**: §3-1 の通り `_candidate_sigma_sources` に実機で有効な属性パスを追加。復元値は全て有限かつ `(0,1)` 内でなければ `None` 扱いになる点に注意（範囲外なら sigma→t 前提が崩れている）。
- **較正（`p_cap(a)` / `z_enter(a)` 調整）**: `probability_models.py` に**新バージョン `sigmoid_band_v0.2` を登録して切り替える**。**v0.1 は書き換えない**。手順: `register("sigmoid_band_v0.2", NewModel())` → `STATE.hareskip_probability_model` を新名に。`skip_pattern.py` は名前で lookup するため変更不要（`generate_skip_pattern(..., probability_model=...)`）。a=1.0 が skip 数不足なら `p_cap` 上限や `z_enter` の負方向拡張を検討。
- **streak 刈り込みが強すぎる場合**: `apply_max_streak_constraint` は run 内 `(p,z,index)` 最小をフルに戻す。`expected_skips_before_streak`（刈り込み前 p 総和）と実 `skip_count` の乖離が大きすぎるなら `ZONE_MAX_STREAK`（danger=1 / middle=2 / safe=3 の 3 ゾーンモデル、2026-07-10 に final ゾーンを廃止済み。§ 下記および `docs/SPEC-alpha.md` §4.3 参照）や safe=3 の緩和を検討。刈り込みは決定論なので同一入力で再現する。**α+1 以降、ゾーン境界 `(low, high)` は UI から調整可能**（既定 `(−4.0, 0.0)`、`hare-zone-boundaries` の dual-thumb）。境界を動かせば各ステップのゾーン分類 → 許容 streak が変わるので、コードを触らず実機で刈り込み強度を較正できる。
- **skip window の調整（α+1）**: skip 対象の progress 範囲は UI の「Skip window (progress)」（`hare-skip-window`、既定 `(0.05, 0.95)`）で調整可能。序盤・終盤をより広く/狭く skip 対象にしたければ window を広げ/狭める。**隠れた末尾セーフティネットは無い**（WYSIWYG、2026-07-10 ユーザー決定）: window を `(0.0, 1.0)` にすれば最終ステップも skip 対象になる（第 1 ステップは patcher の `first_call` 強制フルで実運用上フルのまま——これは window とは独立）。旧 `guard_count` 相当は既定 window で再現される。
- **skip-probability taper の再較正（将来課題・ユーザー決定）**: 2026-07-10 に旧 final ゾーン（`z ≥ 4` で streak=1 に絞る「最終仕上げ保護」）を廃止し、danger/middle/safe の 3 ゾーンに統合した（層別再分析で final 帯域の密度低下が選択バイアスと判明したため、定量的裏付けが無いと判断）。生成終盤の減速は `probability_models.py` の skip-probability taper（`z_exit` / `tau_exit`）**のみ**が担う設計に戻った。taper 自体は温存するが、**その較正方法（`z_exit(a)` / `tau_exit` の値）は現行データでは正当化できず、新規データによる再較正が必要**——これはユーザー決定であり、次セッション以降の作業対象。較正に使えるデータが揃うまでは v0.1 の値のまま据え置く。

---

## 4.5 再キャリブレーション実験計画（2026-07-10ブレインストーム決定）

> **移管注記（2026-07-13）**: 実験計画の正典は [`../../experiment-HareSkip/EXPERIMENT-PLAN.md`](../../experiment-HareSkip/EXPERIMENT-PLAN.md)（リポジトリ外・同親階層の実験ディレクトリ）に移管した。本節は移管時点の歴史的記録であり今後更新しない。

### 背景（短く）

- 現行 `sigmoid_band_v0.1` の立ち下がり側（taper: `z_exit`/`tau_exit`）は UjiCache 実験の偏ったデータ由来。立ち上がり側（序盤が危険）はバイアス非依存の証拠（step2/4 critical、prompt-seed マッチ済み）で支持される。
- 対抗仮説（ユーザー提案）: バンド型ではなく**単調飽和型**。「軌道座標に沿って確率が上がり、収束帯（DiT/flow matching の直線軌道が収束した後）では max skip streak の強制フル演算以外ほぼ全て飛ばす」。将来 `monotone_saturate_v0.1` としてレジストリ登録し `sigmoid_band` と A/B 比較する。
- DiT 性質の理解（妥当と確認、留保2点）: flow matching 系は直線輸送経路が訓練目標で終盤の速度変化は小さい（SDXL/DDIM の「収束後に絵柄が変わる」現象は起きにくい）。留保①収束≠凍結——高周波ディテールは動き続け、そこは LPIPS/SSIM が最も鈍い領域。留保②ER SDE は毎ステップノイズ注入するため ODE 系より終盤スキップの意味が重い。
- safe zone streak=3 の下では p≈1 でも実現スキップ率上限 75% で、終盤の挙動は実質決定論化する。ガチャ多様性の損失は小さい見込みだが、実機で「どれくらい運命論的か」を観測してから対策を検討する（`Hare skipped_steps` infotext から複数 seed のパターン間ハミング距離分布を集計するだけで測れる。追加実装不要）。

### 実行手段（実装済み）

- 本実験計画（第1層の単発スキップ掃引、第2層の `{i}`/`{j}`/`{i,j}` 3点セット）で「どのステップを飛ばすか」を数値で明示指定する実行手段として **Manual Skip mode が実装済み**（2026-07-10、`hareskip/manual_skip.py`、Radio 第3モード、`docs/ManualSkip-spec.md`）。カンマ区切り1始まりステップ列を1テキストボックスに指定し、生成前に `p.steps` に対して検証（非数値・範囲外・step 1 はエラーで生成中止）、`Manual skipped_steps` infotext に実現値を記録する。sigma スケジュール捕捉・確率モデルに非依存。

### 実験設計: 3層構造（組み合わせ爆発 C(30,15)≈1.55億 への回答）

方針: パターン→品質の全写像ではなく低次元構造（帯域別重み＋streak項）を測り、「低次元モデルで十分」自体を検証可能な主張にする。

- **第1層 単発スキップ掃引（主効果）**: 30 ステップ中 1 ステップだけ飛ばすパターン×全30位置×45条件（サンプラー系3: ER SDE-Beta / ER SDE-Simple / Euler-Beta × プロンプト5 × シード3）= 1,350生成。位置ごとの限界損傷曲線 `dQ(z)` を得る。O(30) で列挙不能性と無縁。
- **第2層 相互作用の抽出（加法性の検証）**: `{i}`, `{j}`, `{i,j}` の3点セットを選択的に測定（隣接ペア=streak効果、序盤×中盤ペア=帯域間相互作用、位置は10箇所程度）。「加法＋連続ペナルティ項」で説明できれば約10次元に落ちる。
- **第3層 HareSkip ランダムパターンでの out-of-sample 検証**: HareSkip の役割は探索でなく**検証用データ生成器**。第1-2層で構築した低次元モデルにモデル構築未使用のランダムパターンの品質を予測させ、実測とのランク相関/R² を報告。予測が当たること自体が「列挙しなかったことの正当化」になる。近縁パターン集中への対策: aggressiveness/window を振った層化サンプリング＋パターンハッシュ重複除去＋最小ハミング距離間引き（exact-target API と skip seed offset が道具）。

### 「序盤スキップ実験は不要」を読者に納得させる最低3点セット

1. 単発掃引の損傷曲線（序盤1ステップ単発損傷 >> 終盤5ステップ合計損傷、を実測で）
2. 序盤スキップ1→2→3個の単調性チェック（10パターン程度）
3. negative control: 序盤5連続スキップを数枚実生成し、モデル予測通り崩壊することを画像つきで示す

構成の要点: 「試さなかった弁明」ではなく「全域を予測できるモデル＋held-out（極端例含む）での的中」。

### 評価指標

LPIPS-VGG + SSIM 11p（Nz DoppelPix Judge, port 7870, 既存パイプライン流用）。争点が「指標に映らない高周波ディテール」である以上、高周波感度のある指標かクロップ拡大目視を1軸追加すること。アーカイブの既存 fatal 分析は事前情報としてのみ使用し、論文証拠としては再取得する（係数ファミリーバイアスのため）。

### Shift の扱い（2026-07-13 決定）

- 第1層〜第3層の本実験は**Shift=3（Anima 既定値）に固定**する。
- 固定の理由（3点）:
  - 設計哲学上、Shift はサンプリングスケジュール（sigma 分布）を歪めるだけで、z（logSNR）→損傷の写像 `dQ(z)` は不変のはず、というのが HareSkip の中心仮説（`HareSkip-design.md` 基本方針「sampler / scheduler / shift をまたいで意味を持つ軌道座標上の skip density」）。
  - サンプラー3系統の軸が step→z 対応の変動を既に含むため、「`dQ` が z だけの関数か」は Shift を振らなくても部分的に検証できる。
  - Shift 3水準の追加は 1,350→4,050 生成でコストが3倍になる一方、Shift は実用上頻繁に変わらない条件（Anima 既定=3）。
  - 注記: 過去に「Shift ごとに別係数が必須」と結論されたのは TeaCache の `p(x)` 多項式（rel_l1 空間、`CALIBRATION-RESULTS.md`）であり、z 空間の確率勾配には外挿しない。
- ただし「z が Shift を吸収する」は未検証の中心仮説であり（SPEC-alpha にゾーン境界の shift 別調整が将来課題として残存、`delta_z_i` 補正も未実装）、放置しない。**第1層の `dQ(z)` の形が見えた後に「Shift 転移スポットチェック」を実施する**:
  - 内容: ER SDE-Beta × プロンプト1〜2 × シード3 で、`dQ(z)` の特徴的な位置（崖の縁・谷底など5〜6点）のみを Shift1 または 2 で再測定し、z 軸上で曲線が重なるかを確認する。
  - 判定: 重なれば z 座標の普遍性が主張として強化される。重ならなければ zone 境界の shift 依存化（SPEC-alpha の将来課題）が実際に必要と判明する。数十生成の追加で中心仮説が反証可能になる。
- 経緯注記: 2026-07-13 時点まで §4.5 に Shift の扱いは明文化されていなかった（決定の不在）。本節はその欠落を埋めるもの。

---

## 4.6 Skip seed offset の Forge 準拠化（2026-07-12 決定、未実装）

> 本節は `docs/skip-seed-offset-plan.md` へ切り出し済み。そちらが現役の正典（実装はユーザーのゴーサイン待ち）。

---

## 5. 作業ルール（must-follow）

- **親エージェントは監督役**。実作業はサブエージェントに委譲し、親は方針・裏取りを統括。
- **サブエージェントに Fable5 禁止**（Opus 以下を使用）。
- **仮説 → 裏取り → 実験の順**。コード/実測で確認してから結論を書く。計画書と実装が食い違えば実装を正とする。
- **Forge neo 本体を触らない**。拡張だけで完結させる。
- **拡張導入はユーザー手動**（検証マシンでの正規手順）。
- **UjiCache リポジトリは読み取り専用**。
- **リポジトリ跨ぎの import/配線禁止**。ファイルはコピーしてから改名・編集。
- **同一ファイルの並行編集禁止**（コミット単位で直列、またはファイル分割で並列化）。同一 GPU の衝突も禁止。

---

## 6. 参照資料の場所

- **計画ファイル**: `C:\Users\pheno\.claude\plans\ok-cuddly-hamster.md`（承認済み α実装計画）。詳細版: 同ディレクトリ `ok-cuddly-hamster-agent-a26830f7c2c0e7812.md`。
- **研究アーカイブ**（読み取り専用、リポジトリにコピーしない）:
  ```
  S:\30_OriginalApps\16_HareSkip\ER SDE-Beta 30steps Shift3
  ```
  読解順:
  1. `01_RESEARCH_CONCLUSIONS\doc\stochastic_skip_density_design_2026-06-30.md`（= 本リポの `docs/HareSkip-design.md`）
  2. `trajectory_axis_handoff_2026-06-28.md`（軌道座標分析）
  3. `AnalysisTables\skip_density_gradient\skip_density_gradient_report.md`（自動生成レポート）
  4. `INTERIM-REPORT`（※ §6-1 は更新注記を参照）
- **Nz DoppelPix Judge**: 将来の大量画質評価に使用可能（port 7870、起動スクリプト `run_nz_doppelpix_jobs.ps1`）。a→skip 数較正や pattern 品質の分布評価に有用。
