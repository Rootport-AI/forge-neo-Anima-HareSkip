# HareSkip 次セッション向けハンドアウト

- 作成日: 2026-07-10
- HEAD コミット: `5520a92188be7dfe998633349f3fa4cb5a5ce225`（branch `main`, clean, push 済み）

このファイルは、次の Claude セッション（またはユーザー）が再調査ゼロで作業を引き継ぐためのハンドオフである。設計の正典は [`docs/HareSkip-design.md`](HareSkip-design.md)、要件・設計・仕様の統合資料は [`docs/SPEC-alpha.md`](SPEC-alpha.md)。

---

## 1. 現在地

α版は実装完了・push 済み。**実機検証は未実施**（開発マシンに Forge/GPU が無いため）。

コミット（`main` 直上、8 件、すべて `Co-Authored-By: Claude Fable 5`）:

| hash | 内容 |
| --- | --- |
| `b66723a` | chore: UjiCache ソースを HareSkip スケルトンとして取込 |
| `f2ee257` | refactor: Tea/ResRefine 名前空間分割 |
| `dd13563` | feat: Auto Uji mode → Auto Tea mode リネーム |
| `cc11b22` | feat: 純粋モジュール skip_pattern + probability_models 追加 |
| `07ad471` | feat: patcher に HareSkip stochastic モードディスパッチ追加 |
| `7d9c1ca` | feat: UI 再構成（HareSkip/TeaCache モード selector） |
| `0be9ebb` | feat: infotext/診断を Tea/Hare/ResRefine に分割 |
| `5520a92` | docs: README/AGENTS/CHANGELOG/NOTICE を HareSkip 用に書き直し（= HEAD） |

- **テスト**: `pytest tests/ -q` → **52 件緑**（`test_arg_sync.py` / `test_probability_models.py` / `test_skip_pattern.py`）。gradio/Forge を import せず実行可能。
- **済み**: 純粋モジュール実装・単体テスト、モードディスパッチ、スケジュール捕捉ヘルパ、UI 再構成、29 引数 3 点同期、infotext 分割、docs 一式。
- **未**: 実機検証（拡張ロード、スキップ発火、z 符号、a→skip 数較正、再現性、PNG メタデータ、TeaCache ビット同一性）。§3 のチェックリストが対象。

---

## 2. リポジトリ地図

純粋モジュール群（`constants.py` / `skip_pattern.py` / `probability_models.py`）は **Forge/torch/gradio 非依存で pytest 可能**。それ以外は Forge 実行時に依存。

| ファイル | 役割 |
| --- | --- |
| `scripts/hareskip.py` | Forge エントリ（3 行 re-export）。 |
| `hareskip/__init__.py` | パッケージ初期化。 |
| `hareskip/constants.py` | **純粋**。モード ID、`UI_ARG_ORDER`（29）、`EXPECTED_UI_ARG_COUNT`。 |
| `hareskip/skip_pattern.py` | **純粋**。stochastic pattern 生成（z, ゾーン, guard, streak 刈り込み, seed 導出, exact-target）。 |
| `hareskip/probability_models.py` | **純粋**。`p_skip` レジストリ。`sigmoid_band_v0.1` 組み込み。 |
| `hareskip/script.py` | Gradio UI（`ui()`）と生成時フック、infotext 書き込み。 |
| `hareskip/state.py` | 設定スナップショット、`RuntimeState`、`TEA_PRESET_REGISTRY`、`apply_options`。 |
| `hareskip/patcher.py` | `Anima._forward` monkey patch、モードディスパッチ、`_hareskip_should_calc`、`_hareskip_ensure_pattern`、restore。 |
| `hareskip/resrefine.py` | residual 予測/検証/EMA（patcher から抽出）。 |
| `hareskip/forge_introspection.py` | sigma スケジュール捕捉（`sampling_schedule_t_now`）、モデル構造 introspection。 |
| `hareskip/diagnostics.py` | コンソールスナップショット/サマリ。 |
| `hareskip/calibration_capture.py` | TeaCache 係数再較正用 JSONL 捕捉（Tea 専用デバッグ）。 |
| `hareskip/auto_teacache.py` | Auto Tea mode CSV 解析/適用。 |
| `hareskip/callbacks.py` `settings.py` `model_detect.py` `tensor_dump.py` `timing.py` `logging.py` | コールバック配線 / 設定 / モデル判定 / テンソルダンプ / 計時 / ロガー（`[HareSkip]`）。 |
| `tests/test_arg_sync.py` | 3 点同期の静的検証。 |
| `tests/test_probability_models.py` `tests/test_skip_pattern.py` | 純粋モジュールの単体テスト。 |
| `docs/HareSkip-design.md` | 設計正典（stochastic skip density）。 |
| `docs/SPEC-alpha.md` | 要件・設計・仕様の統合資料。 |

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
7. **PNG infotext**: 保存画像メタデータに `Hare skipped_steps` / `Hare skip_count` / `Hare skip_seed` / `Hare guard_count` / `Hare params` 等が入るか（`postprocess_image` 書き込み）。

---

## 4. 予想される修正ポイント

- **スケジュール捕捉失敗時**: §3-1 の通り `_candidate_sigma_sources` に実機で有効な属性パスを追加。復元値は全て有限かつ `(0,1)` 内でなければ `None` 扱いになる点に注意（範囲外なら sigma→t 前提が崩れている）。
- **較正（`p_cap(a)` / `z_enter(a)` 調整）**: `probability_models.py` に**新バージョン `sigmoid_band_v0.2` を登録して切り替える**。**v0.1 は書き換えない**。手順: `register("sigmoid_band_v0.2", NewModel())` → `STATE.hareskip_probability_model` を新名に。`skip_pattern.py` は名前で lookup するため変更不要（`generate_skip_pattern(..., probability_model=...)`）。a=1.0 が skip 数不足なら `p_cap` 上限や `z_enter` の負方向拡張を検討。
- **streak 刈り込みが強すぎる場合**: `apply_max_streak_constraint` は run 内 `(p,z,index)` 最小をフルに戻す。`expected_skips_before_streak`（刈り込み前 p 総和）と実 `skip_count` の乖離が大きすぎるなら `ZONE_MAX_STREAK`（特に final=1 の taper）や safe=3 の緩和を検討。刈り込みは決定論なので同一入力で再現する。

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
