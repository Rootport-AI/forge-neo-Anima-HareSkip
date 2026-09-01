# Manual Skip mode 設計仕様書

- 設計確定: 2026-07-10
- 実装ステータス: 実装済み（2026-07-10）
- 設計確定時 HEAD コミット: `03e0acc0d3cd12889c11bd34612558630956d69e`
- 対象: HareSkip 拡張への追加モード「Manual Skip」（`hareskip/manual_skip.py` として実装済み）
- 関連文書: [`docs/SPEC-alpha.md`](SPEC-alpha.md)（現行 α版仕様。UI構造・引数同期・infotext方式の正典）、[`docs/archive/HANDOFF-next-session.md`](archive/HANDOFF-next-session.md) §4.5（本機能が奉仕する再キャリブレーション実験計画）
- **v2（複数行化）設計確定・未実装**: 2026-07-13。§10 参照。実装ステータスは v1（単一行、本書 §1-8 記載）とは独立に管理する——v1 は実装済みで実機検証済み（§9）、v2 は設計のみで未実装。

本書は Manual Skip mode の要件定義・入力/検証/動作仕様・設計をまとめた確定仕様である。設計確定時のキー名・関数名は当時のリポジトリ実コード（`hareskip/*.py`, HEAD=`03e0acc`）に対して確認済み。**本機能は `a67f559`（2026-07-10）で実装済み**（`hareskip/manual_skip.py`, `tests/test_manual_skip.py`）であり、以下の §7 実装ガイド・§8 は設計確定時点の記述を保存した記録である。実装後の正典は実コードおよび `docs/SPEC-alpha.md`（3 点同期・infotext）を参照。

---

## 1. 目的・背景

`docs/archive/HANDOFF-next-session.md` §4.5 に記載の再キャリブレーション実験（`sigmoid_band_v0.1` の taper 側較正が現行データでは正当化できないため、新規データで再較正する計画）は、第1段階「単発スキップ掃引」（30 ステップ中 1 ステップだけ飛ばすパターンを全30位置×45条件で測定し位置ごとの限界損傷曲線 `dQ(z)` を得る）と第2段階「相互作用の抽出」（`{i}`, `{j}`, `{i,j}` の3点セットで隣接ペア=streak効果・帯域間相互作用を測る）から成る。どちらも「どのステップを飛ばすか」を実験者が数値で明示指定する必要があり、既存の HareSkip モード（確率的抽選）や TeaCache モード（accumulator 判定）では実現できない。Manual Skip mode はこの実験の実行手段として要求された。

当初案（Auto Hare mode）では、既存の Auto Tea mode（CSV 1行 = 1条件をキュー展開し、`n_iter` を行数倍に増やして順次適用する仕組み）を模倣し、CSV の行ごとに異なるスキップステップ集合を展開してバッチ生成する方式を検討した。しかし CSV パーサ・行展開・`n_iter` 書き換え・seed テンプレート整合など Auto Tea mode 側の複雑な既存機構をそのまま持ち込む設計は「複雑すぎる」としてユーザーに却下され、1回の生成につきスキップ集合を1つ指定する単純なテキストボックス入力に簡素化された。これが本仕様である。行展開・バッチ化が必要な場合は、生成を複数回叩く運用（または将来の別機能）でカバーする。却下されたのは**記法**（ヘッダ行・列ラベル・範囲記法）であり、行の逐次実行機構そのものは後に v2 として採用された（→ §10）。詳細は §10.3 参照。

---

## 2. 要件

- スキップ対象選択方式の第3のモードとして追加する。`Script.ui()` トップの `hareskip_mode` Radio を **HareSkip / TeaCache / Manual Skip** の3択に拡張する（`constants.HARESKIP_MODES` に `MODE_MANUAL` を追加）。
- 3モードは相互に**排他**である。理由: 「スキップすべきステップの選択方法そのものが変わる機能」であり、判定方式そのものが同一の軸上にあるため。Radio による単一選択で、重複 Enable は UI 構造上そもそも発生しない（HareSkip / TeaCache 2モード時と同じ排他パターンを踏襲する）。
- Manual Skip は**独立モジュール** `hareskip/manual_skip.py` として実装する。既存の `hareskip/skip_pattern.py`（HareSkip 確率密度パターン生成）や `hareskip/auto_teacache.py`（Tea CSV 解析）への増設ではなく、新規ファイルに切り出す。
  - 理由（ユーザーの保守性判断）: Manual Skip はロジックが単純で将来作り直す可能性が低い。一方 HareSkip 側（`skip_pattern.py` / `probability_models.py`）は再較正・新確率モデル追加など将来大改修の可能性があるため、コードベースを隔離しておく。
  - 副産物: Manual Skip のパース・検証・スキップ判定は sigma スケジュール捕捉（`forge_introspection.sampling_schedule_t_now`）・確率モデル（`probability_models.py`）・軌道座標 z（`skip_pattern.logsnr_proxy_from_t_now`）のいずれにも依存しない。したがって実機でスケジュール捕捉が失敗する状況（`docs/SPEC-alpha.md` §5-2 の未検証項目）でも Manual Skip は独立して動作し、「HareSkip モードが動かないのはスケジュール捕捉の問題か、それとも Anima パッチ全体の問題か」を切り分ける診断ツールを兼ねる。
- 実験装置としての基本思想: 曖昧な入力に対して「勝手にフォールバックして動く」のではなく「エラーを吐いて生成を止める」。実験でスキップ位置を取り違えたまま生成してしまう事故は、通常のクリエイティブ用途の「フル演算に劣化して安全側に倒れる」（`docs/SPEC-alpha.md` NF-2）とは要求が逆であり、Manual Skip 固有の方針として明記する。

---

## 3. 入力仕様

- Manual Skip グループ内に1行のテキストボックス（`gr.Textbox`）を置く。カンマ区切りの1始まり（1-indexed）ステップ番号列を入力する。例: `10, 12,`
- 空白・末尾カンマ・空トークンを許容する。`10, , 12,` は `10, 12` と等価に解釈する。
- 重複するステップ番号は黙って除去する（エラーにしない）。
- 入力は「飛ばすステップの列挙」であり、位置指定記法（範囲 `a-b` やスライス記法等）ではない。1つずつ数値で書く。
- 記載されなかったステップは自動的にフル演算される。ユーザーが「飛ばさないステップ」のために空欄やプレースホルダを置く必要は無い。
- 入力欄が空文字列の場合は「何も飛ばさない」（全ステップフル演算）として有効に扱う。これはキャリブレーション実験における LPIPS 比較の基準画像（baseline）生成に使う正当なユースケースである。
- スキップ可能なステップ数の上限はコードに焼き込まない。範囲検証は生成時の実際の `p.steps`（StableDiffusionProcessing のステップ数）に対して動的に行う。これにより 20〜40 steps 等、実験計画で使われる steps 数の変化に自動対応する。「30 steps」はキャリブレーション実験計画（`docs/archive/HANDOFF-next-session.md` §4.5 第1段階）側の話であり、本機能の仕様として steps 数を固定するものではない。

---

## 4. 検証仕様（エラーで生成中止）

検証は `Script.before_process` フック（Auto Tea CSV のキュー投入処理が既に走っている、キュー投入後・生成開始前のタイミング）で行い、違反時は `RuntimeError` を送出して生成を中止する。これは Auto Tea mode の CSV パースエラー（`hareskip/script.py` `before_process`: `AutoTeaCsvError` を捕捉し `raise RuntimeError(f"Auto Tea CSV error: {exc}") from exc`）と同一の実績あるパターンであり、Forge UI 上にエラーが表示される。

検証項目:

1. **非数値トークン** — カンマ区切りで分割した各トークン（空白除去・空トークン除去後）が整数として解釈できない場合はエラー。
2. **範囲外** — `step > p.steps` または `step < 1` の場合はエラー。
   - 副次効果: 生成ステップ数をマウス操作ミスで意図せず下げたまま Generate してしまった事故を検出するフールプルーフとしても機能する（指定ステップが現在の `p.steps` を超えていれば必ず弾かれる）。
3. **step 1 の指定はエラー** — `step == 1` を含む入力はエラーとする。ステップ1は residual が未保持のため物理的にスキップ不可能（`patcher._shared_force_full_reason` の `first_call` 理由と同じ技術的制約）。黙って強制フル演算に上書きするのではなく、入力段階で弾いてユーザーに知らせる。

---

## 5. 動作仕様

- 指定されたステップをそのままスキップする（WYSIWYG）。HareSkip モードの skip window・ゾーン境界・streak 上限・確率抽選・ガード機構は Manual Skip には一切適用しない。
- 唯一の例外は既存の**共有強制フル**（`patcher._shared_force_full_reason` が返す `first_call` / `missing_residual`）である。これは HareSkip モード・TeaCache モードと同じく技術的必然（residual が存在する前にスキップしない保証）であり、Manual Skip モードでもディスパッチの先頭で評価される。per-step ログの reason 欄（`_hareskip_log_call` の `reason` 引数、`hareskip_call=` ログ行）で可視化される。
  - 検証仕様 §4-3 で step 1 の指定自体をエラーにしているため、`first_call` による強制フルとユーザー指定のズレは実用上ほぼ発生しない。
- ResRefine（`docs/SPEC-alpha.md` §3.1 の共通セクション、モード外）は Manual Skip モードでも設定通り適用される。スキップ位置の純粋効果を測る実験では `resrefine_formula = Reuse (residual only)` に固定することを**実験手順側の規律**とする。コード側では ResRefine の formula を Manual Skip モードだからといって縛らない — ResRefine 自体の効果を測定する実験が将来ありうるため、モジュール的な独立性を保つ。

---

## 6. メタデータ仕様

Manual Skip モード専用の infotext キーは最小限の2つのみとする（`postprocess_image` フックからの書き込み方式は既存の `Hare skipped_steps` 等と同一）。

| キー | 内容 |
| --- | --- |
| `HareSkip mode` | `"Manual Skip"`（モード共通のアイデンティティキー、既存の `HareSkip mode` を流用） |
| `Manual skipped_steps` | 実現値。1始まりのステップ番号をスペース区切りで列挙。既存の `Hare skipped_steps`（`hareskip/script.py` `_apply_hare_pattern_infotext`: `" ".join(str(step) for step in pattern.skipped_steps)`）と同じ書き込み方式。 |

実験名・threshold・ユーザー指定値そのものの別記録は付与しない。指定値と実現値のズレは、原理的に step 1 の指定がエラーで排除されている（§4-3）ため、`first_call` 強制フル以外では発生しない。したがって実現値（`Manual skipped_steps`）のみで実験記録として十分と判断する。

実装上の実現値の出所（realized-vs-specified の裁定）: `Manual skipped_steps` はユーザー指定リストではなく **実現値** を報告する。具体的には patcher のスキップ分岐が per-step に積む実スキップ記録（`STATE.hareskip_skipped_steps`、0始まりで格納）を1始まりへ変換・昇順化して書き込む。これは HareSkip モードで pattern 由来の値を書く箇所と同じ postprocess_image フックだが、Manual Skip には pattern オブジェクトが存在しないため、実スキップ記録を直接参照する方式を採る。共有強制フル（`first_call`/`missing_residual`）が指定スキップを上書きした稀なケースでも、実際に飛ばしたステップのみが記録されるため真正である（§4-3 で step 1 を弾いているため実務上このズレはほぼ発生しない）。

---

## 7. 設計（実装ガイド）

以下は実装時の指針であり、実装フェーズでの裁量（内部変数名等）を認める。

- **新規モジュール `hareskip/manual_skip.py`**（Forge/gradio/torch 非依存、stdlib のみ、単体テスト可能）:
  - `parse_manual_steps(text: str) -> list[int]` — 純粋パーサ。カンマ分割、空白/空トークン除去、整数変換、重複除去（順序は実装裁量。テスト可能性を優先し決定論的な順序とする）。
  - `validate_manual_steps(steps: list[int], num_steps: int) -> None`（またはエラー内容を持つ例外を送出する関数）— §4 の検証項目（非数値は `parse_manual_steps` 側で先に弾かれる想定、範囲外、step 1）をチェックし、違反時はエラー種別を持つ例外を送出する。
- **`hareskip/constants.py`**:
  - `MODE_MANUAL = "Manual Skip"` を追加。
  - `HARESKIP_MODES` を `[MODE_HARESKIP, MODE_TEACACHE, MODE_MANUAL]` の3要素にする。
  - `UI_ARG_ORDER` に `manual_skip_steps` を追加し、末尾に append する（既存の位置は不変というリポジトリの既存規律を踏襲）。現行 33 引数 → **34 引数**になる。`EXPECTED_UI_ARG_COUNT = len(UI_ARG_ORDER)` は自動追従する。（2026-09-01 に末尾5引数追加で39になった。詳細は `docs/SPEC-alpha.md` §3.4）
- **`hareskip/state.py`**:
  - `RuntimeState` に `manual_skip_steps: str` フィールドを追加（UI からの生テキストをそのまま保持）。
  - `RuntimeState.apply_options` の位置引数シグネチャに `manual_skip_steps: str = ""` を末尾追加（`UI_ARG_ORDER` の並びと1:1で一致させる）。
  - パース済みリスト（`list[int]`）は before_process での検証成功後に STATE 側の別フィールドへ格納する（フィールド名は実装時裁量。例 `manual_skip_steps_parsed` 等）。
- **`hareskip/script.py`**:
  - `hareskip_mode` の `gr.Radio` の `choices` に `HARESKIP_MODES`（3要素化済み）をそのまま渡す（`constants.HARESKIP_MODES` を import 済みのため自動反映）。
  - Manual Skip 用の `gr.Group`（テキストボックス1個）を新設し、`_hareskip_mode_group_updates`（現状 HareSkip/TeaCache の2グループ visibility 切替）を3グループ対応に拡張して表示切替に組み込む。
  - `before_process` に Manual Skip の検証を追加する。`p.steps` に対して `validate_manual_steps` を呼び、`ManualSkipError`（またはそれに類する例外）を捕捉して `raise RuntimeError(...)`（Auto Tea CSV エラーと同一パターン、`STATE.set_error(...)` も踏襲）。
  - `postprocess_image`（`_apply_hare_pattern_infotext` 相当の位置）で Manual Skip モード時に `Manual skipped_steps` を書き込む処理を追加する。
- **`hareskip/patcher.py`**:
  - `_hareskip_forward_body` のモードディスパッチ（現状 `if hareskip_mode: ... else: # MODE_TEACACHE`）に Manual 分岐を1つ追加する。
  - `should_calc = not (step_index + 1 in manual_set)`（1始まり変換に注意。`step_index` は0始まり）とし、**共有強制フル（`_shared_force_full_reason`）を先行評価**する点は HareSkip モードと同じ構造を踏襲する。
  - HareSkip 専用の `_hareskip_ensure_pattern` / `_hareskip_should_calc`（確率密度パターン経路）は Manual Skip モードからは呼ばない。
- **tests**:
  - `hareskip/manual_skip.py` のパーサ・検証関数に対する単体テストを追加する。
    - 正常系: 空欄、空トークン混在（`10, , 12,`）、重複除去。
    - 異常系: 非数値トークン、範囲外（`step > num_steps` および `step < 1`）、`step == 1` の指定。
  - `tests/test_arg_sync.py` の引数数アサーションが 34 に自動追従することを確認する（既存の静的検証機構がそのまま機能する）。

引数34への3点同期（`Script.ui()` の return list / `RuntimeState.apply_options` の位置引数 / `constants.UI_ARG_ORDER`）は `docs/SPEC-alpha.md` §3.4 に記載の既存の3点同期規律と `tests/test_arg_sync.py` の既存機構でそのまま担保される。

---

## 8. 未決事項

なし（設計確定）。実装はユーザーのゴーサイン後、1コミット規模で行う想定。

---

## 9. 実機検証記録（2026-07-13）

- 検証内容: Skip mode を Manual Skip に切替え、`15, 17, 19, 21, 23`（1始まり）を指定して生成。旧 UjiCache 拡張（TeaCache 系のスキップ実行経路そのもの）が同じステップを 0 始まりで（`14, 16, 18, 20, 22`）スキップした過去の生成画像と比較した。
- 結果: **ビット同一**（Nz DoppelPix Judge, port 7870 による測定: LPIPS = 0.000000, SSIM = 1.000000, PSNR = inf）。所要時間も整合（73.1秒 vs 72.5秒、スキップされた各ステップは約0.004秒）。
- 意義: Manual Skip のスキップ実行機構（residual Reuse 経路）は、UjiCache（TeaCache モード）のスキップ実行機構と完全に等価であることが実測で証明された。パース・検証（§3, §4）は Manual Skip 固有だが、「スキップすると決めた後の処理」は両モードで共有されている経路であり、ここに差異が無いことが確認された。
- 検証マシン上の旧 UjiCache 拡張は本検証後に無効化された。
- 関連: `hareskip/manual_skip.py` のロジックそのものは 2026-07-13 時点で以下2件のバグ修正を経ている（本検証はこれらの修正後に実施）。
  - `6dee42e`: `reset_generation()` が毎サンプリングパスで `manual_skip_parsed` を無条件クリアしていたため、`before_process` で検証・格納したリストが patcher に読まれる前に消え、Manual Skip が事実上まったくスキップしない状態になっていた（空リスト → 常にフル演算）。`reset_generation()` からのクリアを削除し、`before_process` 側のクリア＋再検証＋再格納のみに一本化して修正。
  - `5c20835`: ログ上のステップ表記を 1 始まりに統一（`hareskip_call=`/`hareskip_step=`/`hareskip_summary=` の `step=`）。Manual Skip の通常フル演算ステップの reason が誤って Tea 専用語彙の `threshold` になっていた点も `reason=manual_full` に修正。詳細は `CHANGELOG.md` Unreleased 参照。

---

## 10. 複数行化（v2、2026-07-13 設計確定・実装済み）

**実装ステータス: 実装済み（2026-07-13）。実機検証: 完了（2026-07-13〜14）。** 本節は 2026-07-13 にユーザー承認された確定設計であり、実装済み。パーサ層は `hareskip/manual_skip.py` の `parse_manual_lines` / `validate_manual_lines`、本体フックは `hareskip/script.py`（`_prepare_manual_skip_run` / `_apply_manual_skip_seed_template` / `_apply_manual_skip_row_if_needed` / `_finish_manual_skip_run`）、テストは `tests/test_manual_skip.py`。

**実機検証記録（2026-07-13〜14、出典 `experiment-HareSkip/EXPERIMENT-LOG.md`）**: 複数行機構（行展開・同一シード共有・per-pass 行差し替え・infotext 実現値）は実機（RTX 3080 Laptop、Anima baseV10、ER SDE / Beta / 30 steps / CFG 4 / 1536x1536）で検証完了した。3行スモークテスト（`10,11,12` / `14,15,16` / `18,19,20`）でコンソールログ上の行展開・全行同一シード（`manual_skip_seed_template`）・per-pass 行差し替え（`manual_skip_row_start` の index 1→2→3 遷移）を確認（2026-07-13）。翌日（2026-07-14）に2枚目画像の infotext `Manual skipped_steps: 14 15 16` を確認し、複数行機構の検証を完了。同日実施した不正行入力テスト（`Line N:` エラーによる fail-stop の検証）では、拡張自体は設計どおり `RuntimeError` を送出したが、Forge 本体側が例外を握り潰し生成を続行するという既知の制約が判明した（下記「既知の制約」参照）。

さらに、2026-07-22〜29 の再キャリブレーション実験（第2段階・第3段階）では、Manual Skip 経路（Forge API 経由・1パターン=1リクエスト）が第2段階 1,260枚＋第3段階 450枚の計 1,700枚超の生成に使用され、いずれも完走した（固定コミット `b6164214` 製、infotext 全数検査で `Manual skipped_steps` の実現値がファイル名のスキップ集合と完全整合、出典 `experiment-HareSkip/EXPERIMENT-LOG.md` 2026-07-27・07-29エントリ）。

### 10.1 背景・動機

`docs/archive/HANDOFF-next-session.md` §4.5 の第1段階「単発スキップ掃引」実験（30 ステップ中1ステップだけ飛ばすパターンを全30位置にわたって測定）は、現行の Manual Skip（1回の生成につきスキップ集合1つ）では Generate ボタンを30回押す必要がある。複数行化により、この掃引を **1回の Generate クリック**で実行できるようにする。

### 10.2 要件

- Manual skip steps テキストボックスを**複数行**（multiline）にする。
- **空でない各行が1つのジョブ**を表す。1行の文法は現行パーサ（§3）のそれと**完全に同一**（カンマ区切り、1始まり、空白/末尾カンマ許容）。行を跨いだジョブは独立したスキップ集合。
- **ヘッダ行は無い**。列ラベルは無い。**ハイフン/範囲記法（`a-b` 等）は導入しない** — パースは現行の1行文法をそのまま行ごとに適用するのみ。
- **空行は無視する**（ユーザー決定・選択肢(a)採用）。末尾の改行・区切り用の空行はジョブを生成しない。ベースライン（無スキップ）生成が必要な場合は、拡張を無効化するか、単一行に空文字列を置いた入力を別途使う——空行を挟んでベースラインを混ぜる方式は採らない。
- **検証**: 全行を `before_process` で生成開始前に検証する（既存の fail-stop 方針を踏襲）。いずれかの行が不正なら、**その行番号を名指しした**エラーでジョブ全体を中止する（他の行が正しくても部分実行はしない）。
- **ジョブ展開**: Auto Tea mode の展開機構（CSV 1行 = 1条件を `p.n_iter` 倍化して順次適用する仕組み）を再利用する。`p.n_iter` を行数倍にする。
- **seed**: 全行が**同一の seed テンプレート**を共有する（固定仕様、オプションではない — キャリブレーション実験は同一シードでのパターン間比較を要求するため）。
- **メタデータ**: 変更なし。`Manual skipped_steps`（実現値）が各画像のパターンを自己記述するため、行インデックスをキーとして別途記録する必要は無い。
- **単一行入力は現行と完全互換**の挙動を保つ（後方互換）。行分割を各行のパースより**先に**行うことで、複数行テキストが1つのカンマ区切りリストへ意図せず結合してしまう既知の危険（trailing-comma silent-merge hazard）も同時に解消される。

### 10.3 命名・経緯の注記

設計初期に検討され却下された「Auto Hare mode」案（ヘッダ行＋ラベル付き CSV 列＋範囲記法）は、CSV パーサ・行展開・`n_iter` 書き換え・seed テンプレート整合など Auto Tea mode 側の複雑な既存機構をそのまま持ち込む点が「複雑すぎる」として却下された（§1 参照）。今回却下されたのは**その記法（ヘッダ・ラベル付き列・範囲記法）**であり、「行を順次実行してバッチ展開する」こと自体ではない。本 v2 設計は、プレーンな行ごとの現行文法を保ったまま、Auto Tea mode の展開機構（行数倍の `n_iter` 増加・順次適用）のみを再利用することで両立させる。

### 10.x 実装機構の設計（2026-07-13 確定）

本節は §10.2 の要件を実装可能な粒度に落とし込んだ確定設計である。行番号・関数名は本設計確定時点の実コード（HEAD=`796c15f`、コード最新コミットは `5c20835`）を実地に読んで引用している。

#### 10.x.1 行リストの保持先

全行を `before_process` でパースし、**全行を検証してから**（1行目でエラーが出ても即中断せず、全行を検証し尽くしてから）ジョブを開始する。不正な行があれば、その1-based行番号を名指ししたエラーで生成全体を中止する（§10.2 の要件通り）。

保持先は Auto Tea mode の p-attribute パターンを踏襲する。Auto Tea は `_prepare_auto_teacache_run`（`hareskip/script.py:576-611`）で以下の p-attribute 群に状態を保持している（precedent、`_AUTO_HARESKIP_P_ATTRS`, `hareskip/script.py:565-573`）:

- `p._hareskip_auto_rows`（`hareskip/script.py:598`）— 行データのリスト
- `p._hareskip_auto_original_n_iter`（`hareskip/script.py:599`）— 展開前の `n_iter`
- （`_hareskip_auto_original_batch_size` / `_hareskip_auto_seed_template_size` / `_hareskip_auto_seed_template_ready` / `_hareskip_auto_logged_row_index` / `_hareskip_auto_iteration_counter` も同グループ）

Manual Skip v2 はこれをミラーし、同じ命名規約で以下を新設する:

- `p._hareskip_manual_rows` — `parse_manual_steps` を行ごとに適用した `list[list[int]]`（各要素が1行分の1-based skip リスト）。
- `p._hareskip_manual_original_n_iter` — 展開前の `p.n_iter`（Auto Tea の `_hareskip_auto_original_n_iter` と同じ役割）。

`before_process` で `p.n_iter` を行数倍にする（Auto Tea が `total_n_iter = len(rows) * original_n_iter` を `p.n_iter` に代入する箇所、`hareskip/script.py:596,601` と同一パターン）。

#### 10.x.2 iteration→行の適用フック

Auto Tea は生成開始直前の `_begin_generation`（`process_before_every_sampling` に対応、`hareskip/script.py:819-828`）で以下の順序を踏む:

```
_apply_ui_args(script_args)              # script.py:820
_apply_auto_teacache_seed_template(p)    # script.py:821
_apply_auto_teacache_row_if_needed(p)    # script.py:822
if not STATE.active(): ...
start_sampling(source)                   # script.py:827 — STATE.reset_generation() を呼ぶ
```

つまり **`_apply_auto_teacache_row_if_needed` は `start_sampling`/`reset_generation` より前**に呼ばれている。`_apply_auto_teacache_row_if_needed`（`hareskip/script.py:674-704`）の機構: `iteration = _current_generation_iteration(p)`（v2 でリネーム）を求め、`row_index = iteration // original_n_iter` でその回に適用すべき行を選び、`apply_auto_teacache_row_to_state(row)` で `STATE` へ反映する。

Manual Skip v2 はこの構造をミラーするが、**適用箇所を `start_sampling`/`reset_generation` の「後」に置く**（Auto Tea とは順序が逆）。理由: `reset_generation`（`hareskip/state.py:505-556`）は `manual_skip_parsed` を意図的にクリアしない（§8.5 相当の NOTE コメント、`hareskip/state.py:533-539`「NOTE: manual_skip_parsed is deliberately NOT reset here...」）。この非クリアは「`before_process` で検証・格納した値がパッチャに読まれる前に消えるバグ」（`6dee42e`、§9 参照）の再発防止だが、v2 では**行ごとに異なる値をパスごとに再代入する必要がある**ため、`reset_generation` の非クリア方針とは別の理由で「`reset_generation` の後」に代入フックを置く必要がある:

```
_apply_ui_args(script_args)
_apply_auto_teacache_seed_template(p)
_apply_auto_teacache_row_if_needed(p)
_apply_manual_skip_row_if_needed(p)      # 新設。start_sampling の後に置く
if not STATE.active(): ...
start_sampling(source)                   # reset_generation() 呼び出しを含む
_apply_manual_skip_row_if_needed(p)      # ← ここ（reset_generation の後）
...
```

新設フック `_apply_manual_skip_row_if_needed(p)` の機構（`_apply_auto_teacache_row_if_needed` 同型）:

```
rows = getattr(p, "_hareskip_manual_rows", None)
if not rows or STATE.hareskip_mode != MODE_MANUAL or not STATE.hareskip_enabled:
    return
original_n_iter = getattr(p, "_hareskip_manual_original_n_iter", 1)
iteration = _current_generation_iteration(p)  # Auto Tea と共通の iteration ヘルパを再利用
row_index = max(0, min(len(rows) - 1, iteration // original_n_iter))
STATE.manual_skip_parsed = rows[row_index]
```

**実装確定（2026-07-13）**: iteration カウンタは Auto Tea 専用の `_current_auto_teacache_iteration` を共通名 `_current_generation_iteration` にリネームして再利用した（フォールバック属性も `_hareskip_iteration_counter` に一般化）。専用カウンタは不要である — 通常経路は `p.iteration` の副作用なし読み取りのみで、Auto Tea / Manual の両モードは `hareskip_mode` により排他ゲートされるため同一 run で両フックが同時に走ることはなく、フォールバック増分の競合も起きない。実装フックのガードには（Auto Tea 同型 676 行と整合させ）`not STATE.hareskip_enabled` も追加した（上記擬似コードに反映済み）。また `p` は Forge が使い回すため、`postprocess` で `_finish_manual_skip_run(p)` を呼び Manual の p-attribute 群（`_MANUAL_HARESKIP_P_ATTRS`）を後始末する — `seed_template_ready` 等が残留すると次回 run で二重適用事故になるため必須。

この設計が `reset_generation` の非クリア方針と両立する理由を明記する: **`reset_generation` はパス単位（サンプリングパスごと）に呼ばれるが、`manual_skip_parsed` をクリアしない設計は「クリアしてから誰も再設定しない」事故を防ぐためのものであり、「同じフック内で reset の直後に再代入する」こと自体は禁止していない**。v2 のフックは `reset_generation` 呼び出しと同じ `_begin_generation` 内で、その直後に `STATE.manual_skip_parsed` を明示的に上書きするため、非クリア方針の意図（値を持ち主なくクリアしない）に反しない。また単一行入力（v1 互換）でも、1行だけの `rows` リストに対して `row_index` は常に `0` になるため、毎パス同じ値が再代入されるだけで v1 と挙動は完全に一致する。

#### 10.x.3 シードテンプレート

全行が同一の seed テンプレートを共有する（固定仕様、§10.2）。Auto Tea の `_apply_auto_teacache_seed_template`（`hareskip/script.py:632-671`）をミラーする専用ヘルパ `_apply_manual_skip_seed_template(p)`（または既存関数を Auto Tea/Manual 共通に一般化したヘルパ）を新設する。機構: `template_size = original_n_iter * batch_size` を求め、`p.all_seeds`（および存在すれば `p.all_subseeds`）を `_seed_template(...)` で構築したテンプレートを `row_count` 回リピートして書き戻す（`hareskip/script.py:651-664` と同型）。ユーザー設定不可の固定挙動とする。

#### 10.x.4 Auto Teaとの併存ルール

コード確認結果: **現状の Auto Tea 展開経路は `hareskip_mode` でゲートされていない**。

- `_prepare_auto_teacache_run`（`hareskip/script.py:586`）のガードは `if not (STATE.enabled and STATE.auto_teacache_enabled): return` — `auto_teacache_enabled` は独立したチェックボックス（`Enable Auto Tea mode`, `hareskip/script.py:256-260`）であり、`hareskip_mode` を見ていない。
- `_apply_auto_teacache_row_if_needed`（`hareskip/script.py:676`）のガードも `if not rows or not STATE.auto_teacache_active or not STATE.hareskip_enabled: return` — 同様に `hareskip_mode` を見ていない。

つまり理論上、ユーザーが `hareskip_mode = Manual Skip` を選びつつ `Enable Auto Tea mode` チェックボックスも ON にすると、UI 上は Manual Skip のグループが表示されていても Auto Tea の `n_iter` 展開が同時に走り得る（3モードは Radio で排他だが、`Enable Auto Tea mode` はモードとは別軸のチェックボックスであるため）。これは v1 時点でも既に存在する状態だが、v2 で Manual Skip 側も `p.n_iter` を書き換えるようになるため、両者が同時に `n_iter` を展開しようとする衝突が新たに現実的なリスクになる。

**ルール（v2 実装スコープに追加する）**:

1. Manual multiline 展開は **`hareskip_mode == MODE_MANUAL` のときのみ**実行する（`_prepare_manual_skip_run` は既に `STATE.hareskip_mode == MODE_MANUAL` をチェックしている、`hareskip/script.py:623`。行展開の新設コードもこのガードを継承する）。
2. Auto Tea 展開は **`hareskip_mode == MODE_TEACACHE` のときのみ**実行するよう、`_prepare_auto_teacache_run` のガードに `STATE.hareskip_mode == MODE_TEACACHE` を追加する。これは現状に対する**新規のモードゲート追加**であり、v2 実装スコープに含める（fail-stop 方針に沿った防御的変更であり、Auto Tea 単体の既存動作は `hareskip_mode` が既定値 `MODE_TEACACHE` である限り変わらない）。
3. 上記1・2を実装してもなお、両者が同時に `n_iter` を書き換えようとする状態を検出した場合（防御的チェック）は、警告ログを出したうえでエラーとして生成を中止する（fail-stop 方針、§2「勝手にフォールバックして動くのではなくエラーで止める」を踏襲）。

#### 10.x.5 UI変更の具体点

現行の `manual_skip_steps` テキストボックスは `gr.Textbox(label="Manual skip steps", lines=1, max_lines=1, placeholder="e.g. 10, 12", elem_id="manual-skip-steps")`（`hareskip/script.py:269-274`）。

v2 での変更点:

- `max_lines=1` を削除する（複数行入力を可能にする）。
- `lines=6` 程度に変更する（スクロール可能な複数行表示。Auto Tea CSV の `gr.Textbox(..., lines=6, ...)`, `hareskip/script.py:261-265` と同じ値を踏襲）。
- `placeholder` を2〜3行のサンプル例に更新する（例: `"e.g.\n10, 12\n15, 17, 19\n"`）。

コンポーネント自体は変わらず**1つの `gr.Textbox` のまま**である。したがって:

- **引数数は34のまま変わらない**。
- **`UI_ARG_ORDER` も変わらない**（`manual_skip_steps` という単一の位置引数のまま）。
- 複数行テキストは Gradio から単一の文字列として `\n` を含んだまま渡ってくる。行への分割は UI 層・引数同期層では一切行わず、**パーサ層（`hareskip/manual_skip.py` の新規関数、§10.x.7）でのみ行う**。

#### 10.x.6 メタデータの依存関係の明示

§6 の「メタデータ仕様」および §10.2 が述べる「メタデータ: 変更なし」は、**§10.x.2（iteration→行の適用フック、per-pass row 割当て）が実装されていることに依存する**。理由:

- `Manual skipped_steps` の実現値は `STATE.hareskip_skipped_steps` を1始まりへ変換して書き込む方式である（§6 参照）。
- `STATE.hareskip_skipped_steps` は `reset_generation`（`hareskip/state.py:505-556`、`clear()` 呼び出しは `hareskip/state.py:517`）で**サンプリングパスごとにクリアされる**。
- したがって、§10.x.2 のフックが `STATE.manual_skip_parsed` をそのパスの行に正しく差し替えていて初めて、各画像が「自分の行」のスキップ結果を記録する。§10.x.2 が実装されていない状態（例: 全パスが同じ最終行を見てしまう実装ミス）では、`Manual skipped_steps` は正しくない値を記録する。

§10.2 の「メタデータ: 変更なし」という記述は、書き込みコード自体（postprocess_image のロジック）は変更不要という意味であり、**正しい値が書かれるかどうかは §10.x.2 の実装に依存する**、という点を本節で明示的に補足する。

#### 10.x.7 パーサ層

`hareskip/manual_skip.py` に純関数を新設する:

```python
def parse_manual_lines(text: str) -> list[tuple[int, list[int]]]:
    """Split text into non-blank lines, parse each with parse_manual_steps.

    Returns (physical_line_no, steps) tuples; physical_line_no is 1-based.
    """
```

仕様:

- `text` をまず改行 `\n` で分割する（§10.2「行分割を各行のパースより先に行う」、trailing-comma silent-merge hazard 対策）。
- 空行・空白のみの行は無視する（リストに含めない、§10.2 の要件）。ただし物理行番号のカウントには含める（空行を挟んでもエラーメッセージの行番号がテキストボックスの見た目と一致する）。
- 各行に既存の `parse_manual_steps`（`hareskip/manual_skip.py:46-74`）をそのまま適用する。1行の文法は変更しない。
- 戻り値は `(物理行番号(1-based), steps)` のタプルのリスト。物理行番号を steps と一緒に保持するのは、後段の `validate_manual_lines` がエラーメッセージで不正行を名指しするため。
- 検証は新設の `validate_manual_lines(parsed, num_steps)` が既存の `validate_manual_steps`（`hareskip/manual_skip.py:77-115`）を行ごとに呼び出す形にする。パース・検証いずれのエラーメッセージにも1-basedの**物理行番号**を含める（例: `f"Line {physical_line_no}: {inner_message}"`）。
- Forge/gradio/torch に依存しない stdlib のみの実装とし、`tests/test_manual_skip.py` から単体テスト可能にする（既存の `parse_manual_steps`/`validate_manual_steps` と同じテスト容易性を踏襲）。
