# Manual Skip mode 設計仕様書

- 設計確定: 2026-07-10
- 実装ステータス: 実装済み（2026-07-10）
- 設計確定時 HEAD コミット: `03e0acc0d3cd12889c11bd34612558630956d69e`
- 対象: HareSkip 拡張への追加モード「Manual Skip」（`hareskip/manual_skip.py` 新設予定）
- 関連文書: [`docs/SPEC-alpha.md`](SPEC-alpha.md)（現行 α版仕様。UI構造・引数同期・infotext方式の正典）、[`docs/HANDOFF-next-session.md`](HANDOFF-next-session.md) §4.5（本機能が奉仕する再キャリブレーション実験計画）

本書は Manual Skip mode の要件定義・入力/検証/動作仕様・設計をまとめた確定仕様である。すべてのキー名・関数名は本書執筆時点のリポジトリ実コード（`hareskip/*.py`, HEAD=`03e0acc`）に対して確認済み。未決事項は無く、実装はユーザーのゴーサイン後に行う。

---

## 1. 目的・背景

`docs/HANDOFF-next-session.md` §4.5 に記載の再キャリブレーション実験（`sigmoid_band_v0.1` の taper 側較正が現行データでは正当化できないため、新規データで再較正する計画）は、第1層「単発スキップ掃引」（30 ステップ中 1 ステップだけ飛ばすパターンを全30位置×45条件で測定し位置ごとの限界損傷曲線 `dQ(z)` を得る）と第2層「相互作用の抽出」（`{i}`, `{j}`, `{i,j}` の3点セットで隣接ペア=streak効果・帯域間相互作用を測る）から成る。どちらも「どのステップを飛ばすか」を実験者が数値で明示指定する必要があり、既存の HareSkip モード（確率的抽選）や TeaCache モード（accumulator 判定）では実現できない。Manual Skip mode はこの実験の実行手段として要求された。

当初案（Auto Hare mode）では、既存の Auto Tea mode（CSV 1行 = 1条件をキュー展開し、`n_iter` を行数倍に増やして順次適用する仕組み）を模倣し、CSV の行ごとに異なるスキップステップ集合を展開してバッチ生成する方式を検討した。しかし CSV パーサ・行展開・`n_iter` 書き換え・seed テンプレート整合など Auto Tea mode 側の複雑な既存機構をそのまま持ち込む設計は「複雑すぎる」としてユーザーに却下され、1回の生成につきスキップ集合を1つ指定する単純なテキストボックス入力に簡素化された。これが本仕様である。行展開・バッチ化が必要な場合は、生成を複数回叩く運用（または将来の別機能）でカバーする。

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
- スキップ可能なステップ数の上限はコードに焼き込まない。範囲検証は生成時の実際の `p.steps`（StableDiffusionProcessing のステップ数）に対して動的に行う。これにより 20〜40 steps 等、実験計画で使われる steps 数の変化に自動対応する。「30 steps」はキャリブレーション実験計画（`docs/HANDOFF-next-session.md` §4.5 第1層）側の話であり、本機能の仕様として steps 数を固定するものではない。

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
  - `UI_ARG_ORDER` に `manual_skip_steps` を追加し、末尾に append する（既存の位置は不変というリポジトリの既存規律を踏襲）。現行 33 引数 → **34 引数**になる。`EXPECTED_UI_ARG_COUNT = len(UI_ARG_ORDER)` は自動追従する。
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
