# ResRefine 推奨設定テーブル(Load recommended settings ボタン)

> このドキュメントは HareSkip の **ResRefine アコーディオン**にある「Load
> recommended settings」ボタンの正本ドキュメントである。数値の正本は
> `hareskip/recommended_settings.py`(`RECOMMENDED_SETTINGS` とそのdocstring)
> であり、本ドキュメントはこれと完全一致させること。数値が食い違った場合は
> `recommended_settings.py` を正として本ドキュメントを直す。
>
> 機械可読版は [`RECOMMENDED-SETTINGS.json`](RECOMMENDED-SETTINGS.json) を
> 参照(row / calibration / known_limitations / how_to_add_a_row を構造化)。
> `docs/PRESET-COEFFICIENTS.md`/`.json`(TeaCache モードの係数プリセット)の
> ペア方式を踏襲している。

---

## このボタンは何をするか

ResRefine アコーディオンの formula ドロップダウン直上にあるボタン。押すと、
**そのタブ(txt2img/img2img)の本体UIが今表示しているサンプラー・スケジューラ・
ステップ数**を読み、下表を引いて、formula ドロップダウンとスライダー群を
**明示値に書き換える**。

- 完全 WYSIWYG。生成時に何かを裏で自動解決することは一切ない。ボタンは
  クリックした瞬間にだけ値を書き込む、一回きりのUI操作。
- 該当なし(サンプラー・スケジューラの組み合わせが未較正)の場合は
  `Reuse (residual only)`(安全側のデフォルト)を適用し、strength/EMA等の
  スライダーは**変更しない**(Reuse ではそもそも使われない値であり、実験中の
  ユーザー設定を壊さないため)。
- ボタン押下後に本体側(サンプラー・スケジューラ・ステップ数)を変更しても、
  拡張はそれに気づかない。値を再解決したい場合はボタンを**再度**押すこと。
- 詳しい捕獲経路・押下時解決・フォールバック挙動の仕様は
  [`SPEC-alpha.md`](SPEC-alpha.md) の ResRefine 節を参照。

---

## テーブル全行

| # | サンプラー | スケジューラ | formula | prediction_strength | slope_ema_smoothing | use_prediction_after_progress | apply_prediction_from_skip |
|---|---|---|---|---|---|---|---|
| 1 | `Euler` | `Beta` | Linear extrapolation | 0.30 | 0.10 | 0.0 | 2 |
| 2 | `Euler` | `Simple` | Linear extrapolation | 0.30 | 0.00 | 0.0 | 2 |

マッチングはサンプラー名・スケジューラ名の**完全一致・大文字小文字区別**
(例: `Euler a` は `Euler` 行に**マッチしない**)。上表にない組み合わせは
すべて該当なし → Reuse。

### `use_prediction_after_progress` / `apply_prediction_from_skip` を行に含める理由

この2値は「触らない」設定ではなく、全行に明示的に含まれている。理由は、
較正キャンペーンがこの2つの ResRefine 引数を**キャンペーン全体で固定して**
実施したため(STAGE5-SCAN1-HANDOFF §1「その他の ResRefine 引数は既定値
固定」)。このボタンは「推奨設定」を名乗る以上、formula/strength/smoothing
だけでなく較正時の条件そのものを再現する必要がある。

---

## 各行の較正出所

較正キャンペーン: **第5段階応答曲面キャンペーン**(stage-5 response-surface
campaign, 2026-09-05 〜 2026-09-14)。生データは Hugging Face データセット
[`Rootport/HareSkip-calibration`](https://huggingface.co/datasets/Rootport/HareSkip-calibration)
を参照。

- **Euler + Beta** → Linear extrapolation(strength 0.30, EMA smoothing
  0.10)。固定点較正: 15 skips / 30 steps で **+22% アンカー**、
  **p = 0.015**。V1 初見プロンプトで **6/6 改善**を確認。
- **Euler + Simple** → Linear extrapolation(strength 0.30, EMA smoothing
  0.00)。同キャンペーン、平均基準の共通固定点。この固定点の有意性は
  **p = 0.055 で境界線上**であり、Euler+Beta より弱い根拠。それでも
  このペアに対して現状最善のデフォルトだったため採用している。

---

## 行の追加方法

較正データが増え、新しい (sampler, scheduler) 行や将来の帯(例:
スキップ数帯・Shift別条件)を追加する場合は、次の3点セットを**同時に**
更新すること。

1. **`hareskip/recommended_settings.py`**: `RECOMMENDED_SETTINGS` に既存行と
   同じキー構成の dict を追加する。`formula` は文字列リテラルではなく
   `hareskip.state` の `RESREFINE_FORMULA_LINEAR` /
   `RESREFINE_FORMULA_TAYLOR2` / `RESREFINE_FORMULA_REUSE` をインポートして
   使う。モジュール docstring にもキャンペーン名/日付範囲と該当統計量を
   追記する。
2. **本ドキュメントと `RECOMMENDED-SETTINGS.json`**: 同じ行を両方に追記する
   (テーブル・較正出所セクションの両方を md 側で更新)。
3. **`tests/test_recommended_settings.py`**: 新しい (sampler, scheduler) の
   lookup を検証するケースを追加する。

`lookup_recommendation` に `skip_band` のような列は存在しない。各行は
プレーンな dict なので、将来必要になった時点で既存行・新規行にキーを1つ
足すだけで拡張できる(スキーマ移行は不要)。これは意図的な YAGNI 判断であり、
「使うかもしれない」列を先回りして追加しないこと。

---

## 既知の限界

- **N=15・30 steps の較正のみ**: 上記2行はいずれも 15 skips / 30 steps の
  固定点でキャリブレーションされている。ステップ数やスキップ数帯ごとの列は
  まだ存在しない(将来追加されうるが、現時点では意図的に未実装)。
- **ボタン押下後、本体設定を変えても追従しない**: ボタンはクリックされた
  瞬間の本体UI(サンプラー・スケジューラ・ステップ数)を読むだけの一回きりの
  操作。押下後にサンプラーやスケジューラを変更した場合、拡張はそれを検知
  しない。値を再解決するにはボタンを**再度押す**こと。
- **UI並び順(ui_reorder_list)の影響でボタンが無効化される場合がある**:
  Forge Neo の Settings → User interface → Parameter order で `scripts` が
  `sampler` より前に並ぶよう変更されていると、本体のサンプラー/スケジューラ/
  ステップ数コントロールが拡張のUI構築時点でまだ存在せず、捕獲できない。
  この場合ボタンは無効化され、ステータス行に「Settings → User interface →
  Parameter order で 'sampler' が 'scripts' より前にあることを確認して
  ください」という案内が表示される。
