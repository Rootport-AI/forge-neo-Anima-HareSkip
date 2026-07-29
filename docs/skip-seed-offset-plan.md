# Skip seed offset の Forge 準拠化（2026-07-12 決定、未実装）

> **頭注**: 本書は旧 `HANDOFF-next-session.md`（2026-07-13、現在は `docs/archive/` に凍結）の §4.6 を独立させたもの。**未実装・実装はユーザーのゴーサイン待ち**。この機能の仕様正典である。

## 現状の意味論（前提の明確化）

- Skip seed offset は「シード値」ではなく派生計算のオフセットである: `skip_seed = sha256(f"{image_seed}|hareskip|{offset}").hexdigest()` を int 化し `mod 2**63`（実装: `skip_pattern.derive_skip_seed`、`docs/SPEC-alpha.md` §4.6）。
- `offset=0` は「その画像シードに対する標準パターン」を意味する。ランダムでもシード 0 でもない。
- パターンのランダム性の源は画像シードである。画像シードを -1（ランダム）で回せば `offset=0` のままでもガチャは機能する。offset の用途は「画像シードを固定したままパターンだけ引き直す」場合に限られる。

## 新要件（2026-07-12 実機テスト後のユーザー決定、実装ステータス: 未実装）

1. **`offset=-1` = ランダム**（Forge neo のシード作法に準拠）: 生成開始時に `-1` を検出したら乱数で実オフセットを解決する。infotext の `Hare skip_seed_offset` には**解決後の実値**を必ず記録する（再現性のため必須）。
2. **サイコロボタン 🎲**: クリックで入力欄に `-1` をセットする。
3. **リユースボタン ♻ — Forge 完全準拠版を採用**（簡易版＝セッション内最終値の書き戻し、は不採用）: ギャラリーで選択中の画像の PNG メタデータから `Hare skip_seed_offset` を拾って入力欄に復元する。A1111/Forge の `infotext_fields` 登録機構＋JS を利用する。
4. **`infotext_fields` 登録**（③の実装手段であり副産物が本質的価値）: HareSkip 関連の infotext キー（`Hare aggressiveness`, `Hare skip_seed_offset`, `Hare skip_window`, `Hare zone_boundaries`, `HareSkip mode` 等 — 実装時に対象キーを精査）を UI コンポーネントに対応付け登録することで、**PNG Info タブのパラメータ転送で HareSkip 設定一式を復元可能にする**。ガチャで引いた「当たり」PNG からワンクリックで設定再現、という運用が本命の価値。
5. **実装規模の見積もり**: ①②は小、③④は中（JS/`infotext_fields` 連携）。合わせて 1〜2 コミット規模。
6. **設計上の注意（実装時に検討）**:
   - `-1` 解決の乱数はパターン再現の外側にあること（解決後の実値がすべての再現情報であり、乱数の種そのものを再現情報にしてはならない）。
   - バッチ内での解決タイミング（現行パターンは生成バッチ毎に 1 回生成、offset もその粒度で解決すべき）。
   - `apply_options` の正規化（`hareskip_skip_seed_offset`、`hareskip/state.py`）の値域を `-1` 許容に広げる。
