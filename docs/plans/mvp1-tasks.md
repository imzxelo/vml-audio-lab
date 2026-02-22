# MVP1 タスク分解 — 「Mareaを解剖して」が動く

Date: 2026-02-22

## ゴール
`load_track` → 基礎分析 → 構造認識 → 可視化 → Claudeが語る、が一気通貫で動く

## 完了条件
1. キー、BPM、長さが返る
2. セクション区切りが出る
3. エネルギーカーブとスペクトログラムの画像が見える
4. Claudeが「なぜこの曲が気持ちいいか」を語る

---

## タスク一覧（依存順）

### Phase 0: プロジェクト骨格
- [ ] **T01** pyproject.toml 作成（uv init）
  - 依存: なし
  - 内容: プロジェクトメタデータ、依存パッケージ（fastmcp, librosa, matplotlib, essentia, yt-dlp, pytest, ruff）
- [ ] **T02** ディレクトリ構造作成
  - 依存: T01
  - 内容: `src/vml_audio_lab/`, `tools/`, `utils/`, `tests/` の空ファイル + `__init__.py`
- [ ] **T03** FastMCPサーバー骨格（server.py）
  - 依存: T02
  - 内容: `mcp = FastMCP("vml-audio-lab")` + ヘルスチェック用ダミーツール1個 + 起動確認

### Phase 1: 音声ロード
- [ ] **T04** load_track 実装（ローカルファイル）
  - 依存: T03
  - 内容: ファイルパス受け取り → librosa.load → パスとメタデータ返却
  - テスト: サンプル音声ファイルでロード成功
- [ ] **T05** load_track 拡張（YouTube URL）
  - 依存: T04
  - 内容: yt-dlp でダウンロード → 一時ファイル保存 → load_track と同じ流れ
  - テスト: 短い公開動画でダウンロード＋ロード成功

### Phase 2: 基礎分析
- [ ] **T06** detect_bpm 実装
  - 依存: T04
  - 内容: librosa.beat.beat_track → float返却
  - テスト: 既知BPMのサンプルで ±2 BPM 以内
- [ ] **T07** detect_key 実装
  - 依存: T04
  - 内容: essentia KeyExtractor → "Fm" 等の文字列返却
  - テスト: 既知キーのサンプルで一致
- [ ] **T08** energy_curve 実装
  - 依存: T04
  - 内容: librosa RMSエネルギー → matplotlib画像 → Image型返却
  - テスト: 画像がPNG形式で返ること

### Phase 3: 構造認識
- [ ] **T09** detect_structure 実装
  - 依存: T04
  - 内容: librosaのセグメンテーション（self-similarity matrix + spectral clustering）→ セクションリスト
  - テスト: 3分以上の曲で2個以上のセクションが検出される

### Phase 4: 可視化
- [ ] **T10** spectrogram 実装
  - 依存: T04
  - 内容: librosa.display.specshow → Image型
  - テスト: 画像がPNG形式で返ること
- [ ] **T11** waveform_overview 実装
  - 依存: T04, T09
  - 内容: 波形 + セクション境界線のアノテーション付き画像
  - テスト: 画像がPNG形式で返ること

### Phase 5: スキル & 統合
- [ ] **T12** song-anatomy.md スキル作成
  - 依存: T06〜T11
  - 内容: 「この曲を解剖して」で全ツールを順番に呼び出す手順書
  - Claudeに「なぜ気持ちいいか」を語らせるプロンプト含む
- [ ] **T13** E2Eテスト（Mareaで通し）
  - 依存: T12
  - 内容: 実際の曲で load → analyze → structure → visualize → 解説 が通ること
- [ ] **T14** Claude Code MCP設定
  - 依存: T03
  - 内容: `~/.claude.json` にMCPサーバー登録、接続確認

---

## 依存関係図

```
T01 → T02 → T03 → T04 → T05
                    ↓
              ┌─────┼─────┐
              T06   T07   T08
              └─────┼─────┘
                    ↓
                   T09
                    ↓
              ┌─────┼─────┐
              T10         T11
              └─────┼─────┘
                    ↓
              T12 → T13
              T14（T03以降いつでも）
```

## テスト用サンプル音声

- 短いフリー音源（10秒程度）をテストフィクスチャとして `tests/fixtures/` に配置
- E2Eテストは実際の曲（ローカルパス指定）で手動実行
