# Project: VML Audio Lab

## 概要
Claude Codeに「耳」を与えるMCPサーバー + スキル。音楽を分析し、DJ/音楽制作の先生として機能する。

フォーク元: [hugohow/mcp-music-analysis](https://github.com/hugohow/mcp-music-analysis) (MIT License)

## 技術スタック
- 言語: Python 3.11+
- MCP: FastMCP
- 音声分析: librosa, essentia
- 可視化: matplotlib → FastMCP Image型
- YouTube DL: yt-dlp
- テスト: pytest
- パッケージ管理: uv (pyproject.toml)

## 開発コマンド
- MCPサーバー起動: `cd mcp-server && uv run python -m vml_audio_lab.server`
- テスト: `cd mcp-server && uv run pytest`
- リント: `cd mcp-server && uv run ruff check .`
- フォーマット: `cd mcp-server && uv run ruff format .`
- 依存追加: `cd mcp-server && uv add <package>`

## コーディング規約
- インデント: スペース4（Python標準）
- 命名: snake_case（変数・関数）、PascalCase（クラス）
- docstring: Google style、日本語OK
- 型ヒント: 必須（全関数に引数・戻り値の型を書く）
- コメント: 日本語OK、変数名・関数名は英語

## ディレクトリ構造
```
vml-audio-lab/
├── mcp-server/
│   ├── src/vml_audio_lab/
│   │   ├── server.py          # FastMCPエントリポイント
│   │   ├── tools/
│   │   │   ├── loader.py      # load_track
│   │   │   ├── analysis.py    # detect_key, detect_bpm, energy_curve
│   │   │   ├── structure.py   # detect_structure, chord_progression
│   │   │   ├── effects.py     # detect_effects, suggest_effects
│   │   │   └── visualize.py   # spectrogram, waveform, comparison
│   │   └── utils/
│   │       ├── teaching.py    # 分析結果→平易な日本語解説
│   │       └── dj_context.py  # DJ/ミックス特化ヘルパー
│   ├── pyproject.toml
│   └── tests/
├── skills/                    # Claude Code skills
├── notebooks/                 # Jupyter実験用
├── docs/plans/                # 設計ドキュメント
└── CLAUDE.md
```

## Git ワークフロー
- ブランチ: `feature/*`, `fix/*`, `chore/*`
- コミット: Conventional Commits 形式（日本語OK）
  - `feat:` 新機能
  - `fix:` バグ修正
  - `docs:` ドキュメント
  - `refactor:` リファクタ
  - `test:` テスト
  - `chore:` 雑務
- PR前チェック: ruff + pytest 必須

## 重要な決定
- フォーク元の1ファイル構成を tools/ に機能分割
- CSV返却ではなく画像返却（matplotlib → Image型）を標準に
- pytube → yt-dlp（安定性のため）
- MVP段階的アプローチ（MVP1→4）

## MVP段階
- **MVP1**: load_track, detect_key, detect_bpm, energy_curve, detect_structure, spectrogram, waveform_overview + song-anatomy skill
- **MVP2**: frequency_bands, harmonic_percussive_split, chord_progression
- **MVP3**: detect_effects, suggest_effects, compare_sections, detect_transitions + dj-set-review skill
- **MVP4**: comparison_chart + 残りスキル全部

## 注意点
- librosaは重い。ロード時間を考慮してキャッシュを活用
- essentia のインストールはプラットフォーム依存あり（macOS: pip で入る）
- matplotlib の画像生成はメモリ使用に注意（plt.close() 必須）
- yt-dlpは外部ツール依存。`brew install yt-dlp` が別途必要
