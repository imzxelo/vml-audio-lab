# VML Audio Lab — Design Document

Date: 2026-02-22
Author: Makoto + Axel
Status: Approved

---

## What This Is

Claude Codeに「耳」を与えるMCPサーバー + スキル。
音楽を分析し、構造を解剖し、エフェクトを提案し、DJ/音楽制作の先生として機能する。

フォーク元: [hugohow/mcp-music-analysis](https://github.com/hugohow/mcp-music-analysis) (MIT License)

## Why

- 既存のMCPは生データを返すだけ。「だから何？」がない
- DJ/プロデューサー目線の統合ワークフロー（分析→学習→制作→レビュー）は誰も作っていない
- Makotoの音楽理論レベルはゼロからスタート。可視化で「見ればわかる」を狙う
- Claude Codeとの会話の中で完結する。別アプリを開く必要がない

## 背景

- Makotoは2024年2月のロンドンのクラブ体験からDJに目覚めた
- DDJ-FLX4 + Rekordbox 7でDJ活動中
- Vibe Mixing Lab（AI × DJ共創プロジェクト）のMVP1完了済み
- Sunoで生成したサンプル10曲でDJセットを録音・YouTube公開済み
- ジャンル: UKG, Deep House, エレクトロニカ
- リファレンス: Fred Again, Four Tet

## ツールセット（全16ツール、4層構成）

### Layer 1: 基礎分析
| Tool | Purpose | Returns |
|------|---------|---------|
| `load_track` | 音声読み込み（ローカル/YouTube） | audio data path |
| `detect_key` | キー判定（Krumhansl-Schmuckler） | "Fm" etc |
| `detect_bpm` | BPM推定 | float |
| `energy_curve` | RMSエネルギー時系列 | **Image** |
| `frequency_bands` | Low/Mid/High帯域分離+時系列 | **Image** |
| `harmonic_percussive_split` | ハーモニック/パーカッシブ分離 | data paths |

### Layer 2: 構造認識
| Tool | Purpose | Returns |
|------|---------|---------|
| `detect_structure` | セクション自動検出（Intro/Build/Drop/Break/Outro） | section list |
| `detect_transitions` | ミックス録音内のトランジション区間特定 | transition list |
| `chord_progression` | コード進行推定（セクションごと） | chord list |

### Layer 3: エフェクト分析 & 提案
| Tool | Purpose | Returns |
|------|---------|---------|
| `detect_effects` | エフェクト使用の推定（リバーブ/ディレイ/フィルター/コンプ） | effects map |
| `suggest_effects` | 「ここにこのエフェクトを入れるとこうなる」提案 | suggestions |
| `compare_sections` | 2区間のエフェクト差分分析 | diff report |

### Layer 4: 可視化
| Tool | Purpose | Returns |
|------|---------|---------|
| `spectrogram` | スペクトログラム画像 | **Image** |
| `waveform_overview` | 波形+構造アノテーション付き全体図 | **Image** |
| `comparison_chart` | 2曲の特徴比較画像 | **Image** |

## 具体シナリオ: 「Mareaを解剖して」

```
Step 1: load_track → 音声読み込み

Step 2: 基礎分析（並列）
  ├── detect_key      → "Fマイナー"
  ├── detect_bpm      → "124 BPM"
  ├── energy_curve    → [エネルギー推移の画像]
  └── frequency_bands → [Low/Mid/High の画像]

Step 3: 構造認識
  └── detect_structure →
      0:00-0:28  Intro（ボーカルサンプル+薄いパッド）
      0:28-1:12  Build（キック導入、ハイハット密度↑）
      1:12-2:05  Drop（ベースライン全開、エネルギーピーク）
      2:05-2:35  Break（キック抜き、パッドのみ）
      2:35-3:20  Drop 2（パーカッション追加、最大密度）
      3:20-3:48  Outro（フェードアウト）

Step 4: 深掘り
  ├── chord_progression   → "Fm → Db → Ab → Eb"
  ├── harmonic_percussive → "ハーモニック主体、上ネタがリズムを牽引"
  └── detect_effects      →
      - Intro: 深いリバーブ（テール3秒級）
      - Drop直前: リバーブが急激にドライに
      - Drop: サイドチェインコンプ
      - Break: フィルタースイープ（High→Low）

Step 5: AI解説
  「なぜこの曲が気持ちいいか」を平易な日本語で

Step 6: エフェクト提案
  MakotoのVMLサンプルへの応用提案
```

## 技術スタック

| Layer | Tech | Why |
|-------|------|-----|
| MCP Server | Python + FastMCP | フォーク元と同じ。音声分析ライブラリがPython一強 |
| Audio Analysis | librosa + essentia | librosa=基礎、essentia=キー検出・構造分析 |
| Visualization | matplotlib → Image | FastMCPのImage型でClaude Codeに直接画像返却 |
| Effects Estimation | librosa spectral + custom | スペクトル減衰率、ダイナミックレンジ変化から推定 |
| Skills | Claude Code skills (.md) | 分析ツールの使い方と教え方を定義 |
| Notebooks | Jupyter + plotly | インタラクティブ可視化の実験場 |

## プロジェクト構造

```
~/lab/vml-audio-lab/
├── mcp-server/
│   ├── src/vml_audio_lab/
│   │   ├── server.py          # FastMCPエントリポイント
│   │   ├── tools/
│   │   │   ├── loader.py      # load_track, download_youtube
│   │   │   ├── analysis.py    # detect_key, detect_bpm, energy_curve
│   │   │   ├── structure.py   # detect_structure, chord_progression
│   │   │   ├── effects.py     # detect_effects, suggest_effects
│   │   │   └── visualize.py   # spectrogram, waveform, comparison
│   │   └── utils/
│   │       ├── teaching.py    # 分析結果→平易な日本語解説
│   │       └── dj_context.py  # DJ/ミックス特化ヘルパー
│   ├── pyproject.toml
│   └── tests/
├── skills/
│   ├── song-anatomy.md        # 「この曲を解剖して」
│   ├── dj-set-review.md       # 「このミックスをレビューして」
│   ├── sample-scout.md        # 「次に何を作るべき？」
│   ├── music-lesson.md        # 「EQって何？」
│   └── effect-lab.md          # 「ここにエフェクト入れるなら？」
├── notebooks/
│   ├── 001-marea-anatomy.ipynb
│   └── 002-vml-sample-comparison.ipynb
├── docs/plans/
│   └── 2026-02-22-vml-audio-lab-design.md (this file)
└── README.md
```

## フォーク元からの変更方針

- 1ファイル構成 → tools/に機能分割
- CSV返却 → 画像返却を標準に
- コメントアウトされたHPSS → 有効化して拡張
- pytube → yt-dlp（安定性）
- 新規: essentia連携、エフェクト推定、構造分析、教育レイヤー

## MVP段階

### MVP1: 「Mareaを解剖して」が動く（7ツール + 1スキル）
- load_track, detect_key, detect_bpm, energy_curve
- detect_structure, spectrogram, waveform_overview
- song-anatomy.md skill

Success criteria:
1. キー、BPM、長さが返る
2. セクション区切りが出る
3. エネルギーカーブとスペクトログラムの画像が見える
4. Claudeが「なぜこの曲が気持ちいいか」を語る

### MVP2: 解剖の深さ
- frequency_bands, harmonic_percussive_split, chord_progression

### MVP3: エフェクト & DJ特化
- detect_effects, suggest_effects, compare_sections
- detect_transitions, dj-set-review.md

### MVP4: 完全体
- comparison_chart, sample-scout.md, music-lesson.md, effect-lab.md
