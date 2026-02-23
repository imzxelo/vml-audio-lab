# Vocal Sampling & Genre Intelligence Design

**Date**: 2026-02-23
**Status**: Approved

## 概要

VML Audio Lab に「ジャンル別分析」「スマートプレイリスト」「ボーカルサンプリング」機能を追加する。
Hip-Hop / J-Pop のボーカルを素材として抽出し、手持ちの House / Deep House ライブラリから
相性の良いトラックを自動で見つける。

## 背景

現状の分析ツールは EDM (House/Techno) に最適化されている。
Hip-Hop を分析すると BPM が倍テンで検出され (72→143)、構造ラベルも EDM 用語 (Drop/Break) になる。
ユーザーは Hip-Hop のバースや J-Pop のボーカルを Deep House に乗せてプレイしたい。

## パイプライン

```
Step 1: ジャンル特定（最初に必ず）
  load_audio → detect_genre (10ジャンル) → ハーフタイム BPM 補正

Step 2: ジャンル別分析
  ジャンルに応じた構造ラベル + BPM / Key / エネルギー / キュー

Step 3: 自動 USB 出力 + プレイリスト生成
  フォルダ整理 + Rekordbox XML (相性プレイリスト + ムードプレイリスト)

Step 4: クリエイティブ提案（オンデマンド）
  サンプリング提案 / トランジション提案 / ボーカル分離
```

## 対応ジャンル (10種)

### DJ系
| ジャンル | 典型BPM | 構造ラベル | 特徴 |
|---------|---------|-----------|------|
| Deep House | 118-124 | Intro/Build/Drop/Break/Outro | 低域厚め、ミニマルなハイハット |
| House | 120-130 | Intro/Build/Drop/Break/Outro | 4つ打ち、明るめスペクトル |
| Techno | 128-140 | Intro/Build/Drop/Break/Outro | ハードキック、高域ノイズ多め |
| Deep Techno | 125-135 | Intro/Build/Drop/Break/Outro | 低域支配、スパース、ミニマル |
| UK Garage | 130-140 | Intro/Build/Drop/Break/Outro | シャッフル/2ステップパターン |
| Melodic | 120-135 | Intro/Build/Drop/Break/Outro | メロディ成分強、起伏大 |

### リスニング系
| ジャンル | 典型BPM | 構造ラベル | 特徴 |
|---------|---------|-----------|------|
| Hip-Hop | 70-110 (ハーフタイム補正) | Verse/Hook/Bridge/Outro | ボーカル主体、ハーフタイム |
| J-Pop | 100-180 | Aメロ/Bメロ/サビ/落ちサビ/大サビ/Outro | 日本語ボーカル、サビ構造 |
| Electronic | 幅広い | Intro/Build/Drop/Break/Outro | 上記に該当しない電子音楽 |
| Classical | 可変 | 楽章/主題/展開/再現 | アコースティック楽器主体 |

### ジャンル検出ロジック

既存の3ソース投票 (YouTube: 35%, Web: 40%, Audio: 25%) を拡張:
- BPM 重複ジャンルはスペクトル特徴（低域/高域バランス、リズムパターン密度）で区別
- BPM > 120 かつ hiphop 判定 → ハーフタイム補正 (BPM / 2)
- J-Pop 判定: 日本語タイトル/アーティスト + Web 検索

## プレイリスト自動生成

### Rekordbox XML プレイリスト構造

```
PLAYLISTS
├── By Compatibility (ミックス用)
│   ├── Camelot 1A Zone
│   ├── Camelot 4A Zone
│   ├── Camelot 8B Zone
│   └── ...
│
├── By Mood (選曲用)
│   ├── Deep Night        ← minor key + 低エネルギー + 低BPM
│   ├── Melodic Journey   ← メロディ成分強 + 起伏大
│   ├── Peak Time         ← 高エネルギー + BPM 128+ + ドロップ強
│   ├── Groovy & Warm     ← 中エネルギー + major key + シャッフル感
│   └── Chill & Mellow    ← 低〜中エネルギー + major key + BPM低め
│
└── Sampling Ideas (サンプリング候補)
    ├── Vocal for House   ← HipHop/JPop ボーカルが House に合う曲
    └── Transition Pairs  ← 相性◎のペア提案
```

### 相性判定 (Camelot Wheel ベース)

- ◎ 同キー (Fm → Fm)
- ◎ Relative major/minor (Fm → Ab)
- ○ Camelot 隣接 (±1, 同番号 A↔B)
- △ それ以外

### ムード判定ロジック

| ムード | 判定材料 |
|--------|---------|
| Deep Night | minor key + 低エネルギー + 低BPM帯 |
| Melodic Journey | メロディ成分強い + ビルド→ドロップの起伏大 |
| Peak Time | 高エネルギー + BPM 128+ + ドロップのパワー大 |
| Groovy & Warm | 中エネルギー + major key + シャッフル感 |
| Chill & Mellow | 低〜中エネルギー + major key + BPM低め |

## サンプリングフロー（オンデマンド）

### `extract_vocals`

```python
def extract_vocals(y_path: str) -> dict:
    """demucs で4ステム分離。呼んだときだけ動く。"""
    # htdemucs モデル使用
    # 出力: vocals, drums, bass, other の各 wav パス
```

4ステム全部返す。ボーカル以外もドラムパターンやベースラインとして使える。

### `analyze_vocal_stem`

```python
def analyze_vocal_stem(
    vocals_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """ボーカル単体のキー・音域・使えるセクションを判定。"""
    # key: ボーカルのキー
    # pitch_range: 音域 (low, high)
    # usable_sections: 使える区間リスト
    #   - type, start, end, has_clear_vocal, suggestion
    # compatible_bpm_range: 合う BPM 帯
    # compatible_genres: 合うジャンル
```

### `find_compatible_tracks`

```python
def find_compatible_tracks(
    vocal_analysis: dict,
    library_index_path: str,
) -> dict:
    """ライブラリからキー相性でマッチング。"""
    # 1. Key 相性でフィルタ (Camelot ±1)
    # 2. BPM 範囲でフィルタ
    # 3. ムード近似度でソート
    # 出力: matches リスト (rank, track, compatibility, suggestion)
```

suggestion の中身がキモ: 「どのセクション同士を合わせるか」まで提案。

### `scan_library`

```python
def scan_library(
    source: str,  # USB パス or Rekordbox XML パス
) -> dict:
    """ライブラリをインデックス化。キャッシュして差分更新。"""
    # XML → パースして BPM/Key/Genre 取得
    # USB → フォルダ走査 + 軽量分析
    # 出力: JSON インデックスファイル
```

## MVPフェーズ

### MVP-A: ジャンル強化 + 自動 USB/プレイリスト

既存コードの拡張。すぐ使える改善。

- [ ] `detect_genre` → 10ジャンル対応
- [ ] BPM ハーフタイム補正 (Hip-Hop / J-Pop)
- [ ] `analyze_structure` → ジャンル別ラベル
  - DJ系: Intro/Build/Drop/Break/Outro
  - Hip-Hop: Verse/Hook/Bridge
  - J-Pop: Aメロ/Bメロ/サビ/落ちサビ
- [ ] USB 出力を通常フローに自動組み込み
- [ ] Rekordbox XML プレイリスト生成
  - Camelot ゾーン別（相性）
  - ムード別（5カテゴリ）

**依存**: なし（既存ライブラリで完結）

### MVP-B: ライブラリインデックス + マッチング

ライブラリを「知ってる」状態にする。

- [ ] `scan_library` — USB/XML からインデックス構築
- [ ] `find_compatible_tracks` — Key 相性マッチング
- [ ] トランジション提案（「A→B、Break→Drop で繋ぐと Fm→Ab で綺麗」）

**依存**: MVP-A（ジャンル/Key/ムード情報が必要）

### MVP-C: ボーカル分離 + サンプリング提案

demucs 導入。一番重い。

- [ ] `extract_vocals` — demucs 4ステム分離
- [ ] `analyze_vocal_stem` — 使えるセクション判定
- [ ] サンプリングプレイリスト自動生成（「Vocal for House」）
- [ ] `find_compatible_tracks` にボーカル→トラック検索追加

**依存**: MVP-B + PyTorch + demucs

## ファイル配置（予定）

```
mcp-server/src/vml_audio_lab/
├── tools/
│   ├── genre.py          ← 改修: 10ジャンル対応
│   ├── structure.py      ← 改修: ジャンル別ラベル
│   ├── analysis.py       ← 改修: ハーフタイム補正
│   ├── playlist.py       ← NEW: プレイリスト生成
│   ├── mood.py           ← NEW: ムード判定
│   ├── camelot.py        ← NEW: Camelot Wheel ユーティリティ
│   ├── separator.py      ← NEW (MVP-C): demucs ボーカル分離
│   ├── vocal_analysis.py ← NEW (MVP-C): ボーカル素材分析
│   └── library.py        ← NEW (MVP-B): ライブラリインデックス
├── server.py             ← 改修: 新ツール登録
└── utils/
    └── teaching.py       ← 改修: ジャンル別解説
```
