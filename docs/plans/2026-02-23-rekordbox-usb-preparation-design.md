# Rekordbox USB 自動準備パイプライン設計

## 概要

YouTube からダウンロードした楽曲を分析し、キューポイント付きの Rekordbox XML を生成して USB ドライブに書き出す。Rekordbox 7.2.6 で1クリックインポートできる状態まで自動化する。

## 全体フロー

```
prepare_usb_track("https://youtube.com/watch?v=xxx")

1. load_track()            — YouTube DL（既存）
2. detect_bpm()            — BPM検出（既存）
3. detect_key()            — キー検出（既存）
4. detect_genre()          — ジャンル判定（新規）
5. recommend_cues()        — キュー提案（既存）
6. copy_to_usb()           — USB へコピー（新規）
7. update_rekordbox_xml()  — XML にトラック+キュー追記（新規）
```

## ジャンル判定 `detect_genre()`

### ファイル: `tools/genre.py`

3つのソースから判定し、多数決+重み付けで最終決定。

```python
detect_genre(title: str, artist: str, y: np.ndarray, sr: int) -> dict
```

### 判定フロー

1. **YouTube メタデータ解析** — タイトル・説明文からジャンルキーワード抽出
2. **Web検索** — "{artist} {title} genre" で Beatport/Discogs 等を検索
3. **essentia 音声分析** — MusicCNN/EffNet ベースのジャンル分類、上位3候補と confidence 取得

### 出力

```python
{
    "genre": "tech-house",
    "confidence": 0.85,
    "sources": {
        "youtube": "tech-house",
        "web": "tech-house",
        "audio": "house"
    },
    "genre_group": "house"     # フォルダ用大分類
}
```

`genre_group` はフォルダ名に使う大分類: house, techno, dnb, trance, hiphop 等。

## USB コピー `copy_to_usb()`

### ファイル: `tools/usb_export.py`

```python
copy_to_usb(
    audio_path: str,
    genre_group: str,
    bpm: float,
    title: str,
    artist: str,
    usb_path: str = "/Volumes/NONAME"
) -> dict
```

### BPM レンジ

5刻みでグループ化: `120-124`, `125-129`, `130-134` ...

### フォルダ構成

```
/Volumes/NONAME/
├── tracks/
│   ├── house/
│   │   ├── 120-124bpm/
│   │   │   └── Artist - Title.wav
│   │   └── 125-129bpm/
│   ├── techno/
│   │   ├── 130-134bpm/
│   │   └── 135-139bpm/
│   └── dnb/
│       └── 170-174bpm/
└── rekordbox_library.xml
```

### ファイル名正規化

- `Artist - Title.wav` 形式に統一
- YouTube タイトルから artist/title を分離（`" - "`, `" / "` で分割）
- ファイルシステム禁止文字を除去

### 重複チェック

同名ファイルが既にあればスキップ（上書きしない）。戻り値で `skipped: true` を返す。

## Rekordbox XML 生成 `update_rekordbox_xml()`

### ファイル: `tools/usb_export.py`（copy_to_usb と同ファイル）

### ライブラリ選定

**`xml.etree.ElementTree` を直接使用**する。pyrekordbox は Hot Cue の色属性（Red/Green/Blue）に未対応のため不採用。

### 実装上の地雷（調査結果）

| 地雷 | 深刻度 | 対策 |
|------|--------|------|
| `TotalTime` 未設定だとキューが全て無視される | CRITICAL | 必ず秒数を設定 |
| 既存トラック上書きバグ (v5.6.1〜7.x) | HIGH | インポートガイドで回避手順を案内 |
| `Location` は `file://localhost/` + URLエンコード必須 | HIGH | `urllib.parse.quote()` で処理 |
| Memory Cue の色がXMLでは保持されない | LOW | Rekordbox の既知制限、回避不可 |
| `PLAYLISTS` の ROOT ノードが必須 | MEDIUM | `<NODE Type="0" Name="ROOT">` を常に生成 |
| `TrackID` の重複 | HIGH | 既存XML読み込み時に最大ID+1で採番 |

### POSITION_MARK フォーマット

| 種類 | Type | Num | 色 |
|------|------|-----|-----|
| Hot Cue A | `"0"` | `"0"` | RGB指定あり |
| Hot Cue B | `"0"` | `"1"` | RGB指定あり |
| Hot Cue C | `"0"` | `"2"` | RGB指定あり |
| Hot Cue D | `"0"` | `"3"` | RGB指定あり |
| Memory Cue | `"0"` | `"-1"` | なし |

### Hot Cue カラーマッピング

Rekordbox の16色パレットから DJ 用途に適した色を選定:

| キュー | 用途 | R | G | B | 色 |
|--------|------|---|---|---|----|
| A | ミックスイン開始 | 40 | 226 | 20 | 緑 |
| B | 1st Drop | 230 | 40 | 40 | 赤 |
| C | Break/遷移 | 48 | 90 | 255 | 青 |
| D | 2nd Drop/終盤 | 224 | 100 | 27 | オレンジ |

### XML サンプル

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="vml-audio-lab" Version="1.0.0" Company="VML"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Title" Artist="Artist"
           Genre="Tech House" TotalTime="228"
           AverageBpm="128.00" Tonality="4A"
           DateAdded="2026-02-23"
           Location="file://localhost/Volumes/NONAME/tracks/house/125-129bpm/Artist%20-%20Title.wav">
      <TEMPO Inizio="0.095" Bpm="128.00" Metro="4/4" Battito="1"/>
      <POSITION_MARK Name="A" Type="0" Start="28.500" Num="0"
                     Red="40" Green="226" Blue="20"/>
      <POSITION_MARK Name="B" Type="0" Start="72.000" Num="1"
                     Red="230" Green="40" Blue="40"/>
      <POSITION_MARK Name="C" Type="0" Start="125.000" Num="2"
                     Red="48" Green="90" Blue="255"/>
      <POSITION_MARK Name="D" Type="0" Start="155.000" Num="3"
                     Red="224" Green="100" Blue="27"/>
      <POSITION_MARK Name="Intro" Type="0" Start="0.095" Num="-1"/>
      <POSITION_MARK Name="Drop1" Type="0" Start="72.000" Num="-1"/>
    </TRACK>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Name="VML Analysis" Type="1" KeyType="0" Entries="1">
        <TRACK Key="1"/>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
```

### 追記ロジック

1. 既存 XML があれば読み込み、なければ新規作成
2. `COLLECTION` にトラック追加、`Entries` を更新
3. `TrackID` は既存最大ID + 1 で採番
4. プレイリスト `VML Analysis` に `TRACK Key` を追加
5. 保存時は UTF-8 エンコーディング + XML 宣言

## MCP ツール `prepare_usb_track`

### ファイル: `server.py` に追加

```python
@mcp.tool
def prepare_usb_track(
    url: str,
    genre_override: str | None = None,
    usb_path: str = "/Volumes/NONAME"
) -> dict
```

### 戻り値

```python
{
    "status": "success",
    "track": {
        "title": "Artist - Title",
        "bpm": 128.0,
        "key": "4A",
        "genre": "tech-house",
        "genre_group": "house",
        "duration_sec": 228
    },
    "genre_detection": {
        "youtube": "tech-house",
        "web": "tech-house",
        "audio": "house",
        "confidence": 0.85,
        "final": "tech-house"
    },
    "cues": {
        "hot_cues": [...],
        "memory_cues": [...]
    },
    "files": {
        "audio": "/Volumes/NONAME/tracks/house/125-129bpm/Artist - Title.wav",
        "xml": "/Volumes/NONAME/rekordbox_library.xml"
    },
    "rekordbox_import_guide": [
        "1. Rekordbox を開く",
        "2. 環境設定 > 表示 > レイアウト > 'rekordbox xml' を有効化",
        "3. 環境設定 > 詳細 > rekordbox xml でXMLパスを指定:",
        "   /Volumes/NONAME/rekordbox_library.xml",
        "4. 左サイドバー 'rekordbox xml' からトラックを全選択(Cmd+A)",
        "5. 右クリック > 'コレクションにインポート'",
        "   ※ 既存トラック更新時は必ず全選択→強制インポート"
    ]
}
```

### エラーハンドリング

- USB未接続 → `"/Volumes/NONAME が見つかりません"` で即座にエラー
- ダウンロード失敗 → yt-dlp のエラーをそのまま返却
- ジャンル判定失敗 → `"unknown"` フォルダに格納、ユーザーに通知

## 新規ファイル構成

```
mcp-server/src/vml_audio_lab/tools/
├── cues.py          # 既存（キュー算出ロジック）
├── genre.py         # 新規（ジャンル判定: YT meta + Web + essentia）
├── usb_export.py    # 新規（USBコピー + XML生成）
└── ...
```

## Rekordbox インポート手順（ユーザー向け）

1. USB "NONAME" を Mac に接続
2. `prepare_usb_track("https://...")` を実行
3. Rekordbox を開く
4. 環境設定 > 表示 > レイアウト > "rekordbox xml" を有効化
5. 環境設定 > 詳細 > rekordbox xml で XML パスを指定: `/Volumes/NONAME/rekordbox_library.xml`
6. 左サイドバー "rekordbox xml" からトラックを全選択 (Cmd+A)
7. 右クリック > "コレクションにインポート"
8. キューポイント付きのトラックが使える状態になる

**注意:** 既存トラックの更新時は、必ず全選択→強制インポートが必要（Rekordbox の既知バグ）。
