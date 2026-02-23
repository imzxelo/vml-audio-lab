# Rekordbox USB Preparation 実装計画

## 目的

`prepare_usb_track()` を入口に、YouTube楽曲の分析結果を Rekordbox XML と USBフォルダへ自動反映する。

## 設計ドキュメントに対する構造理解

本件は次の3レイヤで構成される。

1. 分析レイヤ  
`load_track / detect_bpm / detect_key / recommend_cues` に `detect_genre` を追加して楽曲メタを完成させる。

2. エクスポートレイヤ  
`copy_to_usb` でジャンル + BPM バケット構成に配置し、重複時はスキップする。

3. Rekordbox互換レイヤ  
`update_rekordbox_xml` で `TotalTime / TrackID / Location / PLAYLISTS ROOT` を満たす XML を生成・追記する。

## 実装フェーズ

1. `genre.py` 新規実装
- YouTubeメタ由来テキスト判定
- Web検索テキスト判定（失敗時フォールバック）
- 音声特徴ベースの判定（モデルなしでも動くヒューリスティック）
- 重み付き投票で最終ジャンルと信頼度を決定

2. `usb_export.py` 新規実装
- ファイル名正規化と `genre_group/BPM` フォルダ振り分け
- 同名重複のスキップ
- Rekordbox XML 追記（トラック・キュー・プレイリスト）

3. `server.py` 連携
- `prepare_usb_track(url, genre_override, usb_path)` を追加
- 既存分析と新規 export 機能をオーケストレーション
- Rekordbox インポート手順をレスポンスに同梱

4. テスト
- `test_genre.py`: ジャンル投票と unknown フォールバック
- `test_usb_export.py`: USBコピー、重複スキップ、XML必須ノード検証
- `test_server_prepare.py`: パイプライン統合（モック）

## 現在の進捗

- [x] フェーズ1: `genre.py` 実装
- [x] フェーズ2: `usb_export.py` 実装
- [x] フェーズ3: `prepare_usb_track` 実装
- [x] フェーズ4: テスト実行と微修正
