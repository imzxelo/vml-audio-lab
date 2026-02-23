"""VML Audio Lab — FastMCP サーバーエントリポイント"""

from pathlib import Path

from fastmcp import FastMCP

from vml_audio_lab import __version__
from vml_audio_lab.tools.analysis import detect_bpm, detect_key, energy_curve
from vml_audio_lab.tools.camelot import (
    compatibility_score,
    compatible_camelot_codes,
    key_to_camelot,
)
from vml_audio_lab.tools.cues import recommend_cues
from vml_audio_lab.tools.genre import canonicalize_genre_slug, detect_genre, genre_group_for
from vml_audio_lab.tools.library import find_compatible_tracks_by_params, scan_library
from vml_audio_lab.tools.loader import DEFAULT_SR, load_track, load_y
from vml_audio_lab.tools.mood import detect_mood
from vml_audio_lab.tools.playlist import generate_playlists, get_compatible_playlists
from vml_audio_lab.tools.structure import detect_structure
from vml_audio_lab.tools.transition import suggest_transition
from vml_audio_lab.tools.usb_export import copy_to_usb, split_artist_title, update_rekordbox_xml
from vml_audio_lab.tools.visualize import spectrogram, waveform_overview

mcp = FastMCP(
    name="vml-audio-lab",
    version=__version__,
    instructions=(
        "音楽分析MCPサーバー。楽曲のBPM・キー・構造・エネルギーを分析し、"
        "スペクトログラムや波形の画像を返す。DJ/音楽制作の先生として機能する。"
        "10ジャンル対応・Camelot Wheel・ムード判定・スマートプレイリスト生成をサポート。"
    ),
)


@mcp.tool
def ping() -> str:
    """ヘルスチェック。サーバーが動作中なら 'pong' を返す。"""
    return "pong"


@mcp.tool
def load_audio(
    file_path: str,
    offset: float = 0.0,
    duration: float | None = None,
) -> dict:
    """音声を読み込み、メタデータとキャッシュパスを返す。

    ローカルファイル（wav, mp3, flac 等）または YouTube URL を指定する。
    以降の分析ツールはここで返る y_path を使う。

    Args:
        file_path: 音声ファイルのパス、または YouTube URL
        offset: 読み込み開始位置（秒）
        duration: 読み込む長さ（秒）。省略で全体
    """
    return load_track(file_path, offset=offset, duration=duration)


@mcp.tool
def analyze_bpm(y_path: str) -> dict:
    """楽曲の BPM を推定する。

    Args:
        y_path: load_audio で返された y_path
    """
    return detect_bpm(y_path)


@mcp.tool
def analyze_key(y_path: str) -> dict:
    """楽曲のキー（調）を推定する。

    essentia KeyExtractor 使用。"Fm", "C" 等のラベルを返す。
    Camelot コードも同時に返す。

    Args:
        y_path: load_audio で返された y_path
    """
    result = detect_key(y_path)
    camelot = key_to_camelot(result["key_label"])
    result["camelot"] = camelot
    return result


@mcp.tool
def analyze_energy(y_path: str) -> bytes:
    """RMS エネルギーカーブの画像を生成する。

    時間軸に沿ったエネルギー推移を可視化。

    Args:
        y_path: load_audio で返された y_path
    """
    return energy_curve(y_path)


@mcp.tool
def analyze_structure(
    y_path: str,
    n_segments: int | None = None,
    genre: str | None = None,
    genre_group: str = "default",
) -> dict:
    """楽曲のセクション構造を自動検出する。

    MFCC + クラスタリングでセクション境界を推定し、ジャンルに応じたラベルを付与する。

    Args:
        y_path: load_audio で返された y_path
        n_segments: セクション数（省略で自動推定）
        genre: ジャンルスラグ (例: "hiphop", "jpop", "house")。
            analyze_genre で得たジャンル名をそのまま渡せる。genre_group より優先。
        genre_group: ジャンルグループ (genre 未指定時に使用)
            - "default": Intro/Build/Drop/Break/Outro (DJ系)
            - "hiphop": Verse/Hook/Bridge/Outro
            - "jpop": Aメロ/Bメロ/サビ/落ちサビ/大サビ/Outro
            - "classical": 主題/展開/再現/コーダ
    """
    return detect_structure(y_path, n_segments=n_segments, genre=genre, genre_group=genre_group)


@mcp.tool
def analyze_genre(
    y_path: str,
    title: str = "",
    artist: str = "",
) -> dict:
    """楽曲ジャンルを推定する（10ジャンル対応）。

    複数ソース（テキスト・Web・音声特徴）を組み合わせて判定。
    Hip-Hop / J-Pop は倍テン BPM を自動補正する。

    対応ジャンル:
        DJ系: deep-house, house, tech-house, melodic, uk-garage, techno, deep-techno
        リスニング系: hiphop, jpop, electronic, classical

    Args:
        y_path: load_audio で返された y_path
        title: 楽曲タイトル（省略可）
        artist: アーティスト名（省略可）
    """

    y = load_y(y_path)
    return detect_genre(title=title, artist=artist, y=y, sr=DEFAULT_SR)


@mcp.tool
def analyze_mood(
    y_path: str,
    bpm: float | None = None,
    scale: str | None = None,
) -> dict:
    """楽曲のムードカテゴリを推定する。

    キー・BPM・エネルギーからムードを判定する。

    ムードカテゴリ:
        - Peak Time: 高エネルギー・BPM高め
        - Deep Night: マイナーキー・低エネルギー・深夜向け
        - Melodic Journey: メロディ成分強・起伏大
        - Groovy & Warm: 中エネルギー・メジャーキー
        - Chill & Mellow: 低〜中エネルギー・BPM低め

    Args:
        y_path: load_audio で返された y_path
        bpm: BPM（省略時は自動推定）
        scale: キースケール "major" または "minor"（省略時は自動推定）
    """

    y = load_y(y_path)

    if bpm is None:
        bpm_result = detect_bpm(y_path)
        bpm = bpm_result["bpm"]

    if scale is None:
        key_result = detect_key(y_path)
        scale = key_result["scale"]

    return detect_mood(y=y, sr=DEFAULT_SR, bpm=bpm, scale=scale)


@mcp.tool
def visualize_spectrogram(y_path: str) -> bytes:
    """メルスペクトログラムの画像を生成する。

    周波数成分の時間変化を可視化。

    Args:
        y_path: load_audio で返された y_path
    """
    return spectrogram(y_path)


@mcp.tool
def visualize_waveform(
    y_path: str,
    sections: list[dict] | None = None,
) -> bytes:
    """波形オーバービュー画像を生成する。

    セクション情報を渡すと境界線とラベルをアノテーションする。

    Args:
        y_path: load_audio で返された y_path
        sections: analyze_structure で返された sections リスト（省略可）
    """
    return waveform_overview(y_path, sections=sections)


@mcp.tool
def suggest_rekordbox_cues(
    y_path: str,
    n_segments: int | None = None,
) -> dict:
    """DJ用のRekordboxキュー案（A/B/C/D + Memory Cue）を生成する。

    Args:
        y_path: load_audio で返された y_path
        n_segments: 構造推定のセグメント数（省略可）

    Returns:
        dict:
            - hot_cues: A/B/C/D
            - memory_cues: Intro/Build/Drop/Break/Outro など
            - sections: 推定構造
            - notes: 運用メモ
    """
    return recommend_cues(y_path, n_segments=n_segments)


@mcp.tool
def camelot_compatibility(
    key_label_a: str,
    key_label_b: str,
) -> dict:
    """2つのキーの Camelot 相性を判定する。

    Args:
        key_label_a: キーラベル A (例: "Fm", "C")
        key_label_b: キーラベル B (例: "Am", "G")

    Returns:
        dict:
            - camelot_a: Camelot コード A
            - camelot_b: Camelot コード B
            - score: 相性スコア (0.0〜1.0)
            - compatible_codes: A と相性の良い Camelot コード一覧
    """
    camelot_a = key_to_camelot(key_label_a)
    camelot_b = key_to_camelot(key_label_b)
    score = 0.0
    if camelot_a and camelot_b:
        score = compatibility_score(camelot_a, camelot_b)

    compatible = compatible_camelot_codes(camelot_a) if camelot_a else []

    return {
        "camelot_a": camelot_a,
        "camelot_b": camelot_b,
        "score": score,
        "compatible_codes": compatible,
    }


def _build_rekordbox_import_guide(xml_path: str) -> list[str]:
    return [
        "1. Rekordbox を開く",
        "2. 環境設定 > 表示 > レイアウト > 'rekordbox xml' を有効化",
        "3. 環境設定 > 詳細 > rekordbox xml でXMLパスを指定:",
        f"   {xml_path}",
        "4. 左サイドバー 'rekordbox xml' からトラックを全選択(Cmd+A)",
        "5. 右クリック > 'コレクションにインポート'",
        "   ※ 既存トラック更新時は必ず全選択→強制インポート",
    ]


@mcp.tool
def prepare_usb_track(
    url: str,
    genre_override: str | None = None,
    usb_path: str = "/Volumes/NONAME",
) -> dict:
    """YouTube楽曲を分析し、USB + Rekordbox XML まで一気通貫で準備する。

    ジャンル判定 → BPM/Key/キュー分析 → USB コピー → Rekordbox XML 更新
    → Camelot ゾーン別プレイリスト + ムード別プレイリストへ自動登録。

    Args:
        url: YouTube URL またはローカルファイルパス
        genre_override: ジャンルを手動指定する場合のジャンル名
        usb_path: USB ドライブのマウントパス
    """

    usb_root = Path(usb_path).expanduser()
    if not usb_root.exists():
        raise FileNotFoundError(f"{usb_path} が見つかりません")

    loaded = load_track(url)
    y_path = loaded["y_path"]
    y = load_y(y_path)
    sr = int(loaded["sr"])

    raw_title = str(loaded.get("title") or Path(str(loaded["file_path"])).stem)
    uploader = str(loaded.get("uploader") or "Unknown Artist")
    artist, title = split_artist_title(raw_title, fallback_artist=uploader)

    bpm_result = detect_bpm(y_path)
    key_result = detect_key(y_path)
    cues_result = recommend_cues(y_path)

    override_value = (genre_override or "").strip()
    if override_value:
        final_genre = canonicalize_genre_slug(override_value)
        genre_result = {
            "genre": final_genre,
            "confidence": 1.0,
            "sources": {"youtube": "override", "web": "override", "audio": "override"},
            "genre_group": genre_group_for(final_genre),
            "halftime_corrected": False,
        }
    else:
        genre_result = detect_genre(title=title, artist=artist, y=y, sr=sr, bpm=bpm_result["bpm"])

    # ハーフタイム補正 BPM を使用
    bpm = genre_result.get("corrected_bpm", bpm_result["bpm"])

    # ムード判定
    mood_result = detect_mood(
        y=y,
        sr=DEFAULT_SR,
        bpm=float(bpm),
        scale=str(key_result["scale"]),
    )

    copied = copy_to_usb(
        audio_path=str(loaded["file_path"]),
        genre_group=str(genre_result["genre_group"]),
        bpm=float(bpm),
        title=title,
        artist=artist,
        usb_path=usb_path,
    )
    audio_export_path = copied["audio_path"]

    xml_file = str((usb_root / "rekordbox_library.xml").resolve())
    xml_result = update_rekordbox_xml(
        xml_path=xml_file,
        audio_path=audio_export_path,
        title=title,
        artist=artist,
        genre=str(genre_result["genre"]),
        bpm=float(bpm),
        key_label=str(key_result["key_label"]),
        duration_sec=float(cues_result["duration_sec"]),
        hot_cues=list(cues_result["hot_cues"]),
        memory_cues=list(cues_result["memory_cues"]),
    )

    # Camelot ゾーン + ムードプレイリストへ自動追加
    track_id_str = str(xml_result["track_id"])
    playlist_result = generate_playlists(
        xml_path=xml_file,
        track_id=track_id_str,
        key_label=str(key_result["key_label"]),
        mood=str(mood_result["mood"]),
    )

    # Camelot コード
    camelot = key_to_camelot(str(key_result["key_label"]))

    return {
        "status": "success",
        "track": {
            "title": f"{artist} - {title}",
            "bpm": bpm,
            "key": key_result["key_label"],
            "camelot": camelot,
            "genre": genre_result["genre"],
            "genre_group": genre_result["genre_group"],
            "mood": mood_result["mood"],
            "duration_sec": cues_result["duration_sec"],
            "halftime_corrected": genre_result.get("halftime_corrected", False),
        },
        "genre_detection": {
            "youtube": genre_result["sources"]["youtube"],
            "web": genre_result["sources"]["web"],
            "audio": genre_result["sources"]["audio"],
            "confidence": genre_result["confidence"],
            "final": genre_result["genre"],
        },
        "cues": {
            "hot_cues": cues_result["hot_cues"],
            "memory_cues": cues_result["memory_cues"],
        },
        "playlists": {
            "camelot": playlist_result.get("camelot_playlist"),
            "mood": playlist_result.get("mood_playlist"),
            "compatible_playlists": get_compatible_playlists(xml_file, str(key_result["key_label"])),
        },
        "files": {
            "audio": audio_export_path,
            "xml": xml_result["xml_path"],
            "copy_skipped": copied["skipped"],
            "xml_added": xml_result["xml_added"],
        },
        "rekordbox_import_guide": _build_rekordbox_import_guide(xml_result["xml_path"]),
    }


@mcp.tool
def index_library(
    source: str,
    cache_path: str | None = None,
) -> dict:
    """USB またはRekordbox XMLからライブラリインデックスを構築する。

    source が .xml ファイルの場合は Rekordbox XML をパースする。
    source がディレクトリの場合はフォルダを走査して軽量分析する。
    フィンガープリントが変わらない場合はキャッシュを返す。

    Args:
        source: USB パス (例: "/Volumes/NONAME") または Rekordbox XML パス
        cache_path: キャッシュ保存先（省略時: /tmp/vml_audio_lab/library_index.json）

    Returns:
        dict:
            - tracks: トラック辞書のリスト (bpm, key_label, camelot_code, genre, mood など)
            - total: トラック数
            - source: スキャン元パス
            - cached: キャッシュから読み込んだか
    """
    return scan_library(source=source, cache_path=cache_path)


@mcp.tool
def match_compatible_tracks(
    key_label: str,
    bpm: float,
    mood: str | None = None,
    genre: str | None = None,
    library_source: str | None = None,
    cache_path: str | None = None,
    max_results: int = 10,
) -> dict:
    """ライブラリからキー・BPM・ムード相性でトラックをマッチングする。

    事前に index_library を実行してキャッシュを作っておくと速い。

    Args:
        key_label: クエリトラックのキーラベル (例: "Fm", "Am")
        bpm: クエリトラックの BPM
        mood: クエリトラックのムード名（省略可）
        genre: クエリトラックのジャンル（省略可、参考情報）
        library_source: ライブラリのソースパス（キャッシュがない場合にスキャン）
        cache_path: キャッシュファイルパス（省略時はデフォルト位置）
        max_results: 返す最大件数（デフォルト: 10）

    Returns:
        dict:
            - matches: マッチ結果リスト (rank, track, bpm, key, camelot, genre, mood,
                        compatibility ◎/○/△, score, suggestion)
            - query: クエリ情報
            - total_scanned: スキャンしたトラック数
            - compatible_count: 相性スコア > 0 のトラック数
    """
    from vml_audio_lab.tools.library import load_index

    library_index = load_index(cache_path)

    return find_compatible_tracks_by_params(
        query_key_label=key_label,
        query_bpm=bpm,
        query_mood=mood,
        query_genre=genre,
        library_index=library_index,
        library_source=library_source if library_index is None else None,
        max_results=max_results,
    )


@mcp.tool
def suggest_track_transition(
    track_a: dict,
    track_b: dict,
) -> dict:
    """2トラック間のベストなトランジション（繋ぎ方）を提案する。

    キー関係・BPM・エネルギー・構造セクションから最適な出点・入点と
    ミキシングテクニックを提案する。

    Args:
        track_a: トラック A の分析結果。以下のキーを含む辞書:
            - key_label (str): キーラベル (例: "Fm")
            - camelot (str, optional): Camelot コード (例: "4A")
            - bpm (float): BPM
            - genre_group (str, optional): ジャンルグループ
            - sections (list[dict], optional): analyze_structure の sections
            - energy_level (float, optional): エネルギーレベル 0〜1
        track_b: トラック B の分析結果（同上）

    Returns:
        dict:
            - compatibility: 相性ラベル (◎/○/△)
            - key_relationship: キー関係の説明
            - bpm_diff: BPM 差
            - suggested_transition: 出点・入点・テクニック・energy_match
            - explanation: 日本語の提案説明文
    """
    return suggest_transition(track_a=track_a, track_b=track_b)


@mcp.tool
def extract_vocals(file_path: str) -> dict:
    """demucsでボーカルを分離する。4ステム(vocals/drums/bass/other)を出力。

    `separate_stems` の薄いラッパー。stems を展開したフラットな dict を返す。
    analyze_vocal_stem にそのまま渡せるボーカルパスが含まれる。

    Args:
        file_path: 音声ファイルのローカルパス (wav/mp3/flac 等)

    Returns:
        dict:
            - vocals: ボーカルステムのWAVパス
            - drums: ドラムステムのWAVパス
            - bass: ベースステムのWAVパス
            - other: その他ステムのWAVパス
            - model: 使用したモデル名
            - cached: キャッシュから返した場合 True
    """
    from vml_audio_lab.tools.separator import extract_vocals as _extract_vocals

    return _extract_vocals(file_path)


@mcp.tool
def separate_stems(
    file_path: str,
    model_name: str = "htdemucs",
) -> dict:
    """音声を4ステム（vocals/drums/bass/other）に分離する。

    demucs htdemucs モデルを使用。初回は数分かかる場合がある。
    ステムはキャッシュに保存され、2回目以降は即時返却。

    Args:
        file_path: 音声ファイルのローカルパス (wav/mp3/flac 等)
        model_name: demucs モデル名（デフォルト: "htdemucs"）

    Returns:
        dict:
            - stems: 各ステムの WAVパス辞書 (vocals, drums, bass, other)
            - model: 使用したモデル名
            - sample_rate: 出力サンプリングレート
            - duration_sec: 楽曲長（秒）
            - processing_time_sec: 処理時間（秒）
            - source_file: 元ファイルのパス
            - cached: キャッシュから返した場合 True
    """
    from vml_audio_lab.tools.separator import separate_stems as _separate_stems

    return _separate_stems(file_path, model_name=model_name)


@mcp.tool
def analyze_vocal(
    vocals_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """ボーカルステムのキー・音域・使えるセクションを判定する。

    separate_stems で得たボーカルステムのパスを渡す。

    Args:
        vocals_path: separate_stems で返されたボーカルステムのWAVパス
        sections: analyze_structure で返されたセクションリスト（省略可）。
            渡すと各セクションのボーカル有無を判定する。

    Returns:
        dict:
            - key: ボーカルのキーラベル (例: "Fm")
            - key_label: key の別名
            - camelot_code: Camelot コード (例: "4A")
            - camelot: camelot_code の別名
            - scale: "major" または "minor"
            - key_strength: キー確信度 (0.0〜1.0)
            - pitch_range: 音域情報 (low, high, low_hz, high_hz)
            - usable_sections: 使えるセクションリスト (vocal_clarity, suggestion 付き)
            - compatible_bpm_range: 合う BPM 帯 [min, max]
            - compatible_genres: 合うジャンルリスト
            - compatible_camelot_codes: 相性の良い Camelot コードリスト
            - sampling_score: 総合サンプリングスコア (0.0〜1.0)
    """
    from vml_audio_lab.tools.vocal_analysis import analyze_vocal as _analyze_vocal

    return _analyze_vocal(vocals_path, sections=sections)


@mcp.tool
def analyze_vocal_stem(
    vocals_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """抽出済みボーカルのキー・音域・使えるセクションを分析する。

    `extract_vocals` や `separate_stems` で得たボーカルステムのパスを渡す。
    `analyze_vocal` と同じ実装。別名として提供。

    Args:
        vocals_path: ボーカルステムのWAVパス (extract_vocals の vocals キー)
        sections: analyze_structure で返されたセクションリスト（省略可）

    Returns:
        dict:
            - key: ボーカルのキーラベル (例: "Fm")
            - key_label: key の別名
            - camelot_code: Camelot コード (例: "4A")
            - camelot: camelot_code の別名
            - scale: "major" または "minor"
            - key_strength: キー確信度 (0.0〜1.0)
            - pitch_range: 音域情報 (low, high Hz / low_note, high_note)
            - usable_sections: 使えるセクションリスト (vocal_clarity, suggestion 付き)
            - compatible_bpm_range: 合う BPM 帯 [min, max]
            - compatible_genres: 合うジャンルリスト
            - compatible_camelot_codes: 相性の良い Camelot コードリスト
            - sampling_score: 総合サンプリングスコア (0.0〜1.0)
    """
    from vml_audio_lab.tools.vocal_analysis import analyze_vocal as _analyze_vocal

    return _analyze_vocal(vocals_path, sections=sections)


@mcp.tool
def find_sampling_candidates(
    file_path: str,
    library_source: str | None = None,
) -> dict:
    """ボーカルを分析し、ライブラリから相性の良いトラックを提案する。

    一気通貫ワークフロー: ボーカル分離 → キー/音域分析 → ライブラリマッチング。
    「このボーカル、どのトラックに合う？」をワンショットで答える。

    ライブラリが未インデックスの場合は library_source でスキャン元を指定する。
    事前に index_library を実行しておくと高速。

    Args:
        file_path: 分析対象のボーカルステム WAVパス
            (extract_vocals や separate_stems で得た vocals パス)
        library_source: ライブラリのスキャン元パス（USB または Rekordbox XML）。
            キャッシュがある場合は省略可。

    Returns:
        dict:
            - vocal_analysis: ボーカル分析結果 (key, pitch_range, usable_sections 等)
            - matches: 相性トラックのリスト (rank, track_id, compatibility, suggestion)
            - total_searched: ライブラリ検索対象トラック数
            - vocal_key: ボーカルのキーラベル
            - vocal_camelot: ボーカルの Camelot コード
            - sampling_score: ボーカルの総合サンプリング適性スコア (0.0〜1.0)
    """
    from vml_audio_lab.tools.library import load_index
    from vml_audio_lab.tools.library import scan_library as _scan_library
    from vml_audio_lab.tools.vocal_analysis import analyze_vocal as _analyze_vocal
    from vml_audio_lab.tools.vocal_analysis import find_compatible_tracks_for_vocal

    # ボーカル分析
    vocal_result = _analyze_vocal(file_path)

    # ライブラリインデックスを取得（キャッシュ優先、なければスキャン）
    library_index = load_index(None)
    if library_index is None and library_source:
        scan_result = _scan_library(source=library_source)
        library_index = {t["file_path"]: t for t in scan_result.get("tracks", [])}

    # インデックスがなければ空で返す
    if library_index is None:
        return {
            "vocal_analysis": vocal_result,
            "matches": [],
            "total_searched": 0,
            "vocal_key": vocal_result.get("key"),
            "vocal_camelot": vocal_result.get("camelot_code"),
            "sampling_score": vocal_result.get("sampling_score", 0.0),
            "note": (
                "ライブラリインデックスが見つかりません。"
                "library_source を指定するか、事前に index_library を実行してください。"
            ),
        }

    # インデックスを一時ファイルに保存して find_compatible_tracks_for_vocal に渡す
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        # library_index が dict の場合はリストに変換
        if isinstance(library_index, dict):
            tracks_list = list(library_index.values())
        else:
            tracks_list = library_index
        json.dump({"tracks": tracks_list}, tmp, ensure_ascii=False)
        tmp_path = tmp.name

    match_result = find_compatible_tracks_for_vocal(vocal_result, tmp_path)

    return {
        "vocal_analysis": vocal_result,
        "matches": match_result.get("matches", []),
        "total_searched": match_result.get("total_searched", 0),
        "vocal_key": vocal_result.get("key"),
        "vocal_camelot": vocal_result.get("camelot_code"),
        "sampling_score": vocal_result.get("sampling_score", 0.0),
    }


@mcp.tool
def find_tracks_for_vocal(
    vocal_analysis: dict,
    library_index_path: str,
) -> dict:
    """ボーカル分析結果からライブラリ内の相性トラックを検索する。

    Camelot キー相性 + BPM 範囲でフィルタし、マッチ度でソートして返す。

    Args:
        vocal_analysis: analyze_vocal で返された分析結果 dict
        library_index_path: scan_library で生成した JSON インデックスのパス

    Returns:
        dict:
            - matches: マッチしたトラックリスト (rank, track, compatibility, suggestion)
            - total_searched: 検索対象トラック数
            - vocal_key: ボーカルのキー
            - vocal_camelot: ボーカルの Camelot コード
    """
    from vml_audio_lab.tools.vocal_analysis import find_compatible_tracks_for_vocal

    return find_compatible_tracks_for_vocal(vocal_analysis, library_index_path)


@mcp.tool
def add_to_vocal_for_house_playlist(
    xml_path: str,
    track_id: str,
) -> dict:
    """「Vocal for House」プレイリストにトラックを追加する。

    HipHop/JPop のボーカルが House に合う曲を Rekordbox XML に登録する。

    Args:
        xml_path: Rekordbox XML のパス
        track_id: トラックID (文字列)

    Returns:
        dict:
            - playlist: 追加されたプレイリスト名
            - xml_path: 更新した XML パス
            - skipped: スキップした場合 True
    """
    from vml_audio_lab.tools.playlist import add_vocal_for_house

    return add_vocal_for_house(xml_path, track_id)


def main() -> None:
    """サーバーを起動する。"""
    mcp.run()


if __name__ == "__main__":
    main()
