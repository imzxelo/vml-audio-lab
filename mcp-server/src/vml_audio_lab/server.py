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
from vml_audio_lab.tools.loader import load_track, load_y
from vml_audio_lab.tools.mood import detect_mood
from vml_audio_lab.tools.playlist import generate_playlists, get_compatible_playlists
from vml_audio_lab.tools.structure import detect_structure
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

    from vml_audio_lab.tools.loader import DEFAULT_SR

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

    from vml_audio_lab.tools.loader import DEFAULT_SR

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

    from vml_audio_lab.tools.loader import DEFAULT_SR

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
        genre_result = detect_genre(title=title, artist=artist, y=y, sr=sr)

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


def main() -> None:
    """サーバーを起動する。"""
    mcp.run()


if __name__ == "__main__":
    main()
