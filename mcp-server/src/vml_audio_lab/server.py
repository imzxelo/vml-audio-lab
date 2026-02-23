"""VML Audio Lab — FastMCP サーバーエントリポイント"""

from pathlib import Path

from fastmcp import FastMCP

from vml_audio_lab import __version__
from vml_audio_lab.tools.analysis import detect_bpm, detect_key, energy_curve
from vml_audio_lab.tools.cues import recommend_cues
from vml_audio_lab.tools.genre import canonicalize_genre_slug, detect_genre, genre_group_for
from vml_audio_lab.tools.loader import load_track, load_y
from vml_audio_lab.tools.structure import detect_structure
from vml_audio_lab.tools.usb_export import copy_to_usb, split_artist_title, update_rekordbox_xml
from vml_audio_lab.tools.visualize import spectrogram, waveform_overview

mcp = FastMCP(
    name="vml-audio-lab",
    version=__version__,
    instructions=(
        "音楽分析MCPサーバー。楽曲のBPM・キー・構造・エネルギーを分析し、"
        "スペクトログラムや波形の画像を返す。DJ/音楽制作の先生として機能する。"
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

    Args:
        y_path: load_audio で返された y_path
    """
    return detect_key(y_path)


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
) -> dict:
    """楽曲のセクション構造を自動検出する。

    MFCC + クラスタリングで Intro/Build/Drop/Break/Outro を推定。

    Args:
        y_path: load_audio で返された y_path
        n_segments: セクション数（省略で自動推定）
    """
    return detect_structure(y_path, n_segments=n_segments)


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
    """YouTube楽曲を分析し、USB + Rekordbox XML まで一気通貫で準備する。"""
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
        }
    else:
        genre_result = detect_genre(title=title, artist=artist, y=y, sr=sr)

    copied = copy_to_usb(
        audio_path=str(loaded["file_path"]),
        genre_group=str(genre_result["genre_group"]),
        bpm=float(bpm_result["bpm"]),
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
        bpm=float(bpm_result["bpm"]),
        key_label=str(key_result["key_label"]),
        duration_sec=float(cues_result["duration_sec"]),
        hot_cues=list(cues_result["hot_cues"]),
        memory_cues=list(cues_result["memory_cues"]),
    )

    return {
        "status": "success",
        "track": {
            "title": f"{artist} - {title}",
            "bpm": bpm_result["bpm"],
            "key": key_result["key_label"],
            "genre": genre_result["genre"],
            "genre_group": genre_result["genre_group"],
            "duration_sec": cues_result["duration_sec"],
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
