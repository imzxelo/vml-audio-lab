"""VML Audio Lab — FastMCP サーバーエントリポイント"""

from fastmcp import FastMCP

from vml_audio_lab import __version__
from vml_audio_lab.tools.analysis import detect_bpm, detect_key, energy_curve
from vml_audio_lab.tools.loader import load_track
from vml_audio_lab.tools.structure import detect_structure
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


def main() -> None:
    """サーバーを起動する。"""
    mcp.run()


if __name__ == "__main__":
    main()
