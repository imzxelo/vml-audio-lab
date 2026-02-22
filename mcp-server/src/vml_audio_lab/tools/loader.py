"""音声ファイルの読み込みとキャッシュ管理"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np

# キャッシュディレクトリ
_CACHE_DIR = Path(tempfile.gettempdir()) / "vml_audio_lab"

# サンプリングレート（librosa デフォルト）
DEFAULT_SR = 22050

# YouTube URL パターン
_YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)


def _ensure_cache_dir() -> Path:
    """キャッシュディレクトリを作成して返す。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_key(source: str) -> str:
    """ソース文字列からキャッシュキーを生成する。"""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _is_youtube_url(source: str) -> bool:
    """YouTube URL かどうかを判定する。"""
    return bool(_YT_PATTERN.match(source))


def _download_youtube(url: str) -> tuple[str, dict]:
    """YouTube 動画の音声をダウンロードし、ファイルパスとメタデータを返す。

    Args:
        url: YouTube の URL

    Returns:
        tuple: (ダウンロードしたファイルパス, メタデータ dict)

    Raises:
        RuntimeError: ダウンロードに失敗した場合
    """
    import yt_dlp

    cache_dir = _ensure_cache_dir()
    key = _cache_key(url)
    output_path = str(cache_dir / f"{key}_yt.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"YouTube からの情報取得に失敗しました: {url}")

    wav_path = str(cache_dir / f"{key}_yt.wav")
    if not Path(wav_path).exists():
        raise RuntimeError(f"ダウンロードしたファイルが見つかりません: {wav_path}")

    meta = {
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "youtube_url": url,
    }
    return wav_path, meta


def load_track(
    file_path: str,
    offset: float = 0.0,
    duration: float | None = None,
) -> dict:
    """音声を読み込み、メタデータとキャッシュパスを返す。

    ローカルファイルパスまたは YouTube URL を受け取る。

    Args:
        file_path: 音声ファイルのパス、または YouTube URL
        offset: 読み込み開始位置（秒）
        duration: 読み込む長さ（秒）。Noneで全体を読み込む

    Returns:
        dict: メタデータとキャッシュパス
            - y_path: 音声データ(.npy)のパス
            - sr: サンプリングレート
            - duration_sec: 楽曲の長さ（秒）
            - n_samples: サンプル数
            - file_path: 元のファイルパス（またはダウンロード先）
            - source: "local" または "youtube"
            - title: (YouTube のみ) 動画タイトル
            - uploader: (YouTube のみ) アップロード者

    Raises:
        FileNotFoundError: ローカルファイルが存在しない場合
        RuntimeError: YouTube ダウンロードに失敗した場合
    """
    yt_meta: dict = {}

    if _is_youtube_url(file_path):
        local_path, yt_meta = _download_youtube(file_path)
        source = "youtube"
    else:
        local_path = str(Path(file_path).expanduser().resolve())
        if not Path(local_path).exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {local_path}")
        source = "local"

    y, sr = librosa.load(
        path=local_path,
        sr=DEFAULT_SR,
        mono=True,
        offset=offset,
        duration=duration,
    )

    # .npy でキャッシュ
    cache_dir = _ensure_cache_dir()
    key = _cache_key(file_path)
    y_path = str(cache_dir / f"{key}_y.npy")
    np.save(y_path, y)

    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    result = {
        "y_path": y_path,
        "sr": sr,
        "duration_sec": round(duration_sec, 2),
        "n_samples": len(y),
        "file_path": local_path,
        "source": source,
    }
    result.update(yt_meta)
    return result
