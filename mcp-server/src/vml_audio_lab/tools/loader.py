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

# 許可する音声ファイル拡張子
_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

# YouTube URL パターン
_YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+$"
)


def _ensure_cache_dir() -> Path:
    """キャッシュディレクトリを作成して返す。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_key(source: str, offset: float = 0.0, duration: float | None = None) -> str:
    """ソース文字列とパラメータからキャッシュキーを生成する。"""
    key_str = f"{source}|offset={offset}|duration={duration}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _is_youtube_url(source: str) -> bool:
    """YouTube URL かどうかを判定する。"""
    return bool(_YT_PATTERN.match(source))


def _validate_local_path(file_path: str) -> str:
    """ローカルファイルパスを検証し、解決済みパスを返す。

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: サポートされていないファイル形式の場合
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")
    if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"サポートされていないファイル形式です: {path.suffix} "
            f"(許可: {', '.join(sorted(_ALLOWED_EXTENSIONS))})"
        )
    return str(path)


def load_y(y_path: str) -> np.ndarray:
    """キャッシュされた音声データを読み込む。

    キャッシュディレクトリ内のパスのみ許可する。

    Args:
        y_path: load_track で返された音声データのパス

    Raises:
        ValueError: キャッシュディレクトリ外のパスが指定された場合
        FileNotFoundError: ファイルが存在しない場合
    """
    path = Path(y_path).resolve()
    cache_dir = _CACHE_DIR.resolve()
    if not path.is_relative_to(cache_dir):
        raise ValueError(f"不正なパスです（キャッシュディレクトリ外）: {y_path}")
    if not path.exists():
        raise FileNotFoundError(f"音声データが見つかりません: {y_path}")
    return np.load(path)


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
    local_path: str = ""

    # キャッシュヒット確認（ダウンロード/ロード前にチェック）
    cache_dir = _ensure_cache_dir()
    key = _cache_key(file_path, offset=offset, duration=duration)
    y_path = str(cache_dir / f"{key}_y.npy")
    cache_hit = Path(y_path).exists()

    if _is_youtube_url(file_path):
        source = "youtube"
        if not cache_hit:
            local_path, yt_meta = _download_youtube(file_path)
    else:
        local_path = _validate_local_path(file_path)
        source = "local"

    if cache_hit:
        y = np.load(y_path)
    else:
        y, _ = librosa.load(
            path=local_path,
            sr=DEFAULT_SR,
            mono=True,
            offset=offset,
            duration=duration,
        )
        np.save(y_path, y)

    duration_sec = float(librosa.get_duration(y=y, sr=DEFAULT_SR))

    result = {
        "y_path": y_path,
        "sr": DEFAULT_SR,
        "duration_sec": round(duration_sec, 2),
        "n_samples": len(y),
        "file_path": local_path,
        "source": source,
    }
    result.update(yt_meta)
    return result
