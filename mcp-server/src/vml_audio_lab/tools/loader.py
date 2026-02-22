"""音声ファイルの読み込みとキャッシュ管理"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import librosa
import numpy as np

# キャッシュディレクトリ
_CACHE_DIR = Path(tempfile.gettempdir()) / "vml_audio_lab"

# 分析用サンプリングレート（精度優先: CD相当）
DEFAULT_SR = 44100

# 許可する音声ファイル拡張子
_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm"}


def _ensure_cache_dir() -> Path:
    """キャッシュディレクトリを作成して返す。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_key(source: str, offset: float = 0.0, duration: float | None = None) -> str:
    """ソース文字列とパラメータからキャッシュキーを生成する。"""
    key_str = f"{source}|offset={offset}|duration={duration}|sr={DEFAULT_SR}|mono=True"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _is_youtube_url(source: str) -> bool:
    """YouTube URL かどうかを判定する。クエリ付きURLにも対応。"""
    try:
        parsed = urlparse(source)
    except Exception:
        return False

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")

    youtube_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
    if host not in youtube_hosts:
        return False

    if host == "youtu.be":
        return len(path.strip("/")) > 0

    # youtube.com系
    if path == "/watch":
        return "v" in query and len(query.get("v", [""])[0]) > 0
    if path.startswith("/shorts/"):
        return len(path.split("/shorts/", 1)[1]) > 0
    return False


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
    """YouTube 動画の音声を高音質でダウンロードし、ファイルパスとメタデータを返す。"""
    import yt_dlp

    cache_dir = _ensure_cache_dir()
    key = _cache_key(url)
    output_tmpl = str(cache_dir / f"{key}_yt.%(ext)s")

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_tmpl,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"YouTube からの情報取得に失敗しました: {url}")

        # 実際に保存されたパスを取得
        local_path = None
        requested = info.get("requested_downloads")
        if isinstance(requested, list) and requested:
            local_path = requested[0].get("filepath")
        if not local_path:
            local_path = info.get("_filename")
        if not local_path:
            local_path = ydl.prepare_filename(info)

    if not local_path or not Path(local_path).exists():
        raise RuntimeError(f"ダウンロードしたファイルが見つかりません: {local_path}")

    meta = {
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "youtube_url": url,
        "audio_ext": info.get("ext", ""),
        "audio_abr_kbps": info.get("abr"),
        "audio_asr_hz": info.get("asr"),
    }
    return str(Path(local_path).resolve()), meta


def load_track(
    file_path: str,
    offset: float = 0.0,
    duration: float | None = None,
) -> dict:
    """音声を読み込み、メタデータとキャッシュパスを返す。

    ローカルファイルパスまたは YouTube URL を受け取る。
    YouTubeの場合もダウンロード音源（高音質）を file_path として返すので、
    そのまま再利用できる。

    Args:
        file_path: 音声ファイルのパス、または YouTube URL
        offset: 読み込み開始位置（秒）
        duration: 読み込む長さ（秒）。Noneで全体を読み込む

    Returns:
        dict: メタデータとキャッシュパス
            - y_path: 分析用音声データ(.npy)のパス
            - sr: 分析サンプリングレート（DEFAULT_SR）
            - duration_sec: 楽曲の長さ（秒）
            - n_samples: サンプル数
            - file_path: 元のファイルパス（またはYouTubeダウンロード先）
            - source: "local" または "youtube"
            - title/uploader/youtube_url: (YouTube のみ)
    """
    yt_meta: dict = {}
    local_path: str = ""

    cache_dir = _ensure_cache_dir()
    key = _cache_key(file_path, offset=offset, duration=duration)
    y_path = str(cache_dir / f"{key}_y.npy")
    meta_path = cache_dir / f"{key}_meta.json"
    cache_hit = Path(y_path).exists()

    source = "youtube" if _is_youtube_url(file_path) else "local"

    if cache_hit:
        y = np.load(y_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            local_path = meta.get("file_path", "")
            yt_meta = meta.get("yt_meta", {})
            source = meta.get("source", source)
    else:
        if source == "youtube":
            local_path, yt_meta = _download_youtube(file_path)
        else:
            local_path = _validate_local_path(file_path)

        y, _ = librosa.load(
            path=local_path,
            sr=DEFAULT_SR,
            mono=True,
            offset=offset,
            duration=duration,
        )
        np.save(y_path, y)

        meta_path.write_text(
            json.dumps(
                {
                    "source": source,
                    "file_path": local_path,
                    "yt_meta": yt_meta,
                    "analysis_sr": DEFAULT_SR,
                    "analysis_mono": True,
                },
                ensure_ascii=False,
            )
        )

    duration_sec = float(librosa.get_duration(y=y, sr=DEFAULT_SR))

    result = {
        "y_path": y_path,
        "sr": DEFAULT_SR,
        "duration_sec": round(duration_sec, 2),
        "n_samples": len(y),
        "file_path": local_path,
        "source": source,
        "analysis_mono": True,
        "analysis_profile": "hq_44k_mono",
    }
    result.update(yt_meta)
    return result
