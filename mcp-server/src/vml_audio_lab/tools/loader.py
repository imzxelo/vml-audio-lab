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


def _extract_youtube_video_id(source: str) -> str | None:
    """YouTube URL から video id を抽出する。非対応URLなら None。"""
    try:
        parsed = urlparse(source)
    except Exception:
        return None

    if parsed.scheme not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")

    youtube_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
    if host not in youtube_hosts:
        return None

    if host == "youtu.be":
        vid = path.strip("/").split("/")[0] if path.strip("/") else ""
        return vid or None

    if path == "/watch":
        vid = query.get("v", [""])[0]
        return vid or None

    if path.startswith("/shorts/"):
        tail = path.split("/shorts/", 1)[1]
        vid = tail.split("/")[0]
        return vid or None

    return None


def _canonical_source(source: str) -> str:
    """キャッシュキー用にソースを正規化する。"""
    vid = _extract_youtube_video_id(source)
    if vid:
        return f"youtube:{vid}"
    return source


def _is_youtube_url(source: str) -> bool:
    """YouTube URL かどうかを判定する。クエリ付きURLにも対応。"""
    return _extract_youtube_video_id(source) is not None


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
    key = _cache_key(_canonical_source(url))
    output_tmpl = str(cache_dir / f"{key}_yt.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_tmpl,
        "noplaylist": True,
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
        selected_format = {}
        if isinstance(requested, list) and requested:
            local_path = requested[0].get("filepath")
            selected_format = requested[0]
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
        "video_id": _extract_youtube_video_id(url),
        "audio_ext": selected_format.get("ext") or info.get("ext", ""),
        "audio_abr_kbps": selected_format.get("abr") if selected_format else info.get("abr"),
        "audio_asr_hz": selected_format.get("asr") if selected_format else info.get("asr"),
        "audio_codec": selected_format.get("acodec") if selected_format else info.get("acodec"),
        "format_id": selected_format.get("format_id") if selected_format else info.get("format_id"),
    }
    return str(Path(local_path).resolve()), meta


def _write_meta(meta_path: Path, source: str, local_path: str, yt_meta: dict) -> None:
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


def load_track(
    file_path: str,
    offset: float = 0.0,
    duration: float | None = None,
) -> dict:
    """音声を読み込み、メタデータとキャッシュパスを返す。

    ローカルファイルパスまたは YouTube URL を受け取る。
    YouTubeの場合もダウンロード音源（高音質）を file_path として返すので、
    そのまま再利用できる。
    """
    yt_meta: dict = {}
    local_path: str = ""

    source = "youtube" if _is_youtube_url(file_path) else "local"
    key_source = _canonical_source(file_path)

    cache_dir = _ensure_cache_dir()
    key = _cache_key(key_source, offset=offset, duration=duration)
    y_path = str(cache_dir / f"{key}_y.npy")
    meta_path = cache_dir / f"{key}_meta.json"
    cache_hit = Path(y_path).exists()

    if cache_hit:
        y = np.load(y_path)

        meta_ok = False
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                source = meta.get("source", source)
                local_path = str(meta.get("file_path", ""))
                yt_meta = meta.get("yt_meta", {})
                if local_path and Path(local_path).exists():
                    meta_ok = True
            except Exception:
                meta_ok = False

        # メタ情報欠損時は復元
        if not meta_ok:
            if source == "youtube":
                local_path, yt_meta = _download_youtube(file_path)
            else:
                local_path = _validate_local_path(file_path)
            _write_meta(meta_path, source, local_path, yt_meta)

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
        _write_meta(meta_path, source, local_path, yt_meta)

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
