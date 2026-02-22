"""基礎分析ツール: BPM検出、キー検出、エネルギーカーブ"""

from __future__ import annotations

import librosa
import numpy as np
from essentia.standard import KeyExtractor

from vml_audio_lab.tools.loader import DEFAULT_SR, load_y
from vml_audio_lab.utils.plotting import fig_to_png, plt


def detect_bpm(y_path: str) -> dict:
    """BPM を推定する。

    Args:
        y_path: load_track で返された音声データのパス

    Returns:
        dict:
            - bpm: 推定BPM (float)
    """
    y = load_y(y_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=DEFAULT_SR)
    bpm = float(np.atleast_1d(tempo)[0])
    return {"bpm": round(bpm, 1)}


def detect_key(y_path: str) -> dict:
    """楽曲のキー（調）を推定する。

    essentia の KeyExtractor を使用（Krumhansl-Schmuckler アルゴリズム）。

    Args:
        y_path: load_track で返された音声データのパス

    Returns:
        dict:
            - key: キー名 (例: "F")
            - scale: スケール (例: "minor")
            - key_label: 表示用 (例: "Fm")
            - strength: 確信度 (0.0〜1.0)
    """
    y = load_y(y_path)
    extractor = KeyExtractor()
    key, scale, strength = extractor(y)
    key_label = f"{key}m" if scale == "minor" else key
    return {
        "key": key,
        "scale": scale,
        "key_label": key_label,
        "strength": round(float(strength), 3),
    }


def energy_curve(y_path: str) -> bytes:
    """RMS エネルギーカーブの画像を生成する。

    Args:
        y_path: load_track で返された音声データのパス

    Returns:
        bytes: PNG 画像データ
    """
    y = load_y(y_path)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=DEFAULT_SR)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, rms, color="#1DB954", linewidth=0.8)
    ax.fill_between(times, rms, alpha=0.3, color="#1DB954")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("Energy Curve")
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()

    return fig_to_png(fig)
