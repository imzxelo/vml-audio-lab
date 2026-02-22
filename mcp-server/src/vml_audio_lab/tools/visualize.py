"""可視化ツール: スペクトログラム、波形オーバービュー"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from vml_audio_lab.tools.loader import DEFAULT_SR

matplotlib.use("Agg")

# 共通カラー設定
_SECTION_COLORS = {
    "Intro": "#4FC3F7",
    "Build": "#FFB74D",
    "Drop": "#E53935",
    "Break": "#81C784",
    "Outro": "#9575CD",
}
_DEFAULT_COLOR = "#BDBDBD"


def _load_y(y_path: str) -> np.ndarray:
    """キャッシュされた音声データを読み込む。"""
    path = Path(y_path)
    if not path.exists():
        raise FileNotFoundError(f"音声データが見つかりません: {y_path}")
    return np.load(y_path)


def _fig_to_png(fig: plt.Figure) -> bytes:
    """matplotlib Figure を PNG bytes に変換して閉じる。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def spectrogram(y_path: str) -> bytes:
    """メルスペクトログラムの画像を生成する。

    Args:
        y_path: load_track で返された音声データのパス

    Returns:
        bytes: PNG 画像データ
    """
    y = _load_y(y_path)
    S = librosa.feature.melspectrogram(y=y, sr=DEFAULT_SR, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        S_dB,
        sr=DEFAULT_SR,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")
    ax.set_title("Mel Spectrogram")
    fig.tight_layout()

    return _fig_to_png(fig)


def waveform_overview(
    y_path: str,
    sections: list[dict] | None = None,
) -> bytes:
    """波形オーバービュー画像を生成する。

    セクション情報が渡された場合、境界線とラベルをアノテーションする。

    Args:
        y_path: load_track で返された音声データのパス
        sections: detect_structure で返された sections リスト（省略可）

    Returns:
        bytes: PNG 画像データ
    """
    y = _load_y(y_path)
    duration = librosa.get_duration(y=y, sr=DEFAULT_SR)
    times = np.linspace(0, duration, len(y))

    fig, ax = plt.subplots(figsize=(12, 3))

    if sections:
        # セクションごとに色付き背景を描画
        for s in sections:
            color = _SECTION_COLORS.get(s.get("label", ""), _DEFAULT_COLOR)
            ax.axvspan(s["start"], s["end"], alpha=0.15, color=color)

            # 境界線
            if s["start"] > 0:
                ax.axvline(x=s["start"], color=color, linewidth=0.8, linestyle="--", alpha=0.7)

            # ラベル
            mid = (s["start"] + s["end"]) / 2
            ax.text(
                mid,
                1.05,
                s.get("label", ""),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax.plot(times, y, color="#1DB954", linewidth=0.3, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform Overview")
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    return _fig_to_png(fig)
