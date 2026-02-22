"""構造認識ツール: セクション自動検出"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from vml_audio_lab.tools.loader import DEFAULT_SR


def _load_y(y_path: str) -> np.ndarray:
    """キャッシュされた音声データを読み込む。"""
    path = Path(y_path)
    if not path.exists():
        raise FileNotFoundError(f"音声データが見つかりません: {y_path}")
    return np.load(y_path)


def _estimate_n_segments(duration_sec: float) -> int:
    """楽曲の長さからセグメント数を推定する。

    目安: 30秒あたり1セグメント、最低2・最大12。
    """
    n = max(2, int(duration_sec / 30))
    return min(n, 12)


def _format_time(seconds: float) -> str:
    """秒数を "M:SS" 形式に変換する。"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def detect_structure(
    y_path: str,
    n_segments: int | None = None,
) -> dict:
    """楽曲のセクション構造を自動検出する。

    MFCC 特徴量 + 凝集型クラスタリングでセクション境界を推定し、
    各セクションにラベル（エネルギーベース）を付与する。

    Args:
        y_path: load_track で返された音声データのパス
        n_segments: セクション数（省略で自動推定）

    Returns:
        dict:
            - sections: セクションのリスト
                - start: 開始時刻 (秒)
                - end: 終了時刻 (秒)
                - start_label: "M:SS" 形式
                - end_label: "M:SS" 形式
                - label: セクションラベル
                - energy: 相対エネルギー (0.0〜1.0)
            - n_segments: セクション数
            - duration_sec: 全体の長さ (秒)
    """
    y = _load_y(y_path)
    duration_sec = float(librosa.get_duration(y=y, sr=DEFAULT_SR))

    if n_segments is None:
        n_segments = _estimate_n_segments(duration_sec)

    # MFCC 特徴量を抽出
    mfcc = librosa.feature.mfcc(y=y, sr=DEFAULT_SR, n_mfcc=13)

    # フレーム数がセグメント数より少ない場合はガード
    if mfcc.shape[1] < n_segments:
        n_segments = max(2, mfcc.shape[1])

    # 凝集型クラスタリングでセグメント境界を検出
    boundaries = librosa.segment.agglomerative(mfcc, k=n_segments)
    boundary_times = librosa.frames_to_time(boundaries, sr=DEFAULT_SR)

    # RMS エネルギーを計算（セクションラベル付与用）
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=DEFAULT_SR)

    # 各セクションのエネルギーを計算
    sections = []
    for i in range(len(boundary_times)):
        start = float(boundary_times[i])
        end = float(boundary_times[i + 1]) if i + 1 < len(boundary_times) else duration_sec

        # セクション内の RMS エネルギー平均
        mask = (rms_times >= start) & (rms_times < end)
        segment_energy = float(np.mean(rms[mask])) if np.any(mask) else 0.0

        sections.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "start_label": _format_time(start),
            "end_label": _format_time(end),
            "energy": segment_energy,
        })

    # エネルギーを 0〜1 に正規化
    max_energy = max(s["energy"] for s in sections) if sections else 1.0
    if max_energy > 0:
        for s in sections:
            s["energy"] = round(s["energy"] / max_energy, 3)

    # エネルギーベースのラベル付与
    _assign_labels(sections)

    return {
        "sections": sections,
        "n_segments": len(sections),
        "duration_sec": round(duration_sec, 2),
    }


def _assign_labels(sections: list[dict]) -> None:
    """エネルギーに基づいてセクションにラベルを付与する。

    - 最初のセクション → Intro
    - 最後のセクション → Outro
    - エネルギー 0.8以上 → Drop / High Energy
    - エネルギー 0.3未満 → Break
    - それ以外 → Build
    """
    for i, s in enumerate(sections):
        if i == 0:
            s["label"] = "Intro"
        elif i == len(sections) - 1:
            s["label"] = "Outro"
        elif s["energy"] >= 0.8:
            s["label"] = "Drop"
        elif s["energy"] < 0.3:
            s["label"] = "Break"
        else:
            s["label"] = "Build"
