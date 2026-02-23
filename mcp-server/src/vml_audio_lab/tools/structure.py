"""構造認識ツール: セクション自動検出"""

from __future__ import annotations

import librosa
import numpy as np

from vml_audio_lab.tools.loader import DEFAULT_SR, load_y

# ジャンルグループ別のラベルセット
# 各エントリ: (intro, high_energy, low_energy, mid_energy, outro)
_GENRE_LABEL_MAP: dict[str, dict[str, str]] = {
    "hiphop": {
        "intro": "Intro",
        "high_energy": "Hook",
        "low_energy": "Bridge",
        "mid_energy": "Verse",
        "outro": "Outro",
    },
    "jpop": {
        "intro": "Intro",
        "high_energy": "サビ",
        "low_energy": "落ちサビ",
        "mid_energy": "Aメロ",
        "outro": "Outro",
    },
    "classical": {
        "intro": "主題",
        "high_energy": "再現",
        "low_energy": "コーダ",
        "mid_energy": "展開",
        "outro": "コーダ",
    },
    # DJ系 (house, techno, melodic 等) は EDM ラベル
    "default": {
        "intro": "Intro",
        "high_energy": "Drop",
        "low_energy": "Break",
        "mid_energy": "Build",
        "outro": "Outro",
    },
}

# ジャンルスラグ → ラベルグループのマッピング
# genre.py の _GENRE_GROUP と対応する。ここでは structure.py 独立で持つ。
_GENRE_SLUG_TO_LABEL_GROUP: dict[str, str] = {
    "hiphop": "hiphop",
    "jpop": "jpop",
    "classical": "classical",
    # DJ系はすべて default
    "house": "default",
    "deep-house": "default",
    "tech-house": "default",
    "techno": "default",
    "deep-techno": "default",
    "uk-garage": "default",
    "melodic": "default",
    "dnb": "default",
    "trance": "default",
    "electronic": "default",
}


def _genre_to_label_group(genre: str) -> str:
    """ジャンルスラグまたはグループ名をラベルグループに変換する。

    Args:
        genre: ジャンルスラグ (例: "hiphop", "house") またはグループ名

    Returns:
        ラベルグループ名 ("hiphop", "jpop", "classical", "default")
    """
    normalized = genre.strip().lower()
    return _GENRE_SLUG_TO_LABEL_GROUP.get(normalized, "default")


def _genre_labels(genre_group: str) -> dict[str, str]:
    """ジャンルグループに対応するラベルセットを返す。"""
    return _GENRE_LABEL_MAP.get(genre_group, _GENRE_LABEL_MAP["default"])


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


def _assign_labels(sections: list[dict], genre_group: str = "default") -> None:
    """エネルギーに基づいてセクションにラベルを付与する。

    J-Pop / Hip-Hop はジャンル固有のラベルを使用する。

    Args:
        sections: セクションのリスト (energy フィールドが必須)
        genre_group: ジャンルグループ名
    """
    labels = _genre_labels(genre_group)
    n = len(sections)

    for i, s in enumerate(sections):
        if i == 0:
            s["label"] = labels["intro"]
        elif i == n - 1:
            s["label"] = labels["outro"]
        elif s["energy"] >= 0.8:
            s["label"] = labels["high_energy"]
        elif s["energy"] < 0.3:
            s["label"] = labels["low_energy"]
        else:
            s["label"] = labels["mid_energy"]

    # J-Pop 特有: 高エネルギーが複数ある場合、後半を大サビ/落ちサビに変換
    if genre_group == "jpop":
        _refine_jpop_labels(sections)


def _refine_jpop_labels(sections: list[dict]) -> None:
    """J-Pop のサビ構造をより詳細に分類する。

    最初のサビ → サビ
    2回目のサビ → 大サビ（最後の高エネルギーセクション）
    """
    sabi_indices = [i for i, s in enumerate(sections) if s["label"] == "サビ"]
    if len(sabi_indices) >= 2:
        # 最後の「サビ」を「大サビ」に変更
        sections[sabi_indices[-1]]["label"] = "大サビ"

    # Bメロ: Aメロとサビの間のセクション
    for i, s in enumerate(sections):
        if s["label"] == "Aメロ":
            # 前のセクションが「Aメロ」で次がサビなら Bメロ
            if i > 0 and i + 1 < len(sections):
                prev_label = sections[i - 1]["label"]
                next_label = sections[i + 1]["label"]
                if prev_label == "Aメロ" and next_label in ("サビ", "大サビ"):
                    s["label"] = "Bメロ"


def detect_structure(
    y_path: str,
    n_segments: int | None = None,
    genre: str | None = None,
    genre_group: str = "default",
) -> dict:
    """楽曲のセクション構造を自動検出する。

    MFCC 特徴量 + 凝集型クラスタリングでセクション境界を推定し、
    ジャンルに応じたラベルを付与する。

    Args:
        y_path: load_track で返された音声データのパス
        n_segments: セクション数（省略で自動推定）
        genre: ジャンルスラグ (例: "hiphop", "jpop", "house")。
            指定するとジャンル別ラベルが適用される。
            genre_group より優先される。
        genre_group: ジャンルグループ (hiphop, jpop, classical, または default)
            genre が None の場合に使用される。
            - "hiphop": Verse/Hook/Bridge/Outro
            - "jpop": Aメロ/Bメロ/サビ/落ちサビ/大サビ/Outro
            - "classical": 主題/展開/再現/コーダ
            - "default": Intro/Build/Drop/Break/Outro (DJ系)

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
            - genre_group: 使用したジャンルグループ
    """
    # genre 引数が指定されたら genre_group に変換する
    if genre is not None:
        genre_group = _genre_to_label_group(genre)
    y = load_y(y_path)
    duration_sec = float(librosa.get_duration(y=y, sr=DEFAULT_SR))

    # 極短音声（1秒未満）はクラスタリングできないので単一セクションを返す
    if duration_sec < 1.0:
        section = {
            "start": 0.0,
            "end": round(duration_sec, 2),
            "start_label": _format_time(0.0),
            "end_label": _format_time(duration_sec),
            "label": "Full Track",
            "energy": 1.0,
        }
        return {
            "sections": [section],
            "n_segments": 1,
            "duration_sec": round(duration_sec, 2),
            "genre_group": genre_group,
        }

    if n_segments is None:
        n_segments = _estimate_n_segments(duration_sec)

    # MFCC 特徴量を抽出
    mfcc = librosa.feature.mfcc(y=y, sr=DEFAULT_SR, n_mfcc=13)

    # フレーム数がセグメント数より少ない場合はガード
    if mfcc.shape[1] < n_segments:
        n_segments = mfcc.shape[1]

    # フレーム数が足りずクラスタリングできない場合は単一セクションを返す
    if n_segments < 2:
        section = {
            "start": 0.0,
            "end": round(duration_sec, 2),
            "start_label": _format_time(0.0),
            "end_label": _format_time(duration_sec),
            "label": "Full Track",
            "energy": 1.0,
        }
        return {
            "sections": [section],
            "n_segments": 1,
            "duration_sec": round(duration_sec, 2),
            "genre_group": genre_group,
        }

    # 凝集型クラスタリングでセグメント境界を検出
    boundaries = librosa.segment.agglomerative(mfcc, k=n_segments)
    boundary_times = librosa.frames_to_time(boundaries, sr=DEFAULT_SR)

    # 先頭が 0.0 でない場合は追加
    if len(boundary_times) == 0 or boundary_times[0] > 0.0:
        boundary_times = np.concatenate([[0.0], boundary_times])

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

    # ジャンル別ラベル付与
    _assign_labels(sections, genre_group=genre_group)

    return {
        "sections": sections,
        "n_segments": len(sections),
        "duration_sec": round(duration_sec, 2),
        "genre_group": genre_group,
    }
