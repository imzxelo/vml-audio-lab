"""ムード判定ツール.

楽曲のキー・BPM・エネルギーからムードカテゴリを推定する。
"""

from __future__ import annotations

import numpy as np

# ムード定義 (優先度順)
# 条件は AND で評価。最初にマッチしたムードを採用する。
_MOOD_RULES: list[dict] = [
    {
        "mood": "Peak Time",
        "description": "高エネルギー・BPM高め・ピークタイム向け",
        "conditions": {
            "energy_min": 0.65,
            "bpm_min": 128.0,
            "scale": None,  # any
        },
    },
    {
        "mood": "Deep Night",
        "description": "マイナーキー・低エネルギー・深夜向け",
        "conditions": {
            "energy_max": 0.50,
            "bpm_max": 125.0,
            "scale": "minor",
        },
    },
    {
        "mood": "Melodic Journey",
        "description": "メロディ成分強・起伏大・マイナーキー",
        "conditions": {
            "melodic_min": 0.35,
            "energy_range": (0.35, 0.80),
            "scale": "minor",
        },
    },
    {
        "mood": "Groovy & Warm",
        "description": "中エネルギー・メジャーキー・グルーヴ感",
        "conditions": {
            "energy_range": (0.35, 0.70),
            "scale": "major",
            "bpm_range": (118.0, 132.0),
        },
    },
    {
        "mood": "Chill & Mellow",
        "description": "低〜中エネルギー・メジャーキー・BPM低め",
        "conditions": {
            "energy_max": 0.55,
            "scale": "major",
        },
    },
]


def _compute_melodic_score(y: np.ndarray, sr: int) -> float:
    """メロディ成分の強さを 0〜1 で推定する。

    クロマグラムのエントロピーが高いほどメロディ成分が強いと判定する。
    """
    import librosa

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # フレームごとのクロマエネルギーを正規化
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)
    # シャノンエントロピー (フレーム平均)
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + 1e-8), axis=0)
    # 最大エントロピー = log(12) ≒ 2.485
    max_entropy = np.log(12)
    score = float(np.mean(entropy) / max_entropy)
    return min(1.0, max(0.0, score))


def _compute_energy_level(y: np.ndarray, sr: int) -> float:
    """RMS エネルギーを 0〜1 に正規化して返す。"""
    import librosa

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    # 経験的な最大値 0.3 で正規化 (クリップ)
    return min(1.0, mean_rms / 0.3)


def _matches_condition(
    condition: dict,
    energy: float,
    bpm: float,
    scale: str,
    melodic_score: float,
) -> bool:
    """単一ムードの条件を評価する。"""
    if "energy_min" in condition and energy < condition["energy_min"]:
        return False
    if "energy_max" in condition and energy > condition["energy_max"]:
        return False
    if "energy_range" in condition:
        lo, hi = condition["energy_range"]
        if not (lo <= energy <= hi):
            return False
    if "bpm_min" in condition and bpm < condition["bpm_min"]:
        return False
    if "bpm_max" in condition and bpm > condition["bpm_max"]:
        return False
    if "bpm_range" in condition:
        lo, hi = condition["bpm_range"]
        if not (lo <= bpm <= hi):
            return False
    if "scale" in condition and condition["scale"] is not None:
        if scale != condition["scale"]:
            return False
    if "melodic_min" in condition and melodic_score < condition["melodic_min"]:
        return False
    return True


def detect_mood(
    y: np.ndarray,
    sr: int,
    bpm: float,
    scale: str,
) -> dict:
    """楽曲のムードカテゴリを推定する。

    Args:
        y: 音声データ (librosa で読み込んだ float32 配列)
        sr: サンプリングレート
        bpm: 検出済みの BPM
        scale: キースケール ("major" または "minor")

    Returns:
        dict:
            - mood: ムード名 (例: "Peak Time")
            - description: ムードの説明
            - energy_level: 正規化エネルギー (0.0〜1.0)
            - melodic_score: メロディ成分スコア (0.0〜1.0)
    """
    energy = _compute_energy_level(y, sr)
    melodic_score = _compute_melodic_score(y, sr)

    for rule in _MOOD_RULES:
        if _matches_condition(rule["conditions"], energy, bpm, scale, melodic_score):
            return {
                "mood": rule["mood"],
                "description": rule["description"],
                "energy_level": round(energy, 3),
                "melodic_score": round(melodic_score, 3),
            }

    # どの条件にもマッチしない場合はエネルギーで大まかに分類
    if energy >= 0.6:
        mood, desc = "Peak Time", "高エネルギー・ピークタイム向け"
    else:
        mood, desc = "Chill & Mellow", "低〜中エネルギー・メジャーキー・BPM低め"

    return {
        "mood": mood,
        "description": desc,
        "energy_level": round(energy, 3),
        "melodic_score": round(melodic_score, 3),
    }
