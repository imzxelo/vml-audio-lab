"""エフェクト検出ツール — 楽曲内のオーディオエフェクトを検出する.

リバーブ、ディレイ、フィルタースウィープ、サイドチェインコンプレッションの
4種類のエフェクトをlibrosaのスペクトル特徴量から推定する。
"""

from __future__ import annotations

import numpy as np


def _detect_reverb(y: np.ndarray, sr: int) -> dict:
    """リバーブ（残響）を検出する。

    スペクトルロールオフの減衰率とトランジェント後のエネルギー残存から推定。
    """
    import librosa

    # スペクトルロールオフ (高周波成分の減衰を見る)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    # RMSエネルギー
    rms = librosa.feature.rms(y=y)[0]

    # オンセット検出
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")

    if len(onset_frames) < 2 or len(rms) < 10:
        return {"type": "reverb", "detected": False, "confidence": 0.0}

    # オンセット後のエネルギー減衰率を計算
    decay_rates: list[float] = []
    hop_length = 512  # librosa default
    for onset_f in onset_frames:
        if onset_f + 10 >= len(rms):
            continue
        # オンセット後10フレームのRMS減衰を見る
        post_onset = rms[onset_f : onset_f + 10]
        if post_onset[0] > 0.01:
            decay = post_onset[-1] / post_onset[0]
            decay_rates.append(decay)

    if not decay_rates:
        return {"type": "reverb", "detected": False, "confidence": 0.0}

    # 高い残存率 = リバーブあり
    avg_decay = float(np.mean(decay_rates))
    # rolloff の標準偏差が低い = 一定の高域残響
    rolloff_stability = 1.0 - min(1.0, float(np.std(rolloff)) / (float(np.mean(rolloff)) + 1e-8))

    # 複合スコア: 減衰が遅い + rolloff安定 = リバーブ
    confidence = min(1.0, avg_decay * 0.6 + rolloff_stability * 0.4)
    detected = confidence > 0.45

    return {
        "type": "reverb",
        "detected": detected,
        "confidence": round(confidence, 3),
        "details": {
            "avg_decay_rate": round(avg_decay, 3),
            "rolloff_stability": round(rolloff_stability, 3),
        },
    }


def _detect_delay(y: np.ndarray, sr: int) -> dict:
    """ディレイ（エコー）を検出する。

    振幅エンベロープの自己相関で等間隔の反復パターンを検出。
    """
    import librosa

    rms = librosa.feature.rms(y=y)[0]
    if len(rms) < 50:
        return {"type": "delay", "detected": False, "confidence": 0.0}

    # 自己相関
    rms_norm = rms - np.mean(rms)
    autocorr = np.correlate(rms_norm, rms_norm, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    if autocorr[0] <= 0:
        return {"type": "delay", "detected": False, "confidence": 0.0}

    autocorr = autocorr / autocorr[0]

    # ピーク検出 (最小間隔 5フレーム、高さ 0.3以上)
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(autocorr[5:], height=0.3, distance=5)
    peaks = peaks + 5  # オフセット補正

    if len(peaks) < 2:
        return {"type": "delay", "detected": False, "confidence": 0.0}

    # ピーク間隔の一貫性を評価
    intervals = np.diff(peaks)
    if len(intervals) == 0:
        return {"type": "delay", "detected": False, "confidence": 0.0}

    interval_std = float(np.std(intervals))
    interval_mean = float(np.mean(intervals))
    regularity = max(0.0, 1.0 - interval_std / (interval_mean + 1e-8))

    # ピーク高さの平均
    peak_heights = float(np.mean(properties["peak_heights"])) if "peak_heights" in properties else 0.3

    confidence = min(1.0, regularity * 0.6 + peak_heights * 0.4)
    detected = confidence > 0.5 and len(peaks) >= 2

    # ディレイタイムの推定 (フレーム → 秒)
    hop_length = 512
    delay_time_sec = float(interval_mean * hop_length / sr) if interval_mean > 0 else 0.0

    return {
        "type": "delay",
        "detected": detected,
        "confidence": round(confidence, 3),
        "details": {
            "delay_time_sec": round(delay_time_sec, 3),
            "regularity": round(regularity, 3),
            "num_echoes": len(peaks),
        },
    }


def _detect_filter_sweep(y: np.ndarray, sr: int) -> dict:
    """フィルタースウィープを検出する。

    スペクトル重心の時系列に単調増加/減少トレンドがあるか調べる。
    """
    import librosa

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    if len(centroid) < 20:
        return {"type": "filter_sweep", "detected": False, "confidence": 0.0}

    # スムージング (移動平均)
    window = min(10, len(centroid) // 4)
    if window < 3:
        window = 3
    kernel = np.ones(window) / window
    smoothed = np.convolve(centroid, kernel, mode="valid")

    if len(smoothed) < 10:
        return {"type": "filter_sweep", "detected": False, "confidence": 0.0}

    # 勾配を計算
    gradient = np.gradient(smoothed)
    centroid_std = float(np.std(centroid))

    # 連続的な上昇/下降区間を検出
    sweep_threshold = centroid_std * 0.3
    rising = gradient > sweep_threshold
    falling = gradient < -sweep_threshold

    # 最長の連続区間を見つける
    max_rising_run = _max_consecutive(rising)
    max_falling_run = _max_consecutive(falling)
    max_run = max(max_rising_run, max_falling_run)

    # 全体の2秒以上（約8-10フレーム）のスウィープがあれば検出
    min_frames = max(8, int(2.0 * sr / 512 / window))
    detected = max_run >= min_frames

    # 信頼度: スウィープの長さと勾配の一貫性
    run_ratio = min(1.0, max_run / len(smoothed))
    confidence = min(1.0, run_ratio * 1.5)

    direction = "rising" if max_rising_run >= max_falling_run else "falling"

    return {
        "type": "filter_sweep",
        "detected": detected,
        "confidence": round(confidence, 3),
        "details": {
            "direction": direction,
            "max_sweep_frames": int(max_run),
            "sweep_duration_est_sec": round(max_run * 512 / sr, 2),
        },
    }


def _detect_sidechain(y: np.ndarray, sr: int) -> dict:
    """サイドチェインコンプレッションを検出する。

    ビート同期の振幅変調（ポンピング効果）を検出。
    """
    import librosa

    # ビートトラッキング
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    if len(beat_frames) < 4:
        return {"type": "sidechain", "detected": False, "confidence": 0.0}

    # RMSエネルギー
    rms = librosa.feature.rms(y=y)[0]

    # 各ビート周辺のRMSパターンを分析
    # サイドチェイン: ビート直後にRMSが下がり、次のビートに向けて回復
    dip_depths: list[float] = []

    for i in range(len(beat_frames) - 1):
        bf = beat_frames[i]
        bf_next = beat_frames[i + 1]

        if bf >= len(rms) or bf_next >= len(rms):
            continue

        segment = rms[bf:bf_next]
        if len(segment) < 3:
            continue

        peak_val = float(segment[0])
        if peak_val < 0.01:
            continue

        min_val = float(np.min(segment[1:]))
        dip = 1.0 - (min_val / peak_val)
        dip_depths.append(dip)

    if not dip_depths:
        return {"type": "sidechain", "detected": False, "confidence": 0.0}

    avg_dip = float(np.mean(dip_depths))
    dip_consistency = 1.0 - min(1.0, float(np.std(dip_depths)) / (avg_dip + 1e-8))

    # 深いディップ + 一貫性 = サイドチェイン
    confidence = min(1.0, avg_dip * 0.5 + dip_consistency * 0.5)
    detected = confidence > 0.4 and avg_dip > 0.2

    return {
        "type": "sidechain",
        "detected": detected,
        "confidence": round(confidence, 3),
        "details": {
            "avg_dip_depth": round(avg_dip, 3),
            "dip_consistency": round(dip_consistency, 3),
            "num_beats_analyzed": len(dip_depths),
        },
    }


def _max_consecutive(arr: np.ndarray) -> int:
    """bool配列の最長連続True区間の長さを返す。"""
    if len(arr) == 0:
        return 0
    max_run = 0
    current_run = 0
    for val in arr:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def detect_effects(
    y_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """楽曲内のエフェクトを検出する。

    Args:
        y_path: load_audio で返された y_path (numpy キャッシュパス)
        sections: analyze_structure のセクションリスト。
            省略時は全体を1セクションとして分析。

    Returns:
        dict:
            - effects: 検出されたエフェクトのリスト
            - effects_summary: セクションごとのエフェクトマップ
            - dominant_effect: 最も目立つエフェクト
    """
    import librosa

    from vml_audio_lab.tools.loader import DEFAULT_SR, load_y

    y, sr = load_y(y_path)
    if sr is None:
        sr = DEFAULT_SR

    if sections:
        return _analyze_with_sections(y, sr, sections)

    # 全体分析
    detectors = [_detect_reverb, _detect_delay, _detect_filter_sweep, _detect_sidechain]
    effects: list[dict] = []

    for detector in detectors:
        result = detector(y, sr)
        if result.get("detected"):
            effects.append({
                "type": result["type"],
                "confidence": result["confidence"],
                "section": "全体",
                "time_range": [0.0, round(len(y) / sr, 2)],
                "details": result.get("details", {}),
            })

    dominant = max(effects, key=lambda e: e["confidence"])["type"] if effects else "none"

    return {
        "effects": effects,
        "effects_summary": {"全体": [e["type"] for e in effects]},
        "dominant_effect": dominant,
    }


def _analyze_with_sections(
    y: np.ndarray,
    sr: int,
    sections: list[dict],
) -> dict:
    """セクション別にエフェクトを分析する。"""
    detectors = [_detect_reverb, _detect_delay, _detect_filter_sweep, _detect_sidechain]
    all_effects: list[dict] = []
    summary: dict[str, list[str]] = {}

    for section in sections:
        start_sec = float(section.get("start", 0))
        end_sec = float(section.get("end", len(y) / sr))
        label = section.get("label", "Unknown")

        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), len(y))

        if end_sample <= start_sample:
            continue

        segment = y[start_sample:end_sample]
        if len(segment) < sr:  # 1秒未満はスキップ
            continue

        section_effects: list[str] = []
        for detector in detectors:
            result = detector(segment, sr)
            if result.get("detected"):
                all_effects.append({
                    "type": result["type"],
                    "confidence": result["confidence"],
                    "section": label,
                    "time_range": [round(start_sec, 2), round(end_sec, 2)],
                    "details": result.get("details", {}),
                })
                section_effects.append(result["type"])

        summary[label] = section_effects

    dominant = max(all_effects, key=lambda e: e["confidence"])["type"] if all_effects else "none"

    return {
        "effects": all_effects,
        "effects_summary": summary,
        "dominant_effect": dominant,
    }
