"""ボーカルステム分析ツール.

分離されたボーカルステムのキー・音域・使えるセクションを判定する。
ライブラリからキー相性でマッチングして、サンプリング提案を生成する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from essentia.standard import KeyExtractor

from vml_audio_lab.tools.camelot import compatibility_score, compatible_camelot_codes, key_to_camelot

# ボーカルエネルギー閾値: この値以上のRMSを「ボーカルあり」と判定
_VOCAL_ENERGY_THRESHOLD = 0.02
# 最小使用可能セクション長（秒）
_MIN_SECTION_DURATION = 4.0
# エネルギーベース検出時の最小セクション長（秒）
_MIN_DETECT_DURATION = 2.0
# デフォルト分析サンプリングレート
_SR = 44100


def _load_audio_sr(file_path: str) -> tuple[np.ndarray, int]:
    """音声ファイルを読み込む。

    Args:
        file_path: WAVファイルのパス

    Returns:
        (y, sr): 音声データとサンプリングレート
    """
    y, sr = librosa.load(file_path, sr=_SR, mono=True)
    return y, int(sr)


def _detect_vocal_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """ボーカルステムのキーを検出する。

    無音や極短音声の場合は ("unknown", "major", 0.0) を返す。

    Returns:
        (key_label, scale, strength)
    """
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-5:
        return "unknown", "major", 0.0

    try:
        extractor = KeyExtractor()
        key, scale, strength = extractor(y)
        key_label = f"{key}m" if scale == "minor" else key
        return key_label, scale, float(strength)
    except Exception:
        return "unknown", "major", 0.0


def _format_time(seconds: float) -> str:
    """秒数を "M:SS" 形式の文字列に変換する。"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _estimate_pitch_range(y: np.ndarray, sr: int) -> dict:
    """ピッチ（音域）の低域・高域を推定する。

    librosa の pyin を使って基本周波数を追跡する。

    Args:
        y: 音声データ
        sr: サンプリングレート

    Returns:
        dict: low (note name), high (note name), low_hz, high_hz
    """
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-5:
        return {"low": None, "high": None, "low_hz": None, "high_hz": None,
                "low_note": None, "high_note": None}

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    except Exception:
        return {"low": "N/A", "high": "N/A", "low_hz": 0.0, "high_hz": 0.0,
                "low_note": "N/A", "high_note": "N/A"}

    if len(voiced_f0) == 0:
        return {"low": 0.0, "high": 0.0, "low_hz": 0.0, "high_hz": 0.0,
                "low_note": "N/A", "high_note": "N/A"}

    low_hz = float(np.percentile(voiced_f0, 5))
    high_hz = float(np.percentile(voiced_f0, 95))

    low_note = librosa.hz_to_note(low_hz) if low_hz > 0 else "N/A"
    high_note = librosa.hz_to_note(high_hz) if high_hz > 0 else "N/A"

    return {
        "low": round(low_hz, 1),   # Hz 値 (数値)
        "high": round(high_hz, 1), # Hz 値 (数値)
        "low_hz": round(low_hz, 1),
        "high_hz": round(high_hz, 1),
        "low_note": low_note,
        "high_note": high_note,
    }


# _detect_pitch_range はテスト互換のエイリアス
def _detect_pitch_range(y: np.ndarray, sr: int) -> dict:
    """ピッチ（音域）の低域・高域を推定する（テスト互換ラッパー）。

    Returns:
        dict: low, high (Hz 値)、low_note, high_note
    """
    result = _estimate_pitch_range(y, sr)
    # low/high を Hz 値に変換してテスト互換に
    return {
        "low": result.get("low_hz"),
        "high": result.get("high_hz"),
        "low_note": result.get("low_note"),
        "high_note": result.get("high_note"),
    }


def _vocal_clarity(rms_segment: np.ndarray) -> float:
    """RMSセグメントからボーカルクラリティスコア（0-1）を計算する。

    RMSの平均と変動係数からノイズ比を推定する。
    """
    if len(rms_segment) == 0 or float(np.mean(rms_segment)) < 1e-8:
        return 0.0
    mean_rms = float(np.mean(rms_segment))
    std_rms = float(np.std(rms_segment))
    # 変動が小さい（安定している）ほど明瞭
    cv = std_rms / (mean_rms + 1e-8)
    clarity = max(0.0, min(1.0, 1.0 - cv))
    # エネルギーレベルで調整
    energy_bonus = min(1.0, mean_rms / 0.1)
    return round(float(clarity * 0.7 + energy_bonus * 0.3), 3)


def _generate_suggestion(section: dict, key: str, energy: float) -> str:
    """セクション特性から日本語サンプリング提案文を生成する。

    Args:
        section: セクション辞書 (type, has_clear_vocal 等)
        key: ボーカルのキーラベル (例: "Fm")
        energy: セクションのエネルギーレベル (0-1)

    Returns:
        日本語の提案文字列
    """
    label = str(section.get("type", section.get("original_label", "Section")))
    has_vocal = bool(section.get("has_clear_vocal", False))

    if not has_vocal:
        return f"{label}: ボーカルエネルギーが低い。インストゥルメンタル区間。"

    label_lower = label.lower()

    # 高エネルギー + フック/サビ
    if energy >= 0.6 and any(kw in label_lower for kw in ("hook", "chorus", "サビ", "drop")):
        return f"{label}: ドロップ前のビルドに使える。エネルギーが高くキャッチー。"

    # 中エネルギー + バース/Aメロ
    if 0.2 <= energy < 0.6 and any(kw in label_lower for kw in ("verse", "aメロ", "bメロ")):
        return f"{label}: Deep Houseのブレイク上に合う。フロー感あり。"

    # 低エネルギー + ブリッジ
    if energy < 0.2 and any(kw in label_lower for kw in ("bridge", "outro", "break")):
        return f"{label}: アンビエント系のブレイクに使える。静かで幻想的。"

    # 汎用判定
    if energy >= 0.5:
        return f"{label}: エネルギーが高い。ドロップやフックのサンプリング候補。"
    if energy >= 0.2:
        return f"{label}: 中程度のエネルギー。ブレイク区間でのループ使いに適している。"
    return f"{label}: 落ち着いた区間。アンビエント要素として使える。"


def _detect_vocal_sections(y: np.ndarray, sr: int) -> list[dict]:
    """RMSエネルギーベースでボーカルセクションを自動検出する。

    2秒未満または極低エネルギーのセクションは除外する。

    Args:
        y: 音声データ
        sr: サンプリングレート

    Returns:
        検出されたセクションのリスト
    """
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    usable: list[dict] = []
    in_vocal = False
    seg_start = 0.0
    seg_start_idx = 0

    for i, (t, r) in enumerate(zip(times, rms)):
        is_vocal = r >= _VOCAL_ENERGY_THRESHOLD
        if is_vocal and not in_vocal:
            seg_start = float(t)
            seg_start_idx = i
            in_vocal = True
        elif not is_vocal and in_vocal:
            seg_end = float(t)
            duration = seg_end - seg_start
            if duration >= _MIN_DETECT_DURATION:
                seg_rms = rms[seg_start_idx:i]
                mean_rms = float(np.mean(seg_rms))
                clarity = _vocal_clarity(seg_rms)
                usable.append({
                    "type": "Vocal Segment",
                    "start": round(seg_start, 2),
                    "end": round(seg_end, 2),
                    "start_label": _format_time(seg_start),
                    "end_label": _format_time(seg_end),
                    "duration": round(duration, 2),
                    "has_clear_vocal": True,
                    "vocal_clarity": clarity,
                    "energy": round(mean_rms, 4),
                    "suggestion": "ボーカルが明確なセグメント。サンプリング候補。",
                })
            in_vocal = False

    # ファイル終端処理
    if in_vocal:
        seg_end = float(times[-1])
        duration = seg_end - seg_start
        if duration >= _MIN_DETECT_DURATION:
            seg_rms = rms[seg_start_idx:]
            mean_rms = float(np.mean(seg_rms))
            clarity = _vocal_clarity(seg_rms)
            usable.append({
                "type": "Vocal Segment",
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "start_label": _format_time(seg_start),
                "end_label": _format_time(seg_end),
                "duration": round(duration, 2),
                "has_clear_vocal": True,
                "vocal_clarity": clarity,
                "energy": round(mean_rms, 4),
                "suggestion": "ボーカルが明確なセグメント。サンプリング候補。",
            })

    return usable


def _find_usable_sections(
    y: np.ndarray,
    sr: int,
    sections: list[dict] | None = None,
) -> list[dict]:
    """使えるボーカルセクションを検出する。

    sections が渡された場合は各セクションのボーカル有無を判定。
    なければ _detect_vocal_sections を使って自動検出する。

    Args:
        y: 音声データ
        sr: サンプリングレート
        sections: detect_structure で返されたセクションリスト（省略可）

    Returns:
        使えるセクションのリスト
    """
    if sections:
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        usable: list[dict] = []

        for section in sections:
            start = float(section.get("start", 0.0))
            end = float(section.get("end", 0.0))
            label = str(section.get("label", "Section"))

            if end - start < _MIN_SECTION_DURATION:
                continue

            mask = (times >= start) & (times <= end)
            if not np.any(mask):
                continue

            seg_rms = rms[mask]
            mean_rms = float(np.mean(seg_rms))
            has_vocal = mean_rms >= _VOCAL_ENERGY_THRESHOLD
            clarity = _vocal_clarity(seg_rms) if has_vocal else 0.0
            energy_norm = min(1.0, mean_rms / 0.1)

            suggestion = _generate_suggestion(
                {"type": label, "has_clear_vocal": has_vocal},
                key="",
                energy=energy_norm,
            )

            usable.append({
                "type": label,
                "original_label": label,
                "start": round(start, 2),
                "end": round(end, 2),
                "start_label": _format_time(start),
                "end_label": _format_time(end),
                "duration": round(end - start, 2),
                "has_clear_vocal": has_vocal,
                "vocal_clarity": clarity,
                "energy": round(mean_rms, 4),
                "suggestion": suggestion,
            })

        return usable

    return _detect_vocal_sections(y, sr)


def _sampling_score(usable_sections: list[dict], key_strength: float) -> float:
    """全体のサンプリングスコア（0-1）を計算する。

    使えるセクション数・クラリティ・キー強度から総合評価する。
    """
    if not usable_sections:
        return 0.0

    vocal_sections = [s for s in usable_sections if s.get("has_clear_vocal")]
    if not vocal_sections:
        return round(key_strength * 0.2, 3)

    avg_clarity = float(np.mean([s.get("vocal_clarity", 0.0) for s in vocal_sections]))
    coverage = min(1.0, len(vocal_sections) / 4.0)
    score = avg_clarity * 0.5 + key_strength * 0.3 + coverage * 0.2
    return round(float(score), 3)


def _compatible_bpm_range(key_label: str) -> list[float]:
    """ボーカルキーに合うBPM帯を返す。

    House / Deep House のテンポ帯を基準とする。

    Returns:
        [min_bpm, max_bpm] の2要素リスト
    """
    return [110.0, 135.0]


def _compatible_genres(key_label: str, scale: str) -> list[str]:
    """ボーカルに合うジャンルリストを返す。"""
    genres = ["deep-house", "house"]
    if scale == "minor":
        genres.append("deep-techno")
    else:
        genres.append("melodic")
    return genres


def analyze_vocal(
    vocals_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """ボーカルステムのキー・音域・使えるセクションを判定する（主要API）。

    Args:
        vocals_path: extract_vocals / separate_stems で返されたボーカルステムのWAVパス
        sections: analyze_structure で返されたセクションリスト（省略可）。
            渡すと各セクションのボーカル有無を判定する。

    Returns:
        dict:
            - key: ボーカルのキーラベル (例: "Fm")
            - key_label: key の別名
            - camelot_code: Camelot コード (例: "4A")
            - camelot: camelot_code の別名
            - scale: "major" または "minor"
            - key_strength: キー確信度 (0.0〜1.0)
            - pitch_range: 音域情報 (low, high, low_hz, high_hz)
            - usable_sections: 使えるセクションリスト
            - compatible_bpm_range: 合う BPM 帯 [min, max]
            - compatible_genres: 合うジャンルリスト
            - compatible_camelot_codes: 相性の良い Camelot コードリスト
            - sampling_score: 総合サンプリングスコア (0.0〜1.0)
    """
    path = Path(vocals_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ボーカルステムが見つかりません: {path}")

    y, sr = _load_audio_sr(str(path))

    key_label, scale, key_strength = _detect_vocal_key(y, sr)
    camelot = key_to_camelot(key_label)
    pitch_range = _estimate_pitch_range(y, sr)
    usable_sections = _find_usable_sections(y, sr, sections)
    bpm_range = _compatible_bpm_range(key_label)
    genres = _compatible_genres(key_label, scale)
    compat_camelot = compatible_camelot_codes(camelot) if camelot else []
    score = _sampling_score(usable_sections, key_strength)

    return {
        "key": key_label,
        "key_label": key_label,
        "camelot_code": camelot,
        "camelot": camelot,
        "scale": scale,
        "key_strength": round(key_strength, 3),
        "pitch_range": pitch_range,
        "usable_sections": usable_sections,
        "compatible_bpm_range": bpm_range,
        "compatible_genres": genres,
        "compatible_camelot_codes": compat_camelot,
        "sampling_score": score,
    }


def analyze_vocal_stem(
    vocals_path: str,
    sections: list[dict] | None = None,
) -> dict:
    """ボーカルステム分析（後方互換ラッパー）。

    `analyze_vocal` の別名。既存コードとテストとの互換を保つために維持。
    """
    return analyze_vocal(vocals_path, sections=sections)


def find_compatible_tracks_for_vocal(
    vocal_analysis: dict,
    library_index_path: str,
) -> dict:
    """ライブラリからボーカルと相性の良いトラックを検索する。

    Camelot キー相性 → BPM範囲フィルタ → スコア降順でソートして返す。

    Args:
        vocal_analysis: analyze_vocal / analyze_vocal_stem で返された分析結果
        library_index_path: scan_library で生成した JSONインデックスのパス

    Returns:
        dict:
            - matches: マッチしたトラックリスト (rank, track, compatibility, suggestion)
            - total_searched: 検索対象トラック数
            - vocal_key: ボーカルのキー
            - vocal_camelot: ボーカルの Camelot コード
    """
    idx_path = Path(library_index_path).expanduser().resolve()
    if not idx_path.exists():
        raise FileNotFoundError(f"ライブラリインデックスが見つかりません: {idx_path}")

    with idx_path.open("r", encoding="utf-8") as f:
        library: dict[str, Any] = json.load(f)

    tracks: list[dict] = library.get("tracks", [])

    vocal_camelot = vocal_analysis.get("camelot_code") or vocal_analysis.get("camelot")
    compat_codes = set(vocal_analysis.get("compatible_camelot_codes", []))
    bpm_range = vocal_analysis.get("compatible_bpm_range", [110.0, 135.0])
    if isinstance(bpm_range, list) and len(bpm_range) == 2:
        min_bpm, max_bpm = float(bpm_range[0]), float(bpm_range[1])
    elif isinstance(bpm_range, dict):
        min_bpm = float(bpm_range.get("min_bpm", 110.0))
        max_bpm = float(bpm_range.get("max_bpm", 135.0))
    else:
        min_bpm, max_bpm = 110.0, 135.0

    matches: list[dict] = []

    for track in tracks:
        track_bpm = float(track.get("bpm", 0.0))
        track_key = str(track.get("key_label", ""))
        track_camelot = key_to_camelot(track_key)

        # BPM範囲フィルタ（±5%のマージン）
        if track_bpm > 0 and not (min_bpm * 0.95 <= track_bpm <= max_bpm * 1.05):
            continue

        # Camelot 相性スコア
        score = 0.0
        if vocal_camelot and track_camelot:
            score = compatibility_score(vocal_camelot, track_camelot)
        elif track_camelot in compat_codes:
            score = 0.5

        if score <= 0.0:
            continue

        suggestion = _build_sampling_suggestion(vocal_analysis, track, score)

        matches.append({
            "track": track,
            "compatibility": score,
            "suggestion": suggestion,
        })

    # 相性スコア降順でソート
    matches.sort(key=lambda x: x["compatibility"], reverse=True)

    # ランク付け
    ranked = [
        {"rank": i + 1, **m}
        for i, m in enumerate(matches[:20])
    ]

    return {
        "matches": ranked,
        "total_searched": len(tracks),
        "vocal_key": vocal_analysis.get("key"),
        "vocal_camelot": vocal_camelot,
    }


def _build_sampling_suggestion(
    vocal_analysis: dict,
    track: dict,
    score: float,
) -> str:
    """サンプリング提案文を生成する。"""
    vocal_key = vocal_analysis.get("key", "?")
    track_title = track.get("title", track.get("file_path", "Unknown"))
    track_key = track.get("key_label", "?")
    track_bpm = track.get("bpm", "?")

    if score >= 1.0:
        compat = "完全一致"
    elif score >= 0.9:
        compat = "relative major/minor"
    elif score >= 0.7:
        compat = "Camelot 隣接"
    else:
        compat = "相性あり"

    sections = vocal_analysis.get("usable_sections", [])
    hook_sections = [
        s for s in sections
        if s.get("has_clear_vocal")
        and any(kw in s.get("type", "").lower() for kw in ("hook", "chorus", "サビ"))
    ]
    if hook_sections:
        best_sec = hook_sections[0]
        sec_text = f"サビ区間 ({best_sec['start']}s〜{best_sec['end']}s)"
    elif sections:
        best_sec = max(sections, key=lambda s: s.get("energy", 0))
        sec_text = f"{best_sec['type']} ({best_sec['start']}s〜{best_sec['end']}s)"
    else:
        sec_text = "全体"

    return (
        f"ボーカル({vocal_key}) × {track_title} ({track_key}, {track_bpm}BPM) — {compat}。"
        f"{sec_text}をループして乗せると効果的。"
    )
