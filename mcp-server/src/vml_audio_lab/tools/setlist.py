"""セットリストビルダー — N曲から最適なDJ再生順を生成する.

Camelot キー互換性、BPM 進行、エネルギーフロー、ジャンル一貫性の
4軸複合スコアで貪欲法による順序最適化を行う。
"""

from __future__ import annotations

import io
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# --- ジャンル互換性マトリクス ---
# 同ジャンル = 1.0 (デフォルト)
_GENRE_COMPATIBILITY: dict[frozenset[str], float] = {
    frozenset({"house", "techno"}): 0.8,
    frozenset({"house", "melodic"}): 0.9,
    frozenset({"house", "uk-garage"}): 0.7,
    frozenset({"house", "electronic"}): 0.8,
    frozenset({"techno", "melodic"}): 0.8,
    frozenset({"techno", "electronic"}): 0.9,
    frozenset({"melodic", "trance"}): 0.7,
    frozenset({"hiphop", "rnb"}): 0.9,
    frozenset({"jpop", "rnb"}): 0.6,
    frozenset({"jpop", "hiphop"}): 0.5,
    frozenset({"dnb", "electronic"}): 0.6,
    frozenset({"classical", "techno"}): 0.1,
    frozenset({"classical", "hiphop"}): 0.1,
    frozenset({"classical", "house"}): 0.1,
}

# デフォルト重み
_DEFAULT_WEIGHTS: dict[str, float] = {
    "w_key": 0.35,
    "w_bpm": 0.25,
    "w_energy": 0.20,
    "w_genre": 0.20,
}


def _get_genre(track: dict) -> str:
    """トラックからジャンルを取得する。"""
    return track.get("genre_group") or track.get("genre", "unknown")


def _get_energy(track: dict) -> float:
    """トラックからエネルギーを取得する。"""
    return float(track.get("energy_level") or track.get("energy", 0.5))


def _get_camelot(track: dict) -> str:
    """トラックからCamelotコードを取得する。"""
    if track.get("camelot"):
        return track["camelot"]
    key_label = track.get("key_label") or track.get("key", "")
    if key_label:
        from vml_audio_lab.tools.camelot import key_to_camelot

        return key_to_camelot(key_label) or ""
    return ""


def _key_score(track_a: dict, track_b: dict) -> float:
    """Camelot キー互換性スコア (0.0-1.0)。"""
    from vml_audio_lab.tools.camelot import compatibility_score

    ca = _get_camelot(track_a)
    cb = _get_camelot(track_b)
    if not ca and not cb:
        return 0.3  # 両方キー情報なしはペナルティ小
    if not ca or not cb:
        return 0.3  # 片方なしもペナルティ小
    return compatibility_score(ca, cb)


def _bpm_score(track_a: dict, track_b: dict) -> float:
    """BPM 差スコア (0.0-1.0)。差20 BPM以上で0。"""
    bpm_a = float(track_a.get("bpm", 0))
    bpm_b = float(track_b.get("bpm", 0))
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5
    return max(0.0, 1.0 - abs(bpm_a - bpm_b) / 20.0)


def _energy_flow_score(
    track_a: dict,
    track_b: dict,
    position_ratio: float,
    strategy: str,
) -> float:
    """エネルギーフロースコア (0.0-1.0)。

    position_ratio: 0.0 (セット開始) ～ 1.0 (セット終了)
    strategy: "build" | "peak" | "wave"
    """
    ea = _get_energy(track_a)
    eb = _get_energy(track_b)
    diff = eb - ea  # 正 = エネルギー上昇

    if strategy == "build":
        # 前半60%: 上昇を報酬、下降をペナルティ
        if position_ratio < 0.6:
            if diff >= 0:
                return min(1.0, 0.7 + diff)
            return max(0.0, 0.5 + diff * 2)  # 下降は強くペナルティ
        # 後半40%: 維持か緩やかな下降はOK
        if diff >= -0.15:
            return 0.7
        return max(0.0, 0.5 + diff)

    if strategy == "peak":
        # 最初からハイエネルギー。下降をペナルティ
        if diff >= 0:
            return 0.8
        return max(0.0, 0.6 + diff * 2)

    if strategy == "wave":
        # 上下動を許容。急激な変化をペナルティ
        return max(0.0, 1.0 - abs(diff) * 2)

    return 0.5  # fallback


def _genre_coherence_score(track_a: dict, track_b: dict) -> float:
    """ジャンル一貫性スコア (0.0-1.0)。"""
    ga = _get_genre(track_a)
    gb = _get_genre(track_b)

    if ga == gb:
        return 1.0
    if ga == "unknown" or gb == "unknown":
        return 0.5

    key = frozenset({ga, gb})
    return _GENRE_COMPATIBILITY.get(key, 0.3)


def _compute_transition_score(
    track_a: dict,
    track_b: dict,
    position_ratio: float,
    energy_strategy: str,
    weights: dict[str, float],
) -> float:
    """2トラック間の遷移スコアを計算する。"""
    ks = _key_score(track_a, track_b)
    bs = _bpm_score(track_a, track_b)
    es = _energy_flow_score(track_a, track_b, position_ratio, energy_strategy)
    gs = _genre_coherence_score(track_a, track_b)

    return (
        weights["w_key"] * ks
        + weights["w_bpm"] * bs
        + weights["w_energy"] * es
        + weights["w_genre"] * gs
    )


def _greedy_order(
    tracks: list[dict],
    energy_strategy: str,
    weights: dict[str, float],
) -> tuple[list[dict], float]:
    """貪欲法で最適順序を構築する。

    各トラックを開始点として試し、最高スコアの順序を返す。

    Returns:
        (ordered_tracks, total_score)
    """
    n = len(tracks)
    if n <= 1:
        return list(tracks), 0.0

    best_order: list[dict] = []
    best_score = -1.0

    for start_idx in range(n):
        order = [tracks[start_idx]]
        remaining = list(tracks[:start_idx]) + list(tracks[start_idx + 1 :])
        total_score = 0.0

        for step in range(n - 1):
            pos_ratio = step / max(1, n - 2)
            best_next_idx = 0
            best_next_score = -1.0

            for j, candidate in enumerate(remaining):
                score = _compute_transition_score(
                    order[-1], candidate, pos_ratio, energy_strategy, weights
                )
                if score > best_next_score:
                    best_next_score = score
                    best_next_idx = j

            total_score += best_next_score
            order.append(remaining.pop(best_next_idx))

        if total_score > best_score:
            best_score = total_score
            best_order = order

    return best_order, best_score


def build_setlist(
    tracks: list[dict],
    energy_curve: str = "build",
    weights: dict[str, float] | None = None,
) -> dict:
    """N曲からDJセットの最適再生順を生成する。

    Args:
        tracks: 分析済みトラック辞書のリスト。各トラックに以下が必要:
            - title (str): トラック名
            - bpm (float): BPM
            - key_label or key (str): キーラベル
            - camelot (str, optional): Camelot コード
            - energy_level or energy (float, optional): エネルギーレベル 0-1
            - genre or genre_group (str, optional): ジャンルスラグ
            - mood (str, optional): ムード名
            - sections (list[dict], optional): セクション構造
        energy_curve: エネルギーカーブ戦略
            - "build": 低→高（クラブ標準）
            - "peak": 最初から高テンション
            - "wave": 上下を繰り返す
        weights: 重みのカスタム (w_key, w_bpm, w_energy, w_genre)

    Returns:
        dict:
            - ordered_tracks: 最適順のトラックリスト
            - transitions: 隣接トラック間のトランジション情報
            - energy_flow: 各ポジションのエネルギー値リスト
            - weak_points: スコアが低いトランジション箇所
            - overall_score: セット全体の平均トランジションスコア
            - stats: 全体の統計
    """
    from vml_audio_lab.utils.dj_context import (
        classify_energy_phase,
        recommend_transition_type,
    )
    from vml_audio_lab.utils.teaching import (
        explain_bpm_transition,
        explain_energy_flow,
        explain_key_transition,
    )

    if not tracks:
        return {
            "ordered_tracks": [],
            "transitions": [],
            "energy_flow": [],
            "weak_points": [],
            "overall_score": 0.0,
            "stats": {},
        }

    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # 貪欲法で最適順序を構築
    ordered, total_score = _greedy_order(tracks, energy_curve, w)
    n = len(ordered)

    # トランジション情報を生成
    transitions: list[dict] = []
    weak_points: list[dict] = []
    energy_flow: list[float] = []

    for i, track in enumerate(ordered):
        energy = _get_energy(track)
        energy_flow.append(energy)
        phase = classify_energy_phase(i, n, energy)
        track["_position"] = i
        track["_phase"] = phase

    for i in range(n - 1):
        a = ordered[i]
        b = ordered[i + 1]
        pos_ratio = i / max(1, n - 2)

        score = _compute_transition_score(a, b, pos_ratio, energy_curve, w)
        transition_type = recommend_transition_type(a, b)

        ca = _get_camelot(a)
        cb = _get_camelot(b)
        key_a = a.get("key_label") or a.get("key", "")
        key_b = b.get("key_label") or b.get("key", "")
        bpm_a = float(a.get("bpm", 0))
        bpm_b = float(b.get("bpm", 0))

        explanation_parts: list[str] = []
        explanation_parts.append(
            explain_key_transition(key_a, ca, key_b, cb, _key_score(a, b))
        )
        explanation_parts.append(explain_bpm_transition(bpm_a, bpm_b))

        transition = {
            "from": a.get("title", f"Track {i + 1}"),
            "to": b.get("title", f"Track {i + 2}"),
            "score": round(score, 3),
            "key_score": round(_key_score(a, b), 2),
            "bpm_diff": round(abs(bpm_a - bpm_b), 1),
            "transition_type": transition_type,
            "explanation": " ".join(explanation_parts),
        }
        transitions.append(transition)

        if score < 0.5:
            weak_point = {
                "position": i,
                "from": a.get("title", f"Track {i + 1}"),
                "to": b.get("title", f"Track {i + 2}"),
                "score": round(score, 3),
                "reason": _diagnose_weakness(a, b),
            }
            weak_points.append(weak_point)

    avg_score = total_score / max(1, n - 1)

    # エネルギーフロー解説
    track_names = [t.get("title", f"Track {i + 1}") for i, t in enumerate(ordered)]
    flow_explanation = explain_energy_flow(energy_flow, track_names)

    # 統計
    bpms = [float(t.get("bpm", 0)) for t in ordered if float(t.get("bpm", 0)) > 0]
    genres = [_get_genre(t) for t in ordered]
    genre_dist = {}
    for g in genres:
        genre_dist[g] = genre_dist.get(g, 0) + 1

    return {
        "ordered_tracks": ordered,
        "transitions": transitions,
        "energy_flow": energy_flow,
        "energy_flow_explanation": flow_explanation,
        "weak_points": weak_points,
        "overall_score": round(avg_score, 3),
        "stats": {
            "total_tracks": n,
            "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else 0,
            "bpm_range": f"{min(bpms):.0f}-{max(bpms):.0f}" if bpms else "N/A",
            "genre_distribution": genre_dist,
        },
    }


def _diagnose_weakness(track_a: dict, track_b: dict) -> str:
    """弱いトランジションの原因を診断する。"""
    reasons: list[str] = []

    ks = _key_score(track_a, track_b)
    if ks < 0.5:
        reasons.append("キーが離れている")

    bpm_a = float(track_a.get("bpm", 0))
    bpm_b = float(track_b.get("bpm", 0))
    if abs(bpm_a - bpm_b) > 8:
        reasons.append(f"BPM差が{abs(bpm_a - bpm_b):.0f}ある")

    ga = _get_genre(track_a)
    gb = _get_genre(track_b)
    if ga != gb:
        pair = frozenset({ga, gb})
        if _GENRE_COMPATIBILITY.get(pair, 0.3) < 0.5:
            reasons.append(f"ジャンル({ga}→{gb})の相性が低い")

    return "。".join(reasons) if reasons else "複合的なスコア低下"


def visualize_setlist_energy(ordered_tracks: list[dict]) -> bytes:
    """セットリストのエネルギーフロー画像を生成する。

    Args:
        ordered_tracks: build_setlist で返された ordered_tracks

    Returns:
        PNG 画像の bytes
    """
    if not ordered_tracks:
        return b""

    n = len(ordered_tracks)
    energies = [_get_energy(t) for t in ordered_tracks]
    titles = [t.get("title", f"#{i + 1}") for i, t in enumerate(ordered_tracks)]
    bpms = [float(t.get("bpm", 0)) for t in ordered_tracks]
    genres = [_get_genre(t) for t in ordered_tracks]

    # ジャンルごとの色マッピング
    genre_colors: dict[str, str] = {
        "house": "#1DB954",
        "techno": "#E74C3C",
        "uk-garage": "#F39C12",
        "melodic": "#9B59B6",
        "electronic": "#3498DB",
        "hiphop": "#E67E22",
        "rnb": "#E91E63",
        "dnb": "#2ECC71",
        "trance": "#00BCD4",
        "jpop": "#FF69B4",
    }
    default_color = "#95A5A6"
    colors = [genre_colors.get(g, default_color) for g in genres]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.2), 8), height_ratios=[3, 1])
    fig.patch.set_facecolor("#1a1a2e")

    # エネルギーカーブ
    x = np.arange(n)
    ax1.bar(x, energies, color=colors, alpha=0.8, width=0.7, edgecolor="white", linewidth=0.5)
    ax1.plot(x, energies, color="#00d4ff", linewidth=2, marker="o", markersize=6, zorder=5)

    # フェーズ背景
    if n >= 4:
        ax1.axvspan(-0.5, n * 0.25, alpha=0.08, color="#3498DB", label="Warm Up")
        ax1.axvspan(n * 0.25, n * 0.6, alpha=0.08, color="#F39C12", label="Build Up")
        ax1.axvspan(n * 0.6, n * 0.85, alpha=0.08, color="#E74C3C", label="Peak")
        ax1.axvspan(n * 0.85, n - 0.5, alpha=0.08, color="#9B59B6", label="Cool Down")

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{i + 1}. {t[:20]}" for i, t in enumerate(titles)],
        rotation=45,
        ha="right",
        fontsize=8,
        color="white",
    )
    ax1.set_ylabel("Energy Level", color="white", fontsize=12)
    ax1.set_title("Setlist Energy Flow", color="white", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("white")
    ax1.spines["left"].set_color("white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if n >= 4:
        ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    # BPMカーブ
    valid_bpms = [(i, b) for i, b in enumerate(bpms) if b > 0]
    if valid_bpms:
        bx, by = zip(*valid_bpms)
        ax2.plot(bx, by, color="#ff6b6b", linewidth=2, marker="s", markersize=5)
        ax2.fill_between(bx, by, alpha=0.2, color="#ff6b6b")
    ax2.set_xticks(x)
    ax2.set_xticklabels([""] * n)
    ax2.set_ylabel("BPM", color="white", fontsize=10)
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("white")
    ax2.spines["left"].set_color("white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
