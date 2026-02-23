"""トランジション提案ツール.

2トラック間のベストな繋ぎ方（出点・入点・テクニック）を提案する。
"""

from __future__ import annotations


def _fmt_time(seconds: float) -> str:
    """秒数を M:SS 形式の文字列に変換する。"""
    total_s = int(round(seconds))
    m = total_s // 60
    s = total_s % 60
    return f"{m}:{s:02d}"


def _compute_key_score(camelot_a: str, camelot_b: str) -> float:
    """2つの Camelot コードの相性スコアを返す。"""
    from vml_audio_lab.tools.camelot import compatibility_score

    return compatibility_score(camelot_a, camelot_b) if (camelot_a and camelot_b) else 0.0


def _key_description(camelot_a: str, camelot_b: str, score: float) -> str:
    """キー関係の日本語説明を返す。"""
    if score >= 1.0:
        return f"同キー ({camelot_a})"
    if score >= 0.9:
        return f"relative major/minor ({camelot_a}→{camelot_b})"
    if score >= 0.7:
        return f"Camelot 隣接 ({camelot_a}→{camelot_b})"
    return f"キー不一致 ({camelot_a}→{camelot_b})"


# セクションラベルの優先度 (出点候補: 低エネルギーのセクションを優先)
_OUT_SECTION_PRIORITY: list[str] = [
    "Break", "Outro", "落ちサビ", "Bridge", "Verse",
    "Build", "Hook", "Bメロ", "Aメロ",
    "Intro", "Drop", "サビ", "大サビ",
]

# セクションラベルの優先度 (入点候補: 低エネルギーのセクションを優先)
_IN_SECTION_PRIORITY: list[str] = [
    "Intro", "Build", "Verse", "Aメロ",
    "Break", "Bridge", "Bメロ",
    "Hook", "サビ", "Drop", "大サビ", "落ちサビ", "Outro",
]


def _pick_transition_section(
    sections: list[dict],
    priority_list: list[str],
    is_out: bool = False,
) -> dict | None:
    """優先度リストに従ってトランジションに適したセクションを選ぶ。

    Args:
        sections: detect_structure が返す sections リスト
        priority_list: 優先するセクションラベルのリスト (先頭が最優先)
        is_out: 出点選択の場合 True (マッチしない場合は末尾を選ぶ)

    Returns:
        選択されたセクション辞書、または None (セクションが空の場合)
    """
    if not sections:
        return None

    for label in priority_list:
        for section in sections:
            if section.get("label", "") == label:
                return section

    # マッチしない場合は末尾(出点)か先頭(入点)
    return sections[-1] if is_out else sections[0]


# ジャンルグループ別のミキシングテクニック
_GENRE_TECHNIQUE: dict[str, str] = {
    "house": "EQスワップで16〜32小節ブレンド",
    "techno": "EQスワップで16〜32小節ブレンド",
    "uk-garage": "EQスワップで8〜16小節ブレンド",
    "melodic": "フィルタースウィープで16〜32小節ブレンド",
    "dnb": "EQスワップで8小節ブレンド",
    "trance": "フィルタースウィープで32小節ブレンド",
    "hiphop": "エコーアウトまたはクイックカット",
    "jpop": "カットまたはフィルタースウィープ",
    "electronic": "EQスワップで8〜16小節ブレンド",
    "classical": "フェードアウト/フェードイン",
}

_DEFAULT_TECHNIQUE = "EQスワップでブレンド"


def _select_technique(genre_a: str, genre_b: str) -> str:
    """トラック A と B のジャンルからミキシングテクニックを選ぶ。

    genre_group または genre キーの値を受け付ける。
    """
    return _GENRE_TECHNIQUE.get(genre_a, _DEFAULT_TECHNIQUE)


def suggest_transition(
    track_a: dict,
    track_b: dict,
) -> dict:
    """2トラック間のベストなトランジションを提案する。

    track_a の出点 (Break/Outro) から track_b の入点 (Intro/Build) へ
    繋ぐ方法をキー関係・BPM・エネルギーから判定する。

    Args:
        track_a: 分析済みトラック情報。以下のキーを含む辞書:
            - key_label または key (str): キーラベル (例: "Fm")
            - camelot (str): Camelot コード (例: "4A")
            - bpm (float): BPM
            - genre または genre_group (str): ジャンル
            - sections (list[dict], optional): detect_structure の sections
            - energy または energy_level (float, optional): エネルギーレベル 0〜1
        track_b: 同上

    Returns:
        dict:
            - key_compatibility: キー相性スコア (0.0〜1.0)
            - key_description: キー関係の日本語説明
            - energy_match: エネルギー一致度 (0.0〜1.0)
            - bpm_diff: BPM 差
            - compatibility: 相性ラベル (◎/○/△)
            - suggestion: 日本語の提案説明文 (セクション・テクニック含む)
            - section_suggestion: セクション詳細情報
    """
    from vml_audio_lab.tools.camelot import key_to_camelot

    # --- キー情報の取得 ---
    # key_label または key キーを両方サポート
    key_label_a = track_a.get("key_label") or track_a.get("key", "")
    key_label_b = track_b.get("key_label") or track_b.get("key", "")
    camelot_a = track_a.get("camelot") or key_to_camelot(key_label_a) or ""
    camelot_b = track_b.get("camelot") or key_to_camelot(key_label_b) or ""

    key_score = _compute_key_score(camelot_a, camelot_b)
    key_desc = _key_description(camelot_a, camelot_b, key_score)

    # --- BPM 差 ---
    bpm_a = float(track_a.get("bpm", 0))
    bpm_b = float(track_b.get("bpm", 0))
    bpm_diff = round(abs(bpm_a - bpm_b), 1) if (bpm_a > 0 and bpm_b > 0) else 0.0

    # --- 出点・入点のセクション選択 ---
    sections_a = track_a.get("sections", [])
    sections_b = track_b.get("sections", [])
    out_section = _pick_transition_section(sections_a, _OUT_SECTION_PRIORITY, is_out=True)
    in_section = _pick_transition_section(sections_b, _IN_SECTION_PRIORITY, is_out=False)

    # --- エネルギー一致度 ---
    # energy または energy_level キーの両方をサポート
    energy_a = float(track_a.get("energy") or track_a.get("energy_level") or 0.5)
    energy_b = float(track_b.get("energy") or track_b.get("energy_level") or 0.5)

    # 出点セクションが Break/Outro の場合はエネルギーが下がる傾向を考慮
    if out_section and out_section.get("label") in ("Break", "Outro", "落ちサビ"):
        effective_energy_a = energy_a * 0.7
    else:
        effective_energy_a = energy_a

    energy_diff = abs(effective_energy_a - energy_b)
    energy_match = round(max(0.0, 1.0 - energy_diff * 2), 2)

    # --- ミキシングテクニック ---
    # genre または genre_group キーの両方をサポート
    genre_a = track_a.get("genre_group") or track_a.get("genre", "")
    genre_b = track_b.get("genre_group") or track_b.get("genre", "")
    technique = _select_technique(genre_a, genre_b)

    # --- 出点・入点の時間情報 ---
    if out_section:
        out_time_sec = float(out_section.get("start", 0))
        out_label = str(out_section.get("label", ""))
    else:
        out_time_sec = 0.0
        out_label = ""

    if in_section:
        in_time_sec = float(in_section.get("start", 0))
        in_label = str(in_section.get("label", ""))
    else:
        in_time_sec = 0.0
        in_label = ""

    # --- 相性ラベル ---
    if key_score >= 0.9:
        compat_label = "◎"
    elif key_score >= 0.7:
        compat_label = "○"
    else:
        compat_label = "△"

    # --- suggestion 文の生成 ---
    suggestion_parts: list[str] = [key_desc]

    if bpm_diff <= 3:
        suggestion_parts.append(f"BPM差{bpm_diff}（ほぼ同速）")
    elif bpm_diff <= 10:
        suggestion_parts.append(f"BPM差{bpm_diff}（許容範囲）")
    else:
        suggestion_parts.append(f"BPM差{bpm_diff}（テンポ調整推奨）")

    if out_label:
        out_str = f"TrackAの{out_label}({_fmt_time(out_time_sec)})"
    else:
        out_str = "TrackA"

    if in_label:
        in_str = f"TrackBの{in_label}({_fmt_time(in_time_sec)})"
    else:
        in_str = "TrackB"

    suggestion_parts.append(f"{out_str}から{in_str}へ、{technique}で繋ぐと自然")

    suggestion = "。".join(suggestion_parts)

    return {
        "key_compatibility": key_score,
        "key_description": key_desc,
        "energy_match": energy_match,
        "bpm_diff": bpm_diff,
        "compatibility": compat_label,
        "suggestion": suggestion,
        "section_suggestion": {
            "out_point": {
                "time_sec": out_time_sec,
                "time_label": _fmt_time(out_time_sec),
                "section": out_label,
            },
            "in_point": {
                "time_sec": in_time_sec,
                "time_label": _fmt_time(in_time_sec),
                "section": in_label,
            },
            "technique": technique,
        },
    }
