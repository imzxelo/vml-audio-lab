"""DJ向けキューポイント生成ツール"""

from __future__ import annotations

from typing import Any

from vml_audio_lab.tools.structure import detect_structure

# ジャンルグループ別のラベル→ロール対応
# mix_in: ミックスイン候補セクション
# peak: ピーク（盛り上がり）セクション
# break_: ブレイク（静かなつなぎ）セクション
_LABEL_ROLES: dict[str, dict[str, str]] = {
    "default": {"mix_in": "Build", "peak": "Drop", "break_": "Break"},
    "rnb": {"mix_in": "Verse", "peak": "Chorus", "break_": "Bridge"},
    "hiphop": {"mix_in": "Verse", "peak": "Hook", "break_": "Bridge"},
    "jpop": {"mix_in": "Aメロ", "peak": "サビ", "break_": "落ちサビ"},
    "classical": {"mix_in": "展開", "peak": "再現", "break_": "コーダ"},
}


def _roles_for(genre_group: str) -> dict[str, str]:
    """ジャンルグループに対応するロール辞書を返す。"""
    return _LABEL_ROLES.get(genre_group, _LABEL_ROLES["default"])


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _pick_first(sections: list[dict[str, Any]], label: str, after: float = -1.0) -> dict[str, Any] | None:
    for s in sections:
        if s.get("label") == label and float(s.get("start", 0.0)) > after:
            return s
    return None


def _to_cue(name: str, sec: float, kind: str = "hot") -> dict[str, Any]:
    return {
        "name": name,
        "time_sec": round(float(sec), 2),
        "time_label": _format_time(float(sec)),
        "type": kind,
    }


def generate_dj_cues_from_sections(
    sections: list[dict[str, Any]],
    duration_sec: float,
    genre_group: str = "default",
) -> dict[str, Any]:
    """構造情報からDJ用キューを生成する。A/B/C/Dは常に返す。

    Args:
        sections: detect_structure が返す sections リスト
        duration_sec: 楽曲の長さ（秒）
        genre_group: ジャンルグループ。ラベル検索に使用する。
    """
    if not sections:
        return {
            "hot_cues": [],
            "memory_cues": [],
            "notes": ["sections が空のためキューを生成できません"],
        }

    roles = _roles_for(genre_group)

    intro = sections[0]
    mix_in1 = _pick_first(sections, roles["mix_in"])
    peak1 = _pick_first(sections, roles["peak"])
    break1 = (
        _pick_first(sections, roles["break_"], after=float(peak1["start"]))
        if peak1
        else _pick_first(sections, roles["break_"])
    )
    peak2 = (
        _pick_first(sections, roles["peak"], after=float(break1["start"]))
        if break1
        else _pick_first(sections, roles["peak"], after=float(peak1["start"]) if peak1 else -1)
    )
    outro = sections[-1]

    # A: ミックスイン推奨（最初の mix_in 開始 / なければ peak 開始 / Intro 開始）
    a_time = float((mix_in1 or peak1 or intro)["start"])

    # B: 1st peak（なければA+16秒）
    b_time = float(peak1["start"]) if peak1 else min(float(duration_sec), a_time + 16.0)

    # D: 2nd peak（なければOutro手前）
    d_time = float(peak2["start"]) if peak2 else max(0.0, float(duration_sec) - 32.0)

    # C: break in（なければBとDの中点）
    if break1:
        c_time = float(break1["start"])
    else:
        c_time = max(b_time + 8.0, (b_time + d_time) / 2.0)
        c_time = min(c_time, max(0.0, float(duration_sec) - 8.0))

    hot_cues: list[dict[str, Any]] = [
        _to_cue("A", a_time, kind="hot"),
        _to_cue("B", b_time, kind="hot"),
        _to_cue("C", c_time, kind="hot"),
        _to_cue("D", d_time, kind="hot"),
    ]

    # Memory cues — ジャンル別のラベル名で記録
    memory_cues: list[dict[str, Any]] = [_to_cue("Intro", float(intro["start"]), kind="memory")]

    peak_label = roles["peak"]
    break_label = roles["break_"]
    mix_in_label = roles["mix_in"]

    boundary_candidates: list[tuple[str, float]] = []
    if mix_in1:
        boundary_candidates.append((mix_in_label, float(mix_in1["start"])))
    if peak1:
        boundary_candidates.append((f"{peak_label}1", float(peak1["start"])))
    if break1:
        boundary_candidates.append((break_label, float(break1["start"])))
    if peak2:
        boundary_candidates.append((f"{peak_label}2", float(peak2["start"])))
    boundary_candidates.append(("Outro", float(outro["start"])))

    seen = {round(float(intro["start"]), 2)}
    for label, sec in boundary_candidates:
        key = round(sec, 2)
        if key in seen:
            continue
        seen.add(key)
        memory_cues.append(_to_cue(label, sec, kind="memory"))

    notes = [
        f"A=ミックスイン開始候補, B=1st {peak_label}, C={break_label}/中間遷移, D=2nd {peak_label}/終盤移行",
        "Rekordboxへは time_sec をそのままホットキューに入力",
    ]

    return {
        "hot_cues": hot_cues,
        "memory_cues": memory_cues,
        "notes": notes,
    }


def recommend_cues(
    y_path: str,
    n_segments: int | None = None,
    genre: str | None = None,
) -> dict[str, Any]:
    """y_pathから構造を推定し、DJ向けキュー案を返す。

    Args:
        y_path: load_track で返された音声データのパス
        n_segments: セグメント数（省略で自動推定）
        genre: ジャンルスラグ。detect_structure と cue 生成に渡す。
    """
    structure = detect_structure(y_path, n_segments=n_segments, genre=genre)
    sections = structure.get("sections", [])
    duration_sec = float(structure.get("duration_sec", 0.0))
    genre_group = str(structure.get("genre_group", "default"))

    cue_pack = generate_dj_cues_from_sections(sections, duration_sec, genre_group=genre_group)
    return {
        "duration_sec": duration_sec,
        "n_segments": structure.get("n_segments", 0),
        "hot_cues": cue_pack["hot_cues"],
        "memory_cues": cue_pack["memory_cues"],
        "notes": cue_pack["notes"],
        "sections": sections,
    }
