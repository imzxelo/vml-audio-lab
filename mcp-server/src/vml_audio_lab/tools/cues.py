"""DJ向けキューポイント生成ツール"""

from __future__ import annotations

from typing import Any

from vml_audio_lab.tools.structure import detect_structure


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


def generate_dj_cues_from_sections(sections: list[dict[str, Any]], duration_sec: float) -> dict[str, Any]:
    """構造情報からDJ用キューを生成する。A/B/C/Dは常に返す。"""
    if not sections:
        return {
            "hot_cues": [],
            "memory_cues": [],
            "notes": ["sections が空のためキューを生成できません"],
        }

    intro = sections[0]
    build1 = _pick_first(sections, "Build")
    drop1 = _pick_first(sections, "Drop")
    break1 = _pick_first(sections, "Break", after=float(drop1["start"])) if drop1 else _pick_first(sections, "Break")
    drop2 = _pick_first(sections, "Drop", after=float(break1["start"])) if break1 else _pick_first(
        sections, "Drop", after=float(drop1["start"]) if drop1 else -1
    )
    outro = sections[-1]

    # A: ミックスイン推奨（最初のBuild開始 / なければDrop開始 / Intro開始）
    a_time = float((build1 or drop1 or intro)["start"])

    # B: 1st drop（なければA+16秒）
    b_time = float(drop1["start"]) if drop1 else min(float(duration_sec), a_time + 16.0)

    # D: 2nd drop（なければOutro手前）
    d_time = float(drop2["start"]) if drop2 else max(0.0, float(duration_sec) - 32.0)

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

    # Memory cues
    memory_cues: list[dict[str, Any]] = [_to_cue("Intro", float(intro["start"]), kind="memory")]

    boundary_candidates: list[tuple[str, float]] = []
    if build1:
        boundary_candidates.append(("Build", float(build1["start"])))
    if drop1:
        boundary_candidates.append(("Drop1", float(drop1["start"])))
    if break1:
        boundary_candidates.append(("Break", float(break1["start"])))
    if drop2:
        boundary_candidates.append(("Drop2", float(drop2["start"])))
    boundary_candidates.append(("Outro", float(outro["start"])))

    seen = {round(float(intro["start"]), 2)}
    for label, sec in boundary_candidates:
        key = round(sec, 2)
        if key in seen:
            continue
        seen.add(key)
        memory_cues.append(_to_cue(label, sec, kind="memory"))

    notes = [
        "A=ミックスイン開始候補, B=1st Drop, C=Break/中間遷移, D=2nd Drop/終盤移行",
        "Rekordboxへは time_sec をそのままホットキューに入力",
    ]

    return {
        "hot_cues": hot_cues,
        "memory_cues": memory_cues,
        "notes": notes,
    }


def recommend_cues(y_path: str, n_segments: int | None = None) -> dict[str, Any]:
    """y_pathから構造を推定し、DJ向けキュー案を返す。"""
    structure = detect_structure(y_path, n_segments=n_segments)
    sections = structure.get("sections", [])
    duration_sec = float(structure.get("duration_sec", 0.0))

    cue_pack = generate_dj_cues_from_sections(sections, duration_sec)
    return {
        "duration_sec": duration_sec,
        "n_segments": structure.get("n_segments", 0),
        "hot_cues": cue_pack["hot_cues"],
        "memory_cues": cue_pack["memory_cues"],
        "notes": cue_pack["notes"],
        "sections": sections,
    }
