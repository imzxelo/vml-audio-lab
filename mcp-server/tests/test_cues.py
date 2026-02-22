"""DJキュー生成ツールのテスト"""

from __future__ import annotations

from vml_audio_lab.tools.cues import generate_dj_cues_from_sections


def test_generate_dj_cues_from_sections_basic() -> None:
    sections = [
        {"start": 0.0, "end": 20.0, "label": "Intro", "energy": 0.5},
        {"start": 20.0, "end": 45.0, "label": "Build", "energy": 0.4},
        {"start": 45.0, "end": 95.0, "label": "Drop", "energy": 1.0},
        {"start": 95.0, "end": 120.0, "label": "Break", "energy": 0.3},
        {"start": 120.0, "end": 180.0, "label": "Drop", "energy": 0.9},
        {"start": 180.0, "end": 210.0, "label": "Outro", "energy": 0.2},
    ]

    result = generate_dj_cues_from_sections(sections, duration_sec=210.0)

    assert "hot_cues" in result
    assert "memory_cues" in result
    assert len(result["hot_cues"]) >= 4

    # A/B/C/D
    cue_names = [c["name"] for c in result["hot_cues"]]
    assert cue_names[0] == "A"
    assert "B" in cue_names
    assert "C" in cue_names
    assert "D" in cue_names

    # B should be first Drop
    b_cue = next(c for c in result["hot_cues"] if c["name"] == "B")
    assert b_cue["time_sec"] == 45.0


def test_generate_dj_cues_handles_empty_sections() -> None:
    result = generate_dj_cues_from_sections([], duration_sec=0.0)
    assert result["hot_cues"] == []
    assert result["memory_cues"] == []
    assert len(result["notes"]) > 0
