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

    # A/B/C/D always present
    cue_names = [c["name"] for c in result["hot_cues"]]
    assert cue_names == ["A", "B", "C", "D"]

    # B should be first Drop
    b_cue = next(c for c in result["hot_cues"] if c["name"] == "B")
    assert b_cue["time_sec"] == 45.0


def test_generate_dj_cues_handles_empty_sections() -> None:
    result = generate_dj_cues_from_sections([], duration_sec=0.0)
    assert result["hot_cues"] == []
    assert result["memory_cues"] == []
    assert len(result["notes"]) > 0


def test_generate_dj_cues_fallback_still_returns_abcd() -> None:
    sections = [
        {"start": 0.0, "end": 30.0, "label": "Intro", "energy": 0.2},
        {"start": 30.0, "end": 60.0, "label": "Build", "energy": 0.4},
        {"start": 60.0, "end": 90.0, "label": "Outro", "energy": 0.2},
    ]
    result = generate_dj_cues_from_sections(sections, duration_sec=90.0)
    names = [c["name"] for c in result["hot_cues"]]
    assert names == ["A", "B", "C", "D"]
    assert all(0.0 <= c["time_sec"] <= 90.0 for c in result["hot_cues"])


# --- R&B ジャンル対応テスト ---


def test_generate_dj_cues_rnb_uses_chorus_labels() -> None:
    """R&B ジャンルで Chorus/Verse/Bridge ベースのキューが生成される。"""
    sections = [
        {"start": 0.0, "end": 20.0, "label": "Intro", "energy": 0.3},
        {"start": 20.0, "end": 50.0, "label": "Verse", "energy": 0.5},
        {"start": 50.0, "end": 90.0, "label": "Chorus", "energy": 0.9},
        {"start": 90.0, "end": 120.0, "label": "Bridge", "energy": 0.3},
        {"start": 120.0, "end": 160.0, "label": "Chorus", "energy": 0.85},
        {"start": 160.0, "end": 190.0, "label": "Outro", "energy": 0.2},
    ]
    result = generate_dj_cues_from_sections(sections, duration_sec=190.0, genre_group="rnb")

    # A/B/C/D は常に返る
    cue_names = [c["name"] for c in result["hot_cues"]]
    assert cue_names == ["A", "B", "C", "D"]

    # B は最初の Chorus (peak)
    b_cue = next(c for c in result["hot_cues"] if c["name"] == "B")
    assert b_cue["time_sec"] == 50.0

    # Memory cues に Chorus/Bridge ラベルが含まれる
    mem_labels = [c["name"] for c in result["memory_cues"]]
    assert any("Chorus" in label for label in mem_labels)

    # notes にも Chorus が使われている
    assert any("Chorus" in note for note in result["notes"])


def test_generate_dj_cues_rnb_a_is_verse() -> None:
    """R&B の A キュー（ミックスイン）は Verse を優先する。"""
    sections = [
        {"start": 0.0, "end": 15.0, "label": "Intro", "energy": 0.2},
        {"start": 15.0, "end": 45.0, "label": "Verse", "energy": 0.5},
        {"start": 45.0, "end": 80.0, "label": "Chorus", "energy": 0.9},
        {"start": 80.0, "end": 100.0, "label": "Outro", "energy": 0.2},
    ]
    result = generate_dj_cues_from_sections(sections, duration_sec=100.0, genre_group="rnb")
    a_cue = next(c for c in result["hot_cues"] if c["name"] == "A")
    assert a_cue["time_sec"] == 15.0  # Verse 開始


def test_generate_dj_cues_hiphop_uses_hook_labels() -> None:
    """Hip-Hop ジャンルで Hook/Verse/Bridge ベースのキューが生成される。"""
    sections = [
        {"start": 0.0, "end": 20.0, "label": "Intro", "energy": 0.3},
        {"start": 20.0, "end": 50.0, "label": "Verse", "energy": 0.5},
        {"start": 50.0, "end": 80.0, "label": "Hook", "energy": 0.9},
        {"start": 80.0, "end": 100.0, "label": "Outro", "energy": 0.2},
    ]
    result = generate_dj_cues_from_sections(sections, duration_sec=100.0, genre_group="hiphop")
    b_cue = next(c for c in result["hot_cues"] if c["name"] == "B")
    assert b_cue["time_sec"] == 50.0  # Hook 開始


def test_generate_dj_cues_default_backward_compatible() -> None:
    """genre_group 未指定（デフォルト）で既存の Drop/Build/Break ロジックが維持される。"""
    sections = [
        {"start": 0.0, "end": 20.0, "label": "Intro", "energy": 0.5},
        {"start": 20.0, "end": 45.0, "label": "Build", "energy": 0.4},
        {"start": 45.0, "end": 95.0, "label": "Drop", "energy": 1.0},
        {"start": 95.0, "end": 120.0, "label": "Break", "energy": 0.3},
        {"start": 120.0, "end": 180.0, "label": "Drop", "energy": 0.9},
        {"start": 180.0, "end": 210.0, "label": "Outro", "energy": 0.2},
    ]
    result = generate_dj_cues_from_sections(sections, duration_sec=210.0)
    b_cue = next(c for c in result["hot_cues"] if c["name"] == "B")
    assert b_cue["time_sec"] == 45.0  # Drop 開始（従来通り）
