"""dj_context.py のテスト."""

import pytest

from vml_audio_lab.utils.dj_context import (
    classify_energy_phase,
    genre_mixing_tips,
    recommend_transition_type,
)


def _track(
    title: str = "Track",
    key: str = "Am",
    camelot: str = "8A",
    bpm: float = 124.0,
    energy_level: float = 0.5,
    genre_group: str = "house",
) -> dict:
    return {
        "title": title,
        "key_label": key,
        "camelot": camelot,
        "bpm": bpm,
        "energy_level": energy_level,
        "genre_group": genre_group,
    }


class TestRecommendTransitionType:
    def test_same_genre_house(self):
        result = recommend_transition_type(
            _track(genre_group="house"),
            _track(genre_group="house"),
        )
        assert result["type"] in ("eq_swap", "beat_match_blend")
        assert "bars" in result
        assert "description_ja" in result
        assert "difficulty" in result

    def test_hiphop_to_rnb_echo_out(self):
        result = recommend_transition_type(
            _track(genre_group="hiphop", bpm=90),
            _track(genre_group="rnb", bpm=85),
        )
        assert result["type"] == "echo_out"

    def test_large_bpm_diff_echo_out(self):
        result = recommend_transition_type(
            _track(bpm=80),
            _track(bpm=128),
        )
        assert result["type"] == "echo_out"

    def test_different_genre_filter_sweep(self):
        result = recommend_transition_type(
            _track(genre_group="house", camelot="4A"),
            _track(genre_group="melodic", camelot="9A"),
        )
        assert result["type"] == "filter_sweep"

    def test_compatible_keys_same_bpm_blend(self):
        result = recommend_transition_type(
            _track(genre_group="house", camelot="8A", bpm=124),
            _track(genre_group="house", camelot="8A", bpm=124),
        )
        assert result["type"] == "beat_match_blend"

    def test_returns_required_fields(self):
        result = recommend_transition_type(_track(), _track())
        assert set(result.keys()) >= {"type", "bars", "description_ja", "difficulty"}


class TestClassifyEnergyPhase:
    def test_first_position_warm_up(self):
        result = classify_energy_phase(0, 10, 0.3)
        assert result["phase"] == "warm_up"

    def test_middle_build_up(self):
        result = classify_energy_phase(3, 10, 0.5)
        assert result["phase"] == "build_up"

    def test_late_position_peak(self):
        result = classify_energy_phase(7, 10, 0.8)
        assert result["phase"] == "peak"

    def test_last_position_cool_down(self):
        result = classify_energy_phase(9, 10, 0.3)
        assert result["phase"] == "cool_down"

    def test_returns_required_fields(self):
        result = classify_energy_phase(0, 5, 0.5)
        assert set(result.keys()) >= {"phase", "label", "description_ja", "tips"}
        assert isinstance(result["tips"], list)

    def test_single_track(self):
        result = classify_energy_phase(0, 1, 0.5)
        assert "phase" in result


class TestGenreMixingTips:
    def test_same_genre_tips(self):
        tips = genre_mixing_tips("house", "house")
        assert isinstance(tips, list)
        assert len(tips) > 0
        assert any("EQ" in t or "ブレンド" in t for t in tips)

    def test_house_techno_specific_tips(self):
        tips = genre_mixing_tips("house", "techno")
        assert isinstance(tips, list)
        assert len(tips) > 0

    def test_unknown_pair_default_tips(self):
        tips = genre_mixing_tips("classical", "dnb")
        assert isinstance(tips, list)
        assert len(tips) > 0

    def test_symmetric(self):
        tips_ab = genre_mixing_tips("house", "techno")
        tips_ba = genre_mixing_tips("techno", "house")
        # frozenset makes them symmetric
        assert tips_ab == tips_ba
