"""ムード判定ツールのテスト."""

from __future__ import annotations

import numpy as np

from vml_audio_lab.tools.mood import _matches_condition, detect_mood


def _make_audio(
    *,
    sr: int = 22050,
    duration: float = 2.0,
    amplitude: float = 0.1,
    freq: float = 440.0,
) -> np.ndarray:
    """テスト用サイン波を生成する。"""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMatchesCondition:
    """_matches_condition: 各条件チェックのユニットテスト."""

    def test_energy_min_passes_when_above(self) -> None:
        assert _matches_condition({"energy_min": 0.5}, 0.7, 130.0, "major", 0.4)

    def test_energy_min_fails_when_below(self) -> None:
        assert not _matches_condition({"energy_min": 0.5}, 0.3, 130.0, "major", 0.4)

    def test_energy_max_passes_when_below(self) -> None:
        assert _matches_condition({"energy_max": 0.5}, 0.4, 120.0, "minor", 0.3)

    def test_energy_max_fails_when_above(self) -> None:
        assert not _matches_condition({"energy_max": 0.5}, 0.6, 120.0, "minor", 0.3)

    def test_energy_range_passes_inside(self) -> None:
        assert _matches_condition({"energy_range": (0.3, 0.7)}, 0.5, 120.0, "major", 0.3)

    def test_energy_range_fails_below(self) -> None:
        assert not _matches_condition({"energy_range": (0.3, 0.7)}, 0.2, 120.0, "major", 0.3)

    def test_energy_range_fails_above(self) -> None:
        assert not _matches_condition({"energy_range": (0.3, 0.7)}, 0.8, 120.0, "major", 0.3)

    def test_bpm_min_passes(self) -> None:
        assert _matches_condition({"bpm_min": 128.0}, 0.7, 130.0, "major", 0.3)

    def test_bpm_min_fails(self) -> None:
        assert not _matches_condition({"bpm_min": 128.0}, 0.7, 125.0, "major", 0.3)

    def test_bpm_max_passes(self) -> None:
        assert _matches_condition({"bpm_max": 125.0}, 0.4, 120.0, "minor", 0.3)

    def test_bpm_max_fails(self) -> None:
        assert not _matches_condition({"bpm_max": 125.0}, 0.4, 130.0, "minor", 0.3)

    def test_bpm_range_passes(self) -> None:
        assert _matches_condition({"bpm_range": (118.0, 132.0)}, 0.5, 125.0, "major", 0.3)

    def test_bpm_range_fails_outside(self) -> None:
        assert not _matches_condition({"bpm_range": (118.0, 132.0)}, 0.5, 115.0, "major", 0.3)

    def test_scale_minor_passes(self) -> None:
        assert _matches_condition({"scale": "minor"}, 0.4, 120.0, "minor", 0.3)

    def test_scale_minor_fails_for_major(self) -> None:
        assert not _matches_condition({"scale": "minor"}, 0.4, 120.0, "major", 0.3)

    def test_scale_none_matches_any(self) -> None:
        assert _matches_condition({"scale": None}, 0.7, 130.0, "major", 0.3)
        assert _matches_condition({"scale": None}, 0.7, 130.0, "minor", 0.3)

    def test_melodic_min_passes(self) -> None:
        assert _matches_condition({"melodic_min": 0.35}, 0.5, 120.0, "minor", 0.5)

    def test_melodic_min_fails(self) -> None:
        assert not _matches_condition({"melodic_min": 0.35}, 0.5, 120.0, "minor", 0.2)

    def test_combined_conditions_all_must_match(self) -> None:
        cond = {"energy_min": 0.65, "bpm_min": 128.0}
        assert _matches_condition(cond, 0.7, 130.0, "major", 0.3)
        assert not _matches_condition(cond, 0.7, 125.0, "major", 0.3)  # bpm_min fails
        assert not _matches_condition(cond, 0.5, 130.0, "major", 0.3)  # energy_min fails


class TestDetectMood:
    """detect_mood: ムード判定統合テスト."""

    def test_peak_time_high_energy_high_bpm(self) -> None:
        """高エネルギー・高BPMは Peak Time."""
        # energy_min=0.65 のため、RMS/0.3 >= 0.65 → RMS >= 0.195 が必要
        # sine wave: RMS = amplitude / sqrt(2), amplitude=0.30 → RMS ≈ 0.212
        y = _make_audio(amplitude=0.30, duration=2.0)
        result = detect_mood(y=y, sr=22050, bpm=132.0, scale="major")
        assert result["mood"] == "Peak Time"

    def test_returns_required_keys(self) -> None:
        """必須キーが含まれる."""
        y = _make_audio(amplitude=0.1)
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        for key in ("mood", "description", "energy_level", "melodic_score"):
            assert key in result, f"Missing key: {key}"

    def test_energy_level_in_range(self) -> None:
        """energy_level は 0.0〜1.0 の範囲。"""
        y = _make_audio(amplitude=0.1)
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert 0.0 <= result["energy_level"] <= 1.0

    def test_melodic_score_in_range(self) -> None:
        """melodic_score は 0.0〜1.0 の範囲。"""
        y = _make_audio(amplitude=0.1)
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert 0.0 <= result["melodic_score"] <= 1.0

    def test_chill_mellow_low_energy_major(self) -> None:
        """低エネルギー・メジャーキーは Chill & Mellow."""
        # 非常に低い振幅でエネルギーを低くする
        y = _make_audio(amplitude=0.01, duration=2.0)
        result = detect_mood(y=y, sr=22050, bpm=100.0, scale="major")
        assert result["mood"] == "Chill & Mellow"

    def test_deep_night_low_energy_minor_low_bpm(self) -> None:
        """低エネルギー・マイナーキー・低BPMは Deep Night."""
        y = _make_audio(amplitude=0.01, duration=2.0)
        result = detect_mood(y=y, sr=22050, bpm=115.0, scale="minor")
        assert result["mood"] == "Deep Night"

    def test_mood_is_string(self) -> None:
        """mood は文字列型。"""
        y = _make_audio()
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert isinstance(result["mood"], str)
        assert len(result["mood"]) > 0

    def test_description_is_string(self) -> None:
        """description は文字列型。"""
        y = _make_audio()
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert isinstance(result["description"], str)

    def test_mood_values_are_valid(self) -> None:
        """mood は定義済みカテゴリのいずれか。"""
        valid_moods = {"Peak Time", "Deep Night", "Melodic Journey", "Groovy & Warm", "Chill & Mellow"}
        y = _make_audio(amplitude=0.1)

        for bpm, scale, amplitude in [
            (132.0, "major", 0.25),
            (110.0, "minor", 0.01),
            (120.0, "major", 0.01),
            (125.0, "major", 0.1),
        ]:
            y = _make_audio(amplitude=amplitude)
            result = detect_mood(y=y, sr=22050, bpm=bpm, scale=scale)
            assert result["mood"] in valid_moods, f"Unexpected mood: {result['mood']}"

    def test_silent_audio_does_not_raise(self) -> None:
        """無音でも例外を出さない."""
        y = np.zeros(22050, dtype=np.float32)
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert "mood" in result
        assert result["energy_level"] == 0.0

    def test_short_audio_does_not_raise(self) -> None:
        """短い音声でも例外を出さない."""
        y = _make_audio(duration=0.1, amplitude=0.1)
        result = detect_mood(y=y, sr=22050, bpm=120.0, scale="major")
        assert "mood" in result
