"""teaching.py のテスト."""

import pytest

from vml_audio_lab.utils.teaching import (
    explain_bpm_transition,
    explain_effects,
    explain_energy_flow,
    explain_genre_compatibility,
    explain_key_transition,
)


class TestExplainKeyTransition:
    def test_same_key_positive(self):
        result = explain_key_transition("Fm", "4A", "Fm", "4A", 1.0)
        assert "同じキー" in result
        assert "スムーズ" in result

    def test_relative_key_dramatic(self):
        result = explain_key_transition("Fm", "4A", "Ab", "4B", 0.9)
        assert "相対キー" in result or "ドラマチック" in result

    def test_adjacent_key_natural(self):
        result = explain_key_transition("Fm", "4A", "Cm", "5A", 0.7)
        assert "隣り合う" in result or "Camelot" in result

    def test_incompatible_key_warning(self):
        result = explain_key_transition("Fm", "4A", "Em", "9A", 0.0)
        assert "離れて" in result
        assert "リセット" in result or "フィルター" in result

    def test_returns_string(self):
        result = explain_key_transition("Am", "8A", "Dm", "7A", 0.7)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExplainBpmTransition:
    def test_same_bpm(self):
        result = explain_bpm_transition(128.0, 128.0)
        assert "同じ" in result or "簡単" in result

    def test_small_increase(self):
        result = explain_bpm_transition(124.0, 126.0)
        assert "上がる" in result or "少し" in result

    def test_moderate_increase(self):
        result = explain_bpm_transition(120.0, 128.0)
        assert "上がる" in result
        assert "エネルギー" in result

    def test_moderate_decrease(self):
        result = explain_bpm_transition(128.0, 122.0)
        assert "下がり" in result or "下がる" in result

    def test_large_difference(self):
        result = explain_bpm_transition(80.0, 128.0)
        assert "差" in result
        assert "エコー" in result or "フィルター" in result


class TestExplainEnergyFlow:
    def test_empty_tracks(self):
        result = explain_energy_flow([], [])
        assert "ありません" in result

    def test_single_track(self):
        result = explain_energy_flow([0.5], ["Track 1"])
        assert "1曲" in result

    def test_build_pattern(self):
        energies = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        names = [f"Track {i}" for i in range(7)]
        result = explain_energy_flow(energies, names)
        assert "上昇" in result or "上がる" in result

    def test_energy_drop_detected(self):
        energies = [0.8, 0.3, 0.7, 0.8]
        names = ["A", "B", "C", "D"]
        result = explain_energy_flow(energies, names)
        assert "急降下" in result or "ブリッジ" in result

    def test_returns_string(self):
        result = explain_energy_flow([0.5, 0.6], ["A", "B"])
        assert isinstance(result, str)


class TestExplainEffects:
    def test_no_effects(self):
        result = explain_effects({"effects": []})
        assert "検出されませんでした" in result or "ドライ" in result

    def test_reverb_detected(self):
        result = explain_effects({
            "effects": [{"type": "reverb", "confidence": 0.8, "section": "Intro"}]
        })
        assert "リバーブ" in result
        assert "Intro" in result

    def test_sidechain_detected(self):
        result = explain_effects({
            "effects": [{"type": "sidechain", "confidence": 0.7}]
        })
        assert "サイドチェイン" in result or "ポンピング" in result

    def test_low_confidence(self):
        result = explain_effects({
            "effects": [{"type": "delay", "confidence": 0.3}]
        })
        assert "弱め" in result

    def test_multiple_effects(self):
        result = explain_effects({
            "effects": [
                {"type": "reverb", "confidence": 0.8},
                {"type": "sidechain", "confidence": 0.7},
            ]
        })
        assert "リバーブ" in result
        assert "サイドチェイン" in result or "ポンピング" in result


class TestExplainGenreCompatibility:
    def test_same_genre(self):
        result = explain_genre_compatibility("house", "house")
        assert "同じジャンル" in result

    def test_house_techno(self):
        result = explain_genre_compatibility("house", "techno")
        assert len(result) > 0

    def test_unknown_pair(self):
        result = explain_genre_compatibility("classical", "dnb")
        assert "ジャンルチェンジ" in result or "ブレイク" in result
