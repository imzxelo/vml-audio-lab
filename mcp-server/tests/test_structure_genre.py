"""ジャンル別構造ラベル付与のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.loader import load_track
from vml_audio_lab.tools.structure import (
    _assign_labels,
    _genre_labels,
    _refine_jpop_labels,
    detect_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"
SINE_WAV = FIXTURES / "sine_440hz_3s.wav"


class TestGenreLabels:
    """_genre_labels: ジャンル別ラベルセット取得テスト."""

    def test_hiphop_label_set(self) -> None:
        labels = _genre_labels("hiphop")
        assert labels["intro"] == "Intro"
        assert labels["high_energy"] == "Hook"
        assert labels["low_energy"] == "Bridge"
        assert labels["mid_energy"] == "Verse"
        assert labels["outro"] == "Outro"

    def test_jpop_label_set(self) -> None:
        labels = _genre_labels("jpop")
        assert labels["intro"] == "Intro"
        assert labels["high_energy"] == "サビ"
        assert labels["low_energy"] == "落ちサビ"
        assert labels["mid_energy"] == "Aメロ"
        assert labels["outro"] == "Outro"

    def test_default_label_set(self) -> None:
        labels = _genre_labels("default")
        assert labels["intro"] == "Intro"
        assert labels["high_energy"] == "Drop"
        assert labels["low_energy"] == "Break"
        assert labels["mid_energy"] == "Build"
        assert labels["outro"] == "Outro"

    def test_house_uses_default_labels(self) -> None:
        """DJ系ジャンルはデフォルトラベルを使う."""
        labels = _genre_labels("house")
        default = _genre_labels("default")
        assert labels == default

    def test_techno_uses_default_labels(self) -> None:
        labels = _genre_labels("techno")
        default = _genre_labels("default")
        assert labels == default

    def test_unknown_genre_falls_back_to_default(self) -> None:
        labels = _genre_labels("nonexistent_genre")
        default = _genre_labels("default")
        assert labels == default


class TestAssignLabels:
    """_assign_labels: ラベル付与ロジックのユニットテスト."""

    def _make_sections(self, energies: list[float]) -> list[dict]:
        """テスト用セクションリストを生成する。"""
        return [
            {"start": float(i * 30), "end": float((i + 1) * 30), "energy": e}
            for i, e in enumerate(energies)
        ]

    def test_dj_default_labels_intro_outro(self) -> None:
        sections = self._make_sections([0.5, 0.8, 0.4, 0.5])
        _assign_labels(sections, genre_group="default")
        assert sections[0]["label"] == "Intro"
        assert sections[-1]["label"] == "Outro"

    def test_dj_high_energy_is_drop(self) -> None:
        sections = self._make_sections([0.5, 0.9, 0.3, 0.5])
        _assign_labels(sections, genre_group="default")
        assert sections[1]["label"] == "Drop"

    def test_dj_low_energy_is_break(self) -> None:
        sections = self._make_sections([0.5, 0.8, 0.2, 0.5])
        _assign_labels(sections, genre_group="default")
        assert sections[2]["label"] == "Break"

    def test_dj_mid_energy_is_build(self) -> None:
        sections = self._make_sections([0.5, 0.5, 0.8, 0.5])
        _assign_labels(sections, genre_group="default")
        assert sections[1]["label"] == "Build"

    def test_hiphop_high_energy_is_hook(self) -> None:
        sections = self._make_sections([0.5, 0.9, 0.4, 0.5])
        _assign_labels(sections, genre_group="hiphop")
        assert sections[0]["label"] == "Intro"
        assert sections[1]["label"] == "Hook"
        assert sections[-1]["label"] == "Outro"

    def test_hiphop_low_energy_is_bridge(self) -> None:
        sections = self._make_sections([0.5, 0.8, 0.2, 0.5])
        _assign_labels(sections, genre_group="hiphop")
        assert sections[2]["label"] == "Bridge"

    def test_hiphop_mid_energy_is_verse(self) -> None:
        sections = self._make_sections([0.5, 0.5, 0.9, 0.5])
        _assign_labels(sections, genre_group="hiphop")
        assert sections[1]["label"] == "Verse"

    def test_jpop_high_energy_is_sabi(self) -> None:
        sections = self._make_sections([0.5, 0.9, 0.4, 0.5])
        _assign_labels(sections, genre_group="jpop")
        assert sections[0]["label"] == "Intro"
        assert sections[1]["label"] == "サビ"
        assert sections[-1]["label"] == "Outro"

    def test_jpop_low_energy_is_ochisabi(self) -> None:
        sections = self._make_sections([0.5, 0.8, 0.2, 0.5])
        _assign_labels(sections, genre_group="jpop")
        assert sections[2]["label"] == "落ちサビ"

    def test_jpop_mid_energy_is_amoto(self) -> None:
        sections = self._make_sections([0.5, 0.5, 0.9, 0.5])
        _assign_labels(sections, genre_group="jpop")
        assert sections[1]["label"] == "Aメロ"

    def test_single_section_still_works(self) -> None:
        """セクションが1つでもエラーにならない."""
        sections = self._make_sections([1.0])
        _assign_labels(sections, genre_group="default")
        # 1セクションは intro も outro も最初と最後で同じ
        assert "label" in sections[0]

    def test_two_sections_intro_and_outro(self) -> None:
        sections = self._make_sections([0.5, 0.8])
        _assign_labels(sections, genre_group="default")
        assert sections[0]["label"] == "Intro"
        assert sections[1]["label"] == "Outro"


class TestRefineJpopLabels:
    """_refine_jpop_labels: J-Pop ラベル細分化テスト."""

    def test_multiple_sabi_last_becomes_daisabi(self) -> None:
        sections = [
            {"label": "Intro", "energy": 0.5},
            {"label": "Aメロ", "energy": 0.4},
            {"label": "サビ", "energy": 0.9},
            {"label": "Aメロ", "energy": 0.4},
            {"label": "サビ", "energy": 0.9},
            {"label": "Outro", "energy": 0.3},
        ]
        _refine_jpop_labels(sections)
        sabi_labels = [s["label"] for s in sections]
        assert sabi_labels.count("サビ") == 1
        assert "大サビ" in sabi_labels

    def test_single_sabi_unchanged(self) -> None:
        sections = [
            {"label": "Intro", "energy": 0.5},
            {"label": "Aメロ", "energy": 0.4},
            {"label": "サビ", "energy": 0.9},
            {"label": "Outro", "energy": 0.3},
        ]
        _refine_jpop_labels(sections)
        assert sections[2]["label"] == "サビ"

    def test_bmelo_assigned_between_amelo_and_sabi(self) -> None:
        sections = [
            {"label": "Intro", "energy": 0.5},
            {"label": "Aメロ", "energy": 0.4},
            {"label": "Aメロ", "energy": 0.5},
            {"label": "サビ", "energy": 0.9},
            {"label": "Outro", "energy": 0.3},
        ]
        _refine_jpop_labels(sections)
        labels = [s["label"] for s in sections]
        assert "Bメロ" in labels

    def test_no_sabi_no_crash(self) -> None:
        """サビがなくても例外を出さない."""
        sections = [
            {"label": "Intro", "energy": 0.5},
            {"label": "Aメロ", "energy": 0.4},
            {"label": "Outro", "energy": 0.3},
        ]
        _refine_jpop_labels(sections)  # should not raise


class TestDetectStructureWithGenre:
    """detect_structure: ジャンル別ラベル付与の統合テスト."""

    @pytest.fixture(scope="class")
    def short_y_path(self) -> str:
        result = load_track(str(SINE_WAV))
        return result["y_path"]

    @pytest.fixture(scope="class")
    def long_y_path(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """3分30秒の構造化音声."""
        sr = 22050
        sections = [
            (60, 0.1, 220),   # quiet - Intro
            (60, 0.5, 440),   # mid
            (60, 1.0, 880),   # high energy
            (30, 0.2, 220),   # fade - Outro
        ]
        parts = []
        for dur, amp, freq in sections:
            n = int(sr * dur)
            t = np.linspace(0, dur, n, endpoint=False)
            y = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
            y += np.random.default_rng(42).normal(0, amp * 0.05, n).astype(np.float32)
            parts.append(y)
        y_full = np.concatenate(parts)
        wav_path = tmp_path_factory.mktemp("audio") / "structured_210s.wav"
        sf.write(str(wav_path), y_full, sr)
        result = load_track(str(wav_path))
        return result["y_path"]

    def test_default_genre_returns_genre_group_field(self, short_y_path: str) -> None:
        result = detect_structure(short_y_path)
        assert "genre_group" in result

    def test_hiphop_structure_uses_hip_hop_labels(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="hiphop")
        valid_labels = {"Intro", "Hook", "Bridge", "Verse", "Outro"}
        for section in result["sections"]:
            assert section["label"] in valid_labels, (
                f"Unexpected Hip-Hop label: {section['label']}"
            )

    def test_jpop_structure_uses_jpop_labels(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="jpop")
        valid_labels = {"Intro", "サビ", "落ちサビ", "Aメロ", "Bメロ", "大サビ", "Outro"}
        for section in result["sections"]:
            assert section["label"] in valid_labels, (
                f"Unexpected J-Pop label: {section['label']}"
            )

    def test_default_structure_uses_dj_labels(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="default")
        valid_labels = {"Intro", "Drop", "Break", "Build", "Outro"}
        for section in result["sections"]:
            assert section["label"] in valid_labels, (
                f"Unexpected DJ label: {section['label']}"
            )

    def test_hiphop_first_section_is_intro(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="hiphop")
        assert result["sections"][0]["label"] == "Intro"

    def test_hiphop_last_section_is_outro(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="hiphop")
        assert result["sections"][-1]["label"] == "Outro"

    def test_jpop_first_section_is_intro(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="jpop")
        assert result["sections"][0]["label"] == "Intro"

    def test_jpop_last_section_is_outro(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, genre_group="jpop")
        assert result["sections"][-1]["label"] == "Outro"

    def test_genre_group_stored_in_result(self, long_y_path: str) -> None:
        for genre in ("hiphop", "jpop", "default"):
            result = detect_structure(long_y_path, genre_group=genre)
            assert result["genre_group"] == genre

    def test_short_audio_hiphop_returns_single_section(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """極短音声 (0.5秒) は genre にかかわらず Full Track を返す."""
        sr = 22050
        y = np.zeros(int(sr * 0.5), dtype=np.float32)
        wav_path = tmp_path_factory.mktemp("audio") / "half_second.wav"
        sf.write(str(wav_path), y, sr)
        result_path = load_track(str(wav_path))["y_path"]
        result = detect_structure(result_path, genre_group="hiphop")
        assert result["n_segments"] == 1
        assert result["sections"][0]["label"] == "Full Track"
