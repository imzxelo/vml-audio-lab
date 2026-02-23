"""ボーカル分析ツール (analyze_vocal_stem) のテスト.

MVP-C: vocal_analysis.py — ボーカルステムのキー・音域・使えるセクションを判定。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.vocal_analysis import analyze_vocal_stem

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vocal_wav(tmp_path_factory: pytest.TempPathFactory) -> str:
    """テスト用ボーカル WAV (A4 = 440Hz サイン波 10秒)。"""
    sr = 44100
    duration = 10.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # 歌声のような高め周波数 (440Hz = A4)、エネルギー十分
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path_factory.mktemp("vocals") / "vocal_a4_10s.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture(scope="module")
def silent_wav(tmp_path_factory: pytest.TempPathFactory) -> str:
    """無音の WAV ファイル。"""
    sr = 44100
    duration = 5.0
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    wav_path = tmp_path_factory.mktemp("vocals") / "silent.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture(scope="module")
def multi_section_vocal(tmp_path_factory: pytest.TempPathFactory) -> str:
    """セクション付きボーカル: 低・高・低のエネルギー変化。"""
    sr = 44100
    parts = []
    # Verse (低め): 5秒
    n1 = int(sr * 5.0)
    t1 = np.linspace(0, 5.0, n1, endpoint=False)
    parts.append((0.05 * np.sin(2 * np.pi * 220 * t1)).astype(np.float32))
    # Hook (高め): 5秒
    n2 = int(sr * 5.0)
    t2 = np.linspace(0, 5.0, n2, endpoint=False)
    parts.append((0.4 * np.sin(2 * np.pi * 880 * t2)).astype(np.float32))
    # Bridge (低め): 5秒
    n3 = int(sr * 5.0)
    t3 = np.linspace(0, 5.0, n3, endpoint=False)
    parts.append((0.05 * np.sin(2 * np.pi * 440 * t3)).astype(np.float32))
    y = np.concatenate(parts)
    wav_path = tmp_path_factory.mktemp("vocals") / "multi_section.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


# ---------------------------------------------------------------------------
# 必須フィールドテスト
# ---------------------------------------------------------------------------


class TestAnalyzeVocalStemReturnSchema:
    """analyze_vocal_stem: 返却スキーマのテスト。"""

    def test_returns_dict(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert isinstance(result, dict)

    def test_has_key_field(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert "key" in result

    def test_has_camelot_field(self, vocal_wav: str) -> None:
        """camelot フィールドが含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        assert "camelot" in result

    def test_has_scale_field(self, vocal_wav: str) -> None:
        """scale フィールドが含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        assert "scale" in result
        assert result["scale"] in ("major", "minor")

    def test_has_key_strength_field(self, vocal_wav: str) -> None:
        """key_strength フィールドが含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        assert "key_strength" in result
        assert 0.0 <= result["key_strength"] <= 1.0

    def test_has_pitch_range(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert "pitch_range" in result
        pr = result["pitch_range"]
        # 実装の pitch_range は low, high, low_note, high_note を持つ
        assert "low" in pr
        assert "high" in pr

    def test_has_usable_sections(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert "usable_sections" in result
        assert isinstance(result["usable_sections"], list)

    def test_has_compatible_bpm_range(self, vocal_wav: str) -> None:
        """compatible_bpm_range フィールドが含まれる ([min, max] のリスト)。"""
        result = analyze_vocal_stem(vocal_wav)
        assert "compatible_bpm_range" in result
        bpm_range = result["compatible_bpm_range"]
        assert isinstance(bpm_range, list)
        assert len(bpm_range) == 2

    def test_has_compatible_genres(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert "compatible_genres" in result
        assert isinstance(result["compatible_genres"], list)

    def test_has_compatible_camelot_codes(self, vocal_wav: str) -> None:
        """compatible_camelot_codes フィールドが含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        assert "compatible_camelot_codes" in result
        assert isinstance(result["compatible_camelot_codes"], list)


# ---------------------------------------------------------------------------
# キー検出テスト
# ---------------------------------------------------------------------------


class TestVocalKeyDetection:
    """analyze_vocal_stem: キー検出テスト。"""

    def test_key_is_string(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert isinstance(result["key"], str)

    def test_key_is_valid_music_key_or_empty(self, vocal_wav: str) -> None:
        """キーは有効なラベル、または空文字/unknown。"""
        result = analyze_vocal_stem(vocal_wav)
        key = result["key"]
        # 基本的なバリデーション: A-G から始まるか空文字/unknown
        valid_starts = set("ABCDEFG")
        assert key == "" or key == "unknown" or key[0] in valid_starts

    def test_camelot_code_present_when_key_detected(self, vocal_wav: str) -> None:
        """キーが検出された場合、camelot フィールドが含まれる (None or str)。"""
        result = analyze_vocal_stem(vocal_wav)
        camelot = result["camelot"]
        assert camelot is None or isinstance(camelot, str)

    def test_compatible_camelot_codes_when_key_detected(self, vocal_wav: str) -> None:
        """camelot が取れた場合、compatible_camelot_codes に 4 コードが返る。"""
        result = analyze_vocal_stem(vocal_wav)
        if result["camelot"]:
            # compatible_camelot_codes は camelot から派生
            codes = result["compatible_camelot_codes"]
            assert len(codes) == 4


# ---------------------------------------------------------------------------
# 音域テスト
# ---------------------------------------------------------------------------


class TestPitchRange:
    """analyze_vocal_stem: 音域 (pitch_range) テスト。"""

    def test_pitch_range_low_is_numeric_or_none(self, vocal_wav: str) -> None:
        """low は数値または None。"""
        result = analyze_vocal_stem(vocal_wav)
        pr = result["pitch_range"]
        assert pr["low"] is None or isinstance(pr["low"], (int, float))

    def test_pitch_range_high_is_numeric_or_none(self, vocal_wav: str) -> None:
        """high は数値または None。"""
        result = analyze_vocal_stem(vocal_wav)
        pr = result["pitch_range"]
        assert pr["high"] is None or isinstance(pr["high"], (int, float))

    def test_pitch_range_low_lte_high(self, vocal_wav: str) -> None:
        """low <= high (両方が数値のとき)。"""
        result = analyze_vocal_stem(vocal_wav)
        pr = result["pitch_range"]
        if pr["low"] is not None and pr["high"] is not None:
            if isinstance(pr["low"], (int, float)) and isinstance(pr["high"], (int, float)):
                if pr["low"] > 0 and pr["high"] > 0:
                    assert pr["low"] <= pr["high"]

    def test_pitch_range_has_note_names(self, vocal_wav: str) -> None:
        """low_note と high_note が含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        pr = result["pitch_range"]
        assert "low_note" in pr
        assert "high_note" in pr

    def test_silent_pitch_range_is_none_or_zero(self, silent_wav: str) -> None:
        """無音の音域は None または 0.0。"""
        result = analyze_vocal_stem(silent_wav)
        pr = result["pitch_range"]
        assert pr["low"] is None or pr["low"] == 0.0
        assert pr["high"] is None or pr["high"] == 0.0


# ---------------------------------------------------------------------------
# ボーカルセクションテスト
# ---------------------------------------------------------------------------


class TestUsableSections:
    """analyze_vocal_stem: usable_sections テスト。"""

    def test_usable_sections_is_list(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        assert isinstance(result["usable_sections"], list)

    def test_each_section_has_start_and_end(self, vocal_wav: str) -> None:
        """各セクションに start と end フィールドがある。"""
        result = analyze_vocal_stem(vocal_wav)
        for section in result["usable_sections"]:
            for field in ("start", "end"):
                assert field in section, f"Missing field: {field}"

    def test_section_start_before_end(self, vocal_wav: str) -> None:
        """start < end。"""
        result = analyze_vocal_stem(vocal_wav)
        for section in result["usable_sections"]:
            assert section["start"] < section["end"]

    def test_section_has_suggestion(self, vocal_wav: str) -> None:
        """各セクションに suggestion フィールドがある。"""
        result = analyze_vocal_stem(vocal_wav)
        for section in result["usable_sections"]:
            assert "suggestion" in section
            assert isinstance(section["suggestion"], str)

    def test_section_has_has_clear_vocal(self, vocal_wav: str) -> None:
        """各セクションに has_clear_vocal フィールドがある。"""
        result = analyze_vocal_stem(vocal_wav)
        for section in result["usable_sections"]:
            assert "has_clear_vocal" in section
            assert isinstance(section["has_clear_vocal"], bool)

    def test_section_has_type(self, vocal_wav: str) -> None:
        """各セクションに type フィールドがある。"""
        result = analyze_vocal_stem(vocal_wav)
        for section in result["usable_sections"]:
            assert "type" in section
            assert isinstance(section["type"], str)

    def test_section_with_explicit_sections_input(self, multi_section_vocal: str) -> None:
        """sections パラメータを渡しても正常動作する。"""
        sections = [
            {"label": "Verse", "start": 0.0, "end": 5.0},
            {"label": "Hook", "start": 5.0, "end": 10.0},
            {"label": "Bridge", "start": 10.0, "end": 15.0},
        ]
        result = analyze_vocal_stem(multi_section_vocal, sections=sections)
        assert isinstance(result["usable_sections"], list)
        # セクション提供時は各セクションに判定結果が付く
        labels = [s["type"] for s in result["usable_sections"]]
        assert any(lbl in labels for lbl in ("Verse", "Hook", "Bridge"))

    def test_silent_audio_empty_or_no_vocal_sections(self, silent_wav: str) -> None:
        """無音は usable_sections が空か has_clear_vocal=False のみ。"""
        result = analyze_vocal_stem(silent_wav)
        for section in result["usable_sections"]:
            assert section["has_clear_vocal"] is False

    def test_no_exception_on_short_audio(self) -> None:
        """極短音声でも例外を出さない。"""
        sr = 44100
        n = int(sr * 1.0)
        y = np.zeros(n, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        sf.write(wav_path, y, sr)
        result = analyze_vocal_stem(wav_path)
        assert "usable_sections" in result


# ---------------------------------------------------------------------------
# 相性 BPM・ジャンルテスト
# ---------------------------------------------------------------------------


class TestCompatibleBpmAndGenres:
    """analyze_vocal_stem: compatible_bpm_range と compatible_genres のテスト。"""

    def test_compatible_bpm_range_is_2_element_list(self, vocal_wav: str) -> None:
        """compatible_bpm_range は [min_bpm, max_bpm] の 2 要素リスト。"""
        result = analyze_vocal_stem(vocal_wav)
        bpm_range = result["compatible_bpm_range"]
        assert isinstance(bpm_range, list)
        assert len(bpm_range) == 2

    def test_compatible_bpm_range_min_lt_max(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        bpm_range = result["compatible_bpm_range"]
        assert bpm_range[0] < bpm_range[1]

    def test_compatible_bpm_range_is_positive(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        bpm_range = result["compatible_bpm_range"]
        assert bpm_range[0] > 0
        assert bpm_range[1] > 0

    def test_compatible_genres_contains_house(self, vocal_wav: str) -> None:
        """House / Deep House はボーカルと相性が良いジャンルに含まれる。"""
        result = analyze_vocal_stem(vocal_wav)
        genres = result["compatible_genres"]
        assert any("house" in g for g in genres)

    def test_compatible_genres_is_list_of_strings(self, vocal_wav: str) -> None:
        result = analyze_vocal_stem(vocal_wav)
        for g in result["compatible_genres"]:
            assert isinstance(g, str)
            assert len(g) > 0

    def test_nonexistent_file_raises_error(self) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_vocal_stem("/nonexistent/vocal.wav")


# ---------------------------------------------------------------------------
# find_compatible_tracks_for_vocal のテスト
# ---------------------------------------------------------------------------


class TestFindCompatibleTracksForVocal:
    """find_compatible_tracks_for_vocal: ボーカル → ライブラリマッチングのテスト。"""

    def test_nonexistent_index_raises_error(self, vocal_wav: str) -> None:
        """存在しないインデックスファイルは FileNotFoundError。"""
        from vml_audio_lab.tools.vocal_analysis import find_compatible_tracks_for_vocal

        vocal_analysis = analyze_vocal_stem(vocal_wav)
        with pytest.raises(FileNotFoundError):
            find_compatible_tracks_for_vocal(vocal_analysis, "/nonexistent/index.json")

    def test_returns_required_fields(self, vocal_wav: str, tmp_path: Path) -> None:
        """返却値に必須フィールドがある。"""
        import json

        from vml_audio_lab.tools.vocal_analysis import find_compatible_tracks_for_vocal

        # ダミーのライブラリインデックスを作成
        index = {
            "tracks": [
                {
                    "id": 1,
                    "title": "Deep House Track",
                    "artist": "",
                    "bpm": 124.0,
                    "key_label": "Am",
                    "camelot_code": "8A",
                    "genre": "deep-house",
                    "mood": "Deep Night",
                    "file_path": "/music/track.wav",
                }
            ],
            "total": 1,
            "fingerprint": "test",
        }
        index_path = tmp_path / "library_index.json"
        index_path.write_text(json.dumps(index))

        vocal_analysis = analyze_vocal_stem(vocal_wav)
        result = find_compatible_tracks_for_vocal(vocal_analysis, str(index_path))

        for field in ("matches", "total_searched", "vocal_key", "vocal_camelot"):
            assert field in result, f"Missing field: {field}"

    def test_matches_is_list(self, vocal_wav: str, tmp_path: Path) -> None:
        """matches は list。"""
        import json

        from vml_audio_lab.tools.vocal_analysis import find_compatible_tracks_for_vocal

        index = {
            "tracks": [],
            "total": 0,
            "fingerprint": "test",
        }
        index_path = tmp_path / "empty_library.json"
        index_path.write_text(json.dumps(index))

        vocal_analysis = analyze_vocal_stem(vocal_wav)
        result = find_compatible_tracks_for_vocal(vocal_analysis, str(index_path))
        assert isinstance(result["matches"], list)
