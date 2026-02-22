"""構造認識ツール（detect_structure）のテスト"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.loader import load_track
from vml_audio_lab.tools.structure import _estimate_n_segments, _format_time, detect_structure

FIXTURES = Path(__file__).parent / "fixtures"
SINE_WAV = FIXTURES / "sine_440hz_3s.wav"


@pytest.fixture(scope="module")
def short_y_path() -> str:
    """短い音声(3秒)"""
    result = load_track(str(SINE_WAV))
    return result["y_path"]


@pytest.fixture(scope="module")
def long_y_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """構造変化のある長い音声(3分30秒)を生成する。

    - 0:00-1:00: 静かなパッド（低エネルギー）
    - 1:00-2:00: ビート追加（中エネルギー）
    - 2:00-3:00: フルエネルギー
    - 3:00-3:30: フェードアウト
    """
    sr = 22050
    sections = [
        (60, 0.1, 220),   # quiet pad
        (60, 0.5, 440),   # mid energy
        (60, 1.0, 880),   # high energy
        (30, 0.2, 220),   # fade out
    ]

    parts = []
    for dur, amp, freq in sections:
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        y = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # ノイズを少し追加（セグメンテーションの特徴量に差を出す）
        y += np.random.default_rng(42).normal(0, amp * 0.05, n).astype(np.float32)
        parts.append(y)

    y_full = np.concatenate(parts)
    wav_path = tmp_path_factory.mktemp("audio") / "structured_210s.wav"
    sf.write(str(wav_path), y_full, sr)

    result = load_track(str(wav_path))
    return result["y_path"]


class TestEstimateNSegments:
    """セグメント数自動推定テスト"""

    def test_short_clip_minimum_2(self) -> None:
        assert _estimate_n_segments(10.0) == 2

    def test_3min_gives_6(self) -> None:
        assert _estimate_n_segments(180.0) == 6

    def test_long_clip_capped_at_12(self) -> None:
        assert _estimate_n_segments(600.0) == 12


class TestFormatTime:
    """時刻フォーマットテスト"""

    def test_zero(self) -> None:
        assert _format_time(0.0) == "0:00"

    def test_1min30(self) -> None:
        assert _format_time(90.5) == "1:30"

    def test_5min(self) -> None:
        assert _format_time(300.0) == "5:00"


@pytest.fixture(scope="module")
def subsecond_y_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """極短音声(0.5秒)"""
    sr = 22050
    y = np.zeros(int(sr * 0.5), dtype=np.float32)
    wav_path = tmp_path_factory.mktemp("audio") / "half_second.wav"
    sf.write(str(wav_path), y, sr)
    result = load_track(str(wav_path))
    return result["y_path"]


class TestDetectStructureSubSecond:
    """極短音声での構造検出テスト"""

    def test_returns_single_section(self, subsecond_y_path: str) -> None:
        result = detect_structure(subsecond_y_path)
        assert result["n_segments"] == 1
        assert result["sections"][0]["label"] == "Full Track"
        assert result["sections"][0]["start"] == 0.0


class TestDetectStructureShort:
    """短い音声での構造検出テスト"""

    def test_returns_required_keys(self, short_y_path: str) -> None:
        result = detect_structure(short_y_path)
        assert "sections" in result
        assert "n_segments" in result
        assert "duration_sec" in result

    def test_at_least_2_sections(self, short_y_path: str) -> None:
        result = detect_structure(short_y_path)
        assert result["n_segments"] >= 2

    def test_sections_have_required_fields(self, short_y_path: str) -> None:
        result = detect_structure(short_y_path)
        for s in result["sections"]:
            for key in ("start", "end", "start_label", "end_label", "label", "energy"):
                assert key in s, f"Missing key: {key}"

    def test_first_section_starts_at_zero(self, short_y_path: str) -> None:
        result = detect_structure(short_y_path)
        assert result["sections"][0]["start"] == 0.0

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            detect_structure("/nonexistent/y.npy")


class TestDetectStructureLong:
    """長い音声（3分30秒）での構造検出テスト"""

    def test_detects_multiple_sections(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path)
        assert result["n_segments"] >= 2

    def test_duration_is_correct(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path)
        assert abs(result["duration_sec"] - 210.0) < 1.0

    def test_labels_include_intro_and_outro(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path)
        labels = [s["label"] for s in result["sections"]]
        assert labels[0] == "Intro"
        assert labels[-1] == "Outro"

    def test_energy_normalized_0_to_1(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path)
        energies = [s["energy"] for s in result["sections"]]
        assert max(energies) == 1.0
        assert all(0.0 <= e <= 1.0 for e in energies)

    def test_sections_cover_full_duration(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path)
        sections = result["sections"]
        assert sections[0]["start"] == 0.0
        assert abs(sections[-1]["end"] - result["duration_sec"]) < 0.5

    def test_explicit_n_segments(self, long_y_path: str) -> None:
        result = detect_structure(long_y_path, n_segments=4)
        assert result["n_segments"] == 4
