"""可視化ツール（spectrogram, waveform_overview）のテスト"""

from pathlib import Path

import pytest

from vml_audio_lab.tools.loader import load_track
from vml_audio_lab.tools.structure import detect_structure
from vml_audio_lab.tools.visualize import spectrogram, waveform_overview

FIXTURES = Path(__file__).parent / "fixtures"
SINE_WAV = FIXTURES / "sine_440hz_3s.wav"


@pytest.fixture(scope="module")
def y_path() -> str:
    result = load_track(str(SINE_WAV))
    return result["y_path"]


@pytest.fixture(scope="module")
def sections(y_path: str) -> list[dict]:
    result = detect_structure(y_path)
    return result["sections"]


class TestSpectrogram:
    """スペクトログラム画像テスト"""

    def test_returns_png_bytes(self, y_path: str) -> None:
        data = spectrogram(y_path)
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_image_not_empty(self, y_path: str) -> None:
        data = spectrogram(y_path)
        assert len(data) > 5000

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            spectrogram("/nonexistent/y.npy")


class TestWaveformOverview:
    """波形オーバービュー画像テスト"""

    def test_returns_png_without_sections(self, y_path: str) -> None:
        data = waveform_overview(y_path)
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_returns_png_with_sections(self, y_path: str, sections: list[dict]) -> None:
        data = waveform_overview(y_path, sections=sections)
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_with_sections_larger_than_without(self, y_path: str, sections: list[dict]) -> None:
        without = waveform_overview(y_path)
        with_sec = waveform_overview(y_path, sections=sections)
        # セクションアノテーション分、画像サイズが大きくなるはず
        assert len(with_sec) > len(without) * 0.8

    def test_image_not_empty(self, y_path: str) -> None:
        data = waveform_overview(y_path)
        assert len(data) > 3000

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            waveform_overview("/nonexistent/y.npy")
