"""基礎分析ツール（detect_bpm, detect_key, energy_curve）のテスト"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.analysis import detect_bpm, detect_key, energy_curve
from vml_audio_lab.tools.loader import load_track

FIXTURES = Path(__file__).parent / "fixtures"
SINE_WAV = FIXTURES / "sine_440hz_3s.wav"


@pytest.fixture(scope="module")
def y_path() -> str:
    """テスト用音声データのキャッシュパス"""
    result = load_track(str(SINE_WAV))
    return result["y_path"]


@pytest.fixture(scope="module")
def beat_fixture(tmp_path_factory: pytest.TempPathFactory) -> str:
    """BPM検出テスト用: 120BPM のクリック音"""
    sr = 22050
    duration = 8.0
    bpm = 120.0
    interval = 60.0 / bpm
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)

    # クリック音を等間隔で配置
    t = 0.0
    while t < duration:
        idx = int(t * sr)
        click_len = min(200, n_samples - idx)
        y[idx : idx + click_len] = 0.8 * np.sin(
            2 * np.pi * 1000 * np.arange(click_len) / sr
        )
        t += interval

    wav_path = tmp_path_factory.mktemp("audio") / "click_120bpm.wav"
    sf.write(str(wav_path), y, sr)

    result = load_track(str(wav_path))
    return result["y_path"]


class TestDetectBpm:
    """BPM 検出テスト"""

    def test_returns_bpm_key(self, y_path: str) -> None:
        result = detect_bpm(y_path)
        assert "bpm" in result

    def test_bpm_is_non_negative(self, y_path: str) -> None:
        """サイン波にはビートがないので 0.0 も許容する"""
        result = detect_bpm(y_path)
        assert result["bpm"] >= 0

    def test_120bpm_click_within_tolerance(self, beat_fixture: str) -> None:
        result = detect_bpm(beat_fixture)
        assert abs(result["bpm"] - 120.0) < 5.0, f"Expected ~120 BPM, got {result['bpm']}"

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            detect_bpm("/nonexistent/y.npy")


class TestDetectKey:
    """キー検出テスト"""

    def test_returns_required_keys(self, y_path: str) -> None:
        result = detect_key(y_path)
        for key in ("key", "scale", "key_label", "strength"):
            assert key in result

    def test_scale_is_valid(self, y_path: str) -> None:
        result = detect_key(y_path)
        assert result["scale"] in ("major", "minor")

    def test_strength_in_range(self, y_path: str) -> None:
        result = detect_key(y_path)
        assert 0.0 <= result["strength"] <= 1.0

    def test_key_label_format(self, y_path: str) -> None:
        result = detect_key(y_path)
        label = result["key_label"]
        # "Am", "C", "F#m" 等
        assert len(label) >= 1
        assert label[0].isupper()

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            detect_key("/nonexistent/y.npy")


class TestEnergyCurve:
    """エネルギーカーブ画像テスト"""

    def test_returns_png_bytes(self, y_path: str) -> None:
        data = energy_curve(y_path)
        assert isinstance(data, bytes)
        # PNG マジックバイト
        assert data[:4] == b"\x89PNG"

    def test_image_not_empty(self, y_path: str) -> None:
        data = energy_curve(y_path)
        assert len(data) > 1000  # 最低限のサイズ

    def test_y_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            energy_curve("/nonexistent/y.npy")
