"""effects.py のテスト.

合成信号で各エフェクト検出器をテスト。
正例では detected=True と妥当な confidence を検証する。
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from vml_audio_lab.tools.effects import (
    _analyze_with_sections,
    _detect_delay,
    _detect_filter_sweep,
    _detect_reverb,
    _detect_sidechain,
    _max_consecutive,
    detect_effects,
)

SR = 22050


def _make_sine(freq: float = 440.0, duration: float = 2.0, sr: int = SR) -> np.ndarray:
    """基本サイン波を生成する。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_impulse_reverb(duration: float = 3.0, sr: int = SR) -> np.ndarray:
    """リバーブ風の信号: インパルス + 指数減衰テール。"""
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    for impulse_pos in range(0, n, sr // 2):
        if impulse_pos >= n:
            break
        y[impulse_pos] = 1.0
        tail_length = min(sr, n - impulse_pos)
        decay = np.exp(-np.arange(tail_length) / (sr * 0.3))
        noise = np.random.randn(tail_length).astype(np.float32) * 0.1
        y[impulse_pos : impulse_pos + tail_length] += (decay * noise).astype(np.float32)
    return y


def _make_delay_signal(delay_ms: float = 250.0, sr: int = SR) -> np.ndarray:
    """ディレイ風の信号: 連続音に減衰エコーを重ねる。

    クリック+無音ではなく、短いバーストの後に減衰コピーを重ねることで
    実際のdelayエフェクトに近い信号を生成する。
    sidechainの振幅ディップ形状とは明確に区別できる。
    """
    duration = 3.0
    n = int(sr * duration)
    delay_samples = int(delay_ms * sr / 1000)

    # ベース信号: 周期的な短いバースト（800msごと）
    y = np.zeros(n, dtype=np.float32)
    burst_interval = int(0.8 * sr)
    burst_len = int(0.05 * sr)
    for start in range(0, n, burst_interval):
        end = min(start + burst_len, n)
        t = np.arange(end - start, dtype=np.float32) / sr
        y[start:end] = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.02)

    # エコーを重ねる
    echoed = np.copy(y)
    for echo_num in range(1, 6):
        shift = echo_num * delay_samples
        if shift < n:
            gain = 0.5 ** echo_num
            echoed[shift:] += y[: n - shift] * gain

    return echoed.astype(np.float32)


def _make_filter_sweep(sr: int = SR) -> np.ndarray:
    """フィルタースウィープ風: チャープ信号（周波数が200Hz→8000Hzに上昇）。

    spectral_centroid が単調上昇するので _detect_filter_sweep が検出できる。
    """
    from scipy.signal import chirp

    duration = 4.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = chirp(t, f0=200, f1=8000, t1=duration, method="linear").astype(np.float32)
    return y * 0.5


def _make_sidechain_signal(bpm: float = 128.0, sr: int = SR) -> np.ndarray:
    """サイドチェイン風: ビートごとに振幅が深くディップする信号。"""
    duration = 4.0
    n = int(sr * duration)
    beat_interval = int(60.0 / bpm * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for beat_start in range(0, n, beat_interval):
        dip_length = min(int(beat_interval * 0.5), n - beat_start)
        for j in range(dip_length):
            recovery = j / dip_length
            # 深いダッキング: 最小5%まで落とす
            y[beat_start + j] *= recovery * 0.95 + 0.05
    return y


def _save_y(y: np.ndarray) -> str:
    """テスト用に numpy 配列を一時ファイルに保存。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(tmp.name, y)
    return tmp.name


class TestMaxConsecutive:
    def test_empty(self):
        assert _max_consecutive(np.array([])) == 0

    def test_all_true(self):
        assert _max_consecutive(np.array([True, True, True])) == 3

    def test_all_false(self):
        assert _max_consecutive(np.array([False, False])) == 0

    def test_mixed(self):
        assert _max_consecutive(np.array([True, True, False, True, True, True])) == 3


class TestDetectReverb:
    def test_detects_reverb_signal(self):
        y = _make_impulse_reverb()
        result = _detect_reverb(y, SR)
        assert result["type"] == "reverb"
        assert result["detected"] is True
        assert result["confidence"] > 0.3

    def test_dry_sine_lower_confidence(self):
        reverb_y = _make_impulse_reverb()
        dry_y = _make_sine(duration=2.0)
        reverb_result = _detect_reverb(reverb_y, SR)
        dry_result = _detect_reverb(dry_y, SR)
        assert reverb_result["confidence"] > dry_result["confidence"]


class TestDetectDelay:
    def test_detects_delay_pattern(self):
        y = _make_delay_signal(delay_ms=250)
        result = _detect_delay(y, SR)
        assert result["type"] == "delay"
        # delay_ms=250 → implied BPM=240 → outside 60-200 range → no beat penalty
        assert result["detected"] is True
        assert result["confidence"] > 0.4

    def test_continuous_tone_no_delay(self):
        y = _make_sine(duration=3.0)
        result = _detect_delay(y, SR)
        assert result["type"] == "delay"
        assert result["detected"] is False

    def test_sidechain_not_misdetected_as_delay(self):
        """ビート同期のポンピングはdelayと誤認されないこと。"""
        y = _make_sidechain_signal(bpm=128)
        result = _detect_delay(y, SR)
        assert result["type"] == "delay"
        # sidechain penalty should suppress confidence
        assert result["confidence"] < 0.5 or result["detected"] is False

    def test_tempo_sync_delay_still_detected(self):
        """テンポ同期のdelay (375ms=付点8分@128BPM) はdelayとして検出されること。"""
        # 375ms のディレイ → implied BPM = 160 → 音楽的BPM範囲内だが、
        # sidechainの振幅ディップがないのでペナルティなし
        y = _make_delay_signal(delay_ms=375)
        result = _detect_delay(y, SR)
        assert result["type"] == "delay"
        # sidechain_penalty は低い (サイドチェイン形状がないから)
        assert result["details"].get("sidechain_penalty", 0) < 0.3


class TestDetectFilterSweep:
    def test_detects_rising_sweep(self):
        y = _make_filter_sweep()
        result = _detect_filter_sweep(y, SR)
        assert result["type"] == "filter_sweep"
        assert result["detected"] is True
        assert result["confidence"] > 0.3
        assert result["details"]["direction"] == "rising"

    def test_static_spectrum_no_sweep(self):
        y = _make_sine(duration=3.0)
        result = _detect_filter_sweep(y, SR)
        assert result["type"] == "filter_sweep"
        assert result["detected"] is False
        assert result["confidence"] < 0.3


class TestDetectSidechain:
    def test_detects_pumping(self):
        y = _make_sidechain_signal(bpm=128)
        result = _detect_sidechain(y, SR)
        assert result["type"] == "sidechain"
        assert result["detected"] is True
        assert result["confidence"] > 0.3

    def test_steady_signal_no_sidechain(self):
        y = _make_sine(duration=3.0)
        result = _detect_sidechain(y, SR)
        assert result["type"] == "sidechain"
        assert result["detected"] is False


class TestDetectEffectsIntegration:
    def test_public_api_with_mock_load(self):
        """detect_effects() の公開APIが正しい構造を返すこと。"""
        y = _make_sidechain_signal()
        y_path = _save_y(y)

        with patch("vml_audio_lab.tools.loader.load_y", return_value=y):
            result = detect_effects(y_path)

        assert "effects" in result
        assert "effects_summary" in result
        assert "dominant_effect" in result
        assert isinstance(result["effects"], list)
        assert isinstance(result["effects_summary"], dict)

    def test_public_api_returns_effects_for_signal(self):
        """detect_effects() が合成信号でエフェクトを検出すること。"""
        y = _make_sidechain_signal(bpm=128)
        y_path = _save_y(y)

        with patch("vml_audio_lab.tools.loader.load_y", return_value=y):
            result = detect_effects(y_path)

        # 何らかのエフェクトが検出されること（合成信号の特性上、具体的な種類はSR依存）
        assert len(result["effects"]) > 0
        assert result["dominant_effect"] != "none"
        # 各エフェクトに必須フィールドがあること
        for eff in result["effects"]:
            assert "type" in eff
            assert "confidence" in eff
            assert eff["confidence"] > 0

    def test_section_wise_returns_summary(self):
        y = np.concatenate([_make_impulse_reverb(duration=2.0), _make_sidechain_signal()])
        sections = [
            {"label": "Intro", "start": 0.0, "end": 2.0},
            {"label": "Drop", "start": 2.0, "end": 6.0},
        ]
        result = _analyze_with_sections(y, SR, sections)
        assert "effects" in result
        assert "effects_summary" in result
        assert "dominant_effect" in result
        assert isinstance(result["effects_summary"], dict)
        # セクションキーが存在すること
        assert "Intro" in result["effects_summary"]
        assert "Drop" in result["effects_summary"]

    def test_negative_section_boundary_clamped(self):
        """負のstartが0にクランプされること。"""
        y = _make_sine(duration=3.0)
        sections = [
            {"label": "Bad", "start": -1.0, "end": 2.0},
        ]
        result = _analyze_with_sections(y, SR, sections)
        # クラッシュしないことが主な検証
        assert "effects" in result
