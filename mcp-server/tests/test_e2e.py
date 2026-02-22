"""E2Eテスト: load → analyze → structure → visualize の一気通貫"""

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.analysis import detect_bpm, detect_key, energy_curve
from vml_audio_lab.tools.loader import load_track
from vml_audio_lab.tools.structure import detect_structure
from vml_audio_lab.tools.visualize import spectrogram, waveform_overview


@pytest.fixture(scope="module")
def realistic_track(tmp_path_factory: pytest.TempPathFactory) -> str:
    """E2Eテスト用: 構造変化のある楽曲風音声（2分）を生成する。

    - 0:00-0:20 Intro: 静かなパッド（Am, 220Hz）
    - 0:20-0:50 Build: ビート追加 + 音量上昇
    - 0:50-1:20 Drop: フルエネルギー（高周波ハーモニクス追加）
    - 1:20-1:40 Break: キック抜き、パッドのみ
    - 1:40-2:00 Outro: フェードアウト
    """
    sr = 22050
    rng = np.random.default_rng(42)

    def make_section(dur: float, amp: float, freq: float, add_beat: bool = False) -> np.ndarray:
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        y = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # ハーモニクス追加
        y += (amp * 0.3) * np.sin(2 * np.pi * freq * 2 * t).astype(np.float32)
        y += (amp * 0.1) * np.sin(2 * np.pi * freq * 3 * t).astype(np.float32)
        if add_beat:
            # 120BPM のクリック
            interval = 0.5  # 120BPM = 0.5s interval
            beat_t = 0.0
            while beat_t < dur:
                idx = int(beat_t * sr)
                click_len = min(100, n - idx)
                if click_len > 0:
                    y[idx : idx + click_len] += amp * 0.5 * np.sin(
                        2 * np.pi * 80 * np.arange(click_len) / sr
                    ).astype(np.float32)
                beat_t += interval
        # 軽いノイズ
        y += rng.normal(0, amp * 0.02, n).astype(np.float32)
        return y

    parts = [
        make_section(20, 0.1, 220),                   # Intro
        make_section(30, 0.4, 220, add_beat=True),     # Build
        make_section(30, 0.9, 440, add_beat=True),     # Drop
        make_section(20, 0.15, 220),                   # Break
        make_section(20, 0.08, 220),                   # Outro
    ]
    y_full = np.concatenate(parts)

    wav_path = tmp_path_factory.mktemp("e2e") / "test_track_2min.wav"
    sf.write(str(wav_path), y_full, sr)
    return str(wav_path)


class TestE2EPipeline:
    """全パイプラインの統合テスト"""

    def test_full_pipeline(self, realistic_track: str) -> None:
        # Step 1: Load
        loaded = load_track(realistic_track)
        assert loaded["source"] == "local"
        assert loaded["duration_sec"] > 100  # 2分 = 120秒
        y_path = loaded["y_path"]

        # Step 2: 基礎分析
        bpm_result = detect_bpm(y_path)
        assert bpm_result["bpm"] >= 0

        key_result = detect_key(y_path)
        assert key_result["key_label"]
        assert key_result["scale"] in ("major", "minor")

        energy_png = energy_curve(y_path)
        assert energy_png[:4] == b"\x89PNG"

        # Step 3: 構造認識
        structure_result = detect_structure(y_path)
        sections = structure_result["sections"]
        assert len(sections) >= 2
        assert sections[0]["label"] == "Intro"
        assert sections[-1]["label"] == "Outro"

        # Step 4: 可視化
        spec_png = spectrogram(y_path)
        assert spec_png[:4] == b"\x89PNG"

        wave_png = waveform_overview(y_path, sections=sections)
        assert wave_png[:4] == b"\x89PNG"

        # Step 5: 解説に必要な情報が全て揃っているか
        assert bpm_result["bpm"] is not None
        assert key_result["key"] is not None
        assert loaded["duration_sec"] is not None
        assert len(sections) > 0
        assert len(energy_png) > 1000
        assert len(spec_png) > 1000
        assert len(wave_png) > 1000

    def test_pipeline_produces_consistent_data(self, realistic_track: str) -> None:
        """2回実行しても同じ結果が返ることを確認"""
        loaded1 = load_track(realistic_track)
        loaded2 = load_track(realistic_track)
        assert loaded1["duration_sec"] == loaded2["duration_sec"]
        assert loaded1["sr"] == loaded2["sr"]

        key1 = detect_key(loaded1["y_path"])
        key2 = detect_key(loaded2["y_path"])
        assert key1["key_label"] == key2["key_label"]
