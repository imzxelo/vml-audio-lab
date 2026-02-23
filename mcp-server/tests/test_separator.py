"""ステム分離ツール (extract_vocals) のテスト.

MVP-C: separator.py — demucs 4ステム分離。
demucs を使う重いテストは @pytest.mark.slow でマーク。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.separator import _all_stems_cached, _stem_cache_dir, extract_vocals

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_wav(tmp_path_factory: pytest.TempPathFactory) -> str:
    """3秒のモノラルサイン波 WAV ファイルを生成して返す。"""
    sr = 22050
    duration = 3.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path_factory.mktemp("audio") / "test_3s.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


# ---------------------------------------------------------------------------
# エラーハンドリングテスト (demucs 不要)
# ---------------------------------------------------------------------------


class TestExtractVocalsErrors:
    """extract_vocals: エラーハンドリングのテスト (demucs なし)。"""

    def test_nonexistent_file_raises_file_not_found(self) -> None:
        """存在しない音声ファイルは FileNotFoundError を発生させる。"""
        with pytest.raises(FileNotFoundError):
            extract_vocals("/nonexistent/audio.wav")

    def test_nonexistent_file_path_absolute(self) -> None:
        """絶対パスが存在しない場合も FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            extract_vocals("/tmp/vml_audio_lab_nonexistent_12345/test.wav")


# ---------------------------------------------------------------------------
# キャッシュヘルパーのユニットテスト (demucs 不要)
# ---------------------------------------------------------------------------


class TestStemCacheHelpers:
    """_stem_cache_dir / _all_stems_cached: キャッシュヘルパーのユニットテスト。"""

    def test_stem_cache_dir_returns_path(self, tmp_path: Path) -> None:
        """_stem_cache_dir は Path を返す。"""
        cache_dir = _stem_cache_dir("/some/audio/file.wav")
        assert isinstance(cache_dir, Path)

    def test_stem_cache_dir_is_deterministic(self) -> None:
        """同じパスに対して毎回同じキャッシュディレクトリを返す。"""
        path = "/music/track.wav"
        assert _stem_cache_dir(path) == _stem_cache_dir(path)

    def test_stem_cache_dir_differs_for_different_files(self) -> None:
        """異なるファイルは異なるキャッシュディレクトリを返す。"""
        dir_a = _stem_cache_dir("/music/track_a.wav")
        dir_b = _stem_cache_dir("/music/track_b.wav")
        assert dir_a != dir_b

    def test_all_stems_cached_false_when_no_files(self, tmp_path: Path) -> None:
        """ステムファイルが存在しない場合は False を返す。"""
        empty_dir = tmp_path / "no_stems"
        empty_dir.mkdir()
        assert _all_stems_cached(empty_dir) is False

    def test_all_stems_cached_true_when_all_present(self, tmp_path: Path) -> None:
        """4つのステムファイルが全て存在する場合は True を返す。"""
        stem_dir = tmp_path / "stems"
        stem_dir.mkdir()
        sr = 22050
        n = int(sr * 1.0)
        y = np.zeros(n, dtype=np.float32)
        for stem in ("vocals", "drums", "bass", "other"):
            sf.write(str(stem_dir / f"{stem}.wav"), y, sr)
        assert _all_stems_cached(stem_dir) is True

    def test_all_stems_cached_false_when_partial(self, tmp_path: Path) -> None:
        """一部のステムしかない場合は False を返す。"""
        stem_dir = tmp_path / "partial_stems"
        stem_dir.mkdir()
        sr = 22050
        n = int(sr * 1.0)
        y = np.zeros(n, dtype=np.float32)
        # vocals だけ作成
        sf.write(str(stem_dir / "vocals.wav"), y, sr)
        assert _all_stems_cached(stem_dir) is False


# ---------------------------------------------------------------------------
# キャッシュ統合テスト (demucs をモックしてキャッシュ動作確認)
# ---------------------------------------------------------------------------


class TestExtractVocalsCachingWithMock:
    """extract_vocals: キャッシュ機能のテスト (demucs モック使用)。"""

    def _create_fake_stems(self, cache_dir: Path) -> None:
        """テスト用のフェイクステムファイルを作成する。"""
        cache_dir.mkdir(parents=True, exist_ok=True)
        sr = 22050
        n = int(sr * 1.0)
        y = np.zeros(n, dtype=np.float32)
        for stem in ("vocals", "drums", "bass", "other"):
            sf.write(str(cache_dir / f"{stem}.wav"), y, sr)

    def test_returns_cached_when_stems_exist(self, short_wav: str) -> None:
        """ステムがキャッシュ済みの場合はキャッシュから返す。"""
        # キャッシュディレクトリを事前に作成
        cache_dir = _stem_cache_dir(short_wav)
        self._create_fake_stems(cache_dir)

        result = extract_vocals(short_wav)
        assert result.get("cached") is True
        assert result["model"] == "htdemucs"

    def test_cached_result_has_4_stems(self, short_wav: str) -> None:
        """キャッシュヒット時も4ステムのパスを返す。"""
        cache_dir = _stem_cache_dir(short_wav)
        self._create_fake_stems(cache_dir)

        result = extract_vocals(short_wav)
        for stem in ("vocals", "drums", "bass", "other"):
            assert stem in result
            assert result[stem].endswith(".wav")

    def test_cached_stem_paths_exist(self, short_wav: str) -> None:
        """キャッシュから返されたパスが実際に存在する。"""
        cache_dir = _stem_cache_dir(short_wav)
        self._create_fake_stems(cache_dir)

        result = extract_vocals(short_wav)
        for stem in ("vocals", "drums", "bass", "other"):
            assert Path(result[stem]).exists(), f"{stem} file does not exist"


# ---------------------------------------------------------------------------
# demucs 実体テスト (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExtractVocalsWithDemucs:
    """extract_vocals: 実際の demucs を使う重いテスト。CI では skip。"""

    def test_real_demucs_4_stems(self, short_wav: str) -> None:
        """実 demucs で4ステムが生成される。"""
        # キャッシュをクリアして再実行
        cache_dir = _stem_cache_dir(short_wav)
        import shutil

        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        result = extract_vocals(short_wav)
        for stem in ("vocals", "drums", "bass", "other"):
            assert stem in result
            assert Path(result[stem]).exists()

    def test_real_demucs_cached_is_false_first_time(self, tmp_path: Path) -> None:
        """初回実行は cached=False を返す。"""
        sr = 22050
        n = int(sr * 3.0)
        t = np.linspace(0, 3.0, n, endpoint=False)
        y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav = tmp_path / "fresh_track.wav"
        sf.write(str(wav), y, sr)

        result = extract_vocals(str(wav))
        assert result.get("cached") is False

    def test_real_demucs_second_call_is_cached(self, short_wav: str) -> None:
        """2回目の呼び出しは cached=True を返す。"""
        extract_vocals(short_wav)  # 1回目
        result2 = extract_vocals(short_wav)  # 2回目
        assert result2.get("cached") is True

    def test_real_demucs_vocals_is_readable_wav(self, short_wav: str) -> None:
        """vocals ステムが読み込み可能な WAV ファイル。"""
        result = extract_vocals(short_wav)
        y, loaded_sr = sf.read(result["vocals"])
        assert len(y) > 0
        assert loaded_sr > 0
