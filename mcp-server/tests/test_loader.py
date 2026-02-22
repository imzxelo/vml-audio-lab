"""load_track のテスト"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.loader import DEFAULT_SR, _cache_key, _canonical_source, _is_youtube_url, load_track, load_y

FIXTURES = Path(__file__).parent / "fixtures"
SINE_WAV = FIXTURES / "sine_440hz_3s.wav"


class TestLoadTrackLocal:
    """ローカルファイルの読み込みテスト"""

    def test_returns_required_keys(self) -> None:
        result = load_track(str(SINE_WAV))
        for key in ("y_path", "sr", "duration_sec", "n_samples", "file_path", "source"):
            assert key in result

    def test_source_is_local(self) -> None:
        result = load_track(str(SINE_WAV))
        assert result["source"] == "local"

    def test_sample_rate(self) -> None:
        result = load_track(str(SINE_WAV))
        assert result["sr"] == DEFAULT_SR

    def test_duration_approximately_3s(self) -> None:
        result = load_track(str(SINE_WAV))
        assert abs(result["duration_sec"] - 3.0) < 0.1

    def test_y_path_exists_and_loadable(self) -> None:
        result = load_track(str(SINE_WAV))
        y = np.load(result["y_path"])
        assert len(y) > 0
        assert y.dtype == np.float32

    def test_offset_and_duration(self) -> None:
        result = load_track(str(SINE_WAV), offset=1.0, duration=1.0)
        assert abs(result["duration_sec"] - 1.0) < 0.1

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_track("/nonexistent/audio.wav")

    def test_file_path_is_resolved(self) -> None:
        result = load_track(str(SINE_WAV))
        assert Path(result["file_path"]).is_absolute()

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "not_audio.txt"
        txt_file.write_text("not audio")
        with pytest.raises(ValueError, match="サポートされていないファイル形式"):
            load_track(str(txt_file))

    def test_cache_hit_returns_same_data(self) -> None:
        result1 = load_track(str(SINE_WAV))
        result2 = load_track(str(SINE_WAV))
        y1 = np.load(result1["y_path"])
        y2 = np.load(result2["y_path"])
        np.testing.assert_array_equal(y1, y2)

    def test_cache_hit_recovers_missing_meta(self) -> None:
        import tempfile

        source = str(SINE_WAV)
        result1 = load_track(source)

        cache_dir = Path(tempfile.gettempdir()) / "vml_audio_lab"
        key = _cache_key(_canonical_source(source))
        meta_path = cache_dir / f"{key}_meta.json"
        if meta_path.exists():
            meta_path.unlink()

        result2 = load_track(source)
        assert result2["file_path"]
        assert Path(result2["file_path"]).exists()
        assert meta_path.exists()


class TestLoadY:
    """load_y セキュリティテスト"""

    def test_loads_valid_cached_data(self) -> None:
        result = load_track(str(SINE_WAV))
        y = load_y(result["y_path"])
        assert len(y) > 0

    def test_rejects_path_outside_cache_dir(self) -> None:
        with pytest.raises(ValueError, match="キャッシュディレクトリ外"):
            load_y("/etc/passwd")

    def test_rejects_nonexistent_path_in_cache_dir(self) -> None:
        import tempfile

        cache_dir = Path(tempfile.gettempdir()) / "vml_audio_lab"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake_path = str(cache_dir / "nonexistent.npy")
        with pytest.raises(FileNotFoundError):
            load_y(fake_path)


class TestIsYoutubeUrl:
    """YouTube URL 判定テスト"""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?si=ABC123",
            "https://www.youtube.com/shorts/abc123DEF-_",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=abc123",
        ],
    )
    def test_valid_youtube_urls(self, url: str) -> None:
        assert _is_youtube_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "/path/to/local/file.wav",
            "https://example.com/video.mp4",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
            "not a url at all",
            "",
        ],
    )
    def test_non_youtube_urls(self, url: str) -> None:
        assert _is_youtube_url(url) is False


class TestLoadTrackYouTube:
    """YouTube ダウンロードのテスト（モック使用）"""

    def test_youtube_url_triggers_download(self, tmp_path: Path) -> None:
        # テスト用の wav ファイルを作成
        sr = 22050
        y_fake = np.zeros(sr * 2, dtype=np.float32)
        wav_path = tmp_path / "fake_yt.wav"
        sf.write(str(wav_path), y_fake, sr)

        fake_info = {
            "title": "Test Video",
            "uploader": "Test Channel",
        }

        # 前回テストのキャッシュが残っている可能性があるので削除
        import tempfile

        url = "https://youtu.be/test123ABC"
        cache_dir = Path(tempfile.gettempdir()) / "vml_audio_lab"
        cache_key = _cache_key(_canonical_source(url))
        cached_npy = cache_dir / f"{cache_key}_y.npy"
        cached_meta = cache_dir / f"{cache_key}_meta.json"
        if cached_npy.exists():
            cached_npy.unlink()
        if cached_meta.exists():
            cached_meta.unlink()

        with patch("vml_audio_lab.tools.loader._download_youtube") as mock_dl:
            mock_dl.return_value = (str(wav_path), {**fake_info, "youtube_url": url})

            result = load_track(url)

        assert result["source"] == "youtube"
        assert result["title"] == "Test Video"
        assert result["uploader"] == "Test Channel"
        assert result["duration_sec"] > 0

    def test_youtube_download_failure(self) -> None:
        with patch("vml_audio_lab.tools.loader._download_youtube") as mock_dl:
            mock_dl.side_effect = RuntimeError("download failed")

            with pytest.raises(RuntimeError, match="download failed"):
                load_track("https://youtu.be/fail123ABC")
