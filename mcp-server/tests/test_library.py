"""ライブラリインデックス (scan_library) のテスト.

MVP-B: library.py — USB/XML からライブラリをインデックス化してキャッシュする。
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vml_audio_lab.tools.library import load_index, save_index, scan_library

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_rekordbox_xml(path: Path, tracks: list[dict]) -> None:
    """最小限の Rekordbox XML を生成する。"""
    root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
    collection = ET.SubElement(root, "COLLECTION", {"Entries": str(len(tracks))})
    for t in tracks:
        attrs: dict[str, str] = {
            "TrackID": str(t.get("id", "1")),
            "Name": str(t.get("name", "Track")),
            "Artist": str(t.get("artist", "")),
            "Genre": str(t.get("genre", "")),
            "TotalTime": str(t.get("duration", 180)),
            "Location": "file://localhost/music/track.mp3",
        }
        if "bpm" in t:
            attrs["AverageBpm"] = f"{t['bpm']:.2f}"
        if "key" in t:
            attrs["Tonality"] = str(t["key"])
        ET.SubElement(collection, "TRACK", attrs)
    ET.SubElement(root, "PLAYLISTS")
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def _make_wav(path: Path, duration: float = 1.0, sr: int = 22050) -> None:
    """テスト用 WAV ファイルを生成する。"""
    n = int(sr * duration)
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)
    sf.write(str(path), y, sr)


# ---------------------------------------------------------------------------
# XML パース系テスト
# ---------------------------------------------------------------------------


class TestScanLibraryFromXml:
    """scan_library: Rekordbox XML のパーステスト."""

    def test_returns_dict_with_tracks_key(self, tmp_path: Path) -> None:
        """scan_library は tracks キーを含む dict を返す。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Test Track", "bpm": 120.0, "key": "Fm"}])
        result = scan_library(str(xml_path))
        assert isinstance(result, dict)
        assert "tracks" in result

    def test_parses_track_id(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 42, "name": "Track A", "bpm": 128.0}])
        result = scan_library(str(xml_path))
        ids = [t["id"] for t in result["tracks"]]
        assert 42 in ids

    def test_parses_track_title(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "My Deep House Track", "bpm": 122.0}])
        result = scan_library(str(xml_path))
        assert result["tracks"][0]["title"] == "My Deep House Track"

    def test_parses_bpm(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "bpm": 125.5}])
        result = scan_library(str(xml_path))
        assert abs(float(result["tracks"][0]["bpm"]) - 125.5) < 0.1

    def test_parses_key_label(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "key": "Fm"}])
        result = scan_library(str(xml_path))
        assert result["tracks"][0]["key_label"] == "Fm"

    def test_parses_camelot_code_from_key(self, tmp_path: Path) -> None:
        """Tonality が Fm → camelot_code が 4A に変換される。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "key": "Fm"}])
        result = scan_library(str(xml_path))
        assert result["tracks"][0]["camelot_code"] == "4A"

    def test_parses_genre(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "genre": "Deep House"}])
        result = scan_library(str(xml_path))
        # canonicalize_genre_slug("Deep House") == "deep-house"
        assert result["tracks"][0]["genre"] == "deep-house"

    def test_parses_multiple_tracks(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        tracks = [
            {"id": i, "name": f"Track {i}", "bpm": 120.0 + i, "key": "Am"}
            for i in range(1, 6)
        ]
        _make_rekordbox_xml(xml_path, tracks)
        result = scan_library(str(xml_path))
        assert len(result["tracks"]) == 5

    def test_empty_library_returns_empty_tracks(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [])
        result = scan_library(str(xml_path))
        assert result["tracks"] == []

    def test_track_missing_bpm_defaults_to_zero(self, tmp_path: Path) -> None:
        """BPM が無いトラックは 0.0 を返す。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track without BPM"}])
        result = scan_library(str(xml_path))
        assert result["tracks"][0]["bpm"] == 0.0

    def test_track_missing_key_returns_empty_string(self, tmp_path: Path) -> None:
        """Key が無いトラックは空文字を返す。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track without Key"}])
        result = scan_library(str(xml_path))
        assert result["tracks"][0]["key_label"] == ""

    def test_returns_total_count(self, tmp_path: Path) -> None:
        """total フィールドが正しいトラック数を返す。"""
        xml_path = tmp_path / "library.xml"
        tracks = [{"id": i, "name": f"Track {i}"} for i in range(1, 4)]
        _make_rekordbox_xml(xml_path, tracks)
        result = scan_library(str(xml_path))
        assert result.get("total") == 3

    def test_camelot_code_converted_from_camelot_tonality(self, tmp_path: Path) -> None:
        """Tonality が Camelot 形式 (4A) の場合もキーラベルに変換される。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "key": "4A"}])
        result = scan_library(str(xml_path))
        track = result["tracks"][0]
        # 4A → Fm
        assert track["key_label"] == "Fm"
        assert track["camelot_code"] == "4A"


# ---------------------------------------------------------------------------
# ディレクトリスキャン系テスト
# ---------------------------------------------------------------------------


class TestScanLibraryFromDirectory:
    """scan_library: USBディレクトリスキャンのテスト."""

    def test_scans_wav_files_in_directory(self, tmp_path: Path) -> None:
        """WAV ファイルがあるディレクトリをスキャンできる。"""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        _make_wav(music_dir / "track1.wav")
        _make_wav(music_dir / "track2.wav")
        result = scan_library(str(music_dir))
        assert len(result["tracks"]) == 2

    def test_scans_nested_directories(self, tmp_path: Path) -> None:
        """ネストされたディレクトリも再帰スキャンする。"""
        music_dir = tmp_path / "music"
        subdir = music_dir / "Deep House"
        subdir.mkdir(parents=True)
        _make_wav(music_dir / "track1.wav")
        _make_wav(subdir / "track2.wav")
        result = scan_library(str(music_dir))
        assert len(result["tracks"]) >= 2

    def test_ignores_non_audio_files(self, tmp_path: Path) -> None:
        """音声ファイル以外は無視する。"""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        _make_wav(music_dir / "track.wav")
        (music_dir / "readme.txt").write_text("not music")
        (music_dir / "image.jpg").write_bytes(b"\xff\xd8\xff")
        result = scan_library(str(music_dir))
        assert len(result["tracks"]) == 1

    def test_empty_directory_returns_empty_tracks(self, tmp_path: Path) -> None:
        """空ディレクトリは空リストを返す。"""
        music_dir = tmp_path / "empty_music"
        music_dir.mkdir()
        result = scan_library(str(music_dir))
        assert result["tracks"] == []

    def test_track_has_file_path(self, tmp_path: Path) -> None:
        """各トラックに file_path フィールドがある。"""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        wav = music_dir / "track.wav"
        _make_wav(wav)
        result = scan_library(str(music_dir))
        assert len(result["tracks"]) == 1
        assert "file_path" in result["tracks"][0]

    def test_track_file_path_points_to_existing_file(self, tmp_path: Path) -> None:
        """file_path が実際のファイルを指している。"""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        wav = music_dir / "track.wav"
        _make_wav(wav)
        result = scan_library(str(music_dir))
        file_path = result["tracks"][0]["file_path"]
        assert Path(file_path).exists()


# ---------------------------------------------------------------------------
# キャッシュ系テスト
# ---------------------------------------------------------------------------


class TestScanLibraryCaching:
    """scan_library: キャッシュ機能のテスト."""

    def test_result_can_be_serialized_to_json(self, tmp_path: Path) -> None:
        """返却値は JSON シリアライズ可能である。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "bpm": 120.0, "key": "Am"}])
        result = scan_library(str(xml_path))
        # JSON シリアライズできること = キャッシュに保存できること
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert len(parsed["tracks"]) == len(result["tracks"])

    def test_fingerprint_present_in_result(self, tmp_path: Path) -> None:
        """fingerprint フィールドが返される。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "bpm": 120.0}])
        result = scan_library(str(xml_path))
        assert "fingerprint" in result
        assert isinstance(result["fingerprint"], str)
        assert len(result["fingerprint"]) > 0

    def test_second_call_returns_cached(self, tmp_path: Path) -> None:
        """同じソースに対して2回呼ぶと2回目はキャッシュから返す。"""
        xml_path = tmp_path / "library.xml"
        _make_rekordbox_xml(xml_path, [{"id": 1, "name": "Track", "bpm": 120.0}])
        cache_path = str(tmp_path / "cache.json")
        result1 = scan_library(str(xml_path), cache_path=cache_path)
        result2 = scan_library(str(xml_path), cache_path=cache_path)
        assert result2.get("cached") is True
        assert result1["tracks"] == result2["tracks"]

    def test_nonexistent_xml_raises_error(self) -> None:
        """存在しない XML はエラーになる。"""
        with pytest.raises(Exception):
            scan_library("/nonexistent/path/to/library.xml")


class TestSaveLoadIndex:
    """save_index / load_index: インデックス保存・読み込みテスト。"""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """保存したインデックスを読み込めること。"""
        index = {"tracks": [{"id": 1, "title": "Test"}], "total": 1, "fingerprint": "abc123"}
        cache_path = str(tmp_path / "index.json")
        save_index(index, cache_path)
        loaded = load_index(cache_path)
        assert loaded is not None
        assert loaded["tracks"] == index["tracks"]

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """存在しないキャッシュは None を返す。"""
        result = load_index(str(tmp_path / "nonexistent.json"))
        assert result is None
