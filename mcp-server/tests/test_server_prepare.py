"""prepare_usb_track ツールのテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vml_audio_lab.server import prepare_usb_track


def _patch_common(monkeypatch, local_audio: Path) -> None:
    monkeypatch.setattr(
        "vml_audio_lab.server.load_track",
        lambda url: {
            "y_path": "/tmp/fake_y.npy",
            "sr": 44100,
            "file_path": str(local_audio),
            "title": "Artist Name - Track Name",
            "uploader": "Uploader",
            "duration_sec": 210.0,
        },
    )
    monkeypatch.setattr("vml_audio_lab.server.load_y", lambda y_path: np.zeros(44100, dtype=np.float32))
    monkeypatch.setattr("vml_audio_lab.server.detect_bpm", lambda y_path: {"bpm": 128.0})
    monkeypatch.setattr(
        "vml_audio_lab.server.detect_key",
        lambda y_path: {"key_label": "Fm", "key": "F", "scale": "minor", "strength": 0.6},
    )
    monkeypatch.setattr(
        "vml_audio_lab.server.recommend_cues",
        lambda y_path: {
            "duration_sec": 210.0,
            "hot_cues": [{"name": "A", "time_sec": 10.0}],
            "memory_cues": [{"name": "Intro", "time_sec": 0.0}],
        },
    )
    monkeypatch.setattr(
        "vml_audio_lab.server.copy_to_usb",
        lambda audio_path, genre_group, bpm, title, artist, usb_path: {
            "audio_path": f"{usb_path}/tracks/{genre_group}/125-129bpm/{artist} - {title}.wav",
            "skipped": False,
        },
    )
    monkeypatch.setattr(
        "vml_audio_lab.server.update_rekordbox_xml",
        lambda **kwargs: {
            "xml_path": kwargs["xml_path"],
            "track_id": 1,
            "xml_added": True,
        },
    )


def test_prepare_usb_track_orchestrates_pipeline(monkeypatch, tmp_path: Path) -> None:
    usb_root = tmp_path / "NONAME"
    usb_root.mkdir(parents=True)
    local_audio = tmp_path / "input.wav"
    local_audio.write_bytes(b"RIFF....WAVE")
    _patch_common(monkeypatch, local_audio)

    monkeypatch.setattr(
        "vml_audio_lab.server.detect_genre",
        lambda title, artist, y, sr, bpm=None: {
            "genre": "tech-house",
            "confidence": 0.85,
            "sources": {"youtube": "tech-house", "web": "tech-house", "audio": "house"},
            "genre_group": "house",
        },
    )

    result = prepare_usb_track("https://youtube.com/watch?v=test001", usb_path=str(usb_root))

    assert result["status"] == "success"
    assert result["track"]["genre"] == "tech-house"
    assert result["track"]["genre_group"] == "house"
    assert result["files"]["copy_skipped"] is False
    assert str(usb_root / "rekordbox_library.xml") in result["files"]["xml"]


def test_prepare_usb_track_normalizes_genre_override(monkeypatch, tmp_path: Path) -> None:
    usb_root = tmp_path / "NONAME"
    usb_root.mkdir(parents=True)
    local_audio = tmp_path / "input.wav"
    local_audio.write_bytes(b"RIFF....WAVE")
    _patch_common(monkeypatch, local_audio)

    def _should_not_call(title, artist, y, sr):
        raise AssertionError("detect_genre should not be called")

    monkeypatch.setattr("vml_audio_lab.server.detect_genre", _should_not_call)

    result = prepare_usb_track(
        "https://youtube.com/watch?v=test001",
        genre_override="tech house",
        usb_path=str(usb_root),
    )

    assert result["track"]["genre"] == "tech-house"
    assert result["track"]["genre_group"] == "house"
    assert result["genre_detection"]["final"] == "tech-house"


def test_prepare_usb_track_blank_override_falls_back_to_detect(monkeypatch, tmp_path: Path) -> None:
    usb_root = tmp_path / "NONAME"
    usb_root.mkdir(parents=True)
    local_audio = tmp_path / "input.wav"
    local_audio.write_bytes(b"RIFF....WAVE")
    _patch_common(monkeypatch, local_audio)

    called = {"detect_genre": False}

    def _detect_genre(title, artist, y, sr, bpm=None):
        called["detect_genre"] = True
        return {
            "genre": "techno",
            "confidence": 0.72,
            "sources": {"youtube": "unknown", "web": "techno", "audio": "techno"},
            "genre_group": "techno",
        }

    monkeypatch.setattr("vml_audio_lab.server.detect_genre", _detect_genre)

    result = prepare_usb_track(
        "https://youtube.com/watch?v=test001",
        genre_override="   ",
        usb_path=str(usb_root),
    )

    assert called["detect_genre"] is True
    assert result["track"]["genre"] == "techno"
    assert result["track"]["genre_group"] == "techno"
