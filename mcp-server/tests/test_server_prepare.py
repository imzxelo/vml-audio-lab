"""prepare_usb_track ツールのテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vml_audio_lab.server import prepare_usb_track


def test_prepare_usb_track_orchestrates_pipeline(monkeypatch, tmp_path: Path) -> None:
    usb_root = tmp_path / "NONAME"
    usb_root.mkdir(parents=True)
    local_audio = tmp_path / "input.wav"
    local_audio.write_bytes(b"RIFF....WAVE")

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
        "vml_audio_lab.server.detect_genre",
        lambda title, artist, y, sr: {
            "genre": "tech-house",
            "confidence": 0.85,
            "sources": {"youtube": "tech-house", "web": "tech-house", "audio": "house"},
            "genre_group": "house",
        },
    )
    monkeypatch.setattr(
        "vml_audio_lab.server.copy_to_usb",
        lambda audio_path, genre_group, bpm, title, artist, usb_path: {
            "audio_path": f"{usb_path}/tracks/house/125-129bpm/{artist} - {title}.wav",
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

    result = prepare_usb_track("https://youtube.com/watch?v=test001", usb_path=str(usb_root))

    assert result["status"] == "success"
    assert result["track"]["genre"] == "tech-house"
    assert result["track"]["genre_group"] == "house"
    assert result["files"]["copy_skipped"] is False
    assert str(usb_root / "rekordbox_library.xml") in result["files"]["xml"]
