"""USB export / Rekordbox XML のテスト."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from vml_audio_lab.tools.usb_export import copy_to_usb, update_rekordbox_xml


def test_copy_to_usb_creates_expected_folder_and_file(tmp_path: Path) -> None:
    usb = tmp_path / "NONAME"
    usb.mkdir(parents=True)
    src = tmp_path / "source.wav"
    src.write_bytes(b"RIFF....WAVE")

    result = copy_to_usb(
        audio_path=str(src),
        genre_group="house",
        bpm=128.0,
        title="Title/01",
        artist="Artist:01",
        usb_path=str(usb),
    )

    assert result["skipped"] is False
    copied = Path(result["audio_path"])
    assert copied.exists()
    assert "tracks/house/125-129bpm" in copied.as_posix()
    assert copied.name == "Artist_01 - Title_01.wav"


def test_copy_to_usb_skips_duplicate_file(tmp_path: Path) -> None:
    usb = tmp_path / "NONAME"
    usb.mkdir(parents=True)
    src = tmp_path / "source.wav"
    src.write_bytes(b"RIFF....WAVE")

    first = copy_to_usb(
        audio_path=str(src),
        genre_group="techno",
        bpm=132.0,
        title="Same",
        artist="Dup",
        usb_path=str(usb),
    )
    second = copy_to_usb(
        audio_path=str(src),
        genre_group="techno",
        bpm=132.0,
        title="Same",
        artist="Dup",
        usb_path=str(usb),
    )

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["audio_path"] == first["audio_path"]


def test_update_rekordbox_xml_creates_collection_and_playlist(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF....WAVE")
    xml_path = tmp_path / "rekordbox_library.xml"

    result = update_rekordbox_xml(
        xml_path=str(xml_path),
        audio_path=str(audio),
        title="My Track",
        artist="My Artist",
        genre="tech-house",
        bpm=128.0,
        key_label="Fm",
        duration_sec=228.0,
        hot_cues=[
            {"name": "A", "time_sec": 28.5},
            {"name": "B", "time_sec": 72.0},
            {"name": "C", "time_sec": 125.0},
            {"name": "D", "time_sec": 155.0},
        ],
        memory_cues=[{"name": "Intro", "time_sec": 0.095}],
    )

    assert result["track_id"] == 1
    assert result["xml_added"] is True
    assert xml_path.exists()

    root = ET.parse(xml_path).getroot()
    track = root.find("./COLLECTION/TRACK")
    assert track is not None
    assert track.get("TotalTime") == "228"
    assert track.get("Location", "").startswith("file://localhost/")
    assert track.get("Genre") == "Tech House"

    playlist_root = root.find("./PLAYLISTS/NODE[@Type='0'][@Name='ROOT']")
    assert playlist_root is not None
    playlist = playlist_root.find("./NODE[@Type='1'][@Name='VML Analysis']")
    assert playlist is not None
    assert len(playlist.findall("TRACK")) == 1

    marks = track.findall("POSITION_MARK")
    assert len(marks) == 5


def test_update_rekordbox_xml_does_not_duplicate_existing_track(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF....WAVE")
    xml_path = tmp_path / "rekordbox_library.xml"

    first = update_rekordbox_xml(
        xml_path=str(xml_path),
        audio_path=str(audio),
        title="Track",
        artist="Artist",
        genre="house",
        bpm=124.0,
        key_label="C",
        duration_sec=180.0,
        hot_cues=[],
        memory_cues=[],
    )
    second = update_rekordbox_xml(
        xml_path=str(xml_path),
        audio_path=str(audio),
        title="Track",
        artist="Artist",
        genre="house",
        bpm=124.0,
        key_label="C",
        duration_sec=180.0,
        hot_cues=[],
        memory_cues=[],
    )

    assert first["track_id"] == second["track_id"]
    assert second["xml_added"] is False

    root = ET.parse(xml_path).getroot()
    tracks = root.findall("./COLLECTION/TRACK")
    assert len(tracks) == 1


def test_update_rekordbox_xml_updates_existing_track_payload(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF....WAVE")
    xml_path = tmp_path / "rekordbox_library.xml"

    first = update_rekordbox_xml(
        xml_path=str(xml_path),
        audio_path=str(audio),
        title="Track V1",
        artist="Artist V1",
        genre="house",
        bpm=124.0,
        key_label="Cm",
        duration_sec=180.0,
        hot_cues=[{"name": "A", "time_sec": 12.0}],
        memory_cues=[{"name": "Intro", "time_sec": 0.0}],
    )
    second = update_rekordbox_xml(
        xml_path=str(xml_path),
        audio_path=str(audio),
        title="Track V2",
        artist="Artist V2",
        genre="tech-house",
        bpm=128.0,
        key_label="Fm",
        duration_sec=210.0,
        hot_cues=[{"name": "B", "time_sec": 72.0}],
        memory_cues=[{"name": "Drop1", "time_sec": 72.0}],
    )

    assert first["track_id"] == second["track_id"]
    assert second["xml_added"] is False

    root = ET.parse(xml_path).getroot()
    tracks = root.findall("./COLLECTION/TRACK")
    assert len(tracks) == 1

    track = tracks[0]
    assert track.get("Name") == "Track V2"
    assert track.get("Artist") == "Artist V2"
    assert track.get("Genre") == "Tech House"
    assert track.get("AverageBpm") == "128.00"
    assert track.get("Tonality") == "Fm"
    assert track.get("TotalTime") == "210"

    tempo_nodes = track.findall("TEMPO")
    assert len(tempo_nodes) == 1
    assert tempo_nodes[0].get("Bpm") == "128.00"

    marks = track.findall("POSITION_MARK")
    assert len(marks) == 2
    assert {m.get("Name") for m in marks} == {"B", "Drop1"}

    playlist_tracks = root.findall(
        "./PLAYLISTS/NODE[@Type='0'][@Name='ROOT']/NODE[@Type='1'][@Name='VML Analysis']/TRACK"
    )
    assert len(playlist_tracks) == 1
