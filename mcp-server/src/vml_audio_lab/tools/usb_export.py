"""USB 書き出しと Rekordbox XML 生成."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import quote

_HOT_CUE_RGB: dict[str, tuple[int, int, int]] = {
    "A": (40, 226, 20),
    "B": (230, 40, 40),
    "C": (48, 90, 255),
    "D": (224, 100, 27),
}


def _sanitize_name(value: str) -> str:
    value = value.strip()
    forbidden = '<>:"/\\|?*'
    table = str.maketrans({c: "_" for c in forbidden})
    cleaned = value.translate(table).replace("\n", " ").replace("\r", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or "Unknown"


def split_artist_title(raw_title: str, fallback_artist: str = "Unknown Artist") -> tuple[str, str]:
    text = raw_title.strip()
    for sep in (" - ", " / "):
        if sep in text:
            left, right = text.split(sep, 1)
            if left.strip() and right.strip():
                return (_sanitize_name(left), _sanitize_name(right))
    return (_sanitize_name(fallback_artist), _sanitize_name(text or "Unknown Title"))


def bpm_bucket_label(bpm: float) -> str:
    if bpm <= 0:
        return "unknown-bpm"
    start = int(bpm // 5) * 5
    end = start + 4
    return f"{start}-{end}bpm"


def _normalize_genre_group(genre_group: str) -> str:
    normalized = _sanitize_name(genre_group.lower())
    return normalized if normalized else "unknown"


def copy_to_usb(
    audio_path: str,
    genre_group: str,
    bpm: float,
    title: str,
    artist: str,
    usb_path: str = "/Volumes/NONAME",
) -> dict:
    """音源を USB のジャンル/BPM フォルダへコピーする。"""
    usb_root = Path(usb_path).expanduser()
    if not usb_root.exists():
        raise FileNotFoundError(f"{usb_path} が見つかりません")

    src = Path(audio_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {src}")

    target_dir = usb_root / "tracks" / _normalize_genre_group(genre_group) / bpm_bucket_label(bpm)
    target_dir.mkdir(parents=True, exist_ok=True)

    ext = src.suffix.lower() or ".wav"
    target_name = f"{_sanitize_name(artist)} - {_sanitize_name(title)}{ext}"
    dst = target_dir / target_name

    if dst.exists():
        return {
            "audio_path": str(dst),
            "skipped": True,
            "reason": "duplicate_filename",
        }

    shutil.copy2(src, dst)
    return {
        "audio_path": str(dst),
        "skipped": False,
    }


def _ensure_xml_base(root: ET.Element) -> tuple[ET.Element, ET.Element, ET.Element]:
    collection = root.find("COLLECTION")
    if collection is None:
        collection = ET.SubElement(root, "COLLECTION", {"Entries": "0"})

    playlists = root.find("PLAYLISTS")
    if playlists is None:
        playlists = ET.SubElement(root, "PLAYLISTS")

    root_node = None
    for node in playlists.findall("NODE"):
        if node.get("Type") == "0" and node.get("Name") == "ROOT":
            root_node = node
            break
    if root_node is None:
        root_node = ET.SubElement(playlists, "NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
    return (collection, playlists, root_node)


def _init_xml_root() -> ET.Element:
    root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
    ET.SubElement(
        root,
        "PRODUCT",
        {
            "Name": "vml-audio-lab",
            "Version": "1.0.0",
            "Company": "VML",
        },
    )
    _ensure_xml_base(root)
    return root


def _next_track_id(collection: ET.Element) -> int:
    max_id = 0
    for track in collection.findall("TRACK"):
        try:
            max_id = max(max_id, int(track.get("TrackID", "0")))
        except ValueError:
            continue
    return max_id + 1


def _to_rekordbox_location(audio_path: str) -> str:
    path = Path(audio_path).expanduser().resolve()
    encoded = quote(path.as_posix(), safe="/")
    return f"file://localhost{encoded}"


def _genre_display_name(genre: str) -> str:
    if not genre:
        return "Unknown"
    return genre.replace("-", " ").title()


def _ensure_playlist(root_node: ET.Element, playlist_name: str) -> ET.Element:
    playlist = None
    for node in root_node.findall("NODE"):
        if node.get("Type") == "1" and node.get("Name") == playlist_name:
            playlist = node
            break
    if playlist is None:
        playlist = ET.SubElement(
            root_node,
            "NODE",
            {
                "Name": playlist_name,
                "Type": "1",
                "KeyType": "0",
                "Entries": "0",
            },
        )
    return playlist


def _append_position_marks(
    track: ET.Element,
    hot_cues: list[dict],
    memory_cues: list[dict],
) -> None:
    for cue in hot_cues:
        name = str(cue.get("name", "")).strip() or "Hot"
        start = float(cue.get("time_sec", 0.0))
        cue_attrs = {
            "Name": name,
            "Type": "0",
            "Start": f"{start:.3f}",
            "Num": str(max(0, ord(name[:1].upper()) - ord("A"))),
        }
        rgb = _HOT_CUE_RGB.get(name[:1].upper())
        if rgb:
            cue_attrs.update({"Red": str(rgb[0]), "Green": str(rgb[1]), "Blue": str(rgb[2])})
        ET.SubElement(track, "POSITION_MARK", cue_attrs)

    for cue in memory_cues:
        name = str(cue.get("name", "")).strip() or "Memory"
        start = float(cue.get("time_sec", 0.0))
        ET.SubElement(
            track,
            "POSITION_MARK",
            {"Name": name, "Type": "0", "Start": f"{start:.3f}", "Num": "-1"},
        )


def update_rekordbox_xml(
    xml_path: str,
    audio_path: str,
    title: str,
    artist: str,
    genre: str,
    bpm: float,
    key_label: str,
    duration_sec: float,
    hot_cues: list[dict],
    memory_cues: list[dict],
    playlist_name: str = "VML Analysis",
) -> dict:
    """Rekordbox XML にトラック情報を追記する。"""
    xml_file = Path(xml_path).expanduser()
    xml_file.parent.mkdir(parents=True, exist_ok=True)

    if xml_file.exists():
        root = ET.parse(xml_file).getroot()
    else:
        root = _init_xml_root()

    collection, _, root_node = _ensure_xml_base(root)
    location = _to_rekordbox_location(audio_path)

    existing = None
    for track in collection.findall("TRACK"):
        if track.get("Location") == location:
            existing = track
            break

    if existing is None:
        track_id_str = str(_next_track_id(collection))
        track = ET.SubElement(
            collection,
            "TRACK",
            {
                "TrackID": track_id_str,
                "DateAdded": date.today().isoformat(),
            },
        )
        xml_added = True
    else:
        track = existing
        track_id_str = str(track.get("TrackID", "")).strip()
        if not track_id_str:
            track_id_str = str(_next_track_id(collection))
            track.set("TrackID", track_id_str)
        if not track.get("DateAdded"):
            track.set("DateAdded", date.today().isoformat())
        xml_added = False

    track.set("Name", _sanitize_name(title))
    track.set("Artist", _sanitize_name(artist))
    track.set("Genre", _genre_display_name(genre))
    track.set("TotalTime", str(max(1, int(round(duration_sec)))))
    track.set("AverageBpm", f"{float(bpm):.2f}")
    track.set("Tonality", key_label)
    track.set("Location", location)

    for child in list(track):
        if child.tag in {"TEMPO", "POSITION_MARK"}:
            track.remove(child)

    ET.SubElement(
        track,
        "TEMPO",
        {
            "Inizio": "0.000",
            "Bpm": f"{float(bpm):.2f}",
            "Metro": "4/4",
            "Battito": "1",
        },
    )
    _append_position_marks(track=track, hot_cues=hot_cues, memory_cues=memory_cues)

    try:
        track_id = int(track_id_str)
    except ValueError:
        track_id = 0

    collection.set("Entries", str(len(collection.findall("TRACK"))))
    playlist = _ensure_playlist(root_node, playlist_name=playlist_name)

    key_value = track_id_str
    existing_keys = {node.get("Key", "") for node in playlist.findall("TRACK")}
    if key_value not in existing_keys:
        ET.SubElement(playlist, "TRACK", {"Key": key_value})
    playlist.set("Entries", str(len(playlist.findall("TRACK"))))
    root_node.set("Count", str(len([n for n in root_node.findall("NODE") if n.get("Type") == "1"])))

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "xml_path": str(xml_file),
        "track_id": track_id,
        "xml_added": xml_added,
    }
