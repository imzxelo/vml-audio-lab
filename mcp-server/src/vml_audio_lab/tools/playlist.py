"""スマートプレイリスト生成ツール.

Rekordbox XML にカメロットゾーン別・ムード別のプレイリストを生成する。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from vml_audio_lab.tools.camelot import compatible_camelot_codes, key_to_camelot


def _ensure_folder_node(parent: ET.Element, folder_name: str) -> ET.Element:
    """フォルダノードを取得または作成する。"""
    for node in parent.findall("NODE"):
        if node.get("Type") == "0" and node.get("Name") == folder_name:
            return node
    folder = ET.SubElement(
        parent,
        "NODE",
        {
            "Name": folder_name,
            "Type": "0",
            "Count": "0",
        },
    )
    return folder


def _ensure_playlist_node(parent: ET.Element, playlist_name: str) -> ET.Element:
    """プレイリストノードを取得または作成する。"""
    for node in parent.findall("NODE"):
        if node.get("Type") == "1" and node.get("Name") == playlist_name:
            return node
    playlist = ET.SubElement(
        parent,
        "NODE",
        {
            "Name": playlist_name,
            "Type": "1",
            "KeyType": "0",
            "Entries": "0",
        },
    )
    return playlist


def _add_track_to_playlist(playlist: ET.Element, track_id: str) -> None:
    """トラックIDをプレイリストに追加する（重複チェック付き）。"""
    existing_keys = {node.get("Key", "") for node in playlist.findall("TRACK")}
    if track_id not in existing_keys:
        ET.SubElement(playlist, "TRACK", {"Key": track_id})
    playlist.set("Entries", str(len(playlist.findall("TRACK"))))


def _update_folder_count(folder: ET.Element) -> None:
    """フォルダのカウントを更新する。"""
    count = len([n for n in folder.findall("NODE") if n.get("Type") == "1"])
    folder.set("Count", str(count))


def add_to_camelot_playlist(
    root_node: ET.Element,
    track_id: str,
    camelot_code: str,
) -> None:
    """Camelot ゾーン別プレイリストにトラックを追加する。

    Args:
        root_node: Rekordbox XML の ROOT ノード
        track_id: トラックID (文字列)
        camelot_code: Camelot コード (例: "4A", "8B")
    """
    if not camelot_code:
        return

    folder = _ensure_folder_node(root_node, "By Compatibility")
    playlist_name = f"Camelot {camelot_code} Zone"
    playlist = _ensure_playlist_node(folder, playlist_name)
    _add_track_to_playlist(playlist, track_id)
    _update_folder_count(folder)


def add_to_mood_playlist(
    root_node: ET.Element,
    track_id: str,
    mood: str,
) -> None:
    """ムード別プレイリストにトラックを追加する。

    Args:
        root_node: Rekordbox XML の ROOT ノード
        track_id: トラックID (文字列)
        mood: ムード名 (例: "Peak Time", "Deep Night")
    """
    if not mood:
        return

    folder = _ensure_folder_node(root_node, "By Mood")
    playlist = _ensure_playlist_node(folder, mood)
    _add_track_to_playlist(playlist, track_id)
    _update_folder_count(folder)


def generate_playlists(
    xml_path: str,
    track_id: str,
    key_label: str,
    mood: str,
) -> dict:
    """トラックを Camelot ゾーン + ムードプレイリストに追加する。

    既存の Rekordbox XML を更新する。XML が存在しない場合はスキップ。

    Args:
        xml_path: Rekordbox XML のパス
        track_id: トラックID (文字列)
        key_label: キーラベル (例: "Fm", "C")
        mood: ムード名 (例: "Peak Time")

    Returns:
        dict:
            - camelot_code: 割り当てられた Camelot コード
            - mood: ムード
            - camelot_playlist: 追加されたプレイリスト名
            - mood_playlist: 追加されたプレイリスト名
            - xml_path: 更新した XML パス
    """
    xml_file = Path(xml_path).expanduser()
    if not xml_file.exists():
        return {
            "camelot_code": None,
            "mood": mood,
            "camelot_playlist": None,
            "mood_playlist": None,
            "xml_path": str(xml_file),
            "skipped": True,
            "reason": "xml_not_found",
        }

    root = ET.parse(xml_file).getroot()

    # ROOT ノードを取得
    playlists_el = root.find("PLAYLISTS")
    if playlists_el is None:
        playlists_el = ET.SubElement(root, "PLAYLISTS")

    root_node = None
    for node in playlists_el.findall("NODE"):
        if node.get("Type") == "0" and node.get("Name") == "ROOT":
            root_node = node
            break
    if root_node is None:
        root_node = ET.SubElement(
            playlists_el,
            "NODE",
            {"Type": "0", "Name": "ROOT", "Count": "0"},
        )

    camelot_code = key_to_camelot(key_label)
    camelot_playlist_name = None
    mood_playlist_name = None

    if camelot_code:
        add_to_camelot_playlist(root_node, track_id, camelot_code)
        camelot_playlist_name = f"Camelot {camelot_code} Zone"

    if mood:
        add_to_mood_playlist(root_node, track_id, mood)
        mood_playlist_name = mood

    # ROOT ノードのカウントを更新
    root_node.set(
        "Count",
        str(len([n for n in root_node.findall("NODE") if n.get("Type") in ("0", "1")])),
    )

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "camelot_code": camelot_code,
        "mood": mood,
        "camelot_playlist": camelot_playlist_name,
        "mood_playlist": mood_playlist_name,
        "xml_path": str(xml_file),
        "skipped": False,
    }


def get_compatible_playlists(
    xml_path: str,
    key_label: str,
) -> list[str]:
    """指定キーと相性の良い Camelot プレイリスト名リストを返す。

    Args:
        xml_path: Rekordbox XML のパス
        key_label: キーラベル (例: "Fm", "C")

    Returns:
        相性の良いプレイリスト名のリスト
    """
    camelot_code = key_to_camelot(key_label)
    if not camelot_code:
        return []

    compatible = compatible_camelot_codes(camelot_code)
    return [f"Camelot {code} Zone" for code in compatible]
