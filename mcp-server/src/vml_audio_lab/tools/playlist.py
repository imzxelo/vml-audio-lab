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
    track_info: dict | None = None,
    vocal_analysis: dict | None = None,
    compatible_tracks: list[dict] | None = None,
) -> dict:
    """トラックを Camelot ゾーン + ムード + Sampling Ideas プレイリストに追加する。

    既存の Rekordbox XML を更新する。XML が存在しない場合はスキップ。

    Args:
        xml_path: Rekordbox XML のパス
        track_id: トラックID (文字列)
        key_label: キーラベル (例: "Fm", "C")
        mood: ムード名 (例: "Peak Time")
        track_info: トラック情報 dict (genre 等)。Sampling Ideas 判定に使用 (省略可)
        vocal_analysis: analyze_vocal() の戻り値 dict (省略可)
        compatible_tracks: マッチしたトラックリスト (Transition Pairs 用、省略可)

    Returns:
        dict:
            - camelot_code: 割り当てられた Camelot コード
            - mood: ムード
            - camelot_playlist: 追加されたプレイリスト名
            - mood_playlist: 追加されたプレイリスト名
            - sampling_added: Sampling Ideas に追加された場合 True (vocal_analysis 指定時のみ)
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
    sampling_added = False

    if camelot_code:
        add_to_camelot_playlist(root_node, track_id, camelot_code)
        camelot_playlist_name = f"Camelot {camelot_code} Zone"

    if mood:
        add_to_mood_playlist(root_node, track_id, mood)
        mood_playlist_name = mood

    # Sampling Ideas: vocal_analysis と track_info が揃っている場合のみ
    if track_info is not None and vocal_analysis is not None:
        if _qualifies_for_vocal_for_house(track_info, vocal_analysis):
            _add_to_sampling_folder(root_node, track_id, _VOCAL_FOR_HOUSE_PLAYLIST)
            sampling_added = True
        for match in (compatible_tracks or []):
            match_id = str(match.get("track_id", ""))
            if match_id:
                _add_to_sampling_folder(root_node, match_id, _TRANSITION_PAIRS_PLAYLIST)

    # ROOT ノードのカウントを更新 (直下の全 NODE 数)
    root_node.set(
        "Count",
        str(len(root_node.findall("NODE"))),
    )

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "camelot_code": camelot_code,
        "mood": mood,
        "camelot_playlist": camelot_playlist_name,
        "mood_playlist": mood_playlist_name,
        "sampling_added": sampling_added,
        "xml_path": str(xml_file),
        "skipped": False,
    }


_SAMPLING_FOLDER = "Sampling Ideas"
_VOCAL_FOR_HOUSE_PLAYLIST = "Vocal for House"
_TRANSITION_PAIRS_PLAYLIST = "Transition Pairs"

# Vocal for House 判定に使うジャンル
_VOCAL_SOURCE_GENRES = {"hiphop", "jpop"}
_HOUSE_TARGET_GENRES = {"house", "deep-house"}


def _add_to_sampling_folder(
    root_node: ET.Element,
    track_id: str,
    playlist_name: str = _VOCAL_FOR_HOUSE_PLAYLIST,
) -> None:
    """Sampling Ideas フォルダのサンプリング候補プレイリストにトラックを追加する（内部用）。

    Args:
        root_node: Rekordbox XML の ROOT ノード
        track_id: トラックID (文字列)
        playlist_name: プレイリスト名 ("Vocal for House" または "Transition Pairs")
    """
    if not track_id or not playlist_name:
        return

    folder = _ensure_folder_node(root_node, _SAMPLING_FOLDER)
    playlist = _ensure_playlist_node(folder, playlist_name)
    _add_track_to_playlist(playlist, track_id)
    _update_folder_count(folder)


def _qualifies_for_vocal_for_house(
    track_info: dict,
    vocal_analysis: dict,
) -> bool:
    """Vocal for House プレイリスト追加基準を満たすか判定する。

    基準:
    - track_info の genre が hiphop または jpop
    - vocal_analysis の usable_sections に vocal_clarity > 0.7 のセクションが存在する
    - vocal_analysis の compatible_genres に house または deep-house が含まれる

    Args:
        track_info: トラック情報 dict (genre キーを含む)
        vocal_analysis: analyze_vocal() の戻り値 dict

    Returns:
        True なら Vocal for House に追加すべき
    """
    genre = str(track_info.get("genre", "")).lower().replace("-", "").replace(" ", "")
    # "hiphop" / "jpop" に正規化してマッチ
    normalized_source_genres = {"hiphop", "jpop"}
    if genre not in normalized_source_genres:
        return False

    usable = vocal_analysis.get("usable_sections", [])
    has_clear_vocal = any(
        float(s.get("vocal_clarity", 0)) > 0.7 for s in usable
    )
    if not has_clear_vocal:
        return False

    compatible = {g.lower() for g in vocal_analysis.get("compatible_genres", [])}
    if not (compatible & _HOUSE_TARGET_GENRES):
        return False

    return True


def add_to_sampling_playlist(
    xml_path: str,
    track_info: dict,
    vocal_analysis: dict,
    compatible_tracks: list[dict],
) -> dict:
    """ボーカル分析結果に基づきサンプリングプレイリストにトラックを追加する。

    "Vocal for House" 追加基準:
    - track_info の genre が hiphop または jpop
    - usable_sections に vocal_clarity > 0.7 のセクションが存在する
    - compatible_genres に house または deep-house が含まれる

    "Transition Pairs" は compatible_tracks の各ペアを追加する。

    Args:
        xml_path: Rekordbox XML のパス
        track_info: ソーストラック情報 dict (track_id, genre 等)
        vocal_analysis: analyze_vocal() の戻り値 dict
        compatible_tracks: find_compatible_tracks_for_vocal() の matches リスト

    Returns:
        dict:
            - added_to_vocal_for_house: bool
            - added_to_transition_pairs: int (追加ペア数)
            - xml_path: 更新した XML パス
            - skipped: スキップした場合 True
            - reason: スキップ理由 (スキップ時のみ)
    """
    xml_file = Path(xml_path).expanduser()
    if not xml_file.exists():
        return {
            "added_to_vocal_for_house": False,
            "added_to_transition_pairs": 0,
            "xml_path": str(xml_file),
            "skipped": True,
            "reason": "xml_not_found",
        }

    root = ET.parse(xml_file).getroot()

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

    track_id = str(track_info.get("track_id", ""))
    added_vocal = False
    pairs_added = 0

    # Vocal for House 判定
    if track_id and _qualifies_for_vocal_for_house(track_info, vocal_analysis):
        _add_to_sampling_folder(root_node, track_id, _VOCAL_FOR_HOUSE_PLAYLIST)
        added_vocal = True

    # Transition Pairs: compatible_tracks の各マッチを追加
    for match in compatible_tracks:
        match_id = str(match.get("track_id", ""))
        if match_id:
            _add_to_sampling_folder(root_node, match_id, _TRANSITION_PAIRS_PLAYLIST)
            pairs_added += 1

    root_node.set("Count", str(len(root_node.findall("NODE"))))

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "added_to_vocal_for_house": added_vocal,
        "added_to_transition_pairs": pairs_added,
        "xml_path": str(xml_file),
        "skipped": False,
    }


def add_vocal_for_house(
    xml_path: str,
    track_id: str,
) -> dict:
    """「Vocal for House」プレイリストにトラックを追加する。

    HipHop/JPop のボーカルが House に合う曲を登録する。

    Args:
        xml_path: Rekordbox XML のパス
        track_id: トラックID (文字列)

    Returns:
        dict:
            - playlist: 追加されたプレイリスト名
            - xml_path: 更新した XML パス
            - skipped: スキップした場合 True
    """
    xml_file = Path(xml_path).expanduser()
    if not xml_file.exists():
        return {
            "playlist": None,
            "xml_path": str(xml_file),
            "skipped": True,
            "reason": "xml_not_found",
        }

    root = ET.parse(xml_file).getroot()

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

    _add_to_sampling_folder(root_node, track_id, _VOCAL_FOR_HOUSE_PLAYLIST)

    root_node.set("Count", str(len(root_node.findall("NODE"))))

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "playlist": _VOCAL_FOR_HOUSE_PLAYLIST,
        "xml_path": str(xml_file),
        "skipped": False,
    }


def generate_sampling_playlists(
    xml_path: str,
    track_id: str,
    playlist_name: str = _VOCAL_FOR_HOUSE_PLAYLIST,
    source_genre: str | None = None,
    target_genre: str | None = None,
) -> dict:
    """Sampling Ideas フォルダ内のサンプリングプレイリストにトラックを追加する。

    Args:
        xml_path: Rekordbox XML のパス
        track_id: トラックID (文字列)
        playlist_name: プレイリスト名 ("Vocal for House" または "Transition Pairs")
        source_genre: ボーカル元のジャンル (例: "hiphop", "jpop"、省略可)
        target_genre: ミックス先のジャンル (例: "deep-house"、省略可)

    Returns:
        dict:
            - playlist_name: 追加されたプレイリスト名
            - track_id: トラックID
            - xml_path: 更新した XML パス
            - skipped: スキップした場合 True
            - reason: スキップ理由 (スキップ時のみ)
    """
    xml_file = Path(xml_path).expanduser()
    if not xml_file.exists():
        return {
            "playlist_name": None,
            "track_id": track_id,
            "xml_path": str(xml_file),
            "skipped": True,
            "reason": "xml_not_found",
        }

    root = ET.parse(xml_file).getroot()

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

    _add_to_sampling_folder(root_node, track_id, playlist_name)
    root_node.set("Count", str(len(root_node.findall("NODE"))))

    ET.indent(root, space="  ")  # type: ignore[arg-type]
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return {
        "playlist_name": playlist_name,
        "track_id": track_id,
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
