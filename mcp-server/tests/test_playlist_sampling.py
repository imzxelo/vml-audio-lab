"""サンプリングプレイリスト生成のテスト.

MVP-C: playlist.py の拡張 — "Vocal for House" などサンプリング用プレイリスト生成。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from vml_audio_lab.tools.playlist import (
    _add_to_sampling_folder,
    add_vocal_for_house,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_minimal_xml(path: Path) -> None:
    """最小限の Rekordbox XML を作成する。"""
    root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
    ET.SubElement(root, "COLLECTION", {"Entries": "0"})
    playlists = ET.SubElement(root, "PLAYLISTS")
    ET.SubElement(playlists, "NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# add_to_sampling_playlist テスト
# ---------------------------------------------------------------------------


class TestAddToSamplingPlaylist:
    """add_to_sampling_playlist: サンプリングプレイリストへのトラック追加テスト。"""

    def test_creates_sampling_ideas_folder(self) -> None:
        """Sampling Ideas フォルダが作成される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        assert folder is not None

    def test_creates_vocal_for_house_playlist(self) -> None:
        """Vocal for House プレイリストが作成される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert playlist is not None

    def test_creates_transition_pairs_playlist(self) -> None:
        """Transition Pairs プレイリストが作成される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Transition Pairs")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Transition Pairs']")
        assert playlist is not None

    def test_adds_track_to_playlist(self) -> None:
        """トラックがプレイリストに追加される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "42", "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert playlist is not None
        track = playlist.find("TRACK[@Key='42']")
        assert track is not None

    def test_no_duplicate_track(self) -> None:
        """同じトラックを2回追加しても重複しない。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert len(playlist.findall("TRACK")) == 1

    def test_multiple_tracks_in_same_playlist(self) -> None:
        """複数トラックを同プレイリストに追加できる。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        for tid in ["1", "2", "3"]:
            _add_to_sampling_folder(root_node, tid, "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert len(playlist.findall("TRACK")) == 3

    def test_different_sampling_playlists_separate(self) -> None:
        """異なるサンプリングプレイリストは別々に管理される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        _add_to_sampling_folder(root_node, "2", "Transition Pairs")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        playlists = folder.findall("NODE[@Type='1']")
        assert len(playlists) == 2

    def test_empty_track_id_does_nothing(self) -> None:
        """空のトラック ID は何もしない。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "", "Vocal for House")
        # Sampling Ideas フォルダが作成されないか、プレイリストにトラックがない
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        if folder is not None:
            playlist = folder.find("NODE[@Name='Vocal for House']")
            if playlist is not None:
                assert len(playlist.findall("TRACK")) == 0

    def test_folder_count_updated_on_new_playlist(self) -> None:
        """新しいプレイリスト追加時にフォルダの Count が更新される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        _add_to_sampling_folder(root_node, "2", "Transition Pairs")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        assert folder.get("Count") == "2"

    def test_entries_updated_on_track_add(self) -> None:
        """トラック追加時に Entries が更新される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1", "Vocal for House")
        _add_to_sampling_folder(root_node, "2", "Vocal for House")
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert playlist.get("Entries") == "2"

    def test_default_playlist_name_is_vocal_for_house(self) -> None:
        """デフォルトのプレイリスト名は 'Vocal for House'。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        _add_to_sampling_folder(root_node, "1")  # デフォルト引数使用
        folder = root_node.find("NODE[@Name='Sampling Ideas']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Vocal for House']")
        assert playlist is not None


# ---------------------------------------------------------------------------
# add_vocal_for_house テスト (XML ファイルベース)
# ---------------------------------------------------------------------------


class TestAddVocalForHouse:
    """add_vocal_for_house: XML ファイルへの追加テスト。"""

    def test_skips_if_xml_not_found(self, tmp_path: Path) -> None:
        """XML が存在しない場合はスキップする。"""
        xml_path = str(tmp_path / "nonexistent.xml")
        result = add_vocal_for_house(xml_path, "1")
        assert result["skipped"] is True
        assert result["reason"] == "xml_not_found"

    def test_adds_to_vocal_for_house_playlist(self, tmp_path: Path) -> None:
        """Vocal for House プレイリストにトラックが追加される。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        result = add_vocal_for_house(str(xml_path), "1")
        assert result["skipped"] is False
        assert result["playlist"] == "Vocal for House"

    def test_xml_is_valid_after_update(self, tmp_path: Path) -> None:
        """XML ファイルが更新後も有効。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        add_vocal_for_house(str(xml_path), "1")
        root = ET.parse(str(xml_path)).getroot()
        assert root.tag == "DJ_PLAYLISTS"

    def test_track_appears_in_sampling_ideas_folder(self, tmp_path: Path) -> None:
        """トラックが Sampling Ideas フォルダ内に存在する。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        add_vocal_for_house(str(xml_path), "99")
        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='Sampling Ideas']"
            "/NODE[@Name='Vocal for House']"
        )
        assert playlist is not None
        track = playlist.find("TRACK[@Key='99']")
        assert track is not None

    def test_multiple_tracks_accumulate(self, tmp_path: Path) -> None:
        """複数回呼ぶとプレイリストにトラックが累積される。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        for tid in ["1", "2", "3"]:
            add_vocal_for_house(str(xml_path), tid)
        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='Sampling Ideas']"
            "/NODE[@Name='Vocal for House']"
        )
        assert playlist is not None
        assert len(playlist.findall("TRACK")) == 3

    def test_returns_required_fields(self, tmp_path: Path) -> None:
        """必須フィールドが全て含まれる。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        result = add_vocal_for_house(str(xml_path), "1")
        for field in ("playlist", "xml_path", "skipped"):
            assert field in result, f"Missing field: {field}"

    def test_xml_path_in_result(self, tmp_path: Path) -> None:
        """結果に xml_path が含まれる。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        result = add_vocal_for_house(str(xml_path), "1")
        assert result["xml_path"] == str(xml_path)

    def test_coexists_with_camelot_and_mood_playlists(self, tmp_path: Path) -> None:
        """Sampling Ideas フォルダが By Compatibility / By Mood フォルダと共存できる。"""
        from vml_audio_lab.tools.playlist import generate_playlists

        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)

        # 先に Camelot / Mood プレイリストを追加
        generate_playlists(str(xml_path), "1", "Am", "Deep Night")

        # Vocal for House プレイリストを追加
        add_vocal_for_house(str(xml_path), "2")

        root = ET.parse(str(xml_path)).getroot()
        root_node = root.find("./PLAYLISTS/NODE[@Name='ROOT']")
        assert root_node is not None

        top_folders = {n.get("Name") for n in root_node.findall("NODE[@Type='0']")}
        assert "By Compatibility" in top_folders
        assert "By Mood" in top_folders
        assert "Sampling Ideas" in top_folders
