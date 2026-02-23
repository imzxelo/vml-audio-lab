"""プレイリスト生成ツールのテスト."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from vml_audio_lab.tools.playlist import (
    _qualifies_for_vocal_for_house,
    add_to_camelot_playlist,
    add_to_mood_playlist,
    add_to_sampling_playlist,
    generate_playlists,
    get_compatible_playlists,
)


def _make_minimal_xml(path: Path) -> None:
    """最小限の Rekordbox XML を作成する。"""
    root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
    ET.SubElement(root, "COLLECTION", {"Entries": "0"})
    playlists = ET.SubElement(root, "PLAYLISTS")
    ET.SubElement(playlists, "NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


class TestAddToCamelotPlaylist:
    """add_to_camelot_playlist のテスト"""

    def test_creates_by_compatibility_folder(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        assert folder is not None

    def test_creates_camelot_zone_playlist(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Camelot 4A Zone']")
        assert playlist is not None

    def test_adds_track_to_playlist(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "42", "8B")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Camelot 8B Zone']")
        assert playlist is not None
        track = playlist.find("TRACK[@Key='42']")
        assert track is not None

    def test_no_duplicate_track(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        add_to_camelot_playlist(root_node, "1", "4A")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Camelot 4A Zone']")
        assert playlist is not None
        assert len(playlist.findall("TRACK")) == 1

    def test_empty_camelot_code_does_nothing(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "")
        assert len(root_node.findall("NODE")) == 0

    def test_multiple_tracks_in_same_zone(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        add_to_camelot_playlist(root_node, "2", "4A")
        add_to_camelot_playlist(root_node, "3", "4A")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        playlist = folder.find("NODE[@Name='Camelot 4A Zone']")
        assert len(playlist.findall("TRACK")) == 3

    def test_different_zones_separate_playlists(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        add_to_camelot_playlist(root_node, "2", "8B")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        playlists = folder.findall("NODE[@Type='1']")
        assert len(playlists) == 2

    def test_folder_count_updated_on_new_zone(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        add_to_camelot_playlist(root_node, "2", "8B")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        assert folder.get("Count") == "2"

    def test_entries_attribute_updated(self) -> None:
        """Entries 属性がトラック追加に応じて更新される。"""
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_camelot_playlist(root_node, "1", "4A")
        add_to_camelot_playlist(root_node, "2", "4A")
        folder = root_node.find("NODE[@Name='By Compatibility']")
        playlist = folder.find("NODE[@Name='Camelot 4A Zone']")
        assert playlist.get("Entries") == "2"


class TestAddToMoodPlaylist:
    """add_to_mood_playlist のテスト"""

    def test_creates_by_mood_folder(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_mood_playlist(root_node, "1", "Peak Time")
        folder = root_node.find("NODE[@Name='By Mood']")
        assert folder is not None

    def test_creates_mood_playlist(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_mood_playlist(root_node, "1", "Deep Night")
        folder = root_node.find("NODE[@Name='By Mood']")
        assert folder is not None
        playlist = folder.find("NODE[@Name='Deep Night']")
        assert playlist is not None

    def test_empty_mood_does_nothing(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_mood_playlist(root_node, "1", "")
        assert len(root_node.findall("NODE")) == 0

    def test_no_duplicate_track_in_mood(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_mood_playlist(root_node, "5", "Peak Time")
        add_to_mood_playlist(root_node, "5", "Peak Time")
        folder = root_node.find("NODE[@Name='By Mood']")
        playlist = folder.find("NODE[@Name='Peak Time']")
        assert len(playlist.findall("TRACK")) == 1

    def test_all_5_moods_in_single_folder(self) -> None:
        """5つのムードカテゴリが同じ By Mood フォルダに収まる."""
        moods = ["Peak Time", "Deep Night", "Melodic Journey", "Groovy & Warm", "Chill & Mellow"]
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        for i, mood in enumerate(moods):
            add_to_mood_playlist(root_node, str(i + 1), mood)
        folder = root_node.find("NODE[@Name='By Mood']")
        playlists = folder.findall("NODE[@Type='1']")
        assert len(playlists) == 5

    def test_track_key_correct(self) -> None:
        root_node = ET.Element("NODE", {"Type": "0", "Name": "ROOT", "Count": "0"})
        add_to_mood_playlist(root_node, "99", "Chill & Mellow")
        folder = root_node.find("NODE[@Name='By Mood']")
        playlist = folder.find("NODE[@Name='Chill & Mellow']")
        track = playlist.find("TRACK[@Key='99']")
        assert track is not None


class TestGeneratePlaylists:
    """generate_playlists のテスト"""

    def test_skips_if_xml_not_found(self, tmp_path: Path) -> None:
        xml_path = str(tmp_path / "nonexistent.xml")
        result = generate_playlists(xml_path, "1", "Fm", "Peak Time")
        assert result["skipped"] is True
        assert result["reason"] == "xml_not_found"

    def test_adds_to_camelot_and_mood_playlists(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)

        result = generate_playlists(str(xml_path), "1", "Fm", "Peak Time")
        assert result["skipped"] is False
        assert result["camelot_code"] == "4A"
        assert result["camelot_playlist"] == "Camelot 4A Zone"
        assert result["mood_playlist"] == "Peak Time"

    def test_xml_is_valid_after_update(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        generate_playlists(str(xml_path), "1", "Am", "Groovy & Warm")

        # XML がパースできることを確認
        root = ET.parse(str(xml_path)).getroot()
        assert root.tag == "DJ_PLAYLISTS"

    def test_unknown_key_skips_camelot(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)

        result = generate_playlists(str(xml_path), "1", "Xm", "Peak Time")
        assert result["camelot_code"] is None
        assert result["camelot_playlist"] is None
        # ムードは追加されていること
        assert result["mood_playlist"] == "Peak Time"

    def test_multiple_tracks_accumulate_in_camelot_zone(self, tmp_path: Path) -> None:
        """複数回呼ぶと Camelot ゾーンにトラックが累積される。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        for tid in ["1", "2", "3"]:
            generate_playlists(str(xml_path), tid, "Am", "Deep Night")
        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='By Compatibility']"
            "/NODE[@Name='Camelot 8A Zone']"
        )
        assert playlist is not None
        assert len(playlist.findall("TRACK")) == 3

    def test_different_keys_go_to_different_zones(self, tmp_path: Path) -> None:
        """異なるキーのトラックが別々の Camelot ゾーンに入る。"""
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        generate_playlists(str(xml_path), "1", "Am", "Deep Night")   # 8A
        generate_playlists(str(xml_path), "2", "Fm", "Deep Night")   # 4A
        root = ET.parse(str(xml_path)).getroot()
        compatibility = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']/NODE[@Name='By Compatibility']"
        )
        playlists = {p.get("Name") for p in compatibility.findall("NODE[@Type='1']")}
        assert "Camelot 8A Zone" in playlists
        assert "Camelot 4A Zone" in playlists

    def test_all_required_return_fields_present(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rekordbox_library.xml"
        _make_minimal_xml(xml_path)
        result = generate_playlists(str(xml_path), "1", "Fm", "Peak Time")
        for field in ("camelot_code", "mood", "camelot_playlist", "mood_playlist", "xml_path", "skipped"):
            assert field in result, f"Missing field: {field}"


class TestGetCompatiblePlaylists:
    """get_compatible_playlists のテスト"""

    def test_returns_list_of_playlist_names(self) -> None:
        result = get_compatible_playlists("/any/path.xml", "Fm")
        assert isinstance(result, list)
        assert len(result) > 0
        for name in result:
            assert "Camelot" in name and "Zone" in name

    def test_unknown_key_returns_empty(self) -> None:
        result = get_compatible_playlists("/any/path.xml", "Xm")
        assert result == []

    def test_includes_same_zone(self) -> None:
        result = get_compatible_playlists("/any/path.xml", "Fm")
        assert "Camelot 4A Zone" in result

    def test_returns_4_playlist_names(self) -> None:
        """4つの相性プレイリスト名を返す (self, relative, ±1)."""
        result = get_compatible_playlists("/any/path.xml", "Am")
        assert len(result) == 4

    def test_includes_relative_zone(self) -> None:
        """relative major/minor ゾーンが含まれる."""
        result = get_compatible_playlists("/any/path.xml", "Am")
        assert "Camelot 8B Zone" in result  # Am の relative は C (8B)

    def test_includes_adjacent_zones(self) -> None:
        """±1 隣接ゾーンが含まれる。"""
        result = get_compatible_playlists("/any/path.xml", "Am")
        assert "Camelot 7A Zone" in result
        assert "Camelot 9A Zone" in result

    def test_fm_compatible_all_4_zones(self) -> None:
        """Fm (4A) の相性プレイリストが正しい 4 つである."""
        result = get_compatible_playlists("/any/path.xml", "Fm")
        expected = {
            "Camelot 4A Zone",  # self
            "Camelot 4B Zone",  # relative (Ab)
            "Camelot 3A Zone",  # -1
            "Camelot 5A Zone",  # +1
        }
        assert set(result) == expected

    def test_xml_path_not_needed_for_key_lookup(self) -> None:
        """キー変換は XML ファイルに依存しない。"""
        result = get_compatible_playlists("/nonexistent/path.xml", "C")
        assert len(result) == 4
        assert "Camelot 8B Zone" in result


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_hiphop_vocal_analysis(vocal_clarity: float = 0.8) -> dict:
    """テスト用の Hip-Hop ボーカル分析結果を返す。"""
    return {
        "usable_sections": [
            {"type": "Hook", "vocal_clarity": vocal_clarity, "has_clear_vocal": True},
        ],
        "compatible_genres": ["house", "deep-house"],
    }


def _make_track_info(genre: str = "hiphop", track_id: str = "1") -> dict:
    return {"track_id": track_id, "genre": genre}


# ---------------------------------------------------------------------------
# _qualifies_for_vocal_for_house テスト
# ---------------------------------------------------------------------------

class TestQualifiesForVocalForHouse:
    """_qualifies_for_vocal_for_house の判定ロジックを検証する。"""

    def test_hiphop_with_clear_vocal_and_house_genre_qualifies(self) -> None:
        track = _make_track_info(genre="hiphop")
        analysis = _make_hiphop_vocal_analysis(vocal_clarity=0.8)
        assert _qualifies_for_vocal_for_house(track, analysis) is True

    def test_jpop_with_clear_vocal_and_house_genre_qualifies(self) -> None:
        track = _make_track_info(genre="jpop")
        analysis = _make_hiphop_vocal_analysis(vocal_clarity=0.9)
        assert _qualifies_for_vocal_for_house(track, analysis) is True

    def test_house_genre_does_not_qualify(self) -> None:
        """ソースジャンルが house の場合は対象外。"""
        track = _make_track_info(genre="house")
        analysis = _make_hiphop_vocal_analysis()
        assert _qualifies_for_vocal_for_house(track, analysis) is False

    def test_low_vocal_clarity_does_not_qualify(self) -> None:
        """vocal_clarity <= 0.7 は対象外。"""
        track = _make_track_info(genre="hiphop")
        analysis = _make_hiphop_vocal_analysis(vocal_clarity=0.5)
        assert _qualifies_for_vocal_for_house(track, analysis) is False

    def test_vocal_clarity_exactly_07_does_not_qualify(self) -> None:
        """境界値 0.7 は含まない (> 0.7 が条件)。"""
        track = _make_track_info(genre="hiphop")
        analysis = _make_hiphop_vocal_analysis(vocal_clarity=0.7)
        assert _qualifies_for_vocal_for_house(track, analysis) is False

    def test_no_house_in_compatible_genres_does_not_qualify(self) -> None:
        """compatible_genres に house / deep-house がない場合は対象外。"""
        track = _make_track_info(genre="hiphop")
        analysis = {
            "usable_sections": [{"vocal_clarity": 0.9, "has_clear_vocal": True}],
            "compatible_genres": ["techno", "melodic"],
        }
        assert _qualifies_for_vocal_for_house(track, analysis) is False

    def test_deep_house_in_compatible_genres_qualifies(self) -> None:
        """deep-house だけでも house 条件を満たす。"""
        track = _make_track_info(genre="hiphop")
        analysis = {
            "usable_sections": [{"vocal_clarity": 0.85, "has_clear_vocal": True}],
            "compatible_genres": ["deep-house"],
        }
        assert _qualifies_for_vocal_for_house(track, analysis) is True

    def test_empty_usable_sections_does_not_qualify(self) -> None:
        track = _make_track_info(genre="hiphop")
        analysis = {"usable_sections": [], "compatible_genres": ["house"]}
        assert _qualifies_for_vocal_for_house(track, analysis) is False


# ---------------------------------------------------------------------------
# add_to_sampling_playlist テスト
# ---------------------------------------------------------------------------

class TestAddToSamplingPlaylist:
    """add_to_sampling_playlist の統合テスト。"""

    def test_skips_if_xml_not_found(self, tmp_path: Path) -> None:
        xml_path = str(tmp_path / "nonexistent.xml")
        result = add_to_sampling_playlist(
            xml_path,
            _make_track_info(),
            _make_hiphop_vocal_analysis(),
            [],
        )
        assert result["skipped"] is True
        assert result["reason"] == "xml_not_found"

    def test_return_fields_present(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = add_to_sampling_playlist(
            str(xml_path), _make_track_info(), _make_hiphop_vocal_analysis(), []
        )
        for field in ("added_to_vocal_for_house", "added_to_transition_pairs", "xml_path", "skipped"):
            assert field in result

    def test_hiphop_with_clear_vocal_added_to_vocal_for_house(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = add_to_sampling_playlist(
            str(xml_path),
            _make_track_info(genre="hiphop", track_id="10"),
            _make_hiphop_vocal_analysis(vocal_clarity=0.85),
            [],
        )
        assert result["added_to_vocal_for_house"] is True

        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='Sampling Ideas']"
            "/NODE[@Name='Vocal for House']"
        )
        assert playlist is not None
        assert playlist.find("TRACK[@Key='10']") is not None

    def test_non_hiphop_not_added_to_vocal_for_house(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = add_to_sampling_playlist(
            str(xml_path),
            _make_track_info(genre="techno", track_id="5"),
            _make_hiphop_vocal_analysis(vocal_clarity=0.9),
            [],
        )
        assert result["added_to_vocal_for_house"] is False

    def test_compatible_tracks_added_to_transition_pairs(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        compatible = [
            {"track_id": "20", "title": "Track A"},
            {"track_id": "21", "title": "Track B"},
        ]
        result = add_to_sampling_playlist(
            str(xml_path),
            _make_track_info(genre="house"),
            _make_hiphop_vocal_analysis(),
            compatible,
        )
        assert result["added_to_transition_pairs"] == 2

        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='Sampling Ideas']"
            "/NODE[@Name='Transition Pairs']"
        )
        assert playlist is not None
        assert len(playlist.findall("TRACK")) == 2

    def test_no_duplicate_in_vocal_for_house(self, tmp_path: Path) -> None:
        """同一トラックを2回追加しても重複しない。"""
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        for _ in range(2):
            add_to_sampling_playlist(
                str(xml_path),
                _make_track_info(genre="hiphop", track_id="99"),
                _make_hiphop_vocal_analysis(),
                [],
            )
        root = ET.parse(str(xml_path)).getroot()
        playlist = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']"
            "/NODE[@Name='Sampling Ideas']"
            "/NODE[@Name='Vocal for House']"
        )
        assert len(playlist.findall("TRACK")) == 1

    def test_xml_valid_after_update(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        add_to_sampling_playlist(
            str(xml_path), _make_track_info(), _make_hiphop_vocal_analysis(), []
        )
        root = ET.parse(str(xml_path)).getroot()
        assert root.tag == "DJ_PLAYLISTS"


# ---------------------------------------------------------------------------
# generate_playlists + Sampling Ideas 統合テスト
# ---------------------------------------------------------------------------

class TestGeneratePlaylistsWithSampling:
    """generate_playlists が vocal_analysis を受け取ると Sampling Ideas を更新する。"""

    def test_sampling_added_field_false_when_no_vocal_analysis(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = generate_playlists(str(xml_path), "1", "Fm", "Peak Time")
        assert result["sampling_added"] is False

    def test_sampling_added_true_when_qualifies(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = generate_playlists(
            str(xml_path), "1", "Fm", "Peak Time",
            track_info=_make_track_info(genre="hiphop", track_id="1"),
            vocal_analysis=_make_hiphop_vocal_analysis(vocal_clarity=0.9),
        )
        assert result["sampling_added"] is True

    def test_sampling_not_added_when_genre_does_not_qualify(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        result = generate_playlists(
            str(xml_path), "1", "Fm", "Peak Time",
            track_info=_make_track_info(genre="house"),
            vocal_analysis=_make_hiphop_vocal_analysis(),
        )
        assert result["sampling_added"] is False

    def test_sampling_folder_created_in_xml(self, tmp_path: Path) -> None:
        xml_path = tmp_path / "rb.xml"
        _make_minimal_xml(xml_path)
        generate_playlists(
            str(xml_path), "5", "Am", "Deep Night",
            track_info=_make_track_info(genre="hiphop", track_id="5"),
            vocal_analysis=_make_hiphop_vocal_analysis(),
        )
        root = ET.parse(str(xml_path)).getroot()
        folder = root.find(
            "./PLAYLISTS/NODE[@Name='ROOT']/NODE[@Name='Sampling Ideas']"
        )
        assert folder is not None
