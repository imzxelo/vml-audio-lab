"""Camelot Wheel ユーティリティのテスト."""

from __future__ import annotations

from vml_audio_lab.tools.camelot import (
    camelot_to_key,
    compatibility_score,
    compatible_camelot_codes,
    key_to_camelot,
)


class TestKeyToCamelot:
    """key_to_camelot: キーラベル → Camelot コード変換テスト."""

    def test_minor_keys_all_24_a_codes(self) -> None:
        """全12マイナーキーが正しい Aコードに変換される."""
        expected = {
            "Am": "8A",
            "Em": "9A",
            "Bm": "10A",
            "F#m": "11A",
            "C#m": "12A",
            "G#m": "1A",
            "D#m": "2A",
            "A#m": "3A",
            "Fm": "4A",
            "Cm": "5A",
            "Gm": "6A",
            "Dm": "7A",
        }
        for key_label, camelot in expected.items():
            assert key_to_camelot(key_label) == camelot, f"Failed for {key_label}"

    def test_major_keys_all_12_b_codes(self) -> None:
        """全12メジャーキーが正しい Bコードに変換される."""
        expected = {
            "C": "8B",
            "G": "9B",
            "D": "10B",
            "A": "11B",
            "E": "12B",
            "B": "1B",
            "F#": "2B",
            "C#": "3B",
            "Ab": "4B",
            "Eb": "5B",
            "Bb": "6B",
            "F": "7B",
        }
        for key_label, camelot in expected.items():
            assert key_to_camelot(key_label) == camelot, f"Failed for {key_label}"

    def test_enharmonic_equivalents_minor(self) -> None:
        """エンハーモニック等価なマイナーキーが同じ Camelot コードを返す."""
        assert key_to_camelot("G#m") == key_to_camelot("Abm") == "1A"
        assert key_to_camelot("D#m") == key_to_camelot("Ebm") == "2A"
        assert key_to_camelot("A#m") == key_to_camelot("Bbm") == "3A"
        assert key_to_camelot("F#m") == key_to_camelot("Gbm") == "11A"
        assert key_to_camelot("C#m") == key_to_camelot("Dbm") == "12A"

    def test_enharmonic_equivalents_major(self) -> None:
        """エンハーモニック等価なメジャーキーが同じ Camelot コードを返す."""
        assert key_to_camelot("F#") == key_to_camelot("Gb") == "2B"
        assert key_to_camelot("C#") == key_to_camelot("Db") == "3B"
        assert key_to_camelot("G#") == key_to_camelot("Ab") == "4B"
        assert key_to_camelot("D#") == key_to_camelot("Eb") == "5B"
        assert key_to_camelot("A#") == key_to_camelot("Bb") == "6B"

    def test_unknown_key_returns_none(self) -> None:
        """認識できないキーは None を返す."""
        assert key_to_camelot("X") is None
        assert key_to_camelot("") is None
        assert key_to_camelot("Hm") is None
        assert key_to_camelot("Zm") is None


class TestCamelotToKey:
    """camelot_to_key: Camelot コード → キーラベル逆引きテスト."""

    def test_all_24_camelot_codes_return_key(self) -> None:
        """全24 Camelot コードがキーラベルを返す."""
        for num in range(1, 13):
            for suffix in ("A", "B"):
                code = f"{num}{suffix}"
                result = camelot_to_key(code)
                assert result is not None, f"camelot_to_key('{code}') returned None"
                assert len(result) >= 1

    def test_specific_mappings(self) -> None:
        """代表的な Camelot コードの逆引きを確認."""
        assert camelot_to_key("8A") == "Am"
        assert camelot_to_key("8B") == "C"
        assert camelot_to_key("4A") == "Fm"
        assert camelot_to_key("4B") == "Ab"
        assert camelot_to_key("1A") == "G#m"
        assert camelot_to_key("1B") == "B"

    def test_case_insensitive(self) -> None:
        """大文字小文字を区別しない."""
        assert camelot_to_key("8a") == camelot_to_key("8A")
        assert camelot_to_key("4b") == camelot_to_key("4B")

    def test_invalid_code_returns_none(self) -> None:
        """無効なコードは None を返す."""
        assert camelot_to_key("0A") is None
        assert camelot_to_key("13A") is None
        assert camelot_to_key("8C") is None
        assert camelot_to_key("") is None
        assert camelot_to_key("XYZ") is None


class TestCompatibilityScore:
    """compatibility_score: 相性スコアテスト."""

    def test_perfect_match_same_code(self) -> None:
        """完全一致は 1.0 を返す."""
        assert compatibility_score("8A", "8A") == 1.0
        assert compatibility_score("4B", "4B") == 1.0
        assert compatibility_score("1A", "1A") == 1.0
        assert compatibility_score("12B", "12B") == 1.0

    def test_relative_major_minor_same_number(self) -> None:
        """同番号 A↔B (relative major/minor) は 0.9 を返す."""
        assert compatibility_score("8A", "8B") == 0.9
        assert compatibility_score("8B", "8A") == 0.9
        assert compatibility_score("4A", "4B") == 0.9
        assert compatibility_score("1A", "1B") == 0.9

    def test_adjacent_number_same_suffix(self) -> None:
        """番号 ±1 の同 suffix は 0.7 を返す."""
        assert compatibility_score("8A", "7A") == 0.7
        assert compatibility_score("8A", "9A") == 0.7
        assert compatibility_score("8B", "7B") == 0.7
        assert compatibility_score("8B", "9B") == 0.7

    def test_wrap_around_12_to_1(self) -> None:
        """12 と 1 は隣接扱い (円環)."""
        assert compatibility_score("12A", "1A") == 0.7
        assert compatibility_score("1A", "12A") == 0.7
        assert compatibility_score("12B", "1B") == 0.7

    def test_unrelated_keys_score_zero(self) -> None:
        """相性なしは 0.0 を返す."""
        assert compatibility_score("8A", "4A") == 0.0
        assert compatibility_score("1A", "6B") == 0.0
        assert compatibility_score("8A", "9B") == 0.0

    def test_invalid_code_score_zero(self) -> None:
        """無効なコードは 0.0 を返す."""
        assert compatibility_score("XYZ", "8A") == 0.0
        assert compatibility_score("8A", "") == 0.0
        assert compatibility_score("", "") == 0.0

    def test_score_symmetry(self) -> None:
        """スコアは対称的 (A→B == B→A)."""
        pairs = [("8A", "8B"), ("4A", "5A"), ("12A", "1A"), ("3B", "7A")]
        for a, b in pairs:
            assert compatibility_score(a, b) == compatibility_score(b, a), (
                f"Score not symmetric for {a}, {b}"
            )


class TestCompatibleCamelotCodes:
    """compatible_camelot_codes: 相性コードリストテスト."""

    def test_returns_4_codes(self) -> None:
        """4つのコードが返される (自分, relative, ±1)."""
        result = compatible_camelot_codes("8A")
        assert len(result) == 4

    def test_includes_self(self) -> None:
        """自分自身が含まれる."""
        assert "8A" in compatible_camelot_codes("8A")
        assert "4B" in compatible_camelot_codes("4B")

    def test_includes_relative(self) -> None:
        """relative major/minor が含まれる."""
        result = compatible_camelot_codes("8A")
        assert "8B" in result

        result = compatible_camelot_codes("4B")
        assert "4A" in result

    def test_includes_adjacent_numbers(self) -> None:
        """番号 ±1 が含まれる."""
        result = compatible_camelot_codes("8A")
        assert "7A" in result
        assert "9A" in result

    def test_wrap_around_at_1(self) -> None:
        """1 の前は 12."""
        result = compatible_camelot_codes("1A")
        assert "12A" in result
        assert "2A" in result

    def test_wrap_around_at_12(self) -> None:
        """12 の次は 1."""
        result = compatible_camelot_codes("12A")
        assert "11A" in result
        assert "1A" in result

    def test_b_suffix_adjacent_are_b(self) -> None:
        """B suffix の隣接コードも B suffix."""
        result = compatible_camelot_codes("6B")
        # 隣接は同じ suffix
        adjacent = [c for c in result if c != "6B" and c != "6A"]
        for code in adjacent:
            assert code.endswith("B"), f"Expected B suffix, got {code}"

    def test_invalid_code_returns_empty(self) -> None:
        """無効なコードは空リストを返す."""
        assert compatible_camelot_codes("XYZ") == []
        assert compatible_camelot_codes("") == []
        assert compatible_camelot_codes("0A") == []
        assert compatible_camelot_codes("13B") == []


class TestCamelotZoneGrouping:
    """Camelot ゾーン別グループ化の統合テスト."""

    def test_relative_pair_fm_ab(self) -> None:
        """Fm (4A) と Ab (4B) は relative pair."""
        fm = key_to_camelot("Fm")
        ab = key_to_camelot("Ab")
        assert fm == "4A"
        assert ab == "4B"
        assert compatibility_score(fm, ab) == 0.9

    def test_adjacent_pair_am_dm(self) -> None:
        """Am (8A) と Dm (7A) は隣接."""
        am = key_to_camelot("Am")
        dm = key_to_camelot("Dm")
        assert compatibility_score(am, dm) == 0.7

    def test_group_all_compatible_from_fm(self) -> None:
        """Fm からの相性グループが正しく構成される."""
        codes = compatible_camelot_codes("4A")
        assert "4A" in codes  # Fm 自身
        assert "4B" in codes  # Ab (relative major)
        assert "3A" in codes  # A#m (-1)
        assert "5A" in codes  # Cm (+1)
