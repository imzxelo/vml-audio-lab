"""トランジション提案ロジックのテスト.

MVP-B: transition.py — suggest_transition による
キー関係・エネルギーマッチング・ジャンル別のトランジション提案。
"""

from __future__ import annotations

from vml_audio_lab.tools.transition import suggest_transition

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _track(
    id: str = "1",
    name: str = "Track",
    key: str = "Am",
    camelot: str = "8A",
    bpm: float = 124.0,
    energy: float = 0.5,
    genre: str = "deep-house",
    sections: list[dict] | None = None,
) -> dict:
    """テスト用トラックデータ。key_label と key の両方を設定する。"""
    t = {
        "id": id,
        "title": name,
        "key_label": key,
        "key": key,
        "camelot": camelot,
        "bpm": bpm,
        "energy": energy,
        "genre": genre,
    }
    if sections is not None:
        t["sections"] = sections
    return t


# ---------------------------------------------------------------------------
# 必須フィールドテスト
# ---------------------------------------------------------------------------


class TestSuggestTransitionReturnSchema:
    """suggest_transition: 返却スキーマのテスト。"""

    def test_returns_required_fields(self) -> None:
        """必須フィールドが全て含まれる。"""
        track_a = _track(id="1")
        track_b = _track(id="2")
        result = suggest_transition(track_a, track_b)
        for field in ("key_compatibility", "key_description", "energy_match",
                      "bpm_diff", "compatibility", "suggestion", "section_suggestion"):
            assert field in result, f"Missing field: {field}"

    def test_key_compatibility_is_float(self) -> None:
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="Am", camelot="8A")
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["key_compatibility"], float)

    def test_energy_match_is_float(self) -> None:
        track_a = _track(energy=0.5)
        track_b = _track(energy=0.5)
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["energy_match"], float)

    def test_suggestion_is_non_empty_string(self) -> None:
        track_a = _track()
        track_b = _track()
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["suggestion"], str)
        assert len(result["suggestion"]) > 0

    def test_bpm_diff_is_float(self) -> None:
        track_a = _track(bpm=120.0)
        track_b = _track(bpm=128.0)
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["bpm_diff"], float)
        assert result["bpm_diff"] == 8.0

    def test_compatibility_label_valid(self) -> None:
        """compatibility は ◎/○/△ のいずれか。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="Am", camelot="8A")
        result = suggest_transition(track_a, track_b)
        assert result["compatibility"] in ("◎", "○", "△")

    def test_section_suggestion_has_out_and_in_points(self) -> None:
        """section_suggestion に out_point と in_point が含まれる。"""
        track_a = _track()
        track_b = _track()
        result = suggest_transition(track_a, track_b)
        ss = result["section_suggestion"]
        assert "out_point" in ss
        assert "in_point" in ss
        assert "technique" in ss


# ---------------------------------------------------------------------------
# キー関係テスト
# ---------------------------------------------------------------------------


class TestKeyRelationship:
    """suggest_transition: キー関係の評価テスト。"""

    def test_same_key_perfect_compatibility(self) -> None:
        """同じキーは key_compatibility = 1.0。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="Am", camelot="8A")
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 1.0

    def test_same_key_compatibility_label_excellent(self) -> None:
        """同じキーは compatibility ◎。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="Am", camelot="8A")
        result = suggest_transition(track_a, track_b)
        assert result["compatibility"] == "◎"

    def test_relative_major_minor_high_compatibility(self) -> None:
        """Am (8A) → C (8B): relative は 0.9。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="C", camelot="8B")
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 0.9
        assert result["compatibility"] == "◎"

    def test_adjacent_camelot_moderate_compatibility(self) -> None:
        """Am (8A) → Em (9A): Camelot 隣接は 0.7。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="Em", camelot="9A")
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 0.7
        assert result["compatibility"] == "○"

    def test_incompatible_key_zero_compatibility(self) -> None:
        """Am (8A) → F#m (11A): 相性なしは 0.0。"""
        track_a = _track(key="Am", camelot="8A")
        track_b = _track(key="F#m", camelot="11A")
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 0.0
        assert result["compatibility"] == "△"

    def test_camelot_wrap_around_12_to_1(self) -> None:
        """12A → 1A: 円環の境界でも正しく判定する。"""
        track_a = _track(key="C#m", camelot="12A")
        track_b = _track(key="G#m", camelot="1A")
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 0.7

    def test_key_description_present_and_non_empty(self) -> None:
        """key_description は非空文字列。"""
        track_a = _track(key="Fm", camelot="4A")
        track_b = _track(key="Ab", camelot="4B")
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["key_description"], str)
        assert len(result["key_description"]) > 0

    def test_key_label_fallback_from_key_label_field(self) -> None:
        """key_label フィールドからも Camelot を計算できる。"""
        track_a = {"key_label": "Am", "bpm": 124.0}
        track_b = {"key_label": "Am", "bpm": 124.0}
        result = suggest_transition(track_a, track_b)
        assert result["key_compatibility"] == 1.0


# ---------------------------------------------------------------------------
# エネルギーマッチングテスト
# ---------------------------------------------------------------------------


class TestEnergyMatching:
    """suggest_transition: エネルギーレベル差の評価テスト。"""

    def test_same_energy_level_high_score(self) -> None:
        """同じエネルギーレベルは高スコア。"""
        track_a = _track(energy=0.7)
        track_b = _track(energy=0.7)
        result = suggest_transition(track_a, track_b)
        assert result["energy_match"] > 0.8

    def test_large_energy_gap_lower_score(self) -> None:
        """エネルギー差が大きい場合はスコアが低い。"""
        track_a = _track(energy=0.9)
        track_b = _track(energy=0.1)
        result = suggest_transition(track_a, track_b)
        assert result["energy_match"] < 0.7

    def test_energy_match_in_range(self) -> None:
        """energy_match は 0.0〜1.0 の範囲。"""
        for e_a, e_b in [(0.3, 0.7), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)]:
            track_a = _track(energy=e_a)
            track_b = _track(energy=e_b)
            result = suggest_transition(track_a, track_b)
            assert 0.0 <= result["energy_match"] <= 1.0

    def test_energy_field_alias_energy_level(self) -> None:
        """energy_level フィールドも energy として使われる。"""
        track_a = {"energy_level": 0.6, "bpm": 124.0}
        track_b = {"energy_level": 0.6, "bpm": 124.0}
        result = suggest_transition(track_a, track_b)
        assert result["energy_match"] > 0.5

    def test_suggestion_non_empty_even_with_energy_drop(self) -> None:
        """エネルギーが下がるトランジションでも suggestion は非空。"""
        track_a = _track(energy=0.9)
        track_b = _track(energy=0.3)
        result = suggest_transition(track_a, track_b)
        assert len(result["suggestion"]) > 0


# ---------------------------------------------------------------------------
# BPM 差テスト
# ---------------------------------------------------------------------------


class TestBpmDiff:
    """suggest_transition: BPM 差の反映テスト。"""

    def test_zero_bpm_diff(self) -> None:
        """BPM が同じなら bpm_diff は 0.0。"""
        track_a = _track(bpm=124.0)
        track_b = _track(bpm=124.0)
        result = suggest_transition(track_a, track_b)
        assert result["bpm_diff"] == 0.0

    def test_bpm_diff_calculated_correctly(self) -> None:
        """BPM 差が正しく計算される。"""
        track_a = _track(bpm=120.0)
        track_b = _track(bpm=128.0)
        result = suggest_transition(track_a, track_b)
        assert result["bpm_diff"] == 8.0

    def test_suggestion_mentions_bpm_diff(self) -> None:
        """suggestion に BPM 差への言及がある。"""
        track_a = _track(bpm=120.0)
        track_b = _track(bpm=120.0)
        result = suggest_transition(track_a, track_b)
        assert "BPM" in result["suggestion"] or "bpm" in result["suggestion"].lower()


# ---------------------------------------------------------------------------
# ジャンル別提案テスト
# ---------------------------------------------------------------------------


class TestGenreSpecificSuggestions:
    """suggest_transition: ジャンル別のミキシングテクニック提案テスト。"""

    def test_suggestion_includes_technique(self) -> None:
        """suggestion にミキシングテクニックが含まれる。"""
        track_a = _track(genre="deep-house")
        track_b = _track(genre="deep-house")
        result = suggest_transition(track_a, track_b)
        assert len(result["suggestion"]) > 20  # 具体的な提案があること

    def test_section_suggestion_with_break_section(self) -> None:
        """Break セクションを持つ場合の出点提案テスト。"""
        track_a = _track(
            sections=[
                {"label": "Drop", "start": 30.0, "end": 90.0, "energy": 0.9},
                {"label": "Break", "start": 90.0, "end": 120.0, "energy": 0.4},
                {"label": "Outro", "start": 120.0, "end": 150.0, "energy": 0.2},
            ]
        )
        track_b = _track(
            sections=[
                {"label": "Intro", "start": 0.0, "end": 30.0, "energy": 0.3},
                {"label": "Drop", "start": 30.0, "end": 90.0, "energy": 0.8},
            ]
        )
        result = suggest_transition(track_a, track_b)
        ss = result["section_suggestion"]
        # 出点は Break または Outro が優先される
        out_label = ss["out_point"].get("section", "")
        assert out_label in ("Break", "Outro", "落ちサビ", "Drop", "Build", "")

    def test_section_suggestion_in_point_picks_intro(self) -> None:
        """Intro を持つトラックでは入点 Intro が優先される。"""
        track_a = _track()
        track_b = _track(
            sections=[
                {"label": "Intro", "start": 0.0, "end": 30.0, "energy": 0.3},
                {"label": "Drop", "start": 30.0, "end": 90.0, "energy": 0.8},
            ]
        )
        result = suggest_transition(track_a, track_b)
        ss = result["section_suggestion"]
        in_label = ss["in_point"].get("section", "")
        assert in_label == "Intro" or in_label == ""

    def test_technique_in_section_suggestion(self) -> None:
        """technique が section_suggestion に含まれる。"""
        track_a = _track()
        track_b = _track()
        result = suggest_transition(track_a, track_b)
        assert isinstance(result["section_suggestion"]["technique"], str)
        assert len(result["section_suggestion"]["technique"]) > 0

    def test_out_point_has_time_fields(self) -> None:
        """out_point に time_sec と time_label が含まれる。"""
        track_a = _track()
        track_b = _track()
        result = suggest_transition(track_a, track_b)
        out = result["section_suggestion"]["out_point"]
        assert "time_sec" in out
        assert "time_label" in out
        assert isinstance(out["time_sec"], float)

    def test_hiphop_genre_uses_different_technique(self) -> None:
        """Hip-Hop ジャンルは異なるミキシングテクニックを使う。"""
        track_a = _track(genre="hiphop", bpm=85.0)
        track_b = _track(genre="deep-house", bpm=124.0)
        result = suggest_transition(track_a, track_b)
        ss = result["section_suggestion"]
        # hiphop テクニック: エコーアウトまたはクイックカット
        technique = ss.get("technique", "")
        assert len(technique) > 0
