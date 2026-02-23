"""find_compatible_tracks のテスト.

MVP-B: library.py — Key相性・BPM・ムードによるマッチングロジック。

API:
    find_compatible_tracks(
        query_key_label, query_bpm,
        query_mood=None, query_genre=None,
        library_index=None, library_source=None,
        max_results=10
    ) -> dict
"""

from __future__ import annotations

import pytest

from vml_audio_lab.tools.library import _bpm_score, _mood_score, find_compatible_tracks

# ---------------------------------------------------------------------------
# helpers — ライブラリインデックスのモックデータ
# ---------------------------------------------------------------------------


def _make_library(tracks: list[dict]) -> dict:
    """テスト用ライブラリインデックスを生成する。"""
    return {
        "tracks": tracks,
        "total": len(tracks),
        "source": "/test/library.xml",
        "cached": False,
        "fingerprint": "test1234",
    }


def _track(
    id: int = 1,
    title: str = "Track",
    artist: str = "",
    bpm: float = 124.0,
    key_label: str = "Am",
    camelot_code: str = "8A",
    genre: str = "deep-house",
    mood: str = "Deep Night",
) -> dict:
    """テスト用トラックデータを生成する。"""
    return {
        "id": id,
        "title": title,
        "artist": artist,
        "bpm": bpm,
        "key_label": key_label,
        "camelot_code": camelot_code,
        "genre": genre,
        "genre_group": genre,
        "mood": mood,
        "file_path": f"/music/{title}.mp3",
        "duration_sec": 300.0,
    }


# ---------------------------------------------------------------------------
# BPM スコアユニットテスト
# ---------------------------------------------------------------------------


class TestBpmScore:
    """_bpm_score: BPM相性スコアのユニットテスト。"""

    def test_same_bpm_perfect_score(self) -> None:
        assert _bpm_score(124.0, 124.0) == 1.0

    def test_within_3bpm_perfect(self) -> None:
        assert _bpm_score(124.0, 126.0) == 1.0

    def test_within_5bpm_good(self) -> None:
        assert _bpm_score(124.0, 129.0) == 0.8

    def test_within_10bpm_moderate(self) -> None:
        assert _bpm_score(124.0, 133.0) == 0.5

    def test_beyond_10bpm_zero(self) -> None:
        assert _bpm_score(124.0, 150.0) == 0.0

    def test_halftime_considered(self) -> None:
        """ハーフタイム BPM (x2 または /2) も相性判定に含まれる。"""
        # 62 * 2 = 124 → 差 0 → score 1.0
        assert _bpm_score(62.0, 124.0) == 1.0

    def test_zero_bpm_returns_zero(self) -> None:
        assert _bpm_score(0.0, 124.0) == 0.0
        assert _bpm_score(124.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# ムードスコアユニットテスト
# ---------------------------------------------------------------------------


class TestMoodScore:
    """_mood_score: ムード相性スコアのユニットテスト。"""

    def test_same_mood_perfect(self) -> None:
        assert _mood_score("Deep Night", "Deep Night") == 1.0

    def test_adjacent_mood_partial(self) -> None:
        # Deep Night の隣接: Melodic Journey, Chill & Mellow
        assert _mood_score("Deep Night", "Melodic Journey") == 0.5

    def test_unrelated_mood_zero(self) -> None:
        assert _mood_score("Deep Night", "Peak Time") == 0.0

    def test_empty_mood_zero(self) -> None:
        assert _mood_score("", "Deep Night") == 0.0
        assert _mood_score("Deep Night", "") == 0.0


# ---------------------------------------------------------------------------
# スコアリング公式テスト
# ---------------------------------------------------------------------------


class TestCompatibilityScoring:
    """find_compatible_tracks: スコアリング公式の確認。"""

    def test_same_key_returns_match(self) -> None:
        """同じキーのトラックはマッチに含まれる。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=124.0),
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert len(result["matches"]) >= 1

    def test_same_key_has_highest_score(self) -> None:
        """同じキーは最高スコア、Camelot 隣接はより低い。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=124.0),
            _track(id=2, key_label="Em", camelot_code="9A", bpm=124.0),  # adjacent
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        matches = result["matches"]
        assert len(matches) >= 2
        # Am (完全一致) は Em (隣接) より高スコア
        scores = [m["score"] for m in matches]
        assert scores[0] >= scores[1]

    def test_relative_major_minor_included(self) -> None:
        """relative major/minor (同番号 A↔B) がマッチに含まれる。"""
        library = _make_library([
            _track(id=1, key_label="C", camelot_code="8B", bpm=124.0),  # Am の relative
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert len(result["matches"]) >= 1

    def test_camelot_adjacent_included(self) -> None:
        """Camelot ±1 隣接 (Em=9A, Dm=7A) がマッチに含まれる。"""
        library = _make_library([
            _track(id=1, key_label="Em", camelot_code="9A", bpm=124.0),
            _track(id=2, key_label="Dm", camelot_code="7A", bpm=124.0),
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert len(result["matches"]) >= 2

    def test_incompatible_key_has_lower_score(self) -> None:
        """相性のないキーは相性ありのキーよりスコアが低い。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=124.0),   # 完全一致
            _track(id=2, key_label="F#m", camelot_code="11A", bpm=124.0), # キー不一致
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        scores = {m["camelot"]: m["score"] for m in result["matches"]}
        if "8A" in scores and "11A" in scores:
            assert scores["8A"] > scores["11A"]

    def test_matches_sorted_by_score_desc(self) -> None:
        """matches は score の降順でソートされる。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=124.0),   # 完全一致
            _track(id=2, key_label="C", camelot_code="8B", bpm=124.0),    # relative
            _track(id=3, key_label="Em", camelot_code="9A", bpm=124.0),   # adjacent
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        scores = [m["score"] for m in result["matches"]]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_required_fields(self) -> None:
        """result が必須フィールドを持つ。"""
        library = _make_library([_track()])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        for field in ("matches", "query", "total_scanned", "compatible_count"):
            assert field in result

    def test_match_has_required_fields(self) -> None:
        """各マッチが必須フィールドを持つ。"""
        library = _make_library([_track(key_label="Am", camelot_code="8A")])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert len(result["matches"]) >= 1
        for m in result["matches"]:
            for field in ("rank", "track", "bpm", "key", "camelot", "genre", "mood",
                          "compatibility", "score", "suggestion"):
                assert field in m, f"Missing field: {field}"

    def test_match_suggestion_is_non_empty_string(self) -> None:
        """suggestion フィールドは非空文字列。"""
        library = _make_library([_track(key_label="Am", camelot_code="8A")])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        for m in result["matches"]:
            assert isinstance(m["suggestion"], str)
            assert len(m["suggestion"]) > 0

    def test_compatibility_label_valid(self) -> None:
        """compatibility は ◎/○/△ のいずれか。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A"),
            _track(id=2, key_label="Em", camelot_code="9A"),
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        for m in result["matches"]:
            assert m["compatibility"] in ("◎", "○", "△")

    def test_query_field_in_result(self) -> None:
        """query フィールドにクエリ情報が含まれる。"""
        library = _make_library([_track()])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert result["query"]["key"] == "Am"
        assert result["query"]["bpm"] == 124.0
        assert result["query"]["camelot"] == "8A"


# ---------------------------------------------------------------------------
# BPM フィルタリングテスト
# ---------------------------------------------------------------------------


class TestBpmFiltering:
    """find_compatible_tracks: BPM 相性のフィルタリング。"""

    def test_same_bpm_highest_combined_score(self) -> None:
        """同 BPM・同 Key は最高合計スコア。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=124.0),
            _track(id=2, key_label="Am", camelot_code="8A", bpm=160.0),  # BPM NG
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        matches = result["matches"]
        # bpm=124 の方が高スコア
        scores = {m["bpm"]: m["score"] for m in matches if m["bpm"] in (124.0, 160.0)}
        if 124.0 in scores and 160.0 in scores:
            assert scores[124.0] > scores[160.0]

    def test_halftime_bpm_also_matches(self) -> None:
        """ハーフタイム BPM でもマッチする (62 BPM ≈ 124/2)。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", bpm=62.0),
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert len(result["matches"]) >= 1


# ---------------------------------------------------------------------------
# ムードフィルタリングテスト
# ---------------------------------------------------------------------------


class TestMoodFiltering:
    """find_compatible_tracks: ムード相性のフィルタリング。"""

    def test_same_mood_increases_score(self) -> None:
        """同ムードのトラックはスコアが高い。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A", mood="Deep Night"),
            _track(id=2, key_label="Am", camelot_code="8A", mood="Peak Time"),  # 異なるムード
        ])
        result = find_compatible_tracks("Am", 124.0, query_mood="Deep Night", library_index=library)
        matches = result["matches"]
        moods = {m["mood"]: m["score"] for m in matches}
        if "Deep Night" in moods and "Peak Time" in moods:
            assert moods["Deep Night"] >= moods["Peak Time"]

    def test_query_mood_stored_in_result(self) -> None:
        """query フィールドにムードが含まれる。"""
        library = _make_library([_track()])
        result = find_compatible_tracks("Am", 124.0, query_mood="Deep Night", library_index=library)
        assert result["query"]["mood"] == "Deep Night"


# ---------------------------------------------------------------------------
# 空ライブラリ / エッジケーステスト
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """find_compatible_tracks: エッジケース。"""

    def test_empty_library_returns_empty_matches(self) -> None:
        """空ライブラリは空の matches を返す。"""
        library = _make_library([])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert result["matches"] == []
        assert result["total_scanned"] == 0

    def test_max_results_limits_output(self) -> None:
        """max_results でマッチ数を制限できる。"""
        library = _make_library([
            _track(id=i, key_label="Am", camelot_code="8A") for i in range(1, 20)
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library, max_results=5)
        assert len(result["matches"]) <= 5

    def test_no_library_raises_error(self) -> None:
        """library_index も library_source も指定しない場合はエラー。"""
        with pytest.raises(ValueError):
            find_compatible_tracks("Am", 124.0)

    def test_unknown_key_has_lower_match_quality(self) -> None:
        """不正なキーはキー相性スコアが 0 になる (BPM スコアは残る)。"""
        library = _make_library([_track(key_label="Am", camelot_code="8A", bpm=124.0)])
        result = find_compatible_tracks("Xm", 124.0, library_index=library)
        # Xm はキー変換できないのでキー相性スコア = 0
        # BPM スコアは残るので matches に入る可能性があるが、
        # 正常なキーでのマッチより低スコアになる
        for m in result["matches"]:
            # key スコア = 0 なので max score = 0.3 (BPM only)
            assert m["score"] <= 0.3

    def test_compatible_count_matches_scored_tracks(self) -> None:
        """compatible_count がスコア > 0 のトラック数と一致する。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A"),   # 相性あり
            _track(id=2, key_label="F#m", camelot_code="11A"),  # 相性なし
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        assert result["compatible_count"] == len(result["matches"])

    def test_rank_starts_at_1(self) -> None:
        """rank は 1 から始まる。"""
        library = _make_library([
            _track(id=1, key_label="Am", camelot_code="8A"),
            _track(id=2, key_label="Em", camelot_code="9A"),
        ])
        result = find_compatible_tracks("Am", 124.0, library_index=library)
        ranks = [m["rank"] for m in result["matches"]]
        assert ranks[0] == 1


# ---------------------------------------------------------------------------
# BPM パラメータパススルーテスト (genre.py のレビュー修正確認)
# ---------------------------------------------------------------------------


class TestGenreBpmPassthrough:
    """genre.detect_genre: bpm パラメータのパススルー確認 (reviewer fix)."""

    def test_bpm_param_skips_librosa_beat_track(self, monkeypatch) -> None:
        """bpm 引数を指定すると librosa.beat.beat_track が呼ばれない。"""
        import numpy as np

        from vml_audio_lab.tools.genre import detect_genre

        call_count = {"n": 0}

        def mock_beat_track(*args, **kwargs):
            call_count["n"] += 1
            return np.array([120.0]), np.array([0])

        monkeypatch.setattr("vml_audio_lab.tools.genre._fetch_web_text", lambda *a, **k: "house music")
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("house", 0.52),
        )
        monkeypatch.setattr("librosa.beat.beat_track", mock_beat_track)

        y = np.zeros(44100, dtype=np.float32)
        detect_genre(title="House Track", artist="DJ", y=y, sr=44100, bpm=124.0)
        assert call_count["n"] == 0, "beat_track should not be called when bpm is provided"

    def test_bpm_param_used_for_halftime_correction(self, monkeypatch) -> None:
        """bpm 引数が halftime 補正に使われる。"""
        import numpy as np

        from vml_audio_lab.tools.genre import detect_genre

        monkeypatch.setattr("vml_audio_lab.tools.genre._fetch_web_text", lambda *a, **k: "hip hop rap")
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("hiphop", 0.50),
        )

        y = np.zeros(44100, dtype=np.float32)
        # bpm=143.0 → hiphop → halftime → 71.5
        result = detect_genre(title="Hip Hop", artist="MC", y=y, sr=44100, bpm=143.0)
        assert result["halftime_corrected"] is True
        assert abs(result.get("corrected_bpm", 0) - 71.5) < 0.1
