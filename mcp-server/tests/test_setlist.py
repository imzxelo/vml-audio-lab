"""setlist.py のテスト."""

import pytest

from vml_audio_lab.tools.setlist import (
    _bpm_score,
    _compute_transition_score,
    _genre_coherence_score,
    _greedy_order,
    _key_score,
    build_setlist,
    visualize_setlist_energy,
)


def _track(
    title: str = "Track",
    key: str = "Am",
    camelot: str = "8A",
    bpm: float = 124.0,
    energy_level: float = 0.5,
    genre: str = "house",
    genre_group: str = "house",
    mood: str = "Groovy & Warm",
) -> dict:
    return {
        "title": title,
        "key_label": key,
        "camelot": camelot,
        "bpm": bpm,
        "energy_level": energy_level,
        "genre": genre,
        "genre_group": genre_group,
        "mood": mood,
    }


class TestKeyScore:
    def test_same_key_max(self):
        assert _key_score(_track(camelot="8A"), _track(camelot="8A")) == 1.0

    def test_relative_key_high(self):
        assert _key_score(_track(camelot="8A"), _track(camelot="8B")) == 0.9

    def test_adjacent_key(self):
        assert _key_score(_track(camelot="8A"), _track(camelot="7A")) == 0.7

    def test_incompatible_key_zero(self):
        assert _key_score(_track(camelot="8A"), _track(camelot="2A")) == 0.0

    def test_missing_camelot_fallback(self):
        # Both camelot and key_label empty = no key info
        t = {"title": "NoKey", "bpm": 124}
        score = _key_score(t, _track(camelot="8A"))
        assert score == 0.3  # fallback


class TestBpmScore:
    def test_same_bpm_max(self):
        assert _bpm_score(_track(bpm=128), _track(bpm=128)) == 1.0

    def test_small_diff(self):
        score = _bpm_score(_track(bpm=124), _track(bpm=128))
        assert 0.7 < score < 1.0

    def test_large_diff_low(self):
        score = _bpm_score(_track(bpm=80), _track(bpm=128))
        assert score == 0.0

    def test_20_bpm_diff_zero(self):
        score = _bpm_score(_track(bpm=100), _track(bpm=120))
        assert score == 0.0


class TestGenreCoherenceScore:
    def test_same_genre_max(self):
        assert _genre_coherence_score(_track(genre_group="house"), _track(genre_group="house")) == 1.0

    def test_related_genres(self):
        score = _genre_coherence_score(_track(genre_group="house"), _track(genre_group="techno"))
        assert score == 0.8

    def test_unknown_genre_fallback(self):
        score = _genre_coherence_score(_track(genre_group="unknown"), _track(genre_group="house"))
        assert score == 0.5


class TestComputeTransitionScore:
    def test_perfect_transition_high(self):
        weights = {"w_key": 0.35, "w_bpm": 0.25, "w_energy": 0.20, "w_genre": 0.20}
        a = _track(camelot="8A", bpm=124, energy_level=0.5, genre_group="house")
        b = _track(camelot="8A", bpm=124, energy_level=0.6, genre_group="house")
        score = _compute_transition_score(a, b, 0.3, "build", weights)
        assert score > 0.7

    def test_bad_transition_low(self):
        weights = {"w_key": 0.35, "w_bpm": 0.25, "w_energy": 0.20, "w_genre": 0.20}
        a = _track(camelot="8A", bpm=80, energy_level=0.8, genre_group="classical")
        b = _track(camelot="2B", bpm=140, energy_level=0.2, genre_group="techno")
        score = _compute_transition_score(a, b, 0.3, "build", weights)
        assert score < 0.3


class TestGreedyOrder:
    def test_single_track(self):
        tracks = [_track(title="A")]
        order, score = _greedy_order(tracks, "build", {"w_key": 0.35, "w_bpm": 0.25, "w_energy": 0.20, "w_genre": 0.20})
        assert len(order) == 1
        assert order[0]["title"] == "A"

    def test_three_tracks_ordered(self):
        tracks = [
            _track(title="Low", camelot="8A", bpm=120, energy_level=0.3),
            _track(title="Mid", camelot="8A", bpm=124, energy_level=0.5),
            _track(title="High", camelot="8A", bpm=128, energy_level=0.8),
        ]
        order, score = _greedy_order(tracks, "build", {"w_key": 0.35, "w_bpm": 0.25, "w_energy": 0.20, "w_genre": 0.20})
        assert len(order) == 3
        # Build strategy should prefer low → high energy
        assert order[0]["title"] == "Low"

    def test_preserves_all_tracks(self):
        tracks = [_track(title=f"T{i}") for i in range(5)]
        order, _ = _greedy_order(tracks, "build", {"w_key": 0.35, "w_bpm": 0.25, "w_energy": 0.20, "w_genre": 0.20})
        assert len(order) == 5
        titles = {t["title"] for t in order}
        assert titles == {f"T{i}" for i in range(5)}


class TestBuildSetlist:
    def test_empty_tracks(self):
        result = build_setlist([])
        assert result["ordered_tracks"] == []
        assert result["overall_score"] == 0.0

    def test_returns_required_fields(self):
        tracks = [
            _track(title="A", energy_level=0.3, bpm=120),
            _track(title="B", energy_level=0.5, bpm=124),
            _track(title="C", energy_level=0.8, bpm=128),
        ]
        result = build_setlist(tracks)
        assert "ordered_tracks" in result
        assert "transitions" in result
        assert "energy_flow" in result
        assert "weak_points" in result
        assert "overall_score" in result
        assert "stats" in result

    def test_ordered_tracks_length(self):
        tracks = [_track(title=f"T{i}") for i in range(5)]
        result = build_setlist(tracks)
        assert len(result["ordered_tracks"]) == 5

    def test_transitions_length(self):
        tracks = [_track(title=f"T{i}") for i in range(4)]
        result = build_setlist(tracks)
        assert len(result["transitions"]) == 3  # N-1 transitions

    def test_energy_flow_length(self):
        tracks = [_track(title=f"T{i}") for i in range(4)]
        result = build_setlist(tracks)
        assert len(result["energy_flow"]) == 4

    def test_weak_points_detected(self):
        tracks = [
            _track(title="House", camelot="8A", bpm=124, genre_group="house"),
            _track(title="Classical", camelot="2B", bpm=60, genre_group="classical"),
        ]
        result = build_setlist(tracks)
        # Such a mismatch should produce weak points
        assert len(result["weak_points"]) > 0

    def test_stats_populated(self):
        tracks = [
            _track(title="A", bpm=120),
            _track(title="B", bpm=128),
        ]
        result = build_setlist(tracks)
        assert result["stats"]["total_tracks"] == 2
        assert result["stats"]["avg_bpm"] > 0

    def test_energy_curve_build_vs_peak(self):
        tracks = [
            _track(title="Low", energy_level=0.2, bpm=120),
            _track(title="High", energy_level=0.9, bpm=130),
        ]
        build_result = build_setlist(tracks, energy_curve="build")
        peak_result = build_setlist(tracks, energy_curve="peak")
        # Both should return valid results
        assert len(build_result["ordered_tracks"]) == 2
        assert len(peak_result["ordered_tracks"]) == 2


class TestVisualizeSetlistEnergy:
    def test_empty_returns_empty_bytes(self):
        assert visualize_setlist_energy([]) == b""

    def test_returns_png_bytes(self):
        tracks = [
            _track(title="A", energy_level=0.3, bpm=120),
            _track(title="B", energy_level=0.6, bpm=124),
            _track(title="C", energy_level=0.8, bpm=128),
        ]
        result = visualize_setlist_energy(tracks)
        assert isinstance(result, bytes)
        assert len(result) > 0
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_single_track(self):
        tracks = [_track(title="Solo")]
        result = visualize_setlist_energy(tracks)
        assert isinstance(result, bytes)
        assert len(result) > 0
