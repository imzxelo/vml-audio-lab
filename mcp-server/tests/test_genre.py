"""ジャンル判定ツールのテスト."""

from __future__ import annotations

import numpy as np

from vml_audio_lab.tools.genre import (
    _apply_halftime_correction,
    canonicalize_genre_slug,
    detect_genre,
)


def test_detect_genre_votes_across_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "vml_audio_lab.tools.genre._fetch_web_text",
        lambda artist, title: "Beatport says this is tech house.",
    )
    monkeypatch.setattr(
        "vml_audio_lab.tools.genre._detect_from_audio",
        lambda y, sr: ("house", 0.51),
    )

    y = np.zeros(44100, dtype=np.float32)
    result = detect_genre(title="Tech House Anthem", artist="DJ Test", y=y, sr=44100)

    assert result["genre"] == "tech-house"
    assert result["genre_group"] == "house"
    assert result["sources"]["web"] == "tech-house"
    assert result["confidence"] > 0


def test_detect_genre_unknown_when_all_sources_fail(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("vml_audio_lab.tools.genre._fetch_web_text", _raise)
    monkeypatch.setattr(
        "vml_audio_lab.tools.genre._detect_from_audio",
        lambda y, sr: ("unknown", 0.23),
    )

    y = np.zeros(44100, dtype=np.float32)
    result = detect_genre(title="No metadata", artist="", y=y, sr=44100)

    assert result["genre"] == "unknown"
    assert result["genre_group"] == "unknown"
    assert result["confidence"] == 0.23


def test_canonicalize_genre_slug_aliases() -> None:
    assert canonicalize_genre_slug("tech house") == "tech-house"
    assert canonicalize_genre_slug("techhouse") == "tech-house"
    assert canonicalize_genre_slug("Tech-House") == "tech-house"
    assert canonicalize_genre_slug("drum and bass") == "dnb"
    assert canonicalize_genre_slug("drum & bass") == "dnb"
    assert canonicalize_genre_slug("hip hop") == "hiphop"
    assert canonicalize_genre_slug("hip-hop") == "hiphop"


def test_canonicalize_new_genres() -> None:
    """10ジャンル対応のテスト."""
    assert canonicalize_genre_slug("deep house") == "deep-house"
    assert canonicalize_genre_slug("uk garage") == "uk-garage"
    assert canonicalize_genre_slug("deep techno") == "deep-techno"
    assert canonicalize_genre_slug("melodic techno") == "melodic"
    assert canonicalize_genre_slug("j-pop") == "jpop"
    assert canonicalize_genre_slug("j pop") == "jpop"


def test_halftime_correction_applied_to_hiphop() -> None:
    """Hip-Hop の倍テン BPM が補正されること。"""
    corrected = _apply_halftime_correction(143.0, "hiphop")
    assert corrected == 71.5


def test_halftime_correction_applied_to_jpop() -> None:
    """J-Pop の倍テン BPM が補正されること。"""
    corrected = _apply_halftime_correction(140.0, "jpop")
    assert corrected == 70.0


def test_halftime_correction_not_applied_below_120(monkeypatch) -> None:
    """BPM <= 120 は補正しない。"""
    corrected = _apply_halftime_correction(95.0, "hiphop")
    assert corrected == 95.0


def test_halftime_correction_not_applied_to_house() -> None:
    """House は補正しない。"""
    corrected = _apply_halftime_correction(150.0, "house")
    assert corrected == 150.0


def test_detect_genre_includes_halftime_corrected_field(monkeypatch) -> None:
    """detect_genre がハーフタイム補正フィールドを含むこと。"""
    monkeypatch.setattr("vml_audio_lab.tools.genre._fetch_web_text", lambda *a, **k: "hip hop rap")
    monkeypatch.setattr(
        "vml_audio_lab.tools.genre._detect_from_audio",
        lambda y, sr: ("hiphop", 0.50),
    )

    y = np.zeros(44100, dtype=np.float32)
    result = detect_genre(title="Boom Bap Beat", artist="MC Test", y=y, sr=44100)

    assert "halftime_corrected" in result
