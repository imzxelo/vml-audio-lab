"""ジャンル判定ツールのテスト."""

from __future__ import annotations

import numpy as np

from vml_audio_lab.tools.genre import detect_genre


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
