"""ジャンル判定ツール拡張（10ジャンル対応・ハーフタイム補正）のテスト."""

from __future__ import annotations

import numpy as np

from vml_audio_lab.tools.genre import (
    _apply_halftime_correction,
    _detect_from_audio,
    _detect_from_text,
    canonicalize_genre_slug,
    detect_genre,
    genre_group_for,
)


class TestCanonicalize10Genres:
    """10ジャンル対応 canonicalize_genre_slug テスト."""

    def test_deep_house_aliases(self) -> None:
        assert canonicalize_genre_slug("deep house") == "deep-house"
        assert canonicalize_genre_slug("deephouse") == "deep-house"
        assert canonicalize_genre_slug("Deep House") == "deep-house"

    def test_uk_garage_aliases(self) -> None:
        assert canonicalize_genre_slug("uk garage") == "uk-garage"
        assert canonicalize_genre_slug("ukg") == "uk-garage"
        assert canonicalize_genre_slug("2step") == "uk-garage"

    def test_deep_techno_aliases(self) -> None:
        assert canonicalize_genre_slug("deep techno") == "deep-techno"

    def test_melodic_aliases(self) -> None:
        assert canonicalize_genre_slug("melodic techno") == "melodic"
        assert canonicalize_genre_slug("melodic house") == "melodic"
        assert canonicalize_genre_slug("organic house") == "melodic"

    def test_jpop_aliases(self) -> None:
        assert canonicalize_genre_slug("j-pop") == "jpop"
        assert canonicalize_genre_slug("j pop") == "jpop"
        assert canonicalize_genre_slug("japanese pop") == "jpop"
        assert canonicalize_genre_slug("city pop") == "jpop"

    def test_existing_genres_unchanged(self) -> None:
        assert canonicalize_genre_slug("tech house") == "tech-house"
        assert canonicalize_genre_slug("hip hop") == "hiphop"
        assert canonicalize_genre_slug("drum and bass") == "dnb"
        assert canonicalize_genre_slug("drum & bass") == "dnb"

    def test_all_10_genre_slugs_are_canonical(self) -> None:
        """全10ジャンルが正規化済みスラグを返す."""
        genres_10 = [
            "deep-house", "house", "tech-house", "techno", "deep-techno",
            "uk-garage", "melodic", "hiphop", "jpop", "electronic", "classical",
        ]
        for genre in genres_10:
            result = canonicalize_genre_slug(genre)
            assert result == genre, f"Expected {genre}, got {result}"


class TestGenreGroupFor10Genres:
    """genre_group_for: 10ジャンルの大分類テスト."""

    def test_house_variants_group_to_house(self) -> None:
        assert genre_group_for("deep-house") == "house"
        assert genre_group_for("house") == "house"
        assert genre_group_for("tech-house") == "house"

    def test_techno_variants_group_to_techno(self) -> None:
        assert genre_group_for("techno") == "techno"
        assert genre_group_for("deep-techno") == "techno"

    def test_other_genres_group_to_themselves(self) -> None:
        assert genre_group_for("hiphop") == "hiphop"
        assert genre_group_for("jpop") == "jpop"
        assert genre_group_for("electronic") == "electronic"
        assert genre_group_for("classical") == "classical"
        assert genre_group_for("uk-garage") == "uk-garage"
        assert genre_group_for("melodic") == "melodic"
        assert genre_group_for("dnb") == "dnb"


class TestDetectFromText10Genres:
    """_detect_from_text: テキストから10ジャンルを検出するテスト."""

    def test_detects_jpop_from_title(self) -> None:
        result = _detect_from_text("J-Pop Summer Hit", "")
        assert result == "jpop"

    def test_detects_classical_from_text(self) -> None:
        result = _detect_from_text("Symphony No.5", "Beethoven orchestra")
        assert result == "classical"

    def test_detects_melodic_from_text(self) -> None:
        # organic house と melodic house の2つがヒットすれば melodic が勝つ
        result = _detect_from_text("organic house melodic bass set", "")
        assert result == "melodic"

    def test_detects_uk_garage_from_text(self) -> None:
        # UK Garage だけ明確にヒットするキーワード
        result = _detect_from_text("UK Garage UKG Speed Garage 2step mix", "")
        assert result == "uk-garage"

    def test_detects_deep_house_from_text(self) -> None:
        # "deephouse" は deep-house キーワードのみにヒットする
        result = _detect_from_text("deephouse deep tech house", "")
        assert result == "deep-house"

    def test_unknown_when_no_keywords(self) -> None:
        result = _detect_from_text("Track 1", "Artist")
        assert result == "unknown"


class TestDetectFromAudio10Genres:
    """_detect_from_audio: 音声特徴による10ジャンル判定テスト."""

    def test_returns_tuple_of_genre_and_confidence(self) -> None:
        y = np.zeros(22050, dtype=np.float32)
        genre, conf = _detect_from_audio(y, 22050)
        assert isinstance(genre, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_empty_audio_returns_unknown(self) -> None:
        y = np.array([], dtype=np.float32)
        genre, conf = _detect_from_audio(y, 22050)
        assert genre == "unknown"
        assert conf == 0.0

    def test_confidence_is_valid_range(self) -> None:
        """さまざまな音声でconfidenceが 0〜1 の範囲に収まる."""
        for amp in [0.01, 0.1, 0.5]:
            y = (amp * np.random.default_rng(42).standard_normal(22050)).astype(np.float32)
            genre, conf = _detect_from_audio(y, 22050)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for amplitude={amp}"

    def test_genre_is_known_slug(self) -> None:
        """返されるジャンルが正規化済みスラグである."""
        valid_genres = {
            "deep-house", "house", "tech-house", "techno", "deep-techno",
            "uk-garage", "melodic", "hiphop", "jpop", "electronic",
            "classical", "dnb", "trance", "unknown",
        }
        y = np.zeros(44100, dtype=np.float32)
        genre, _ = _detect_from_audio(y, 22050)
        assert genre in valid_genres, f"Unexpected genre: {genre}"


class TestHalftimeBpmCorrection:
    """_apply_halftime_correction: ハーフタイム BPM 補正テスト."""

    def test_hiphop_bpm_above_120_corrected(self) -> None:
        assert _apply_halftime_correction(143.0, "hiphop") == 71.5

    def test_hiphop_bpm_at_boundary_120_corrected(self) -> None:
        """120 より大きい場合のみ補正."""
        assert _apply_halftime_correction(121.0, "hiphop") == 60.5

    def test_hiphop_bpm_at_120_not_corrected(self) -> None:
        """BPM == 120 は補正しない (条件: > 120)."""
        assert _apply_halftime_correction(120.0, "hiphop") == 120.0

    def test_hiphop_bpm_below_120_not_corrected(self) -> None:
        assert _apply_halftime_correction(90.0, "hiphop") == 90.0

    def test_jpop_bpm_above_120_corrected(self) -> None:
        assert _apply_halftime_correction(160.0, "jpop") == 80.0

    def test_jpop_bpm_below_120_not_corrected(self) -> None:
        assert _apply_halftime_correction(100.0, "jpop") == 100.0

    def test_house_not_corrected_even_above_120(self) -> None:
        """DJ ジャンルは補正しない."""
        assert _apply_halftime_correction(130.0, "house") == 130.0

    def test_techno_not_corrected(self) -> None:
        assert _apply_halftime_correction(135.0, "techno") == 135.0

    def test_dnb_not_corrected(self) -> None:
        assert _apply_halftime_correction(170.0, "dnb") == 170.0

    def test_unknown_genre_not_corrected(self) -> None:
        assert _apply_halftime_correction(140.0, "unknown") == 140.0

    def test_correction_rounds_to_one_decimal(self) -> None:
        """補正後は小数点1桁に丸まる."""
        result = _apply_halftime_correction(143.5, "hiphop")
        assert result == round(143.5 / 2.0, 1)

    def test_alias_hip_hop_also_corrected(self) -> None:
        """エイリアス形式 hip-hop でも補正される."""
        assert _apply_halftime_correction(143.0, "hip-hop") == 71.5


class TestDetectGenreWithHalftime:
    """detect_genre: ハーフタイム補正込みの統合テスト."""

    def test_returns_halftime_corrected_field(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._fetch_web_text",
            lambda artist, title: "hip hop rap lyrics",
        )
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("hiphop", 0.46),
        )
        y = np.zeros(44100, dtype=np.float32)
        result = detect_genre(title="Hip Hop Track", artist="Rapper", y=y, sr=44100)
        assert "halftime_corrected" in result

    def test_dj_genre_halftime_corrected_is_false(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._fetch_web_text",
            lambda artist, title: "Beatport says this is tech house.",
        )
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("house", 0.52),
        )
        y = np.zeros(44100, dtype=np.float32)
        result = detect_genre(title="House Anthem", artist="DJ", y=y, sr=44100)
        assert result["halftime_corrected"] is False
        assert "corrected_bpm" not in result

    def test_detect_genre_returns_all_required_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._fetch_web_text",
            lambda artist, title: "house music",
        )
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("house", 0.52),
        )
        y = np.zeros(44100, dtype=np.float32)
        result = detect_genre(title="Track", artist="Artist", y=y, sr=44100)
        for field in ("genre", "confidence", "sources", "genre_group", "halftime_corrected"):
            assert field in result, f"Missing field: {field}"

    def test_detect_genre_new_genres_recognized(self, monkeypatch) -> None:
        """新ジャンル (deep-house, melodic 等) が返される."""
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._fetch_web_text",
            lambda artist, title: "deep house deephouse",
        )
        monkeypatch.setattr(
            "vml_audio_lab.tools.genre._detect_from_audio",
            lambda y, sr: ("deep-house", 0.55),
        )
        y = np.zeros(44100, dtype=np.float32)
        result = detect_genre(title="Deep House Groove", artist="DJ", y=y, sr=44100)
        assert result["genre"] == "deep-house"
        assert result["genre_group"] == "house"
