"""ジャンル推定ツール."""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import librosa
import numpy as np

_GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech-house": ("tech house", "tech-house", "minimal house", "groovy house"),
    "house": ("house", "deep house", "bass house", "jackin"),
    "techno": ("techno", "hard techno", "melodic techno", "peak time"),
    "dnb": ("dnb", "drum and bass", "drum & bass", "liquid dnb", "neurofunk"),
    "trance": ("trance", "uplifting trance", "progressive trance", "psytrance"),
    "hiphop": ("hip hop", "hip-hop", "rap", "trap", "boom bap"),
}

_GENRE_GROUP: dict[str, str] = {
    "tech-house": "house",
    "house": "house",
    "techno": "techno",
    "dnb": "dnb",
    "trance": "trance",
    "hiphop": "hiphop",
    "unknown": "unknown",
}

_GENRE_ALIASES: dict[str, str] = {
    "tech house": "tech-house",
    "techhouse": "tech-house",
    "drum and bass": "dnb",
    "drum & bass": "dnb",
    "drum n bass": "dnb",
    "drumandbass": "dnb",
    "d&b": "dnb",
    "hip hop": "hiphop",
    "hip-hop": "hiphop",
}


def canonicalize_genre_slug(genre: str) -> str:
    """ジャンル文字列を canonical slug に正規化する。"""
    normalized = _normalize_text(genre).replace("_", "-")
    if not normalized:
        return "unknown"

    if normalized in _GENRE_GROUP:
        return normalized
    if normalized in _GENRE_ALIASES:
        return _GENRE_ALIASES[normalized]

    spaced = normalized.replace("-", " ")
    if spaced in _GENRE_ALIASES:
        return _GENRE_ALIASES[spaced]

    compact = spaced.replace(" ", "")
    if compact in _GENRE_ALIASES:
        return _GENRE_ALIASES[compact]
    return normalized


def genre_group_for(genre: str) -> str:
    """ジャンル名から保存用の大分類を返す。"""
    return _GENRE_GROUP.get(canonicalize_genre_slug(genre), "unknown")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _detect_from_text(title: str, artist: str) -> str:
    merged = _normalize_text(f"{artist} {title}")
    if not merged:
        return "unknown"

    scores = Counter()
    for genre, keywords in _GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in merged:
                scores[genre] += 1

    if not scores:
        return "unknown"
    return scores.most_common(1)[0][0]


def _fetch_web_text(artist: str, title: str) -> str:
    query = quote_plus(f"{artist} {title} genre beatport discogs")
    url = f"https://duckduckgo.com/html/?q={query}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=4) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _detect_from_web(artist: str, title: str) -> str:
    if not title and not artist:
        return "unknown"
    try:
        body = _fetch_web_text(artist, title)
    except Exception:
        return "unknown"
    return _detect_from_text(body, "")


def _detect_from_audio(y: np.ndarray, sr: int) -> tuple[str, float]:
    if len(y) == 0:
        return ("unknown", 0.0)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    low_rms = float(np.sqrt(np.mean(np.square(y))))

    # 音響モデル未導入環境でも動く、BPM中心のヒューリスティック判定
    if bpm >= 160:
        return ("dnb", 0.62)
    if bpm >= 136:
        return ("techno", 0.55 if centroid > 1200 else 0.5)
    if 125 <= bpm <= 134:
        return ("tech-house", 0.57)
    if 118 <= bpm <= 128:
        return ("house", 0.52)
    if 130 <= bpm <= 145:
        return ("trance", 0.48 if centroid > 1500 else 0.44)
    if 78 <= bpm <= 110 and low_rms < 0.22:
        return ("hiphop", 0.46)
    return ("unknown", 0.25)


def _vote_genre(sources: dict[str, str]) -> tuple[str, float]:
    weights = {"youtube": 0.35, "web": 0.4, "audio": 0.25}
    tally: dict[str, float] = {}
    total = 0.0
    for name, genre in sources.items():
        if not genre or genre == "unknown":
            continue
        weight = weights.get(name, 0.0)
        total += weight
        tally[genre] = tally.get(genre, 0.0) + weight

    if not tally:
        return ("unknown", 0.0)

    winner = max(tally.items(), key=lambda item: item[1])
    confidence = winner[1] / total if total > 0 else 0.0
    return (winner[0], round(float(confidence), 2))


def detect_genre(title: str, artist: str, y: np.ndarray, sr: int) -> dict:
    """楽曲ジャンルを複数ソースから推定する。"""
    yt_guess = canonicalize_genre_slug(_detect_from_text(title=title, artist=artist))
    web_guess = canonicalize_genre_slug(_detect_from_web(artist=artist, title=title))
    audio_guess, audio_conf = _detect_from_audio(y=y, sr=sr)
    audio_guess = canonicalize_genre_slug(audio_guess)

    final_genre, confidence = _vote_genre({"youtube": yt_guess, "web": web_guess, "audio": audio_guess})
    final_genre = canonicalize_genre_slug(final_genre)
    if final_genre == "unknown":
        confidence = round(max(audio_conf, confidence), 2)

    return {
        "genre": final_genre,
        "confidence": confidence,
        "sources": {
            "youtube": yt_guess,
            "web": web_guess,
            "audio": audio_guess,
        },
        "genre_group": genre_group_for(final_genre),
    }
