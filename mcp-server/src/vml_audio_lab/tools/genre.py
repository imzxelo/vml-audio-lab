"""ジャンル推定ツール."""

from __future__ import annotations

import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import librosa
import numpy as np

_GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech-house": ("tech house", "tech-house", "minimal house", "groovy house"),
    "house": ("house", "bass house", "jackin"),
    "deep-house": ("deep house", "deephouse", "deep tech house"),
    "techno": ("techno", "hard techno", "peak time techno", "industrial techno"),
    "deep-techno": ("deep techno", "minimal techno", "dark techno", "atmospheric techno"),
    "uk-garage": ("uk garage", "2step", "2-step", "ukg", "speed garage"),
    "melodic": (
        "melodic techno",
        "melodic house",
        "melodic dubstep",
        "melodic bass",
        "organic house",
        "afro house",
    ),
    "dnb": ("dnb", "drum and bass", "drum & bass", "liquid dnb", "neurofunk"),
    "trance": ("trance", "uplifting trance", "progressive trance", "psytrance"),
    "hiphop": ("hip hop", "hip-hop", "rap", "trap", "boom bap", "r&b", "rnb"),
    "jpop": ("j-pop", "jpop", "j pop", "japanese pop", "city pop", "j-rock", "anime"),
    "electronic": ("electronic", "edm", "electronica", "synth pop", "future bass"),
    "classical": ("classical", "orchestra", "symphony", "piano", "chamber music", "opera"),
}

_GENRE_GROUP: dict[str, str] = {
    "tech-house": "house",
    "house": "house",
    "deep-house": "house",
    "techno": "techno",
    "deep-techno": "techno",
    "uk-garage": "uk-garage",
    "melodic": "melodic",
    "dnb": "dnb",
    "trance": "trance",
    "hiphop": "hiphop",
    "jpop": "jpop",
    "electronic": "electronic",
    "classical": "classical",
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
    "deep house": "deep-house",
    "deephouse": "deep-house",
    "uk garage": "uk-garage",
    "ukg": "uk-garage",
    "2step": "uk-garage",
    "deep techno": "deep-techno",
    "melodic techno": "melodic",
    "melodic house": "melodic",
    "organic house": "melodic",
    "j-pop": "jpop",
    "j pop": "jpop",
    "japanese pop": "jpop",
    "city pop": "jpop",
}

# ジャンル別の典型 BPM レンジ
_GENRE_BPM_RANGE: dict[str, tuple[float, float]] = {
    "deep-house": (118.0, 124.0),
    "house": (120.0, 130.0),
    "tech-house": (124.0, 132.0),
    "melodic": (120.0, 135.0),
    "uk-garage": (130.0, 140.0),
    "techno": (128.0, 140.0),
    "deep-techno": (125.0, 135.0),
    "hiphop": (70.0, 110.0),
    "jpop": (100.0, 180.0),
    "dnb": (160.0, 180.0),
    "trance": (128.0, 145.0),
    "electronic": (80.0, 160.0),
    "classical": (40.0, 200.0),
}

# ハーフタイム BPM 補正が必要なジャンル
_HALFTIME_GENRES: frozenset[str] = frozenset({"hiphop", "jpop"})


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

    # キーワードの長さをスコアに使う（長いほど具体的 = 高スコア）
    scores: dict[str, float] = {}
    for genre, keywords in _GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in merged:
                scores[genre] = scores.get(genre, 0.0) + len(kw)

    if not scores:
        return "unknown"
    return max(scores.items(), key=lambda item: item[1])[0]


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


def _detect_spectral_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """スペクトル特徴量を計算する（ジャンル区別に使用）。"""
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))

    # 低域 (0-200 Hz) / 高域 (4000 Hz+) のエネルギー比率
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_mask = freqs <= 200.0
    high_mask = freqs >= 4000.0
    total_energy = float(np.sum(stft)) + 1e-8
    low_ratio = float(np.sum(stft[low_mask, :])) / total_energy
    high_ratio = float(np.sum(stft[high_mask, :])) / total_energy

    return {
        "centroid": centroid,
        "rolloff": rolloff,
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
    }


def _detect_from_audio(y: np.ndarray, sr: int) -> tuple[str, float]:
    if len(y) == 0:
        return ("unknown", 0.0)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    features = _detect_spectral_features(y, sr)
    centroid = features["centroid"]
    low_ratio = features["low_ratio"]
    high_ratio = features["high_ratio"]

    # BPM + スペクトル特徴によるヒューリスティック判定
    if bpm >= 160:
        return ("dnb", 0.62)

    if bpm >= 136:
        # Techno 系: 高域ノイズ多め
        if high_ratio > 0.18:
            return ("techno", 0.57)
        # Deep Techno: 低域支配
        if low_ratio > 0.12:
            return ("deep-techno", 0.52)
        return ("techno", 0.50)

    if 130 <= bpm <= 145:
        # UK Garage: シャッフル感はスペクトルでは難しいので centroid で区別
        if centroid < 1800:
            return ("uk-garage", 0.48)
        return ("trance", 0.46)

    if 124 <= bpm <= 132:
        # Tech-House vs Techno の境界
        return ("tech-house", 0.57)

    if 120 <= bpm <= 135:
        # Melodic: メロディ成分強い (centroid 高め)
        if centroid > 2000:
            return ("melodic", 0.50)
        # Deep House: 低域厚め
        if low_ratio > 0.10:
            return ("deep-house", 0.52)
        return ("house", 0.52)

    if 118 <= bpm <= 124:
        # Deep House の典型 BPM
        if low_ratio > 0.10:
            return ("deep-house", 0.55)
        return ("house", 0.50)

    if 100 <= bpm <= 118:
        # ゆっくりめの電子音楽
        return ("electronic", 0.38)

    if 70 <= bpm <= 110:
        # Hip-Hop: 低域厚め
        if low_ratio > 0.12:
            return ("hiphop", 0.46)
        # J-Pop: 高めの centroid (ボーカル成分)
        if centroid > 2500:
            return ("jpop", 0.38)
        return ("hiphop", 0.40)

    if bpm < 70:
        return ("classical", 0.30)

    return ("unknown", 0.25)


def _apply_halftime_correction(bpm: float, genre: str) -> float:
    """Hip-Hop / J-Pop の倍テン BPM を補正する。

    BPM > 120 かつ halftime ジャンルの場合、BPM / 2 を返す。

    Args:
        bpm: 検出されたBPM
        genre: 正規化済みジャンル名

    Returns:
        補正後のBPM
    """
    slug = canonicalize_genre_slug(genre)
    if slug in _HALFTIME_GENRES and bpm > 120.0:
        return round(bpm / 2.0, 1)
    return bpm


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
    """楽曲ジャンルを複数ソースから推定する。

    10ジャンル対応。Hip-Hop / J-Pop はハーフタイム BPM 補正を行う。

    Args:
        title: 楽曲タイトル
        artist: アーティスト名
        y: 音声データ
        sr: サンプリングレート

    Returns:
        dict:
            - genre: 正規化ジャンル名
            - confidence: 確信度 (0.0〜1.0)
            - sources: 各ソースの判定結果
            - genre_group: 大分類ジャンル
            - halftime_corrected: BPM補正有無
            - corrected_bpm: 補正後 BPM (補正した場合のみ)
    """
    yt_guess = canonicalize_genre_slug(_detect_from_text(title=title, artist=artist))
    web_guess = canonicalize_genre_slug(_detect_from_web(artist=artist, title=title))
    audio_guess, audio_conf = _detect_from_audio(y=y, sr=sr)
    audio_guess = canonicalize_genre_slug(audio_guess)

    final_genre, confidence = _vote_genre({"youtube": yt_guess, "web": web_guess, "audio": audio_guess})
    final_genre = canonicalize_genre_slug(final_genre)
    if final_genre == "unknown":
        confidence = round(max(audio_conf, confidence), 2)

    # ハーフタイム BPM 補正
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    raw_bpm = float(np.atleast_1d(tempo)[0])
    corrected_bpm = _apply_halftime_correction(raw_bpm, final_genre)
    halftime_corrected = corrected_bpm != raw_bpm

    result: dict = {
        "genre": final_genre,
        "confidence": confidence,
        "sources": {
            "youtube": yt_guess,
            "web": web_guess,
            "audio": audio_guess,
        },
        "genre_group": genre_group_for(final_genre),
        "halftime_corrected": halftime_corrected,
    }
    if halftime_corrected:
        result["corrected_bpm"] = corrected_bpm

    return result
