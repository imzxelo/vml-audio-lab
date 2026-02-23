"""ライブラリインデックスとトラックマッチング.

USB ドライブまたは Rekordbox XML をスキャンしてトラックインデックスを構築し、
キー・BPM・ムード相性でマッチングを行う。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

# ライブラリインデックスのデフォルトキャッシュ先
_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "vml_audio_lab"

# スキャン対象の音声拡張子
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".aac", ".webm", ".opus", ".ogg"}
)

# ムード隣接マップ (同ムード = 1.0, 隣接 = 0.5 で評価する)
_MOOD_ADJACENCY: dict[str, frozenset[str]] = {
    "Peak Time": frozenset({"Melodic Journey", "Groovy & Warm"}),
    "Deep Night": frozenset({"Melodic Journey", "Chill & Mellow"}),
    "Melodic Journey": frozenset({"Peak Time", "Deep Night"}),
    "Groovy & Warm": frozenset({"Peak Time", "Chill & Mellow"}),
    "Chill & Mellow": frozenset({"Groovy & Warm", "Deep Night"}),
}


# ---------------------------------------------------------------------------
# キャッシュ I/O
# ---------------------------------------------------------------------------


def save_index(index: dict, cache_path: str | None = None) -> str:
    """インデックスを JSON ファイルに保存する。

    Args:
        index: scan_library が返すインデックス辞書
        cache_path: 保存先パス。省略時は /tmp/vml_audio_lab/library_index.json

    Returns:
        保存したファイルの絶対パス文字列
    """
    if cache_path is None:
        _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dest = _DEFAULT_CACHE_DIR / "library_index.json"
    else:
        dest = Path(cache_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)

    dest.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return str(dest)


def load_index(cache_path: str | None = None) -> dict | None:
    """キャッシュ済みインデックスを読み込む。

    Args:
        cache_path: キャッシュファイルパス。省略時はデフォルト位置を使用。

    Returns:
        インデックス辞書、またはキャッシュが存在しない場合は None
    """
    if cache_path is None:
        path = _DEFAULT_CACHE_DIR / "library_index.json"
    else:
        path = Path(cache_path).expanduser()

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ソース判定とハッシュ
# ---------------------------------------------------------------------------


def _source_fingerprint(source: str) -> str:
    """ソースのフィンガープリントを生成する（変更検知用）。

    ディレクトリの場合: パス + ファイル数 + 合計サイズのハッシュ。
    XML ファイルの場合: パス + ファイルサイズ + 最終更新時刻のハッシュ。
    """
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        audio_files = sorted(
            f for ext in _AUDIO_EXTENSIONS for f in path.rglob(f"*{ext}")
        )
        count = len(audio_files)
        total_size = sum(f.stat().st_size for f in audio_files if f.exists())
        fingerprint = f"{path}|count={count}|size={total_size}"
    else:
        stat = path.stat()
        fingerprint = f"{path}|size={stat.st_size}|mtime={stat.st_mtime}"

    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Rekordbox XML パーサー
# ---------------------------------------------------------------------------


def _location_to_path(location: str) -> str:
    """Rekordbox の Location 属性をファイルシステムパスに変換する。

    例: "file://localhost/Volumes/USB/track.mp3" → "/Volumes/USB/track.mp3"
    """
    try:
        parsed = urlparse(location)
        if parsed.scheme == "file":
            # localhost パート除去 + URL デコード
            path = unquote(parsed.path)
            # Windows パス対応: 先頭の / を除去 (C:/... 形式)
            if len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return path
    except Exception:
        pass
    return location


def _tonality_to_key_label(tonality: str) -> str:
    """Rekordbox の Tonality 属性をキーラベルに変換する。

    Rekordbox: "Fm", "4A", "C", "8B" 等の形式が混在する。
    Camelot 形式 ("4A") はキーラベルに変換する。
    """
    from vml_audio_lab.tools.camelot import camelot_to_key

    val = tonality.strip()
    if not val:
        return ""

    # Camelot 形式の検出: 数字 + A/B
    if len(val) >= 2 and val[-1].upper() in ("A", "B"):
        try:
            int(val[:-1])
            converted = camelot_to_key(val)
            return converted if converted else val
        except ValueError:
            pass

    return val


def _parse_rekordbox_xml(xml_path: str) -> list[dict]:
    """Rekordbox XML からトラックリストを抽出する。

    Args:
        xml_path: DJ_PLAYLISTS 形式の XML ファイルパス

    Returns:
        トラック辞書のリスト
    """
    from vml_audio_lab.tools.camelot import key_to_camelot
    from vml_audio_lab.tools.genre import canonicalize_genre_slug, genre_group_for

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Rekordbox XML のパースに失敗しました: {xml_path}") from exc
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        return []

    tracks: list[dict] = []
    for i, elem in enumerate(collection.findall("TRACK")):
        title = elem.get("Name", "")
        artist = elem.get("Artist", "")
        bpm_str = elem.get("AverageBpm", "0")
        tonality = elem.get("Tonality", "")
        genre_raw = elem.get("Genre", "")
        location = elem.get("Location", "")
        total_time = elem.get("TotalTime", "0")
        track_id = elem.get("TrackID", str(i + 1))

        try:
            bpm = round(float(bpm_str), 1)
        except (ValueError, TypeError):
            bpm = 0.0

        try:
            duration_sec = float(total_time)
        except (ValueError, TypeError):
            duration_sec = 0.0

        key_label = _tonality_to_key_label(tonality)
        camelot_code = key_to_camelot(key_label) or ""
        genre = canonicalize_genre_slug(genre_raw)
        genre_group = genre_group_for(genre)
        file_path = _location_to_path(location)

        tracks.append({
            "id": int(track_id),
            "title": title,
            "artist": artist,
            "bpm": bpm,
            "key_label": key_label,
            "camelot_code": camelot_code,
            "genre": genre,
            "genre_group": genre_group,
            "mood": "",  # XML には mood 情報がないため空
            "file_path": file_path,
            "duration_sec": round(duration_sec, 2),
        })

    return tracks


# ---------------------------------------------------------------------------
# ディレクトリスキャナー (軽量分析)
# ---------------------------------------------------------------------------


def _analyze_audio_file(file_path: str, track_id: int) -> dict:
    """単一音声ファイルを librosa で軽量分析してトラック辞書を返す。"""
    import librosa
    import numpy as np

    from vml_audio_lab.tools.analysis import detect_key
    from vml_audio_lab.tools.camelot import key_to_camelot
    from vml_audio_lab.tools.genre import canonicalize_genre_slug, genre_group_for
    from vml_audio_lab.tools.loader import _CACHE_DIR, DEFAULT_SR, _cache_key  # noqa: PLC2701
    from vml_audio_lab.tools.mood import detect_mood

    path = Path(file_path)
    stem = path.stem

    # BPM 検出
    try:
        y, _ = librosa.load(str(path), sr=DEFAULT_SR, mono=True, duration=60.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=DEFAULT_SR)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    except Exception:
        y = np.zeros(DEFAULT_SR, dtype=np.float32)
        bpm = 0.0

    # キー検出 (y_path 経由で detect_key を呼ぶためキャッシュに書く)
    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(str(path))
    y_path = str(cache_dir / f"{cache_key}_y.npy")
    np.save(y_path, y)

    try:
        key_result = detect_key(y_path)
        key_label = key_result["key_label"]
        scale = key_result["scale"]
    except Exception:
        key_label = ""
        scale = "major"

    camelot_code = key_to_camelot(key_label) or ""

    # ムード判定
    try:
        mood_result = detect_mood(y=y, sr=DEFAULT_SR, bpm=bpm, scale=scale)
        mood = mood_result["mood"]
    except Exception:
        mood = ""

    # ジャンルはファイル名から推定 (精度は低い)
    genre = canonicalize_genre_slug(stem)
    if genre == "unknown":
        genre = ""
    genre_group = genre_group_for(genre) if genre else "unknown"

    # duration_sec: soundfile でメタデータから取得 (librosa.load は duration=60.0 で切るため)
    try:
        import soundfile as sf
        sf_info = sf.info(str(path))
        duration_sec = round(float(sf_info.duration), 2)
    except Exception:
        duration_sec = round(float(librosa.get_duration(y=y, sr=DEFAULT_SR)), 2)

    return {
        "id": track_id,
        "title": stem,
        "artist": "",
        "bpm": bpm,
        "key_label": key_label,
        "camelot_code": camelot_code,
        "genre": genre,
        "genre_group": genre_group,
        "mood": mood,
        "file_path": str(path.resolve()),
        "duration_sec": duration_sec,
    }


def _scan_directory(dir_path: str) -> list[dict]:
    """ディレクトリをスキャンして音声ファイルを軽量分析する。

    Args:
        dir_path: スキャン対象ディレクトリパス

    Returns:
        トラック辞書のリスト
    """
    root = Path(dir_path).expanduser().resolve()
    audio_files = sorted(
        f
        for ext in _AUDIO_EXTENSIONS
        for f in root.rglob(f"*{ext}")
    )

    tracks: list[dict] = []
    for i, audio_file in enumerate(audio_files, start=1):
        try:
            track = _analyze_audio_file(str(audio_file), track_id=i)
            tracks.append(track)
        except Exception:
            # 分析失敗はスキップ
            continue

    return tracks


# ---------------------------------------------------------------------------
# scan_library (公開 API)
# ---------------------------------------------------------------------------


def scan_library(source: str, cache_path: str | None = None) -> dict:
    """ライブラリをスキャンしてインデックスを構築する。

    source が .xml ファイルの場合は Rekordbox XML をパースする。
    source がディレクトリの場合はフォルダを走査して軽量分析する。
    フィンガープリントが一致する場合はキャッシュを返す。

    Args:
        source: USB パスまたは Rekordbox XML ファイルパス
        cache_path: キャッシュ保存先パス（省略時はデフォルト位置）

    Returns:
        dict:
            - tracks: トラック辞書のリスト
            - total: トラック数
            - source: スキャン元パス
            - cached: キャッシュから読み込んだか
            - fingerprint: ソースのフィンガープリント
    """
    src_path = Path(source).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"ライブラリソースが見つかりません: {src_path}")
    fingerprint = _source_fingerprint(str(src_path))

    # キャッシュ確認
    cached = load_index(cache_path)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return {**cached, "cached": True}

    # スキャン実行
    is_xml = src_path.suffix.lower() == ".xml"
    if is_xml:
        tracks = _parse_rekordbox_xml(str(src_path))
    else:
        tracks = _scan_directory(str(src_path))

    index = {
        "tracks": tracks,
        "total": len(tracks),
        "source": str(src_path),
        "cached": False,
        "fingerprint": fingerprint,
    }

    save_index(index, cache_path)
    return index


# ---------------------------------------------------------------------------
# BPM 相性スコア
# ---------------------------------------------------------------------------


def _bpm_score(bpm_a: float, bpm_b: float) -> float:
    """2つの BPM の相性スコアを計算する。

    スコア:
        1.0: ±3 BPM 以内
        0.8: ±5 BPM 以内
        0.5: ±10 BPM 以内
        0.0: それ以外

    ハーフタイム (bpm_a * 2 または / 2) も考慮する。
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0

    def _raw_score(a: float, b: float) -> float:
        diff = abs(a - b)
        if diff <= 3.0:
            return 1.0
        if diff <= 5.0:
            return 0.8
        if diff <= 10.0:
            return 0.5
        return 0.0

    # 通常 + ハーフタイム比較
    return max(
        _raw_score(bpm_a, bpm_b),
        _raw_score(bpm_a * 2, bpm_b),
        _raw_score(bpm_a / 2, bpm_b),
    )


# ---------------------------------------------------------------------------
# ムード相性スコア
# ---------------------------------------------------------------------------


def _mood_score(mood_a: str, mood_b: str) -> float:
    """2つのムードの相性スコアを計算する。

    スコア:
        1.0: 同ムード
        0.5: 隣接ムード
        0.0: それ以外
    """
    if not mood_a or not mood_b:
        return 0.0
    if mood_a == mood_b:
        return 1.0
    if mood_b in _MOOD_ADJACENCY.get(mood_a, frozenset()):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# 相性ラベル
# ---------------------------------------------------------------------------


def _compatibility_label(key_score: float) -> str:
    """キー相性スコアから表示ラベルを生成する。"""
    if key_score >= 0.9:
        return "◎"
    if key_score >= 0.7:
        return "○"
    return "△"


# ---------------------------------------------------------------------------
# サジェスト文生成
# ---------------------------------------------------------------------------


def _build_suggestion(
    key_score: float,
    bpm_score: float,
    mood_score: float,
    query_bpm: float,
    track_bpm: float,
    query_mood: str | None,
    track_mood: str,
) -> str:
    """マッチング結果の日本語サジェスト文を生成する。"""
    parts: list[str] = []

    if key_score >= 1.0:
        parts.append("同キー")
    elif key_score >= 0.9:
        parts.append("同番号 A↔B (relative)")
    elif key_score >= 0.7:
        parts.append("Camelot 隣接 (±1)")
    else:
        parts.append("キー不一致")

    bpm_diff = abs(query_bpm - track_bpm) if query_bpm > 0 and track_bpm > 0 else 0.0
    if bpm_score >= 1.0:
        parts.append(f"BPM差{bpm_diff:.1f}（完璧）")
    elif bpm_score >= 0.8:
        parts.append(f"BPM差{bpm_diff:.1f}（良好）")
    elif bpm_score >= 0.5:
        parts.append(f"BPM差{bpm_diff:.1f}（許容範囲）")
    else:
        parts.append("BPM帯が異なる（ハーフタイム検討）")

    if query_mood:
        if mood_score >= 1.0:
            parts.append("ムード一致")
        elif mood_score >= 0.5:
            parts.append(f"ムード近似（{track_mood}）")

    return "、".join(parts)


# ---------------------------------------------------------------------------
# find_compatible_tracks (公開 API)
# ---------------------------------------------------------------------------


def _track_camelot(track: dict) -> str:
    """トラック辞書から Camelot コードを取得する (複数キー名を許容)。"""
    return track.get("camelot") or track.get("camelot_code") or ""


def _track_key_label(track: dict) -> str:
    """トラック辞書からキーラベルを取得する (複数キー名を許容)。"""
    return track.get("key_label") or track.get("key") or ""


def _bpm_range_score(bpm: float, bpm_range: list[float] | None) -> float:
    """BPM がレンジ内かどうかでスコアを返す。

    bpm_range がない場合はフィルタなし (1.0 を返す)。
    """
    if bpm_range is None or len(bpm_range) < 2:
        return 1.0
    lo, hi = float(bpm_range[0]), float(bpm_range[1])
    if lo <= bpm <= hi:
        return 1.0
    # レンジ外でも近い場合は部分スコア
    dist = min(abs(bpm - lo), abs(bpm - hi))
    if dist <= 5.0:
        return 0.5
    if dist <= 10.0:
        return 0.2
    return 0.0


def _genre_bonus(track_genre: str, compatible_genres: list[str] | None) -> float:
    """トラックのジャンルが compatible_genres に含まれる場合にボーナスを返す。"""
    if not compatible_genres or not track_genre:
        return 0.0
    return 0.1 if track_genre in compatible_genres else 0.0


def _load_index_from_path(library_index_path: str) -> dict:
    """JSON ファイルからライブラリインデックスを読み込む。"""
    path = Path(library_index_path)
    if not path.exists():
        raise FileNotFoundError(f"ライブラリインデックスが見つかりません: {library_index_path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"ライブラリインデックスの読み込みに失敗しました: {exc}") from exc


def find_compatible_tracks(
    query_key_label: str,
    query_bpm: float,
    query_mood: str | None = None,
    query_genre: str | None = None,
    library_index: dict | None = None,
    library_source: str | None = None,
    max_results: int = 10,
) -> dict:
    """ライブラリからキー・BPM・ムード相性でトラックをマッチングする。

    library_index または library_source のいずれかを指定する。

    Args:
        query_key_label: クエリトラックのキーラベル (例: "Fm", "Am")
        query_bpm: クエリトラックの BPM
        query_mood: クエリトラックのムード名（省略可）
        query_genre: クエリトラックのジャンル（省略可）
        library_index: scan_library が返すインデックス辞書（直接渡す場合）
        library_source: ソースパス（インデックス未作成時）
        max_results: 返す最大件数

    Returns:
        dict:
            - matches: マッチ結果リスト (rank, track, bpm, key, camelot, genre, mood,
                        compatibility ◎/○/△, score, suggestion)
            - query: クエリ情報 (key, camelot, bpm, mood, genre)
            - total_scanned: スキャンしたトラック数
            - compatible_count: 相性スコア > 0 のトラック数
    """
    from vml_audio_lab.tools.camelot import compatibility_score, key_to_camelot

    if library_index is None:
        if library_source is None:
            raise ValueError("library_index または library_source のいずれかを指定してください")
        library_index = scan_library(library_source)

    query_camelot = key_to_camelot(query_key_label) or ""
    tracks = library_index.get("tracks", [])
    scored: list[tuple[float, dict]] = []

    for track in tracks:
        track_camelot = _track_camelot(track)
        track_bpm = float(track.get("bpm", 0))
        track_mood = track.get("mood", "")

        k_score = (
            compatibility_score(query_camelot, track_camelot)
            if (query_camelot and track_camelot)
            else 0.0
        )
        b_score = _bpm_score(query_bpm, track_bpm)
        m_score = _mood_score(query_mood or "", track_mood)
        combined = k_score * 0.6 + b_score * 0.3 + m_score * 0.1

        if combined > 0:
            scored.append((combined, track))

    scored.sort(key=lambda x: x[0], reverse=True)
    compatible_count = len(scored)

    matches: list[dict] = []
    for rank, (score, track) in enumerate(scored[:max_results], start=1):
        track_camelot = _track_camelot(track)
        track_bpm = float(track.get("bpm", 0))
        track_mood = track.get("mood", "")

        k_score = (
            compatibility_score(query_camelot, track_camelot)
            if (query_camelot and track_camelot)
            else 0.0
        )
        b_score = _bpm_score(query_bpm, track_bpm)
        m_score = _mood_score(query_mood or "", track_mood)
        suggestion = _build_suggestion(
            k_score, b_score, m_score,
            query_bpm, track_bpm,
            query_mood, track_mood,
        )

        matches.append({
            "rank": rank,
            "track": track,
            "bpm": track_bpm,
            "key": _track_key_label(track),
            "camelot": track_camelot,
            "genre": track.get("genre", ""),
            "mood": track_mood,
            "compatibility": _compatibility_label(k_score),
            "score": round(score, 3),
            "suggestion": suggestion,
        })

    return {
        "matches": matches,
        "query": {
            "key": query_key_label,
            "camelot": query_camelot,
            "bpm": query_bpm,
            "mood": query_mood,
            "genre": query_genre,
        },
        "total_scanned": len(tracks),
        "compatible_count": compatible_count,
    }


def find_compatible_tracks_by_vocal(
    vocal_analysis: dict,
    library_index_path: str,
    max_results: int = 10,
) -> dict:
    """ボーカル分析結果からライブラリ内の相性トラックを検索する。

    ボーカル分析辞書とライブラリインデックス JSON のパスを受け取り、
    相性の良いトラックを降順でランク付けして返す。

    Args:
        vocal_analysis: ボーカル/クエリトラックの分析情報。以下のキーを参照:
            - key (str): キーラベル (例: "Am", "Fm")
            - camelot (str, optional): Camelot コード。なければ key から変換
            - compatible_bpm_range (list[float], optional): [min_bpm, max_bpm]
            - compatible_genres (list[str], optional): 相性の良いジャンルリスト
            - mood (str, optional): ムード名
        library_index_path: scan_library で生成した JSON インデックスのパス
        max_results: 返す最大件数

    Returns:
        dict:
            - matches: マッチ結果リスト。各要素:
                - track: 元のトラック辞書
                - compatibility: キー相性スコア (0.0〜1.0)
                - suggestion: 日本語の提案説明文
                - score: 総合スコア (0.0〜1.0)
            - total_scanned: スキャンしたトラック数
            - query: クエリ情報
    """
    from vml_audio_lab.tools.camelot import compatibility_score, key_to_camelot

    # クエリ情報の取得
    query_key = _track_key_label(vocal_analysis)
    query_camelot = vocal_analysis.get("camelot") or key_to_camelot(query_key) or ""
    bpm_range: list[float] | None = vocal_analysis.get("compatible_bpm_range")
    query_mood: str = vocal_analysis.get("mood") or ""
    compatible_genres: list[str] | None = vocal_analysis.get("compatible_genres")

    # インデックス読み込み
    index = _load_index_from_path(library_index_path)
    tracks = index.get("tracks", [])

    scored: list[tuple[float, float, dict]] = []

    for track in tracks:
        track_camelot = _track_camelot(track)
        track_bpm = float(track.get("bpm", 0))
        track_mood = track.get("mood") or ""
        track_genre = track.get("genre") or ""

        k_score = (
            compatibility_score(query_camelot, track_camelot)
            if (query_camelot and track_camelot)
            else 0.0
        )
        b_score = _bpm_range_score(track_bpm, bpm_range)
        m_score = _mood_score(query_mood, track_mood)
        g_bonus = _genre_bonus(track_genre, compatible_genres)
        total = k_score * 0.6 + b_score * 0.3 + m_score * 0.05 + g_bonus

        if total > 0:
            scored.append((total, k_score, track))

    scored.sort(key=lambda x: x[0], reverse=True)

    matches: list[dict] = []
    for total_score, k_score, track in scored[:max_results]:
        track_bpm = float(track.get("bpm", 0))
        track_mood = track.get("mood") or ""
        query_bpm_center = (
            (bpm_range[0] + bpm_range[1]) / 2.0 if bpm_range and len(bpm_range) >= 2 else 0.0
        )
        b_score = _bpm_range_score(track_bpm, bpm_range)
        m_score = _mood_score(query_mood, track_mood)
        suggestion = _build_suggestion(
            k_score, b_score, m_score,
            query_bpm_center, track_bpm,
            query_mood or None, track_mood,
        )

        matches.append({
            "track": track,
            "compatibility": k_score,
            "suggestion": suggestion,
            "score": round(total_score, 3),
        })

    return {
        "matches": matches,
        "total_scanned": len(tracks),
        "query": {
            "key": query_key,
            "camelot": query_camelot,
            "bpm_range": bpm_range,
            "mood": query_mood,
        },
    }


# Alias for backward compatibility with MVP-C vocal analysis
find_compatible_tracks_by_params = find_compatible_tracks
