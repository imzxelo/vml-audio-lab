"""Camelot Wheel ユーティリティ.

Camelot Wheel はDJミックス時のキー相性判定に使う円形システム。
各キーに番号(1-12)と A/B の記号を割り当て、隣接キーとの相性を示す。

参考: https://mixedinkey.com/camelot-wheel/
"""

from __future__ import annotations

# キーラベル → Camelot コード のマッピング
# A = マイナーキー、B = メジャーキー
_KEY_TO_CAMELOT: dict[str, str] = {
    # Minor keys (A)
    "Abm": "1A",
    "G#m": "1A",
    "Ebm": "2A",
    "D#m": "2A",
    "Bbm": "3A",
    "A#m": "3A",
    "Fm": "4A",
    "Cm": "5A",
    "Gm": "6A",
    "Dm": "7A",
    "Am": "8A",
    "Em": "9A",
    "Bm": "10A",
    "F#m": "11A",
    "Gbm": "11A",
    "C#m": "12A",
    "Dbm": "12A",
    # Major keys (B)
    "B": "1B",
    "Gb": "2B",
    "F#": "2B",
    "Db": "3B",
    "C#": "3B",
    "Ab": "4B",
    "G#": "4B",
    "Eb": "5B",
    "D#": "5B",
    "Bb": "6B",
    "A#": "6B",
    "F": "7B",
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
}

# Camelot コード → キーラベル のマッピング (逆引き用、主要エンハーモニック)
_CAMELOT_TO_KEY: dict[str, str] = {
    "1A": "G#m",
    "2A": "D#m",
    "3A": "A#m",
    "4A": "Fm",
    "5A": "Cm",
    "6A": "Gm",
    "7A": "Dm",
    "8A": "Am",
    "9A": "Em",
    "10A": "Bm",
    "11A": "F#m",
    "12A": "C#m",
    "1B": "B",
    "2B": "F#",
    "3B": "C#",
    "4B": "Ab",
    "5B": "Eb",
    "6B": "Bb",
    "7B": "F",
    "8B": "C",
    "9B": "G",
    "10B": "D",
    "11B": "A",
    "12B": "E",
}


def key_to_camelot(key_label: str) -> str | None:
    """キーラベルを Camelot コードに変換する。

    Args:
        key_label: キーラベル (例: "Fm", "C", "F#m", "Am")

    Returns:
        Camelot コード (例: "4A", "8B") または None（認識できない場合）
    """
    return _KEY_TO_CAMELOT.get(key_label)


def camelot_to_key(camelot: str) -> str | None:
    """Camelot コードをキーラベルに変換する。

    Args:
        camelot: Camelot コード (例: "4A", "8B")

    Returns:
        キーラベル (例: "Fm", "C") または None（認識できない場合）
    """
    return _CAMELOT_TO_KEY.get(camelot.upper())


def _parse_camelot(camelot: str) -> tuple[int, str] | None:
    """Camelot コードを (番号, A/B) に分解する。"""
    code = camelot.strip().upper()
    if len(code) < 2:
        return None
    suffix = code[-1]
    if suffix not in ("A", "B"):
        return None
    try:
        number = int(code[:-1])
    except ValueError:
        return None
    if not (1 <= number <= 12):
        return None
    return (number, suffix)


def compatible_camelot_codes(camelot: str) -> list[str]:
    """指定 Camelot コードと相性の良いコードリストを返す。

    相性ルール:
        - ◎ 同じコード (完全一致)
        - ◎ 同番号の A↔B (relative major/minor)
        - ○ 番号 ±1 の同じ A/B (Camelot 隣接)

    Args:
        camelot: Camelot コード (例: "4A")

    Returns:
        相性の良い Camelot コードのリスト (同コード含む)
    """
    parsed = _parse_camelot(camelot)
    if parsed is None:
        return []

    number, suffix = parsed
    other = "B" if suffix == "A" else "A"

    # 隣接番号 (1-12 の円環)
    prev_num = 12 if number == 1 else number - 1
    next_num = 1 if number == 12 else number + 1

    return [
        f"{number}{suffix}",   # 同コード
        f"{number}{other}",    # 同番号 A↔B (relative)
        f"{prev_num}{suffix}", # -1 隣接
        f"{next_num}{suffix}", # +1 隣接
    ]


def compatibility_score(camelot_a: str, camelot_b: str) -> float:
    """2つの Camelot コードの相性スコアを返す。

    スコア:
        1.0: 完全一致
        0.9: 同番号 A↔B (relative major/minor)
        0.7: 番号 ±1 の同 A/B
        0.0: それ以外

    Args:
        camelot_a: Camelot コード A
        camelot_b: Camelot コード B

    Returns:
        相性スコア (0.0〜1.0)
    """
    parsed_a = _parse_camelot(camelot_a)
    parsed_b = _parse_camelot(camelot_b)
    if parsed_a is None or parsed_b is None:
        return 0.0

    num_a, suf_a = parsed_a
    num_b, suf_b = parsed_b

    if num_a == num_b and suf_a == suf_b:
        return 1.0

    if num_a == num_b and suf_a != suf_b:
        return 0.9

    if suf_a == suf_b and abs(num_a - num_b) in (1, 11):
        # 11 = 12-1 の円環差 (1A ↔ 12A など)
        return 0.7

    return 0.0
