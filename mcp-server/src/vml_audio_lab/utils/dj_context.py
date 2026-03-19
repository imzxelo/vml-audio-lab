"""DJ コンテキストヘルパー — DJ特化の推薦・分類ロジック.

トランジションタイプの推薦、エネルギーフェーズ分類、ジャンル別ミキシングTips。
"""

from __future__ import annotations


# ジャンルペア → 推奨トランジションタイプのマッピング
_GENRE_TRANSITION_OVERRIDES: dict[frozenset[str], str] = {
    frozenset({"hiphop", "rnb"}): "echo_out",
    frozenset({"hiphop", "jpop"}): "echo_out",
    frozenset({"classical", "house"}): "quick_cut",
    frozenset({"classical", "techno"}): "quick_cut",
}

# ジャンル → デフォルトのトランジションタイプ
_GENRE_DEFAULT_TRANSITION: dict[str, str] = {
    "house": "eq_swap",
    "techno": "eq_swap",
    "uk-garage": "eq_swap",
    "melodic": "filter_sweep",
    "trance": "filter_sweep",
    "dnb": "eq_swap",
    "hiphop": "echo_out",
    "rnb": "echo_out",
    "jpop": "quick_cut",
    "electronic": "eq_swap",
}

# トランジションタイプ → 詳細情報
_TRANSITION_DETAILS: dict[str, dict] = {
    "eq_swap": {
        "description_ja": (
            "EQスワップ: トラックAの低域を徐々に下げながら、"
            "トラックBの低域を上げていく。"
            "ハイハットが重なるタイミングでフェーダーを切り替える。"
        ),
        "bars": 16,
        "difficulty": "easy",
    },
    "filter_sweep": {
        "description_ja": (
            "フィルタースウィープ: トラックAにローパスフィルターをかけて"
            "徐々に音を籠らせつつ、トラックBを裏で入れる。"
            "フィルターを戻すタイミングでBがメインになる。"
        ),
        "bars": 32,
        "difficulty": "medium",
    },
    "echo_out": {
        "description_ja": (
            "エコーアウト: トラックAにエコー/ディレイをかけてフェードアウト。"
            "残響が消える前にトラックBを入れる。"
            "ジャンルが変わるときに使いやすい。"
        ),
        "bars": 8,
        "difficulty": "easy",
    },
    "quick_cut": {
        "description_ja": (
            "クイックカット: トラックAのブレイクや小節の頭で"
            "一気にトラックBに切り替える。"
            "インパクトがあるが失敗すると目立つ。"
        ),
        "bars": 1,
        "difficulty": "medium",
    },
    "beat_match_blend": {
        "description_ja": (
            "ビートマッチブレンド: 両曲のBPMを完全に合わせて、"
            "長い時間をかけてゆっくり混ぜる。"
            "キーが合っているときに最も効果的。"
        ),
        "bars": 32,
        "difficulty": "hard",
    },
}


def recommend_transition_type(
    track_a: dict,
    track_b: dict,
) -> dict:
    """2トラック間の最適なトランジションタイプを推薦する。

    Args:
        track_a: 分析済みトラック辞書。キー: genre/genre_group, bpm, camelot, energy_level
        track_b: 同上

    Returns:
        dict:
            - type: トランジションタイプ名
            - bars: 推奨小節数
            - description_ja: 日本語での操作手順
            - difficulty: "easy"|"medium"|"hard"
    """
    genre_a = track_a.get("genre_group") or track_a.get("genre", "")
    genre_b = track_b.get("genre_group") or track_b.get("genre", "")
    bpm_a = float(track_a.get("bpm", 0))
    bpm_b = float(track_b.get("bpm", 0))

    # ジャンルペアのオーバーライドをチェック
    genre_pair = frozenset({genre_a, genre_b})
    if genre_pair in _GENRE_TRANSITION_OVERRIDES:
        t_type = _GENRE_TRANSITION_OVERRIDES[genre_pair]
    elif abs(bpm_a - bpm_b) > 8:
        # BPM差が大きい場合はエコーアウトかクイックカット
        t_type = "echo_out"
    elif genre_a == genre_b:
        # 同ジャンルはジャンルデフォルト
        t_type = _GENRE_DEFAULT_TRANSITION.get(genre_a, "eq_swap")
    else:
        # 異ジャンルはフィルタースウィープ
        t_type = "filter_sweep"

    # キー互換性が高い場合はロングブレンドも選択肢
    from vml_audio_lab.tools.camelot import compatibility_score, key_to_camelot

    camelot_a = track_a.get("camelot") or key_to_camelot(
        track_a.get("key_label") or track_a.get("key", "")
    ) or ""
    camelot_b = track_b.get("camelot") or key_to_camelot(
        track_b.get("key_label") or track_b.get("key", "")
    ) or ""
    key_score = compatibility_score(camelot_a, camelot_b) if (camelot_a and camelot_b) else 0.0

    if key_score >= 0.9 and abs(bpm_a - bpm_b) <= 3 and t_type == "eq_swap":
        t_type = "beat_match_blend"

    details = _TRANSITION_DETAILS.get(t_type, _TRANSITION_DETAILS["eq_swap"])

    return {
        "type": t_type,
        "bars": details["bars"],
        "description_ja": details["description_ja"],
        "difficulty": details["difficulty"],
    }


# フェーズ定義
_PHASES: list[dict] = [
    {
        "phase": "warm_up",
        "label": "ウォームアップ",
        "description_ja": "フロアがまだ温まっていない時間帯。ゆっくり始めて雰囲気を作る。",
        "tips": [
            "BPM120以下のチルい曲で始める",
            "マイナーキーで深い雰囲気を作ると◎",
            "まだ踊ってない人が多いのでグルーヴ重視",
        ],
    },
    {
        "phase": "build_up",
        "label": "ビルドアップ",
        "description_ja": "エネルギーを徐々に上げていく時間帯。BPMとテンションを少しずつ上昇。",
        "tips": [
            "BPMを2-4ずつ上げていく",
            "認知度の高い曲を混ぜてフロアの反応を掴む",
            "EQスワップで丁寧に繋ぐ",
        ],
    },
    {
        "phase": "peak",
        "label": "ピーク",
        "description_ja": "最もエネルギーが高い時間帯。フロアが最高潮。キラー曲はここで投入。",
        "tips": [
            "一番盛り上がる曲をここに配置",
            "BPM 126-132 が最も踊りやすい",
            "ドロップが印象的な曲を連続で繋ぐ",
        ],
    },
    {
        "phase": "cool_down",
        "label": "クールダウン",
        "description_ja": "セット終盤。徐々にエネルギーを下げて次のDJに渡す、または余韻を残す。",
        "tips": [
            "BPMを少しずつ下げる",
            "メロディックな曲で余韻を作る",
            "最後の曲は印象に残るものを選ぶ",
        ],
    },
]


def classify_energy_phase(
    position: int,
    total: int,
    energy: float,
) -> dict:
    """セットリスト内のポジションからフェーズを分類する。

    Args:
        position: 現在のトラック位置 (0-indexed)
        total: トラック総数
        energy: このトラックのエネルギーレベル (0.0-1.0)

    Returns:
        dict:
            - phase: フェーズ名 ("warm_up"|"build_up"|"peak"|"cool_down")
            - label: 日本語ラベル
            - description_ja: 日本語説明
            - tips: DJ向けTipsリスト
    """
    if total <= 0:
        return _PHASES[0]  # fallback

    ratio = position / max(1, total - 1)

    # エネルギーとポジションの両方を考慮
    if ratio <= 0.2 or (ratio <= 0.3 and energy < 0.4):
        phase_data = _PHASES[0]  # warm_up
    elif ratio <= 0.5 or (ratio <= 0.6 and energy < 0.6):
        phase_data = _PHASES[1]  # build_up
    elif ratio <= 0.8 and energy >= 0.5:
        phase_data = _PHASES[2]  # peak
    elif ratio > 0.8 or energy < 0.4:
        phase_data = _PHASES[3]  # cool_down
    else:
        phase_data = _PHASES[2]  # peak (default for high energy mid-set)

    return {
        "phase": phase_data["phase"],
        "label": phase_data["label"],
        "description_ja": phase_data["description_ja"],
        "tips": phase_data["tips"],
    }


# ジャンルペア → ミキシングTips
_GENRE_MIXING_TIPS: dict[frozenset[str], list[str]] = {
    frozenset({"house", "techno"}): [
        "HouseからTechnoへの移行はEQスワップで16小節がベスト",
        "低域を先に入れ替え、ハイハットで繋ぐとスムーズ",
        "Techno側のキックが入った瞬間にHouse側の低域を抜く",
    ],
    frozenset({"house", "uk-garage"}): [
        "UK GarageはHouseの親戚。BPMが近ければ自然に繋がる",
        "UKGのシャッフルビートに注意 — ストレートビートとの混在は短めに",
        "8-16小節のEQスワップが安全",
    ],
    frozenset({"house", "melodic"}): [
        "Melodicへの移行はブレイクを使うとドラマチック",
        "フィルタースウィープで32小節かけるとエモーショナル",
        "メロディ同士がぶつからないよう、片方のMidをカットする",
    ],
    frozenset({"techno", "melodic"}): [
        "TechnoからMelodicへはエモーショナルな転換ポイント",
        "Techno側のビルドアップの頂点でMelodicのブレイクを入れる",
        "フィルタースウィープで32小節が美しい",
    ],
    frozenset({"hiphop", "rnb"}): [
        "Hip-HopとR&Bは兄弟ジャンル。フェードで自然に繋がる",
        "エコーアウトが定番テクニック",
        "ボーカルが重ならないタイミングで切り替える",
    ],
    frozenset({"hiphop", "house"}): [
        "ジャンルが大きく変わるのでブレイクを挟む",
        "Hip-Hop側をエコーアウト → 空白2-4小節 → House投入",
        "BPM差が大きいので段階的なテンポ調整が必要",
    ],
}

_DEFAULT_MIXING_TIPS: list[str] = [
    "異ジャンル間はブレイクやエコーアウトで一旦リセットしてから繋ぐ",
    "BPM差が大きい場合はフェードアウト→フェードインが安全",
    "ボーカル同士がぶつからないタイミングを選ぶ",
]


def genre_mixing_tips(genre_a: str, genre_b: str) -> list[str]:
    """ジャンル間のミキシングTipsを返す。

    Args:
        genre_a: トラックAのジャンル (genre_group)
        genre_b: トラックBのジャンル (genre_group)

    Returns:
        日本語のTipsリスト
    """
    if genre_a == genre_b:
        return [
            f"同じジャンル（{genre_a}）なのでEQスワップが最も自然",
            "キーが合っていればロングブレンドも◎",
            "キックが重ならないように低域の切り替えを丁寧に",
        ]

    key = frozenset({genre_a, genre_b})
    return _GENRE_MIXING_TIPS.get(key, _DEFAULT_MIXING_TIPS)
