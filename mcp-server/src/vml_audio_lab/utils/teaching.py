"""Teaching レイヤー — 分析結果を平易な日本語に変換する.

DJ初心者にも分かる言葉で「なぜこの繋ぎが気持ちいいか」を説明する。
"""

from __future__ import annotations


def explain_key_transition(
    key_a: str,
    camelot_a: str,
    key_b: str,
    camelot_b: str,
    score: float,
) -> str:
    """キー遷移を平易な日本語で説明する。

    Args:
        key_a: トラックAのキーラベル (例: "Fm")
        camelot_a: トラックAのCamelotコード (例: "4A")
        key_b: トラックBのキーラベル
        camelot_b: トラックBのCamelotコード
        score: camelot.compatibility_score() の値 (0.0-1.0)

    Returns:
        日本語の解説文
    """
    if score >= 1.0:
        return (
            f"同じキー {key_a}({camelot_a}) 同士なので、"
            "どこで繋いでも違和感なくスムーズに移行できます。"
        )
    if score >= 0.9:
        return (
            f"{key_a}({camelot_a}) → {key_b}({camelot_b}) は"
            "相対キー（メジャー⇔マイナー）の関係です。"
            "明暗が入れ替わるのでドラマチックな展開になります。"
        )
    if score >= 0.7:
        return (
            f"{key_a}({camelot_a}) → {key_b}({camelot_b}) は"
            "Camelot Wheelで隣り合うキーです。"
            "自然にキーが移動するので、EQスワップで繋ぐと綺麗に流れます。"
        )
    # 非互換
    return (
        f"{key_a}({camelot_a}) → {key_b}({camelot_b}) は"
        "キーが離れています。"
        "ブレイクやフィルタースウィープを挟んで一旦リセットしてから繋ぐと、"
        "キーの不一致が目立ちにくくなります。"
    )


def explain_bpm_transition(bpm_a: float, bpm_b: float) -> str:
    """BPM変化を平易な日本語で説明する。

    Args:
        bpm_a: トラックAのBPM
        bpm_b: トラックBのBPM

    Returns:
        日本語の解説文
    """
    diff = bpm_b - bpm_a
    abs_diff = abs(diff)

    if abs_diff <= 1.0:
        return f"BPMはほぼ同じ({bpm_a:.0f}→{bpm_b:.0f})。ビートマッチが簡単です。"

    if abs_diff <= 3.0:
        direction = "上がる" if diff > 0 else "下がる"
        return (
            f"BPMが{bpm_a:.0f}→{bpm_b:.0f}に少し{direction}ので、"
            "テンポ同期すれば自然に繋がります。"
        )

    if abs_diff <= 8.0:
        if diff > 0:
            return (
                f"BPMが{bpm_a:.0f}→{bpm_b:.0f}に上がるので、"
                "じわじわエネルギーが上がる流れになります。"
                "マスターテンポで徐々に合わせましょう。"
            )
        return (
            f"BPMが{bpm_a:.0f}→{bpm_b:.0f}に下がります。"
            "ブレイクで速度を落とすと自然です。"
        )

    # 大きなBPM差
    return (
        f"BPM差が{abs_diff:.0f}({bpm_a:.0f}→{bpm_b:.0f})あります。"
        "エコーアウトやフィルターで一旦空白を作ってから"
        "次の曲を入れると、テンポの変化が目立ちません。"
    )


def explain_energy_flow(
    energy_values: list[float],
    track_names: list[str],
) -> str:
    """セットリスト全体のエネルギーフローを解説する。

    Args:
        energy_values: 各トラックのエネルギーレベル (0.0-1.0)
        track_names: 各トラックの名前

    Returns:
        日本語の解説文
    """
    if not energy_values:
        return "トラックがありません。"

    n = len(energy_values)
    if n == 1:
        return f"1曲のみのセット。エネルギーは{_energy_label(energy_values[0])}です。"

    # 全体のトレンドを判定
    third = max(1, n // 3)
    first_third = energy_values[:third]
    last_third = energy_values[n - third :] if n > third else energy_values[-1:]
    avg_start = sum(first_third) / len(first_third)
    avg_end = sum(last_third) / max(1, len(last_third))

    # ピーク位置
    peak_idx = energy_values.index(max(energy_values))
    peak_ratio = peak_idx / (n - 1) if n > 1 else 0.5
    peak_name = track_names[peak_idx] if peak_idx < len(track_names) else f"#{peak_idx + 1}"

    parts: list[str] = []

    if avg_start < 0.4 and avg_end > 0.6:
        parts.append("序盤はゆったりスタートし、後半に向けてエネルギーが上昇する理想的な構成です")
    elif avg_start > 0.6 and avg_end < 0.4:
        parts.append("序盤からハイエネルギーで始まり、後半でクールダウンする構成です")
    elif avg_start > 0.6 and avg_end > 0.6:
        parts.append("序盤から最後までハイエネルギーを維持するアグレッシブな構成です")
    else:
        parts.append("穏やかなエネルギーで統一された落ち着いた構成です")

    if 0.4 <= peak_ratio <= 0.8:
        parts.append(f"ピークは{peak_name}（{peak_idx + 1}曲目）で、セット後半に訪れます")
    elif peak_ratio < 0.4:
        parts.append(
            f"ピークが{peak_name}（{peak_idx + 1}曲目）と早めです。"
            "後半に向けてもう一段上げる曲があるとベター"
        )

    # エネルギーの急降下を検出
    drops: list[str] = []
    for i in range(1, n):
        drop = energy_values[i - 1] - energy_values[i]
        if drop > 0.25:
            name_before = track_names[i - 1] if i - 1 < len(track_names) else f"#{i}"
            name_after = track_names[i] if i < len(track_names) else f"#{i + 1}"
            drops.append(f"{name_before}→{name_after}")

    if drops:
        parts.append(f"エネルギーの急降下が{'、'.join(drops)}で起きています。間にブリッジ曲を挟むと滑らかになります")

    return "。".join(parts) + "。"


def explain_effects(effects_result: dict) -> str:
    """検出されたエフェクトを平易に説明する。

    Args:
        effects_result: detect_effects() の返り値

    Returns:
        日本語の解説文
    """
    effects = effects_result.get("effects", [])
    if not effects:
        return "目立ったエフェクトは検出されませんでした。ドライなサウンドです。"

    _EFFECT_DESCRIPTIONS: dict[str, str] = {
        "reverb": "リバーブ（残響）が効いており、空間の広がりを感じさせるサウンドです",
        "delay": "ディレイ（やまびこ効果）がリズムに乗って反復し、奥行きを作っています",
        "filter_sweep": "フィルタースウィープで周波数が徐々に変化し、盛り上がり/展開を演出しています",
        "sidechain": "サイドチェインコンプレッションでキックに合わせて音量がうねる「ポンピング」効果が出ています",
    }

    parts: list[str] = []
    for effect in effects:
        effect_type = effect.get("type", "")
        section = effect.get("section", "")
        confidence = effect.get("confidence", 0.0)

        desc = _EFFECT_DESCRIPTIONS.get(effect_type, f"{effect_type}が検出されました")
        if section:
            desc = f"{section}セクションで{desc}"
        if confidence < 0.5:
            desc += "（弱め）"

        parts.append(desc)

    return "。".join(parts) + "。"


def explain_genre_compatibility(genre_a: str, genre_b: str) -> str:
    """ジャンル間の相性を説明する。

    Args:
        genre_a: トラックAのジャンル
        genre_b: トラックBのジャンル

    Returns:
        日本語の解説文
    """
    if genre_a == genre_b:
        return f"同じジャンル（{genre_a}）なので、音の質感が揃って自然に繋がります。"

    _GENRE_TIPS: dict[frozenset[str], str] = {
        frozenset({"house", "techno"}): (
            "HouseからTechnoへの移行はよくあるパターンです。"
            "低域を先に入れ替えて、ハイハットで繋ぐとスムーズ。"
        ),
        frozenset({"house", "uk-garage"}): (
            "HouseとUK Garageは親戚のようなジャンルです。"
            "BPMが近ければEQスワップで自然に繋がります。"
        ),
        frozenset({"house", "melodic"}): (
            "HouseからMelodicへの移行は雰囲気を変える良いタイミング。"
            "ブレイクで切り替えるとドラマチックです。"
        ),
        frozenset({"techno", "melodic"}): (
            "TechnoからMelodicへはエモーショナルな転換になります。"
            "フィルタースウィープで32小節かけて移行すると美しい。"
        ),
        frozenset({"hiphop", "rnb"}): (
            "Hip-HopとR&Bは兄弟ジャンル。"
            "フェードやエコーアウトで繋ぐのが定番です。"
        ),
    }

    key = frozenset({genre_a, genre_b})
    if key in _GENRE_TIPS:
        return _GENRE_TIPS[key]

    return (
        f"{genre_a}から{genre_b}へのジャンルチェンジです。"
        "一旦ブレイクやエコーアウトで空白を作ってから次の曲を入れると、"
        "ジャンルの切り替えが自然に感じられます。"
    )


def _energy_label(energy: float) -> str:
    """エネルギー値を日本語ラベルに変換する。"""
    if energy >= 0.7:
        return "ハイエネルギー"
    if energy >= 0.4:
        return "ミドルエネルギー"
    return "ローエネルギー"
