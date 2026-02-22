"""共通プロットユーティリティ"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def fig_to_png(fig: plt.Figure) -> bytes:
    """matplotlib Figure を PNG bytes に変換して閉じる。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
