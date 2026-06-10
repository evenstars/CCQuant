"""Market-regime filter (trend brake).

When the benchmark (SPY) trades above its long moving average we run at full
target exposure; when it closes below, we scale exposure down. This is the
long-only book's only mechanism for stepping aside in a momentum crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_exposure(
    benchmark_close: pd.Series,
    ma_window: int = 200,
    below_ma_exposure: float = 0.5,
) -> pd.Series:
    """Target gross exposure (0..1) per date from the benchmark vs its SMA.

    Above (or equal to) the moving average -> 1.0; below -> ``below_ma_exposure``.
    Before the moving average is defined (first ``ma_window``-1 days) we default
    to full exposure so early backtest dates aren't silently dropped. The MA at
    date t uses only data up to t (trailing rolling mean) -> no look-ahead.
    """
    ma = benchmark_close.rolling(ma_window).mean()
    above = benchmark_close >= ma
    exp = pd.Series(np.where(above, 1.0, below_ma_exposure), index=benchmark_close.index)
    exp[ma.isna()] = 1.0
    return exp
