"""Momentum factor computation.

Everything here is computed on *month-end* prices so the signal is naturally
point-in-time: the score at month-end ``m`` only ever references prices at or
before ``m``. There is deliberately no use of the most recent month's price in
the score (the "skip"), which both follows the classic 12-1 construction and
removes the short-term-reversal contamination.
"""

from __future__ import annotations

import pandas as pd

from ccquant.utils.calendar import month_end_dates


def month_end_prices(close: pd.DataFrame) -> pd.DataFrame:
    """Sub-sample a daily (date x ticker) close panel to month-end rows."""
    me = month_end_dates(close.index)
    return close.reindex(me)


def momentum_scores(
    close: pd.DataFrame,
    lookback_months: int = 12,
    skip_recent_months: int = 1,
) -> pd.DataFrame:
    """12-1 style momentum score at each month-end.

    score(m) = price(m - skip) / price(m - lookback) - 1

    With defaults that is the return from 12 months ago to 1 month ago. Returns a
    (month-end x ticker) DataFrame; NaN where a name lacks enough history.

    Note the shifts are strictly backward, so ``score`` on row ``m`` cannot see
    any price after ``m`` -> no look-ahead by construction.
    """
    if skip_recent_months >= lookback_months:
        raise ValueError("skip_recent_months must be < lookback_months")
    mep = month_end_prices(close)
    return mep.shift(skip_recent_months) / mep.shift(lookback_months) - 1.0
