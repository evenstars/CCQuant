"""Trading-calendar helpers derived from actual data, not an external calendar.

We avoid a market-calendar dependency by deriving month-end / rebalance dates
from the dates that actually appear in the price data. The last available trading
day of each month is treated as that month's signal date.
"""

from __future__ import annotations

import pandas as pd


def _as_index(dates) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return idx.unique().sort_values()


def month_end_dates(dates) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month present in ``dates``."""
    idx = _as_index(dates)
    if len(idx) == 0:
        return idx
    periods = idx.to_period("M")
    last = pd.Series(idx, index=periods).groupby(level=0).last()
    return pd.DatetimeIndex(last.to_numpy()).sort_values()


def rebalance_dates(dates, freq: str = "monthly") -> pd.DatetimeIndex:
    """Signal/rebalance dates for the given frequency.

    Only ``monthly`` is supported in v1 (matches the strategy spec); the argument
    exists so callers can pass ``cfg.strategy.rebalance`` directly.
    """
    if freq != "monthly":
        raise ValueError(f"Unsupported rebalance frequency: {freq!r}")
    return month_end_dates(dates)


def next_trading_day(dates, after: pd.Timestamp) -> pd.Timestamp | None:
    """First available trading day strictly after ``after`` (the execution day).

    Returns None if there is no later day in ``dates`` (e.g. the most recent
    signal date has no next-open yet).
    """
    idx = _as_index(dates)
    after = pd.Timestamp(after)
    later = idx[idx > after]
    return later[0] if len(later) else None
