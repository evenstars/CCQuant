"""Trading-calendar helper tests."""

import pandas as pd

from ccquant.utils.calendar import (
    month_end_dates,
    next_trading_day,
    rebalance_dates,
)


def test_month_end_dates_picks_last_trading_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    me = month_end_dates(idx)
    # Jan/Feb/Mar 2020 last business days
    assert list(me) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-31"),
    ]


def test_month_end_handles_gaps_and_dupes():
    idx = pd.DatetimeIndex(
        ["2021-01-04", "2021-01-04", "2021-01-29", "2021-02-26"]
    )
    me = month_end_dates(idx)
    assert list(me) == [pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-26")]


def test_rebalance_monthly_equals_month_end():
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    assert list(rebalance_dates(idx, "monthly")) == list(month_end_dates(idx))


def test_rebalance_rejects_unsupported_freq():
    import pytest

    with pytest.raises(ValueError):
        rebalance_dates(pd.bdate_range("2020-01-01", "2020-01-10"), "weekly")


def test_next_trading_day():
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    assert next_trading_day(idx, pd.Timestamp("2020-01-31")) is None
    assert next_trading_day(idx, pd.Timestamp("2020-01-06")) == pd.Timestamp("2020-01-07")


def test_empty_input():
    assert len(month_end_dates([])) == 0
