"""PricePanel and data-hygiene helper tests."""

import numpy as np
import pandas as pd

from ccquant.data.panel import (
    PricePanel,
    liquidity_eligible,
    sufficient_history_mask,
)


def test_panel_aligns_frames():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"AAA": range(5), "BBB": range(5)}, index=idx)
    # open missing a column and a row on purpose
    open_ = pd.DataFrame({"AAA": range(5)}, index=idx)
    volume = pd.DataFrame({"AAA": range(5), "BBB": range(5)}, index=idx)
    panel = PricePanel(close=close, open=open_, volume=volume)
    # all frames share the same columns/index after alignment
    assert panel.tickers == ["AAA", "BBB"]
    assert list(panel.open.columns) == ["AAA", "BBB"]
    assert panel.open["BBB"].isna().all()  # filled with NaN, not dropped


def test_dollar_volume_and_returns(synthetic_panel):
    dv = synthetic_panel.dollar_volume()
    assert dv.shape == synthetic_panel.close.shape
    assert (dv.dropna() > 0).all().all()
    rets = synthetic_panel.daily_returns()
    assert np.isnan(rets.iloc[0]).all()  # first row undefined


def test_sufficient_history_mask():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame(
        {"AAA": [1, 2, 3, 4, 5, 6], "BBB": [np.nan, np.nan, 3, 4, 5, 6]},
        index=idx,
        dtype=float,
    )
    mask = sufficient_history_mask(close, window=3)
    # AAA has full history from row index 2 onward
    assert mask["AAA"].iloc[2]
    # BBB only has 3 valid points by row index 4
    assert not mask["BBB"].iloc[3]
    assert mask["BBB"].iloc[4]


def test_liquidity_eligible():
    idx = pd.bdate_range("2020-01-01", periods=25)
    close = pd.DataFrame({"RICH": 100.0, "PENNY": 2.0}, index=idx)
    volume = pd.DataFrame({"RICH": 1_000_000.0, "PENNY": 1_000_000.0}, index=idx)
    panel = PricePanel(close=close, open=close, volume=volume)
    elig = liquidity_eligible(panel, min_adv_usd=5_000_000, min_price=5, adv_window=20)
    last = elig.iloc[-1]
    assert last["RICH"]  # 100 * 1M = 100M ADV, price 100 -> ok
    assert not last["PENNY"]  # 2 * 1M = 2M ADV < 5M and price < 5


def test_slice(synthetic_panel):
    sub = synthetic_panel.slice("2020-02-01", "2020-02-29")
    assert sub.dates.min() >= pd.Timestamp("2020-02-01")
    assert sub.dates.max() <= pd.Timestamp("2020-02-29")
