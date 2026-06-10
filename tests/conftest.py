"""Shared test fixtures: synthetic market data, no network access."""

import numpy as np
import pandas as pd
import pytest

from ccquant.data.panel import PricePanel


@pytest.fixture
def trading_index() -> pd.DatetimeIndex:
    """~3 months of business days."""
    return pd.bdate_range("2020-01-01", "2020-03-31")


@pytest.fixture
def momentum_index() -> pd.DatetimeIndex:
    """~3 years of business days — long enough for 12-1 momentum."""
    return pd.bdate_range("2018-01-01", "2021-06-30")


@pytest.fixture
def momentum_panel(momentum_index) -> PricePanel:
    """5 tickers with distinct, persistent trends so ranking is well-defined."""
    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    n = len(momentum_index)
    close = {}
    for i, t in enumerate(tickers):
        drift = 1.0 + i * 0.0004  # EEE strongest, AAA weakest
        path = drift ** np.arange(n) * (1 + rng.normal(0, 0.001, n)).cumprod()
        close[t] = 100.0 * path
    close = pd.DataFrame(close, index=momentum_index)
    open_ = close.shift(1).bfill()
    volume = pd.DataFrame(1e7, index=momentum_index, columns=tickers)
    return PricePanel(close=close, open=open_, volume=volume)


@pytest.fixture
def momentum_benchmark(momentum_index) -> pd.Series:
    """Uptrending benchmark (stays above its MA most of the time)."""
    return pd.Series(100.0 * 1.0003 ** np.arange(len(momentum_index)), index=momentum_index)


@pytest.fixture
def synthetic_panel(trading_index) -> PricePanel:
    """Deterministic 3-ticker panel with a known, gently trending shape."""
    rng = np.random.default_rng(42)
    tickers = ["AAA", "BBB", "CCC"]
    n = len(trading_index)
    data = {}
    vol = {}
    for i, t in enumerate(tickers):
        drift = 1.0 + (i + 1) * 0.001  # CCC trends fastest -> highest momentum
        path = drift ** np.arange(n) * (1 + rng.normal(0, 0.002, n)).cumprod()
        data[t] = 100.0 * path
        vol[t] = rng.integers(1_000_000, 5_000_000, n).astype(float)
    close = pd.DataFrame(data, index=trading_index)
    open_ = close.shift(1).bfill()
    volume = pd.DataFrame(vol, index=trading_index)
    return PricePanel(close=close, open=open_, volume=volume)
