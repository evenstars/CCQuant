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
