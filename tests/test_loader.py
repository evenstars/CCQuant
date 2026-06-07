"""PriceLoader tests: reshape logic + abstraction contract (no network)."""

import pandas as pd

from ccquant.data.loader import PriceLoader, YFinanceLoader, _cache_key
from ccquant.data.panel import PricePanel


def _fake_yf_multi(tickers, idx):
    """Build a yfinance-style MultiIndex (ticker, field) frame."""
    cols = pd.MultiIndex.from_product([tickers, ["Open", "Close", "Volume"]])
    df = pd.DataFrame(1.0, index=idx, columns=cols)
    for t in tickers:
        df[(t, "Close")] = range(len(idx))
        df[(t, "Open")] = range(len(idx))
        df[(t, "Volume")] = 1_000_000
    return df


def test_reshape_multi_ticker():
    idx = pd.bdate_range("2020-01-01", periods=4)
    raw = _fake_yf_multi(["AAA", "BBB"], idx)
    close, open_, volume = YFinanceLoader._reshape(raw, ["AAA", "BBB"])
    assert list(close.columns) == ["AAA", "BBB"]
    assert close.shape == (4, 2)
    assert (volume["AAA"] == 1_000_000).all()


def test_reshape_single_ticker():
    idx = pd.bdate_range("2020-01-01", periods=4)
    raw = pd.DataFrame(
        {"Open": range(4), "Close": range(4), "Volume": [10] * 4}, index=idx
    )
    close, open_, volume = YFinanceLoader._reshape(raw, ["AAA"])
    assert list(close.columns) == ["AAA"]
    assert close.shape == (4, 1)


def test_reshape_empty():
    close, open_, volume = YFinanceLoader._reshape(pd.DataFrame(), ["AAA"])
    assert close.empty


def test_cache_key_is_order_stable():
    k1 = _cache_key("yf", ["BBB", "AAA"], "2005-01-01", "2026-01-01", "close")
    k2 = _cache_key("yf", ["AAA", "BBB"], "2005-01-01", "2026-01-01", "close")
    assert k1 == k2  # ticker order must not change the key


def test_loader_contract_with_fake(synthetic_panel):
    class FakeLoader(PriceLoader):
        def load(self, tickers, start, end) -> PricePanel:
            return synthetic_panel.slice(start, end)

    loader = FakeLoader()
    panel = loader.load(["AAA"], "2020-01-01", "2020-03-31")
    assert isinstance(panel, PricePanel)
    assert len(panel.dates) > 0
