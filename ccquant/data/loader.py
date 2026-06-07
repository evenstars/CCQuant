"""Price loaders.

``PriceLoader`` is the abstraction the rest of the system depends on. Swapping
yfinance for Polygon / Norgate / a point-in-time source later means writing a new
subclass — nothing upstream changes.

The yfinance import is deferred to call time so the package (and the test suite /
CI) imports cleanly without the optional ``data`` extra installed. To actually
download, install it: ``uv sync --extra data``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import pandas as pd

from ccquant.data.cache import DataCache
from ccquant.data.panel import PricePanel
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.data.loader")


class PriceLoader(ABC):
    """Load adjusted (total-return) OHLCV for a set of tickers into a PricePanel."""

    @abstractmethod
    def load(self, tickers: list[str], start, end) -> PricePanel:
        """Return a PricePanel covering [start, end] for ``tickers``."""
        raise NotImplementedError


def _cache_key(prefix: str, tickers: list[str], start, end, field: str) -> str:
    base = "|".join(sorted(tickers)) + f"|{start}|{end}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}/{digest}/{field}"


class YFinanceLoader(PriceLoader):
    """Total-return OHLCV from Yahoo Finance via the ``yfinance`` package.

    ``auto_adjust=True`` makes Open/High/Low/Close split- and dividend-adjusted
    (total return). Results are cached to disk so repeated runs don't re-download.
    """

    def __init__(self, cache: DataCache | None = None, use_cache: bool = True) -> None:
        self.cache = cache or DataCache()
        self.use_cache = use_cache

    def load(self, tickers: list[str], start, end) -> PricePanel:
        tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
        if self.use_cache:
            cached = self._load_cached(tickers, start, end)
            if cached is not None:
                _log.info("loaded %d tickers from cache", len(tickers))
                return cached

        raw = self._download(tickers, start, end)
        close, open_, volume = self._reshape(raw, tickers)
        panel = PricePanel(close=close, open=open_, volume=volume)

        if self.use_cache:
            for field, df in (("close", close), ("open", open_), ("volume", volume)):
                self.cache.put_df(_cache_key("yf", tickers, start, end, field), df)
        return panel

    # -- internals ---------------------------------------------------------- #
    def _load_cached(self, tickers, start, end) -> PricePanel | None:
        frames = {}
        for field in ("close", "open", "volume"):
            df = self.cache.get_df(_cache_key("yf", tickers, start, end, field))
            if df is None:
                return None
            frames[field] = df
        return PricePanel(close=frames["close"], open=frames["open"], volume=frames["volume"])

    @staticmethod
    def _download(tickers, start, end):
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "yfinance is required for YFinanceLoader. "
                "Install it with: uv sync --extra data"
            ) from exc
        _log.info("downloading %d tickers %s..%s from yfinance", len(tickers), start, end)
        return yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

    @staticmethod
    def _reshape(raw: pd.DataFrame, tickers: list[str]):
        """Turn yfinance output into three wide (date x ticker) frames."""
        if raw is None or raw.empty:
            empty = pd.DataFrame()
            return empty, empty.copy(), empty.copy()

        # Single ticker -> flat columns; multi -> MultiIndex (ticker, field).
        if not isinstance(raw.columns, pd.MultiIndex):
            t = tickers[0]
            close = raw[["Close"]].rename(columns={"Close": t})
            open_ = raw[["Open"]].rename(columns={"Open": t})
            volume = raw[["Volume"]].rename(columns={"Volume": t})
            return close, open_, volume

        def field(name: str) -> pd.DataFrame:
            cols = [(t, name) for t in tickers if (t, name) in raw.columns]
            sub = raw.loc[:, cols].copy()
            sub.columns = [t for t, _ in cols]
            return sub

        return field("Close"), field("Open"), field("Volume")
