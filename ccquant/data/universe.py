"""Investable universe providers.

``UniverseProvider.get_constituents(as_of)`` returns the tickers that belong to
the universe on a given date. The interface is point-in-time by design.

MVP limitation (accepted, see docs/design.md open question #1):
    ``SP500Provider`` only knows the *current* S&P 500 membership, so it returns
    the same list regardless of ``as_of``. That bakes in survivorship bias —
    delisted names and past constituents are missing. Fine for wiring up the
    pipeline; must be replaced before trusting backtest results.

    TODO(M1.1): add a point-in-time provider with historical constituents +
    delisted tickers (e.g. Norgate, or a maintained membership CSV).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from ccquant.data.cache import DataCache
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.data.universe")

_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class UniverseProvider(ABC):
    """Returns the set of tickers in the universe as of a given date."""

    @abstractmethod
    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:
        raise NotImplementedError


class StaticUniverseProvider(UniverseProvider):
    """A fixed ticker list, independent of date. Used in tests and as a fallback."""

    def __init__(self, tickers: list[str]) -> None:
        self._tickers = list(dict.fromkeys(tickers))

    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:  # noqa: ARG002
        return list(self._tickers)


class SP500Provider(UniverseProvider):
    """Current S&P 500 membership scraped from Wikipedia (survivorship-biased).

    The list is cached after the first fetch. ``get_constituents`` ignores
    ``as_of`` — see the module docstring for the MVP limitation.
    """

    def __init__(self, cache: DataCache | None = None, use_cache: bool = True) -> None:
        self.cache = cache or DataCache()
        self.use_cache = use_cache
        self._warned = False

    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:  # noqa: ARG002
        if not self._warned:
            _log.warning(
                "SP500Provider returns CURRENT members only -> survivorship bias. "
                "Replace with a point-in-time source before trusting results."
            )
            self._warned = True
        return self._members()

    def _members(self) -> list[str]:
        key = "universe/sp500_current"
        if self.use_cache:
            cached = self.cache.get_obj(key)
            if cached:
                return list(cached)
        tickers = self._fetch()
        if self.use_cache:
            self.cache.put_obj(key, tickers)
        return tickers

    @staticmethod
    def _fetch() -> list[str]:
        try:
            tables = pd.read_html(_WIKI_SP500)
        except ImportError as exc:  # pragma: no cover - needs lxml/html5lib
            raise ImportError(
                "Reading the Wikipedia table needs lxml. Install with: uv sync --extra data"
            ) from exc
        df = tables[0]
        # Yahoo uses '-' where the exchange symbol uses '.', e.g. BRK.B -> BRK-B.
        tickers = (
            df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        )
        _log.info("fetched %d S&P 500 constituents from Wikipedia", len(tickers))
        return tickers
