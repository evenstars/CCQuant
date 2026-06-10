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
from pathlib import Path

import pandas as pd

from ccquant.data.cache import DataCache
from ccquant.utils import paths
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.data.universe")

_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Free historical S&P 500 membership (date -> full member snapshot on that date).
# Local file is tried first; the URL is a fallback. The repo occasionally renames
# the file with its last-updated date, so if the URL 404s, download the CSV once
# to config/sp500_historical.csv (two columns: date, tickers).
_PIT_LOCAL = paths.CONFIG_DIR / "sp500_historical.csv"
_PIT_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)


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
        # Wikipedia rejects the default urllib User-Agent (HTTP 403), so send a
        # browser-like UA. read_html forwards storage_options to the HTTP layer.
        ua = "Mozilla/5.0 (compatible; CCQuant/0.1; +https://github.com/evenstars/CCQuant)"
        try:
            tables = pd.read_html(_WIKI_SP500, storage_options={"User-Agent": ua})
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


def _normalize(ticker: str) -> str:
    """Yahoo uses '-' where the exchange symbol uses '.', e.g. BRK.B -> BRK-B."""
    return str(ticker).strip().replace(".", "-")


def build_membership_mask(
    provider: "UniverseProvider",
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """Boolean (date x ticker): True where the ticker was a member on that date.

    Computed at each date in ``dates`` (typically rebalance dates) so the engine
    can restrict selection to genuine point-in-time constituents.
    """
    cols = list(tickers)
    rows = {}
    for d in dates:
        members = set(provider.get_constituents(d))
        rows[d] = pd.Series([t in members for t in cols], index=cols)
    return pd.DataFrame(rows).T.reindex(columns=cols).fillna(False)


class HistoricalSP500Provider(UniverseProvider):
    """Point-in-time S&P 500 membership from a free historical-components CSV.

    The CSV has two columns: ``date`` and ``tickers`` (a comma-separated snapshot
    of every member on that date; rows appear when membership changes). This fixes
    the *selection* side of survivorship bias — at each rebalance we only consider
    names that were actually in the index then.

    Remaining limitation (M1.1, accepted): we still rely on yfinance for prices,
    which lacks most delisted tickers, so a name that left the index after a crash
    will be missing its pre-delisting prices. The *price* side of survivorship bias
    is only fully fixed by a paid source (Polygon/Norgate) -> TODO(M1.2).
    """

    def __init__(self, source: str | Path | None = None, cache: DataCache | None = None) -> None:
        self.source = source
        self.cache = cache or DataCache()
        self._snapshots: list[tuple[pd.Timestamp, list[str]]] | None = None

    def get_constituents(self, as_of: date | pd.Timestamp) -> list[str]:
        snaps = self._load()
        as_of = pd.Timestamp(as_of)
        members: list[str] = []
        for dt, tickers in snaps:  # snapshots are sorted ascending by date
            if dt <= as_of:
                members = tickers
            else:
                break
        return list(members)

    def all_members(self, start, end) -> list[str]:
        """Union of every member that appears in snapshots within [start, end].

        Includes the snapshot in effect at ``start`` so names present at the very
        beginning are not dropped. Use this to decide which prices to download.
        """
        snaps = self._load()
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        union: set[str] = set()
        # seed with the snapshot active at start
        union.update(self.get_constituents(start))
        for dt, tickers in snaps:
            if start <= dt <= end:
                union.update(tickers)
        return sorted(union)

    # -- internals ---------------------------------------------------------- #
    def _load(self) -> list[tuple[pd.Timestamp, list[str]]]:
        if self._snapshots is not None:
            return self._snapshots
        cached = self.cache.get_obj("universe/sp500_historical")
        if cached is not None:
            self._snapshots = cached
            return cached
        df = self._read_csv()
        date_col = df.columns[0]
        tick_col = df.columns[1]
        df = df[[date_col, tick_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        snaps = [
            (
                pd.Timestamp(row[date_col]),
                [_normalize(t) for t in str(row[tick_col]).split(",") if t.strip()],
            )
            for _, row in df.iterrows()
        ]
        self.cache.put_obj("universe/sp500_historical", snaps)
        self._snapshots = snaps
        _log.info("loaded %d historical S&P 500 membership snapshots", len(snaps))
        return snaps

    def _read_csv(self) -> pd.DataFrame:
        candidates = [self.source] if self.source else [_PIT_LOCAL, _PIT_URL]
        last_err: Exception | None = None
        for c in candidates:
            if c is None:
                continue
            try:
                if isinstance(c, Path) and not c.exists():
                    continue
                return pd.read_csv(c)
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                last_err = exc
                _log.warning("could not read membership CSV from %s: %s", c, exc)
        raise FileNotFoundError(
            "Could not load historical S&P 500 membership. Download the CSV "
            "(columns: date,tickers) to config/sp500_historical.csv. "
            f"Last error: {last_err}"
        )
