"""Price panel container and data-hygiene helpers.

A ``PricePanel`` holds aligned wide DataFrames (index = trading dates, columns =
tickers) for adjusted close / open / volume. "Adjusted" means total-return:
split- and dividend-adjusted, so momentum is measured on total return as the
design requires.

All prices flowing through the backtest and live signal engine use this single
container, so signal code never has to care where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PricePanel:
    """Aligned adjusted OHLCV-ish panel. ``close``/``open`` are total-return adjusted."""

    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        # Align all three frames to a common (sorted) index and column set so
        # downstream vectorised ops never silently misalign.
        cols = self.close.columns.union(self.open.columns).union(self.volume.columns)
        idx = self.close.index.union(self.open.index).union(self.volume.index).sort_values()
        self.close = self.close.reindex(index=idx, columns=cols).sort_index()
        self.open = self.open.reindex(index=idx, columns=cols).sort_index()
        self.volume = self.volume.reindex(index=idx, columns=cols).sort_index()

    # -- accessors ---------------------------------------------------------- #
    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    # -- derived series ----------------------------------------------------- #
    def dollar_volume(self) -> pd.DataFrame:
        """Daily traded dollar value = adjusted close * volume."""
        return self.close * self.volume

    def daily_returns(self) -> pd.DataFrame:
        # fill_method=None: never forward-fill across gaps (avoids look-ahead and
        # pandas' deprecated default behaviour).
        return self.close.pct_change(fill_method=None)

    def slice(self, start=None, end=None) -> PricePanel:
        s = pd.Timestamp(start) if start is not None else None
        e = pd.Timestamp(end) if end is not None else None
        return PricePanel(
            close=self.close.loc[s:e],
            open=self.open.loc[s:e],
            volume=self.volume.loc[s:e],
        )


def sufficient_history_mask(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Boolean (date x ticker): True where the trailing ``window`` rows are all present.

    Guards momentum computation against tickers that aren't listed yet or have
    data gaps — we only score names with a complete lookback window.
    """
    return close.notna().rolling(window).sum() == window


def liquidity_eligible(
    panel: PricePanel,
    min_adv_usd: float,
    min_price: float,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Boolean (date x ticker): True where the name passes the liquidity filter.

    Eligible if the trailing ``adv_window``-day average dollar volume is at least
    ``min_adv_usd`` and the adjusted close is at least ``min_price``.
    """
    adv = panel.dollar_volume().rolling(adv_window).mean()
    return (adv >= min_adv_usd) & (panel.close >= min_price)
