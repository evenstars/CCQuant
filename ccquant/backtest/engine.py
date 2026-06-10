"""Monthly cross-sectional momentum backtest engine.

Timing contract (the thing that makes or breaks a backtest):
  * Signals are computed on month-end closes (``r``) using only data up to ``r``.
  * Trades execute at the NEXT trading day's OPEN (``exec_day``). The engine never
    fills at the same close it used to decide — that would be look-ahead.
  * Between rebalances the share counts are held fixed (positions drift); daily
    equity is marked at the close.

Costs (commission + slippage) are deducted in cash at each trade. Taxes, when
enabled, are settled at each year boundary and deducted from cash, so the after-
tax run compounds on a genuinely smaller capital base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ccquant.backtest.costs import CostModel
from ccquant.backtest.tax import TaxLedger
from ccquant.data.panel import PricePanel, liquidity_eligible, sufficient_history_mask
from ccquant.strategy.factors import momentum_scores
from ccquant.strategy.portfolio import (
    select_names,
    select_names_with_hysteresis,
    target_weights,
)
from ccquant.strategy.regime import regime_exposure
from ccquant.utils.calendar import next_trading_day
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.backtest.engine")

_TRADING_DAYS_PER_MONTH = 21  # lookback window for sufficient-history / liquidity


@dataclass
class BacktestResult:
    equity: pd.Series
    turnover: pd.Series  # one-way turnover per rebalance date
    total_tax_paid: float
    final_value: float
    apply_tax: bool


def run_backtest(
    panel: PricePanel,
    benchmark_close: pd.Series,
    *,
    n_holdings: int = 15,
    lookback_months: int = 12,
    skip_recent_months: int = 1,
    ma_window: int = 200,
    below_ma_exposure: float = 0.5,
    max_weight: float = 0.08,
    min_adv_usd: float = 5_000_000,
    min_price: float = 5.0,
    cost_model: CostModel | None = None,
    apply_tax: bool = False,
    st_rate: float = 0.45,
    lt_rate: float = 0.25,
    long_term_days: int = 365,
    initial_capital: float = 100_000.0,
    hysteresis_exit: int | None = None,
    rebalance_band: float = 0.0,
    tax_aware_lots: bool = False,
    membership: pd.DataFrame | None = None,
    dividend_drag_annual: float = 0.0,
    start=None,
    end=None,
) -> BacktestResult:
    cost_model = cost_model or CostModel()
    if start is not None or end is not None:
        panel = panel.slice(start, end)
        benchmark_close = benchmark_close.loc[
            (pd.Timestamp(start) if start else benchmark_close.index[0]) :
            (pd.Timestamp(end) if end else benchmark_close.index[-1])
        ]

    close, open_ = panel.close, panel.open
    dates = panel.dates

    # -- precompute signals (all point-in-time) ----------------------------- #
    scores = momentum_scores(close, lookback_months, skip_recent_months)
    hist_ok = sufficient_history_mask(close, lookback_months * _TRADING_DAYS_PER_MONTH)
    liq_ok = liquidity_eligible(panel, min_adv_usd, min_price)
    eligible = hist_ok & liq_ok
    exposure = regime_exposure(benchmark_close, ma_window, below_ma_exposure)

    # map each signal (month-end) date to its execution day (next open)
    exec_to_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for r in scores.index:
        ex = next_trading_day(dates, r)
        if ex is not None:
            exec_to_signal[ex] = r
    if not exec_to_signal:
        raise ValueError("No executable rebalance dates in range")

    first_exec = min(exec_to_signal)
    sim_days = dates[dates >= first_exec]
    _log.info(
        "backtest: %d rebalances, %d trading days, apply_tax=%s",
        len(exec_to_signal), len(sim_days), apply_tax,
    )

    # -- state -------------------------------------------------------------- #
    cash = float(initial_capital)
    shares: dict[str, float] = {}
    ledger = TaxLedger(st_rate, lt_rate, long_term_days) if apply_tax else None
    equity = pd.Series(index=sim_days, dtype=float)
    turnover_rows: dict[pd.Timestamp, float] = {}
    total_tax = 0.0
    cur_year = sim_days[0].year

    for d in sim_days:
        # year-boundary tax settlement (deducts from cash before today's mark)
        if apply_tax and d.year != cur_year:
            tax = ledger.settle_year(cur_year)
            cash -= tax
            total_tax += tax
            cur_year = d.year

        if d in exec_to_signal:
            r = exec_to_signal[d]
            cash, traded_value, pre_equity = _rebalance(
                d, r, scores, eligible, exposure, close, open_,
                cash, shares, n_holdings, max_weight, cost_model, ledger,
                hysteresis_exit, rebalance_band, tax_aware_lots, membership,
            )
            # one-way turnover relative to pre-trade equity
            turnover_rows[d] = (traded_value / pre_equity) if pre_equity > 0 else 0.0

        equity[d] = equity_value(cash, shares, close, d)

    # settle the final (partial) year against the last equity point
    if apply_tax:
        tax = ledger.settle_year(cur_year)
        total_tax += tax
        equity.iloc[-1] -= tax

    equity = equity.dropna()
    # approximate a constant dividend-withholding drag (HK/NRA live scenario):
    # compounds a daily haircut over the curve. Assumes a flat effective yield;
    # set 0 for a Roth IRA (no withholding). See docs/design.md v0.2.
    if dividend_drag_annual > 0 and len(equity) > 0:
        daily = 1.0 - dividend_drag_annual / 252.0
        decay = pd.Series(daily ** np.arange(len(equity)), index=equity.index)
        equity = equity * decay

    return BacktestResult(
        equity=equity,
        turnover=pd.Series(turnover_rows).sort_index(),
        total_tax_paid=total_tax,
        final_value=float(equity.iloc[-1]),
        apply_tax=apply_tax,
    )


def equity_value(cash: float, shares: dict[str, float], close: pd.DataFrame, d) -> float:
    if not shares:
        return cash
    row = close.loc[d]
    val = 0.0
    for t, sh in shares.items():
        px = row.get(t, np.nan)
        if not np.isnan(px):
            val += sh * px
    return cash + val


def _rebalance(
    d, r, scores, eligible, exposure, close, open_,
    cash, shares, n_holdings, max_weight, cost_model, ledger,
    hysteresis_exit=None, rebalance_band=0.0, tax_aware_lots=False, membership=None,
):
    """Trade to target weights at day ``d`` open.

    Returns (new_cash, traded_value, pre_trade_equity).

    Tax-reduction levers:
      * ``hysteresis_exit``: keep held names ranked within this band (cuts churn).
      * ``rebalance_band``: don't trim/top-up a *kept* name whose value is within
        this fraction of its target (lets winners run -> defers gains).
      * ``tax_aware_lots``: sell loss/long-term lots before short-term ones.
    """
    open_row = open_.loc[d]

    # pre-trade equity marked at today's open
    equity = cash
    for t, sh in shares.items():
        px = open_row.get(t, np.nan)
        equity += sh * px if not np.isnan(px) else sh * close.loc[d].get(t, 0.0)

    elig_row = eligible.loc[r] if r in eligible.index else pd.Series(True, index=scores.columns)
    # point-in-time membership: only consider names in the index on this date
    if membership is not None and r in membership.index:
        elig_row = elig_row & membership.loc[r].reindex(elig_row.index, fill_value=False)
    if hysteresis_exit is not None:
        sel = select_names_with_hysteresis(
            scores.loc[r], elig_row, n_holdings, hysteresis_exit, list(shares.keys())
        )
    else:
        sel = select_names(scores.loc[r], elig_row, n_holdings)
    # only trade names with a usable open price today
    sel = [t for t in sel if not np.isnan(open_row.get(t, np.nan))]
    exp = float(exposure.get(r, 1.0))
    weights = target_weights(sel, exp, n_holdings, max_weight)

    desired_shares: dict[str, float] = {}
    for t in set(list(shares.keys()) + list(weights.keys())):
        px = open_row.get(t, np.nan)
        if np.isnan(px) or px <= 0:
            desired_shares[t] = shares.get(t, 0.0)  # can't trade -> hold
            continue
        target = (equity * weights.get(t, 0.0)) / px
        cur = shares.get(t, 0.0)
        # drift tolerance: skip trimming/topping a kept name within the band
        if (
            rebalance_band > 0.0
            and cur > 0
            and weights.get(t, 0.0) > 0
            and abs(target - cur) * px <= rebalance_band * (equity * weights[t])
        ):
            target = cur
        desired_shares[t] = target

    traded_value = 0.0
    for t, target in desired_shares.items():
        px = open_row.get(t, np.nan)
        if np.isnan(px) or px <= 0:
            continue
        delta = target - shares.get(t, 0.0)
        if abs(delta) * px < 1e-6:
            continue
        cost = cost_model.total(delta, px)
        traded_value += abs(delta) * px
        if delta > 0:
            cash -= delta * px + cost
            if ledger is not None:
                ledger.buy(t, delta, px, d)
        else:
            cash += (-delta) * px - cost
            if ledger is not None:
                ledger.sell(t, -delta, px, d, tax_aware=tax_aware_lots)
        shares[t] = target

    # drop ~zero positions
    for t in [t for t, sh in shares.items() if abs(sh) < 1e-9]:
        del shares[t]

    return cash, traded_value, equity
