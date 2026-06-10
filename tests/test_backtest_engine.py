"""Backtest engine guard tests: execution timing, look-ahead, tax plumbing."""

import numpy as np
import pandas as pd

from ccquant.backtest.costs import CostModel
from ccquant.backtest.engine import run_backtest
from ccquant.data.panel import PricePanel


def _single_ticker_panel(open_px=99.0, close_px=100.0):
    idx = pd.bdate_range("2020-01-01", "2020-07-31")
    close = pd.DataFrame({"AAA": close_px}, index=idx)
    open_ = pd.DataFrame({"AAA": open_px}, index=idx)
    volume = pd.DataFrame({"AAA": 1e7}, index=idx)
    return PricePanel(close=close, open=open_, volume=volume), idx


def test_execution_at_next_open_not_signal_close():
    """If we fill at the t+1 OPEN (99) not the close (100), final equity must
    reflect buying cheaper than the mark. close-execution would give exactly
    initial; open-execution gives initial * 100/99."""
    panel, idx = _single_ticker_panel(open_px=99.0, close_px=100.0)
    bench = pd.Series(100.0, index=idx)
    res = run_backtest(
        panel, bench,
        n_holdings=1, lookback_months=2, skip_recent_months=1,
        ma_window=20, max_weight=1.0, min_adv_usd=0, min_price=0,
        cost_model=CostModel(0, 0, 0), apply_tax=False, initial_capital=100_000.0,
    )
    expected = 100_000.0 * 100.0 / 99.0
    assert abs(res.final_value - expected) < 1.0
    assert abs(res.final_value - 100_000.0) > 100.0  # distinct from close-fill


def test_no_lookahead_in_equity(momentum_panel, momentum_benchmark):
    """Poisoning prices after a cutoff must not change equity before it."""
    common = dict(
        n_holdings=3, lookback_months=12, skip_recent_months=1,
        ma_window=100, max_weight=0.5, apply_tax=False,
        cost_model=CostModel(0, 0, 0),
    )
    res1 = run_backtest(momentum_panel, momentum_benchmark, **common)

    cutoff = res1.equity.index[len(res1.equity) // 2]
    cl = momentum_panel.close.copy()
    op = momentum_panel.open.copy()
    fut = cl.index > cutoff
    cl.loc[fut] = cl.loc[fut] * 50 + 13
    op.loc[fut] = op.loc[fut] * 50 + 13
    poisoned = PricePanel(close=cl, open=op, volume=momentum_panel.volume)
    res2 = run_backtest(poisoned, momentum_benchmark, **common)

    e1 = res1.equity.loc[:cutoff]
    e2 = res2.equity.loc[:cutoff]
    pd.testing.assert_series_equal(e1, e2)


def test_after_tax_below_pretax(momentum_panel, momentum_benchmark):
    """On an uptrending book that realises gains, tax must reduce final value."""
    common = dict(
        n_holdings=3, lookback_months=12, skip_recent_months=1,
        ma_window=100, max_weight=0.5,
    )
    pre = run_backtest(momentum_panel, momentum_benchmark, apply_tax=False, **common)
    post = run_backtest(
        momentum_panel, momentum_benchmark, apply_tax=True,
        st_rate=0.45, lt_rate=0.25, **common,
    )
    assert np.isfinite(pre.final_value) and np.isfinite(post.final_value)
    assert post.final_value <= pre.final_value
    assert post.total_tax_paid >= 0.0


def test_dividend_drag_reduces_final_value(momentum_panel, momentum_benchmark):
    """A positive dividend-withholding drag must lower the final value."""
    common = dict(
        n_holdings=3, lookback_months=12, skip_recent_months=1, ma_window=100,
        max_weight=0.5, cost_model=CostModel(0, 0, 0), apply_tax=False,
    )
    no_drag = run_backtest(momentum_panel, momentum_benchmark, **common)
    drag = run_backtest(
        momentum_panel, momentum_benchmark, dividend_drag_annual=0.01, **common
    )
    assert drag.final_value < no_drag.final_value


def test_turnover_recorded(momentum_panel, momentum_benchmark):
    res = run_backtest(
        momentum_panel, momentum_benchmark,
        n_holdings=3, lookback_months=12, skip_recent_months=1, ma_window=100,
    )
    assert len(res.turnover) > 0
    assert (res.turnover >= 0).all()
    # capital gets deployed once momentum has warmed up (early rebalances may be
    # all-NaN scores -> no trade -> 0 turnover, which is correct)
    assert (res.turnover > 0).any()


def test_membership_restricts_selection(momentum_panel, momentum_benchmark):
    """With a membership mask excluding the strongest name, it must never be bought."""
    from ccquant.utils.calendar import month_end_dates

    rebal = month_end_dates(momentum_panel.dates)
    tickers = momentum_panel.tickers
    # EEE is the strongest trender in the fixture; mark it as never a member
    mask = pd.DataFrame(True, index=rebal, columns=tickers)
    mask["EEE"] = False

    common = dict(
        n_holdings=3, lookback_months=12, skip_recent_months=1, ma_window=100,
        max_weight=0.5, cost_model=CostModel(0, 0, 0), apply_tax=False,
    )
    base = run_backtest(momentum_panel, momentum_benchmark, **common)
    masked = run_backtest(momentum_panel, momentum_benchmark, membership=mask, **common)
    # excluding a consistently-selected name must change the result
    assert masked.final_value > 0
    assert abs(masked.final_value - base.final_value) > 1.0


def test_turnover_levers_reduce_turnover(momentum_panel, momentum_benchmark):
    """Hysteresis + drift band must not increase average turnover."""
    common = dict(
        n_holdings=3, lookback_months=12, skip_recent_months=1, ma_window=100,
        max_weight=0.5, cost_model=CostModel(0, 0, 0),
    )
    base = run_backtest(momentum_panel, momentum_benchmark, **common)
    damped = run_backtest(
        momentum_panel, momentum_benchmark,
        hysteresis_exit=5, rebalance_band=0.5, **common,
    )
    assert damped.turnover.mean() <= base.turnover.mean() + 1e-9
