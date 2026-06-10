#!/usr/bin/env python3
"""Turnover / tax-drag experiment.

M2 showed taxes cut CAGR from ~24.5% to ~16% because the strategy turns over ~7x
a year at the 45% short-term rate. This script measures how much of that drag the
tax-reduction levers recover, by running the same backtest under increasingly
aggressive settings and comparing after-tax results.

    uv sync --dev --extra data
    uv run python scripts/turnover_experiment.py

Levers (cumulative):
  baseline              -> monthly top-N, re-equal-weight every month (M2 default)
  +hysteresis           -> keep held names until they fall out of rank_buffer_exit
  +drift_band(0.5)      -> don't trim a kept name within 50% of its target weight
  +tax_aware_lots       -> sell loss/long-term lots before short-term ones
"""

from __future__ import annotations

import pandas as pd

from ccquant.backtest import CostModel, compute_metrics, run_backtest
from ccquant.data import SP500Provider, YFinanceLoader
from ccquant.utils.config import load_config
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.turnover_experiment", to_file=True)

START, END = "2005-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
CAPITAL = 100_000.0


def main() -> None:
    cfg = load_config()
    s, r = cfg.strategy, cfg.risk

    loader = YFinanceLoader()
    universe = SP500Provider().get_constituents(as_of=pd.Timestamp(END))
    _log.info("loading %d names...", len(universe))
    panel = loader.load(universe, START, END)
    spy_close = loader.load(["SPY"], START, END).close["SPY"].dropna()

    cm = CostModel(
        commission_per_share=s.costs.commission_per_share,
        min_commission=s.costs.min_commission,
        slippage_bps=s.costs.slippage_bps,
    )
    base = dict(
        n_holdings=s.n_holdings,
        lookback_months=s.lookback_months,
        skip_recent_months=s.skip_recent_months,
        ma_window=s.regime_filter.ma_window,
        below_ma_exposure=s.regime_filter.below_ma_exposure,
        max_weight=r.max_position_weight,
        min_adv_usd=s.liquidity.min_adv_usd,
        min_price=s.liquidity.min_price,
        cost_model=cm,
        initial_capital=CAPITAL,
        start=START,
        end=END,
        st_rate=r.tax.st_rate,
        lt_rate=r.tax.lt_rate,
        long_term_days=r.tax.long_term_days,
    )

    variants = {
        "baseline": {},
        "+hysteresis": {"hysteresis_exit": r.tax.rank_buffer_exit},
        "+drift_band": {"hysteresis_exit": r.tax.rank_buffer_exit, "rebalance_band": 0.5},
        "+tax_aware_lots": {
            "hysteresis_exit": r.tax.rank_buffer_exit,
            "rebalance_band": 0.5,
            "tax_aware_lots": True,
        },
    }

    rows = {}
    for name, overrides in variants.items():
        _log.info("running variant: %s", name)
        pre = run_backtest(panel, spy_close, apply_tax=False, **base, **overrides)
        post = run_backtest(panel, spy_close, apply_tax=True, **base, **overrides)
        mpre = compute_metrics(pre.equity)
        mpost = compute_metrics(post.equity)
        rows[name] = {
            "AvgTurnover/mo": pre.turnover.mean(),
            "Pretax_CAGR": mpre["CAGR"],
            "Aftertax_CAGR": mpost["CAGR"],
            "Aftertax_Sharpe": mpost["Sharpe"],
            "Aftertax_MaxDD": mpost["MaxDD"],
            "TaxPaid": post.total_tax_paid,
            "Aftertax_Final": mpost["FinalValue"],
        }

    table = pd.DataFrame(rows).T
    pd.set_option("display.width", 160)
    print("\n============== TURNOVER / TAX-DRAG EXPERIMENT ==============")
    print(f"Period: {START} -> {END}  |  capital ${CAPITAL:,.0f}")
    print(f"ST rate {r.tax.st_rate:.0%} / LT rate {r.tax.lt_rate:.0%}\n")
    print(table.round({
        "AvgTurnover/mo": 3, "Pretax_CAGR": 3, "Aftertax_CAGR": 3,
        "Aftertax_Sharpe": 3, "Aftertax_MaxDD": 3, "TaxPaid": 0, "Aftertax_Final": 0,
    }).to_string())
    print("\nLook for: lower turnover -> higher Aftertax_CAGR. How close can we get")
    print("after-tax back to pre-tax? (Reminder: survivorship-biased universe.)")


if __name__ == "__main__":
    main()
