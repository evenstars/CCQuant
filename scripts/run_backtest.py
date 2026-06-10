#!/usr/bin/env python3
"""Run the full M2 backtest on real S&P 500 data and compare to benchmarks.

This is the M2 *gate* harness: it answers "does the momentum strategy beat its
benchmarks after tax?" Requires network + the data extra:

    uv sync --dev --extra data
    uv run python scripts/run_backtest.py
    # options:
    uv run python scripts/run_backtest.py --start 2005-01-01 --capital 100000

Notes / known limitations (by design, see docs/design.md):
  * Universe = CURRENT S&P 500 members (yfinance) -> survivorship bias. Results
    are optimistic; treat as a pipeline check, not a verdict, until a point-in-
    time universe lands (TODO M1.1).
  * Benchmarks: SPY (cap-weighted), RSP (equal-weight proxy), MTUM (momentum
    ETF, data only from ~2013). Each buy-&-hold curve starts at its first
    available date on/after the strategy start.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ccquant.backtest import (
    CostModel,
    buy_and_hold_equity,
    metrics_table,
    run_backtest,
    save_report,
)
from ccquant.data import (
    HistoricalSP500Provider,
    SP500Provider,
    YFinanceLoader,
    build_membership_mask,
)
from ccquant.utils.calendar import month_end_dates
from ccquant.utils.config import load_config
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.run_backtest", to_file=True)

BENCHMARKS = ["SPY", "RSP", "MTUM"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCQuant momentum backtest")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--name", default=None, help="report subfolder name")
    p.add_argument(
        "--pit",
        action="store_true",
        help="point-in-time universe (historical members, reduces survivorship bias)",
    )
    p.add_argument(
        "--account",
        choices=["roth_ira", "us_taxable", "nra_hk"],
        default=None,
        help="override config/account.yaml account type for this run",
    )
    p.add_argument(
        "--dividend-drag",
        type=float,
        default=None,
        help="override the nra_hk dividend-withholding drag, e.g. 0.0036 "
        "(~1.2%% yield x 30%%).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    s, r, acct = cfg.strategy, cfg.risk, cfg.account
    # CLI overrides the config switch when provided
    acct_type = args.account or acct.type
    apply_tax = acct_type == "us_taxable"
    div_drag = args.dividend_drag if args.dividend_drag is not None else acct.dividend_drag_annual
    report_name = args.name or ("m1_1_pit" if args.pit else "m2_sp500")

    loader = YFinanceLoader()
    membership = None
    if args.pit:
        provider = HistoricalSP500Provider()
        universe = provider.all_members(args.start, args.end)
        _log.info("PIT universe: %d names ever in S&P 500 over the period", len(universe))
    else:
        universe = SP500Provider().get_constituents(as_of=pd.Timestamp(args.end))
        _log.info("CURRENT universe: %d names (survivorship-biased)", len(universe))
    _log.info("benchmarks: %s", BENCHMARKS)

    _log.info("downloading universe prices (cached after first run)...")
    panel = loader.load(universe, args.start, args.end)
    bpanel = loader.load(BENCHMARKS, args.start, args.end)
    spy_close = bpanel.close["SPY"].dropna()

    if args.pit:
        rebal_dates = month_end_dates(panel.dates)
        membership = build_membership_mask(provider, rebal_dates, panel.tickers)
        _log.info("built membership mask: %s", membership.shape)

    cm = CostModel(
        commission_per_share=s.costs.commission_per_share,
        min_commission=s.costs.min_commission,
        slippage_bps=s.costs.slippage_bps,
    )
    common = dict(
        n_holdings=s.n_holdings,
        lookback_months=s.lookback_months,
        skip_recent_months=s.skip_recent_months,
        ma_window=s.regime_filter.ma_window,
        below_ma_exposure=s.regime_filter.below_ma_exposure,
        max_weight=r.max_position_weight,
        min_adv_usd=s.liquidity.min_adv_usd,
        min_price=s.liquidity.min_price,
        cost_model=cm,
        initial_capital=args.capital,
        membership=membership,
        start=args.start,
        end=args.end,
    )

    # PRIMARY run = the configured account (the tax switch decides apply_tax).
    _log.info("running primary backtest for account=%s (apply_tax=%s)", acct_type, apply_tax)
    primary = run_backtest(
        panel, spy_close, apply_tax=apply_tax,
        st_rate=r.tax.st_rate, lt_rate=r.tax.lt_rate, long_term_days=r.tax.long_term_days,
        dividend_drag_annual=div_drag, **common,
    )
    primary_label = f"Momentum_{acct_type}"

    start_date = primary.equity.index[0]
    equities = {primary_label: primary.equity}
    turnovers = {primary_label: float(primary.turnover.mean())}

    # For a tax-free primary, add a US-taxable reference row for contrast.
    if not apply_tax:
        ref = run_backtest(
            panel, spy_close, apply_tax=True,
            st_rate=r.tax.st_rate, lt_rate=r.tax.lt_rate, long_term_days=r.tax.long_term_days,
            **common,
        )
        equities["Momentum_us_taxable_ref"] = ref.equity
        turnovers["Momentum_us_taxable_ref"] = float(ref.turnover.mean())

    for b in BENCHMARKS:
        series = bpanel.close[b].loc[start_date:].dropna()
        if series.empty:
            _log.warning("no data for benchmark %s in range; skipping", b)
            continue
        equities[b] = buy_and_hold_equity(series, initial=args.capital)

    table = metrics_table(equities, benchmark_name="SPY", turnovers=turnovers)
    outdir = save_report(table, equities, name=report_name)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    mode = "POINT-IN-TIME (selection bias reduced)" if args.pit else "CURRENT MEMBERS"
    drag = f"{div_drag:.2%}/yr" if div_drag else "none"
    print("\n================ BACKTEST SUMMARY ================")
    print(f"Account: {acct_type}  (tax {'ON' if apply_tax else 'OFF'})  |  dividend drag: {drag}")
    print(f"Universe mode: {mode}")
    print(f"Period: {start_date.date()} -> {primary.equity.index[-1].date()}")
    print(f"Initial capital: ${args.capital:,.0f}\n")
    print(table.round(3).to_string())
    if apply_tax:
        print(f"\nTotal tax paid: ${primary.total_tax_paid:,.0f}")
    print(f"Report + CSV + plot: {outdir}")
    print(f"\nGATE: does {primary_label} beat SPY / RSP / MTUM on CAGR & Sharpe?")
    if args.pit:
        print("(PIT fixes WHICH names; delisted-price gaps remain -> still mildly optimistic.)")
    else:
        print("(Reminder: survivorship-biased universe -> these numbers are optimistic.)")


if __name__ == "__main__":
    main()
