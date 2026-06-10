"""Backtest engine: costs, tax, simulation, metrics, reporting."""

from ccquant.backtest.costs import CostModel
from ccquant.backtest.engine import BacktestResult, run_backtest
from ccquant.backtest.metrics import compute_metrics
from ccquant.backtest.report import buy_and_hold_equity, metrics_table, save_report
from ccquant.backtest.tax import TaxLedger

__all__ = [
    "CostModel",
    "run_backtest",
    "BacktestResult",
    "compute_metrics",
    "metrics_table",
    "save_report",
    "buy_and_hold_equity",
    "TaxLedger",
]
