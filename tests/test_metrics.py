"""Performance metric tests on curves with known properties."""

import numpy as np
import pandas as pd

from ccquant.backtest.metrics import cagr, compute_metrics, max_drawdown


def _equity(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_cagr_known():
    # exactly double over ~1 year of business days
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    eq = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    # ~1 year, doubling -> CAGR ~ 100%
    assert 0.95 < cagr(eq) < 1.05


def test_max_drawdown():
    eq = _equity([100, 120, 60, 90, 150])
    # peak 120 -> trough 60 = -50%
    assert abs(max_drawdown(eq) - (-0.5)) < 1e-9


def test_compute_metrics_keys():
    eq = _equity(list(np.linspace(100, 130, 252)))
    bench = _equity(list(np.linspace(100, 110, 252)))
    m = compute_metrics(eq, benchmark_equity=bench, avg_turnover=0.4)
    for k in ["CAGR", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar", "IR_vs_bench", "AvgTurnover"]:
        assert k in m
    assert np.isfinite(m["CAGR"])
