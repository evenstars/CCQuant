"""Performance metrics computed from a daily equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAYS = 252


def _years(equity: pd.Series) -> float:
    span = (equity.index[-1] - equity.index[0]).days
    return max(span / 365.25, 1e-9)


def cagr(equity: pd.Series) -> float:
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / _years(equity)) - 1.0


def ann_vol(equity: pd.Series) -> float:
    return equity.pct_change().dropna().std() * np.sqrt(_DAYS)


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = equity.pct_change().dropna()
    excess = r - rf / _DAYS
    sd = excess.std()
    return (excess.mean() / sd * np.sqrt(_DAYS)) if sd > 0 else np.nan


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = equity.pct_change().dropna() - rf / _DAYS
    downside = r[r < 0].std()
    return (r.mean() / downside * np.sqrt(_DAYS)) if downside > 0 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return dd.min()


def max_drawdown_days(equity: pd.Series) -> int:
    peak = equity.cummax()
    underwater = equity < peak
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return longest


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return (cagr(equity) / mdd) if mdd > 0 else np.nan


def monthly_win_rate(equity: pd.Series) -> float:
    m = equity.resample("ME").last().pct_change().dropna()
    return (m > 0).mean() if len(m) else np.nan


def information_ratio(equity: pd.Series, benchmark_equity: pd.Series) -> float:
    a = equity.pct_change().dropna()
    b = benchmark_equity.pct_change().dropna()
    active = (a - b).dropna()
    sd = active.std()
    return (active.mean() / sd * np.sqrt(_DAYS)) if sd > 0 else np.nan


def compute_metrics(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    avg_turnover: float | None = None,
) -> dict[str, float]:
    out = {
        "CAGR": cagr(equity),
        "AnnVol": ann_vol(equity),
        "Sharpe": sharpe(equity),
        "Sortino": sortino(equity),
        "MaxDD": max_drawdown(equity),
        "MaxDD_days": max_drawdown_days(equity),
        "Calmar": calmar(equity),
        "MonthlyWinRate": monthly_win_rate(equity),
        "FinalValue": float(equity.iloc[-1]),
    }
    if benchmark_equity is not None:
        bench = benchmark_equity.reindex(equity.index).ffill()
        out["IR_vs_bench"] = information_ratio(equity, bench)
    if avg_turnover is not None:
        out["AvgTurnover"] = avg_turnover
    return out
