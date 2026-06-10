"""Reporting: turn equity curves into a comparison table + saved artifacts.

Plotting is optional — if matplotlib (the ``viz`` extra) isn't installed we still
produce the metrics table and CSVs, so CI and headless runs never fail on it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ccquant.backtest.metrics import compute_metrics
from ccquant.utils import paths
from ccquant.utils.logging import get_logger

_log = get_logger("ccquant.backtest.report")


def buy_and_hold_equity(price: pd.Series, initial: float = 100_000.0) -> pd.Series:
    """Equity curve of buying ``price`` at the first date and holding."""
    price = price.dropna()
    return initial * price / price.iloc[0]


def metrics_table(
    equities: dict[str, pd.Series],
    benchmark_name: str | None = None,
    turnovers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One row of metrics per named equity curve.

    ``benchmark_name`` (if present in ``equities``) is used as the IR benchmark
    for every other series.
    """
    turnovers = turnovers or {}
    bench = equities.get(benchmark_name) if benchmark_name else None
    rows = {
        name: compute_metrics(
            eq,
            benchmark_equity=bench if name != benchmark_name else None,
            avg_turnover=turnovers.get(name),
        )
        for name, eq in equities.items()
    }
    return pd.DataFrame(rows).T


def save_report(
    table: pd.DataFrame,
    equities: dict[str, pd.Series],
    outdir: Path | None = None,
    name: str = "backtest",
) -> Path:
    """Write the metrics table + combined equity CSV (and a plot if possible)."""
    outdir = outdir or (paths.REPORTS_DIR / name)
    outdir.mkdir(parents=True, exist_ok=True)

    table.to_csv(outdir / "metrics.csv")
    eq_df = pd.DataFrame(equities)
    eq_df.to_csv(outdir / "equity_curves.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1])
        eq_df.plot(ax=ax1, logy=True, title="Equity curves (log scale)")
        ax1.set_ylabel("Equity")
        for col in eq_df:
            dd = eq_df[col] / eq_df[col].cummax() - 1.0
            ax2.plot(dd.index, dd, label=col)
        ax2.set_ylabel("Drawdown")
        ax2.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "equity_curves.png", dpi=120)
        plt.close(fig)
    except ImportError:
        _log.info("matplotlib not installed; skipping plot (install the 'viz' extra)")

    _log.info("report written to %s", outdir)
    return outdir
