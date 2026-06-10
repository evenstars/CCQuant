"""Strategy signals: factors, regime filter, portfolio construction.

These modules are shared verbatim between the backtest and the live engine so the
two can never disagree on what the strategy *is*.
"""

from ccquant.strategy.factors import momentum_scores, month_end_prices
from ccquant.strategy.portfolio import (
    build_target_weights,
    select_names,
    select_names_with_hysteresis,
    target_weights,
)
from ccquant.strategy.regime import regime_exposure

__all__ = [
    "momentum_scores",
    "month_end_prices",
    "regime_exposure",
    "select_names",
    "select_names_with_hysteresis",
    "target_weights",
    "build_target_weights",
]
