"""Selection and target-weight construction.

Given momentum scores + an eligibility mask + a regime exposure, produce the
target portfolio weights for a rebalance date. Used identically by the backtest
and (later) the live engine.
"""

from __future__ import annotations

import pandas as pd


def select_names(
    scores_row: pd.Series,
    eligible_row: pd.Series,
    n: int,
) -> list[str]:
    """Top-``n`` eligible names by momentum, with a deterministic tie-break.

    Ties are broken by ticker name (ascending) so results are reproducible — a
    silent non-determinism here would make backtests un-repeatable.
    """
    elig = eligible_row.reindex(scores_row.index, fill_value=False)
    valid = scores_row[elig & scores_row.notna()]
    if valid.empty:
        return []
    ranked = sorted(valid.index, key=lambda t: (-valid[t], str(t)))
    return ranked[:n]


def select_names_with_hysteresis(
    scores_row: pd.Series,
    eligible_row: pd.Series,
    n_holdings: int,
    n_exit: int,
    current: list[str] | set[str],
) -> list[str]:
    """Membership with a buffer band to cut edge churn (and thus tax).

    A held name is kept as long as it stays eligible and ranked within ``n_exit``;
    a new name is only added if ranked within ``n_holdings`` (the enter threshold).
    Kept names take priority; remaining slots are filled with the best newcomers.
    Deterministic given the (deterministic) ranking.
    """
    elig = eligible_row.reindex(scores_row.index, fill_value=False)
    valid = scores_row[elig & scores_row.notna()]
    if valid.empty:
        return []
    ranked = sorted(valid.index, key=lambda t: (-valid[t], str(t)))
    rank_of = {t: i for i, t in enumerate(ranked)}  # 0-based rank

    kept = [t for t in current if t in rank_of and rank_of[t] < n_exit][:n_holdings]
    result = list(kept)
    for t in ranked:  # fill remaining slots with best newcomers within enter band
        if len(result) >= n_holdings:
            break
        if t not in result and rank_of[t] < n_holdings:
            result.append(t)
    return result


def target_weights(
    selected: list[str],
    exposure: float,
    n_holdings: int,
    max_weight: float,
) -> dict[str, float]:
    """Equal-weight the selected names, scaled by ``exposure``, capped per name.

    Weight per name = exposure / n_holdings (not / len(selected)) so that when
    fewer than ``n_holdings`` names qualify, the shortfall sits in cash rather
    than over-concentrating the few that did. The remainder (1 - sum) is cash.
    """
    if not selected or exposure <= 0:
        return {}
    w = min(exposure / n_holdings, max_weight)
    return {t: w for t in selected}


def build_target_weights(
    scores: pd.DataFrame,
    eligible: pd.DataFrame,
    exposure: pd.Series,
    n_holdings: int,
    max_weight: float,
) -> pd.DataFrame:
    """Vectorised (rebalance-date x ticker) target weights, for analysis/tests.

    ``scores`` index is the set of rebalance dates. ``eligible`` and ``exposure``
    are sampled at those dates.
    """
    rows = {}
    for dt, srow in scores.iterrows():
        elig = eligible.loc[dt] if dt in eligible.index else pd.Series(True, index=srow.index)
        exp = float(exposure.get(dt, 1.0)) if exposure is not None else 1.0
        sel = select_names(srow, elig, n_holdings)
        rows[dt] = target_weights(sel, exp, n_holdings, max_weight)
    return pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)
