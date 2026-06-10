"""Strategy signal tests: momentum factor, selection, weights, regime."""

import numpy as np
import pandas as pd

from ccquant.strategy.factors import momentum_scores
from ccquant.strategy.portfolio import (
    select_names,
    select_names_with_hysteresis,
    target_weights,
)
from ccquant.strategy.regime import regime_exposure


# --- factor known-answer ---------------------------------------------------- #
def test_momentum_known_answer():
    """Hand-computed 12-1 momentum on constant-within-month prices.

    Each month's price is a known level, so price at any month-end equals that
    level and score(m) = level[m-1] / level[m-12] - 1 exactly.
    """
    # 14 months of monthly levels for a single ticker
    levels = [10, 11, 12, 11, 13, 14, 15, 14, 16, 17, 18, 19, 20, 22]
    idx = pd.bdate_range("2020-01-01", periods=14 * 21)  # ~14 months
    # assign each business day the level of its month-of-sequence
    month_pos = idx.to_period("M")
    uniq = list(dict.fromkeys(month_pos))
    level_map = {p: levels[i] for i, p in enumerate(uniq[: len(levels)])}
    close = pd.DataFrame(
        {"X": [level_map[p] for p in month_pos]}, index=idx, dtype=float
    )
    scores = momentum_scores(close, lookback_months=12, skip_recent_months=1)
    # last month-end: level[-2]/level[-13] - 1 = 20/11 - 1
    last = scores["X"].dropna().iloc[-1]
    assert abs(last - (20 / 11 - 1)) < 1e-9


def test_momentum_no_lookahead(momentum_panel):
    """Mutating future prices must not change past momentum scores."""
    close = momentum_panel.close
    base = momentum_scores(close, 12, 1)
    cutoff = base.dropna(how="all").index[len(base.dropna(how="all")) // 2]

    poisoned = close.copy()
    poisoned.loc[poisoned.index > cutoff] = poisoned.loc[poisoned.index > cutoff] * 99 + 7
    after = momentum_scores(poisoned, 12, 1)

    a = base.loc[base.index <= cutoff]
    b = after.loc[after.index <= cutoff]
    pd.testing.assert_frame_equal(a, b)


# --- selection -------------------------------------------------------------- #
def test_select_deterministic_tiebreak():
    scores = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": 0.5, "DDD": 0.1})
    elig = pd.Series(True, index=scores.index)
    # all top-3 tie at 0.5 -> alphabetical
    assert select_names(scores, elig, 3) == ["AAA", "BBB", "CCC"]


def test_select_respects_eligibility_and_nan():
    scores = pd.Series({"AAA": 0.9, "BBB": np.nan, "CCC": 0.3})
    elig = pd.Series({"AAA": False, "BBB": True, "CCC": True})
    # AAA ineligible, BBB NaN -> only CCC
    assert select_names(scores, elig, 5) == ["CCC"]


def test_hysteresis_keeps_held_name_in_buffer():
    # 6 names ranked by score desc: F(0.6) E(0.5) D(0.4) C(0.3) B(0.2) A(0.1)
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6})
    elig = pd.Series(True, index=scores.index)
    # hold n=2, exit band 4. Currently hold C (rank 2, within exit band of 4)
    sel = select_names_with_hysteresis(scores, elig, n_holdings=2, n_exit=4, current=["C"])
    # C is kept (rank 2 < 4); one new slot filled by top newcomer F (rank 0)
    assert sel == ["C", "F"]


def test_hysteresis_drops_name_outside_exit_band():
    scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6})
    elig = pd.Series(True, index=scores.index)
    # holding A (rank 5) with exit band 3 -> A is dropped; refill with F, E
    sel = select_names_with_hysteresis(scores, elig, n_holdings=2, n_exit=3, current=["A"])
    assert sel == ["F", "E"]


# --- weights ---------------------------------------------------------------- #
def test_target_weights_equal_and_exposure():
    w = target_weights(["A", "B", "C"], exposure=1.0, n_holdings=10, max_weight=0.5)
    assert all(abs(v - 0.1) < 1e-12 for v in w.values())  # exposure/n = 1/10
    # shortfall (3 names of 10) -> rest is cash, weights don't sum to 1
    assert abs(sum(w.values()) - 0.3) < 1e-12


def test_target_weights_regime_scaling_and_cap():
    half = target_weights(["A", "B"], exposure=0.5, n_holdings=10, max_weight=1.0)
    assert all(abs(v - 0.05) < 1e-12 for v in half.values())  # 0.5/10
    capped = target_weights(["A"], exposure=1.0, n_holdings=2, max_weight=0.4)
    assert abs(capped["A"] - 0.4) < 1e-12  # 1/2=0.5 capped to 0.4


# --- regime ----------------------------------------------------------------- #
def test_regime_exposure():
    s = pd.Series(
        [100] * 30 + [80] * 5 + [120] * 5,
        index=pd.bdate_range("2020-01-01", periods=40),
        dtype=float,
    )
    exp = regime_exposure(s, ma_window=20, below_ma_exposure=0.5)
    assert exp.iloc[0] == 1.0  # before MA defined -> full
    assert exp.iloc[34] == 0.5  # 80 is below the ~100 MA
    assert exp.iloc[-1] == 1.0  # 120 back above MA
