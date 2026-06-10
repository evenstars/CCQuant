"""Cost model and tax ledger tests."""

import pandas as pd

from ccquant.backtest.costs import CostModel
from ccquant.backtest.tax import TaxLedger


def test_commission_minimum():
    cm = CostModel(commission_per_share=0.0035, min_commission=0.35, slippage_bps=0)
    assert cm.commission(10) == 0.35  # 10*0.0035=0.035 -> floored at min
    assert abs(cm.commission(1000) - 3.5) < 1e-12
    assert cm.commission(0) == 0.0


def test_slippage():
    cm = CostModel(commission_per_share=0, min_commission=0, slippage_bps=5)
    assert abs(cm.slippage(10_000) - 5.0) < 1e-12  # 5bps of 10k


def test_tax_short_vs_long_term():
    led = TaxLedger(st_rate=0.40, lt_rate=0.20, long_term_days=365)
    # short-term gain: buy then sell 100 days later
    led.buy("AAA", 10, 100.0, pd.Timestamp("2020-01-01"))
    led.sell("AAA", 10, 110.0, pd.Timestamp("2020-04-10"))  # ~100 days, ST
    tax_2020 = led.settle_year(2020)
    assert abs(tax_2020 - (10 * 10) * 0.40) < 1e-9  # gain 100 * ST rate

    # long-term gain in a new ledger
    led2 = TaxLedger(st_rate=0.40, lt_rate=0.20)
    led2.buy("BBB", 10, 100.0, pd.Timestamp("2020-01-01"))
    led2.sell("BBB", 10, 110.0, pd.Timestamp("2021-06-01"))  # >365 days, LT
    assert abs(led2.settle_year(2021) - 100 * 0.20) < 1e-9


def test_tax_fifo_order():
    led = TaxLedger(st_rate=0.50, lt_rate=0.20)
    led.buy("AAA", 10, 100.0, pd.Timestamp("2020-01-01"))
    led.buy("AAA", 10, 120.0, pd.Timestamp("2020-02-01"))
    # sell 10 -> should consume the first (cheaper) lot first
    gain = led.sell("AAA", 10, 130.0, pd.Timestamp("2020-03-01"))
    assert abs(gain - (130 - 100) * 10) < 1e-9


def test_tax_aware_sell_realises_losses_first():
    led = TaxLedger(st_rate=0.50, lt_rate=0.20)
    # two lots: one will be a loss, one a gain, at sell price 100
    led.buy("AAA", 10, 120.0, pd.Timestamp("2020-01-01"))  # loss lot @120
    led.buy("AAA", 10, 80.0, pd.Timestamp("2020-02-01"))   # gain lot @80
    # sell 10 (partial) tax-aware -> should consume the LOSS lot first
    gain = led.sell("AAA", 10, 100.0, pd.Timestamp("2020-03-01"), tax_aware=True)
    assert abs(gain - (100 - 120) * 10) < 1e-9  # -200 loss realised, gain lot kept
    # remaining lot is the gain lot
    assert len(led.lots["AAA"]) == 1
    assert abs(led.lots["AAA"][0].basis - 80.0) < 1e-9


def test_tax_loss_carryforward():
    led = TaxLedger(st_rate=0.50, lt_rate=0.20)
    # year 1: realise a loss
    led.buy("AAA", 10, 100.0, pd.Timestamp("2020-01-01"))
    led.sell("AAA", 10, 80.0, pd.Timestamp("2020-06-01"))  # -200 ST loss
    assert led.settle_year(2020) == 0.0  # no tax on a loss
    # year 2: realise a gain that the carryforward should partly offset
    led.buy("BBB", 10, 100.0, pd.Timestamp("2021-01-01"))
    led.sell("BBB", 10, 150.0, pd.Timestamp("2021-06-01"))  # +500 ST gain
    tax = led.settle_year(2021)
    assert abs(tax - (500 - 200) * 0.50) < 1e-9  # 300 taxable after carryforward
