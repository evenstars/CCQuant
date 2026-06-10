"""Config loading and validation tests."""

import pytest
from pydantic import ValidationError

from ccquant.utils.config import (
    AccountConfig,
    RiskConfig,
    StrategyConfig,
    load_config,
)


def test_load_config_from_yaml() -> None:
    cfg = load_config()
    # strategy.yaml values
    assert cfg.strategy.universe == "SP500"
    assert cfg.strategy.lookback_months == 12
    assert cfg.strategy.skip_recent_months == 1
    assert cfg.strategy.n_holdings == 15
    assert cfg.strategy.regime_filter.ma_window == 200
    assert cfg.strategy.regime_filter.below_ma_exposure == 0.5
    assert cfg.strategy.ramp_in_tranches == 3
    # risk.yaml values
    assert cfg.risk.max_position_weight == 0.08
    assert cfg.risk.max_sector_weight == 0.35
    assert cfg.risk.tax.wash_sale_window_days == 30
    assert cfg.risk.tax.rank_buffer_exit >= cfg.risk.tax.rank_buffer_enter
    # secrets default (no secrets.env present -> model defaults apply)
    assert cfg.secrets.ib_port == 4002
    assert cfg.secrets.ib_host == "127.0.0.1"


def test_account_switch_drives_tax() -> None:
    assert AccountConfig(type="roth_ira").apply_tax is False
    assert AccountConfig(type="nra_hk").apply_tax is False
    assert AccountConfig(type="us_taxable").apply_tax is True  # the heavy-tax mode


def test_account_default_is_tax_free() -> None:
    cfg = load_config()
    # config/account.yaml ships as roth_ira -> tax-free by default
    assert cfg.account.type == "roth_ira"
    assert cfg.account.apply_tax is False


def test_account_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AccountConfig(type="brokerage_xyz")


def test_strategy_lookback_must_exceed_skip() -> None:
    with pytest.raises(ValidationError):
        StrategyConfig(lookback_months=3, skip_recent_months=3)


def test_risk_weights_bounded() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(max_position_weight=1.5)


def test_tax_buffer_ordering() -> None:
    with pytest.raises(ValidationError):
        # exit buffer smaller than enter buffer is invalid
        RiskConfig.model_validate({"tax": {"rank_buffer_enter": 25, "rank_buffer_exit": 15}})
