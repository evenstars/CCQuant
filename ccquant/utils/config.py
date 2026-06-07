"""Typed configuration loading.

Strategy and risk parameters live in YAML (`config/*.yaml`) and are validated with
pydantic models. Secrets (API keys, account numbers) come from environment variables
/ `config/secrets.env` and are kept separate so they never land in version control.

Usage:
    from ccquant.utils.config import load_config
    cfg = load_config()
    cfg.strategy.n_holdings   # -> 15
    cfg.risk.max_position_weight
    cfg.secrets.ib_port
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccquant.utils import paths


# --------------------------------------------------------------------------- #
# Strategy config (config/strategy.yaml)
# --------------------------------------------------------------------------- #
class RegimeFilterConfig(BaseModel):
    benchmark: str = "SPY"
    ma_window: int = Field(200, gt=0)
    below_ma_exposure: float = Field(0.5, ge=0.0, le=1.0)


class LiquidityConfig(BaseModel):
    min_adv_usd: float = Field(5_000_000, ge=0)
    min_price: float = Field(5.0, ge=0)


class ExecutionConfig(BaseModel):
    signal_price: Literal["close", "open"] = "close"
    fill_price: Literal["next_open", "close", "next_close"] = "next_open"
    total_return: bool = True


class StrategyConfig(BaseModel):
    universe: str = "SP500"
    lookback_months: int = Field(12, gt=0)
    skip_recent_months: int = Field(1, ge=0)
    n_holdings: int = Field(15, gt=0)
    weighting: Literal["equal"] = "equal"
    rebalance: Literal["monthly"] = "monthly"
    regime_filter: RegimeFilterConfig = Field(default_factory=RegimeFilterConfig)
    ramp_in_tranches: int = Field(3, ge=1)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _check_lookback(self) -> StrategyConfig:
        if self.skip_recent_months >= self.lookback_months:
            raise ValueError("skip_recent_months must be < lookback_months")
        return self


# --------------------------------------------------------------------------- #
# Risk config (config/risk.yaml)
# --------------------------------------------------------------------------- #
class TaxConfig(BaseModel):
    wash_sale_window_days: int = Field(30, ge=0)
    prefer_long_term: bool = True
    rank_buffer_enter: int = Field(15, gt=0)
    rank_buffer_exit: int = Field(25, gt=0)

    @model_validator(mode="after")
    def _check_buffer(self) -> TaxConfig:
        if self.rank_buffer_exit < self.rank_buffer_enter:
            raise ValueError("rank_buffer_exit must be >= rank_buffer_enter")
        return self


class RiskConfig(BaseModel):
    max_position_weight: float = Field(0.08, gt=0, le=1.0)
    max_sector_weight: float = Field(0.35, gt=0, le=1.0)
    cash_buffer: float = Field(0.02, ge=0, le=1.0)
    portfolio_drawdown_alert: float = Field(0.25, gt=0, le=1.0)
    order_price_deviation_reject: float = Field(0.03, gt=0, le=1.0)
    vol_target_annual: float | None = None
    tax: TaxConfig = Field(default_factory=TaxConfig)


# --------------------------------------------------------------------------- #
# Secrets (env / config/secrets.env) — never committed
# --------------------------------------------------------------------------- #
class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(paths.SECRETS_ENV),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 1
    ib_account: str = ""

    polygon_api_key: str = ""
    norgate_api_key: str = ""

    alert_email: str = ""
    slack_webhook_url: str = ""


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    strategy: StrategyConfig
    risk: RiskConfig
    secrets: Secrets


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(
    strategy_path: Path | None = None,
    risk_path: Path | None = None,
) -> AppConfig:
    """Load and validate strategy + risk YAML and environment secrets."""
    strategy = StrategyConfig.model_validate(
        _load_yaml(strategy_path or paths.STRATEGY_CONFIG)
    )
    risk = RiskConfig.model_validate(_load_yaml(risk_path or paths.RISK_CONFIG))
    secrets = Secrets()  # reads env / secrets.env
    return AppConfig(strategy=strategy, risk=risk, secrets=secrets)
