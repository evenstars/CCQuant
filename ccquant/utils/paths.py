"""Centralized filesystem paths for the project.

All paths are derived from the project root so the code works regardless of the
current working directory.
"""

from __future__ import annotations

from pathlib import Path

# ccquant/utils/paths.py -> project root is two parents up from this file's dir.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_CACHE_DIR: Path = DATA_DIR / "cache"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOG_DIR: Path = PROJECT_ROOT / "logs"

STRATEGY_CONFIG: Path = CONFIG_DIR / "strategy.yaml"
RISK_CONFIG: Path = CONFIG_DIR / "risk.yaml"
ACCOUNT_CONFIG: Path = CONFIG_DIR / "account.yaml"
SECRETS_ENV: Path = CONFIG_DIR / "secrets.env"


def ensure_dirs() -> None:
    """Create runtime directories if they do not yet exist."""
    for d in (DATA_CACHE_DIR, REPORTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
