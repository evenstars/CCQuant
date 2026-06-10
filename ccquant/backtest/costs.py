"""Transaction-cost model.

Two components per trade:
  * commission: per-share, with a per-order minimum (IBKR-tiered shape).
  * slippage: a fraction of traded notional, in basis points, applied per side.

``slippage`` is returned separately from ``commission`` so the engine can apply
slippage as an effective worse fill price and commission as a cash fee — but for
P&L purposes both are simply costs subtracted from the account.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0035
    min_commission: float = 0.35
    slippage_bps: float = 5.0

    def commission(self, shares: float) -> float:
        """Commission for trading ``shares`` (absolute count) in one order."""
        shares = abs(shares)
        if shares == 0:
            return 0.0
        return max(self.min_commission, self.commission_per_share * shares)

    def slippage(self, notional: float) -> float:
        """Slippage cost for ``notional`` dollars traded (absolute)."""
        return abs(notional) * self.slippage_bps / 10_000.0

    def total(self, shares: float, price: float) -> float:
        """Total cost (commission + slippage) for a single trade."""
        return self.commission(shares) + self.slippage(abs(shares) * price)
