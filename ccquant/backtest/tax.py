"""Simplified lot-level capital-gains tax accounting.

Scope (per the M2 decision):
  * FIFO lots; each sale realises gains classified short-term (<= ``long_term_days``
    held) or long-term.
  * Annual settlement: within a year, short- and long-term are netted; a loss in
    one character offsets gains in the other; any remaining net loss carries
    forward to future years (applied to short-term gains first, the higher rate).
  * Cost basis = fill price (commissions are handled as cash fees by the engine,
    a minor simplification).

Deferred to M3 (risk module): wash-sale disallowance, tax-loss harvesting, and
sell-lot optimisation. This model is therefore a slightly *pessimistic* estimate
of after-tax return for names sold at a gain, which is the right side to err on
for the M2 gate.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Lot:
    shares: float
    basis: float  # per-share cost
    buy_date: object  # pd.Timestamp-like (has .year and supports subtraction)


class TaxLedger:
    def __init__(self, st_rate: float, lt_rate: float, long_term_days: int = 365) -> None:
        self.st_rate = st_rate
        self.lt_rate = lt_rate
        self.long_term_days = long_term_days
        self.lots: dict[str, deque[Lot]] = defaultdict(deque)
        self.realized: list[dict] = []  # {year, gain, long_term}
        self._carryforward = 0.0  # >= 0, dollars of unused losses
        self._settled: set[int] = set()

    # -- trading ------------------------------------------------------------ #
    def buy(self, ticker: str, shares: float, price: float, date) -> None:
        if shares <= 0:
            return
        self.lots[ticker].append(Lot(shares=shares, basis=price, buy_date=date))

    def sell(
        self, ticker: str, shares: float, price: float, date, tax_aware: bool = False
    ) -> float:
        """Sell ``shares``; record realised gains by character. Returns total gain.

        Default lot order is FIFO. With ``tax_aware=True`` we instead consume lots
        in the order that minimises tax: realised losses first (harvest), then
        long-term gains (lower rate), then short-term gains (highest rate) — only
        relevant for partial sells, since a full exit liquidates every lot anyway.
        """
        lots = self.lots[ticker]
        original = list(lots)  # FIFO order, for rebuilding survivors

        if tax_aware:
            def priority(lot: Lot) -> tuple[int, int]:
                gain = price - lot.basis
                long_term = (date - lot.buy_date).days > self.long_term_days
                if gain < 0:
                    return (0, 0)  # losses first
                return (1, 0 if long_term else 1)  # LT gains before ST gains

            order = sorted(original, key=priority)
        else:
            order = original

        remaining = shares
        total_gain = 0.0
        for lot in order:
            if remaining <= 1e-9:
                break
            take = min(lot.shares, remaining)
            gain = take * (price - lot.basis)
            held_days = (date - lot.buy_date).days
            self.realized.append(
                {"year": date.year, "gain": gain, "long_term": held_days > self.long_term_days}
            )
            total_gain += gain
            lot.shares -= take
            remaining -= take

        # rebuild the deque with surviving lots, preserving FIFO order
        self.lots[ticker] = deque(lot for lot in original if lot.shares > 1e-9)
        return total_gain

    # -- settlement --------------------------------------------------------- #
    def settle_year(self, year: int) -> float:
        """Tax owed for ``year``. Updates loss carryforward. Idempotent per year."""
        if year in self._settled:
            return 0.0
        self._settled.add(year)
        st = sum(r["gain"] for r in self.realized if r["year"] == year and not r["long_term"])
        lt = sum(r["gain"] for r in self.realized if r["year"] == year and r["long_term"])

        # cross-character offsetting of within-year losses
        if st < 0:
            lt += st
            st = 0.0
        if lt < 0:
            st += lt
            lt = 0.0

        # apply prior-year loss carryforward to positive gains, short-term first
        avail = self._carryforward
        if st > 0:
            use = min(avail, st)
            st -= use
            avail -= use
        if lt > 0:
            use = min(avail, lt)
            lt -= use
            avail -= use

        # bank remaining carryforward and any net loss generated this year
        net_loss = min(st, 0.0) + min(lt, 0.0)  # <= 0
        self._carryforward = avail + (-net_loss)
        st = max(st, 0.0)
        lt = max(lt, 0.0)
        return st * self.st_rate + lt * self.lt_rate
