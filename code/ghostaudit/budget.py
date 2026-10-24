"""Call and spend circuit breaker. On breach the run continues engine-only."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# USD per million tokens (first-party API list price).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_calls: int = int(os.environ.get("GHOSTAUDIT_MAX_CALLS", "200"))
    max_usd: float = float(os.environ.get("GHOSTAUDIT_MAX_USD", "10"))
    calls: int = 0
    usd: float = 0.0
    tripped: bool = False
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def check(self) -> None:
        if self.tripped or self.calls >= self.max_calls or self.usd >= self.max_usd:
            self.tripped = True
            raise BudgetExceeded(f"budget reached: {self.calls} calls, ${self.usd:.2f}")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
        self.calls += 1
        self.usd += cost
        m = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        m["calls"] += 1
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["usd"] += cost
        return cost
