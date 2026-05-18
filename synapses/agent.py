"""Deterministic agent model.

The :class:`Agent` class keeps only three pieces of internal state and exposes a
single ``act`` method. The decision rules are intentionally simple so the module
is easy to test and safe to reuse without hidden dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class Agent:
    """A small deterministic agent with basic wellbeing attributes.

    Attributes:
        wealth: Numeric measure of available resources.
        health: Numeric measure of physical condition.
        satisfaction: Numeric measure of contentment.
        position: Current grid-cell position used by spatial simulations.
    """

    wealth: float = 50.0
    health: float = 50.0
    satisfaction: float = 50.0
    position: tuple[int, int] = (0, 0)

    def act(self, environment_state: Mapping[str, Any] | None) -> dict[str, str]:
        """Choose one deterministic action from the current environment state.

        Args:
            environment_state: Optional mapping with simple environmental
                signals. Recognized keys are ``risk``, ``opportunity``, and
                ``social``. Missing keys default to neutral values.

        Returns:
            Action dictionary with an ``action`` name and short ``reason``.
        """
        state = environment_state or {}
        risk = _as_float(state.get("risk", 0))
        opportunity = _as_float(state.get("opportunity", 0))
        social = _as_float(state.get("social", 0))

        if self.health < 40 or risk >= 70:
            return {"action": "rest", "reason": "protect_health"}

        if self.wealth < 40 and opportunity >= risk:
            return {"action": "work", "reason": "increase_wealth"}

        if self.satisfaction < 40 or social >= 60:
            return {"action": "socialize", "reason": "increase_satisfaction"}

        return {"action": "maintain", "reason": "balanced_state"}


def _as_float(value: Any) -> float:
    """Convert simple numeric environment values to floats.

    Non-numeric values are treated as zero to keep decisions deterministic and
    avoid coupling the agent to external validation components.
    """
    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    return 0.0
