"""User-facing immutable environment object."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from heart.config import SINGLE, SingleDealRules
from heart.rules import legal_action_mask, observe, reset, step, step_unchecked
from heart.types import Info, Observation, State

SIMPLEST_V0 = "simplest-v0"
AVAILABLE_MODES = (SIMPLEST_V0,)


@dataclass(frozen=True)
class HeartEnv:
    """A stateless handle for one HEART ruleset."""

    rules: SingleDealRules = SINGLE
    mode: str = SIMPLEST_V0

    def reset(self, key: Array) -> tuple[State, Observation]:
        return reset(key, self.rules)

    def step(
        self, state: State, action: Array | int
    ) -> tuple[State, Observation, Array, Array, Info]:
        return step(state, action, self.rules)

    def step_unchecked(
        self, state: State, action: Array | int
    ) -> tuple[State, Observation, Array, Array, Info]:
        """Fast path for an integer action already selected from the mask."""

        return step_unchecked(state, action, self.rules)

    def observe(self, state: State, player: Array | int) -> Observation:
        return observe(state, player, self.rules)

    def legal_action_mask(self, state: State) -> Array:
        return legal_action_mask(state, self.rules)


def make(mode: str = SIMPLEST_V0, **rule_overrides: object) -> HeartEnv:
    """Create an environment by stable mode name."""

    if mode not in AVAILABLE_MODES:
        raise ValueError(f"unknown mode {mode!r}; available modes: {AVAILABLE_MODES}")
    rules = SingleDealRules(**rule_overrides)
    return HeartEnv(rules=rules, mode=mode)
