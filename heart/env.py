"""Environment construction, and a direct handle on the deal core."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from heart.config import SINGLE, SingleDealRules
from heart.rules import legal_action_mask, observe, reset, step, step_unchecked
from heart.types import Info, Observation, State

CLASSIC_V0 = "classic-v0"
AVAILABLE_MODES = (CLASSIC_V0,)


@dataclass(frozen=True)
class DealEnv:
    """One deal of Hearts: 52 plays, no passing, no running score.

    This is the core `classic-v0` is built on, not an environment of its own.
    It used to be published as `simplest-v0`, with its own single-learner
    adapter, rule document, examples and benchmarks; nothing was ever built on
    it, and a second contract is a second thing to keep true. What remains is
    the handle that makes the core testable and measurable on its own.
    """

    rules: SingleDealRules = SINGLE

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


def make(mode: str = CLASSIC_V0, **rule_overrides: object):
    """Create an environment by stable mode name."""

    if mode not in AVAILABLE_MODES:
        raise ValueError(f"unknown mode {mode!r}; available modes: {AVAILABLE_MODES}")
    from heart.classic import make_classic

    return make_classic(**rule_overrides)
