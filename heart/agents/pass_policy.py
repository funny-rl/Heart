"""JIT-compatible rule baselines for choosing three cards to pass."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from heart.cards import CARD_RANKS, CARD_SUITS, HEARTS, QUEEN_OF_SPADES
from heart.classic import NUM_PASS_ACTIONS, PASS_COMBINATIONS, ClassicObservation

PASS_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class RulePassPolicy:
    """Select one of the 286 unordered three-slot pass combinations."""

    difficulty: str = "medium"

    def __post_init__(self) -> None:
        if self.difficulty not in PASS_DIFFICULTIES:
            raise ValueError(
                f"unknown pass difficulty {self.difficulty!r}; "
                f"available: {PASS_DIFFICULTIES}"
            )

    def __call__(self, observation: ClassicObservation, key: Array) -> Array:
        hand_cards = jnp.nonzero(
            observation.game.hand,
            size=13,
            fill_value=0,
        )[0]
        ranks = CARD_RANKS[hand_cards].astype(jnp.float32)
        suits = CARD_SUITS[hand_cards]
        combos = PASS_COMBINATIONS.astype(jnp.int32)

        if self.difficulty == "easy":
            logits = jax.random.uniform(key, (NUM_PASS_ACTIONS,))
        else:
            danger = ranks
            danger += jnp.where(hand_cards == QUEEN_OF_SPADES, 30.0, 0.0)
            danger += jnp.where(
                (suits == 2) & (ranks >= 10),
                8.0,
                0.0,
            )
            danger += jnp.where(suits == HEARTS, 3.0 + ranks / 4.0, 0.0)
            logits = jnp.sum(danger[combos], axis=1)
            if self.difficulty == "hard":
                suit_counts = jnp.bincount(suits, length=4)
                selected_suits = suits[combos]
                selected_counts = jnp.stack(
                    [jnp.sum(selected_suits == suit, axis=1) for suit in range(4)],
                    axis=1,
                )
                creates_void = (suit_counts[None, :] > 0) & (
                    selected_counts == suit_counts[None, :]
                )
                logits += 12.0 * jnp.sum(creates_void, axis=1)
            logits += jax.random.uniform(key, (NUM_PASS_ACTIONS,), maxval=1e-3)

        masked = jnp.where(observation.pass_action_mask, logits, -jnp.inf)
        return jnp.argmax(masked).astype(jnp.int32)


def make_rule_pass_policy(difficulty: str = "medium") -> RulePassPolicy:
    return RulePassPolicy(difficulty=difficulty)
