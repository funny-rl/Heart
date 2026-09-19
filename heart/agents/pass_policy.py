"""JIT-compatible rule baselines for choosing three cards to pass."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from heart.cards import (
    CARD_RANKS,
    CARD_SUITS,
    CLUBS,
    DIAMONDS,
    HEARTS,
    QUEEN_OF_SPADES,
    SPADES,
    TWO_OF_CLUBS,
)
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

        danger = ranks
        danger += jnp.where(hand_cards == QUEEN_OF_SPADES, 30.0, 0.0)
        danger += jnp.where((suits == SPADES) & (ranks >= 10), 8.0, 0.0)
        danger += jnp.where(suits == HEARTS, 3.0 + ranks / 4.0, 0.0)

        if self.difficulty != "easy":
            spade_count = jnp.sum(suits == SPADES)
            owns_queen = observation.game.hand[QUEEN_OF_SPADES]
            queen_is_exposed = owns_queen & (spade_count <= 4)
            queen_adjustment = jnp.where(queen_is_exposed, 50.0, -55.0)
            danger += jnp.where(hand_cards == QUEEN_OF_SPADES, queen_adjustment, 0.0)

            # Retain low spades as cover for a held Q, K, or A.
            direction_weight = jnp.asarray((1.0, 1.15, 1.1, 0.0))[
                observation.pass_direction
            ]
            high_spade = (suits == SPADES) & (ranks >= 11)
            danger += jnp.where(high_spade, 16.0 * direction_weight, 0.0)

        selected_suits = suits[combos]
        logits = jnp.sum(danger[combos], axis=1)
        if self.difficulty != "easy":
            selected_low_spades = jnp.sum(
                (selected_suits == SPADES) & (ranks[combos] <= 7), axis=1
            )
            needs_spade_cover = observation.game.hand[QUEEN_OF_SPADES] | jnp.any(
                (suits == SPADES) & (ranks >= 11)
            )
            logits -= jnp.where(needs_spade_cover, 9.0 * selected_low_spades, 0.0)

        if self.difficulty == "hard":
            suit_counts = jnp.bincount(suits, length=4)
            selected_counts = jnp.stack(
                [jnp.sum(selected_suits == suit, axis=1) for suit in range(4)],
                axis=1,
            )
            creates_void = (suit_counts[None, :] > 0) & (
                selected_counts == suit_counts[None, :]
            )
            # Avoid passing 2C when creating a club void.
            logits += 13.0 * creates_void[:, DIAMONDS]
            logits += (
                8.0 * creates_void[:, CLUBS] * (~observation.game.hand[TWO_OF_CLUBS])
            )

        logits += jax.random.uniform(key, (NUM_PASS_ACTIONS,), maxval=1e-3)

        masked = jnp.where(observation.pass_action_mask, logits, -jnp.inf)
        return jnp.argmax(masked).astype(jnp.int32)


def make_rule_pass_policy(difficulty: str = "medium") -> RulePassPolicy:
    return RulePassPolicy(difficulty=difficulty)
