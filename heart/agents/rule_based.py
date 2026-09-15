"""JIT-compatible rule-based baseline policies."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from heart.cards import (
    CARD_RANKS,
    CARD_SUITS,
    HEARTS,
    NUM_CARDS,
    POINT_CARD_MASK,
    QUEEN_OF_SPADES,
)
from heart.types import Observation

DIFFICULTIES = ("easy", "medium", "hard")


def _random_tiebreak(key: Array) -> Array:
    return jax.random.uniform(key, (NUM_CARDS,), minval=0.0, maxval=1e-3)


def _easy_scores(observation: Observation, key: Array) -> Array:
    del observation
    return -CARD_RANKS.astype(jnp.float32) + _random_tiebreak(key)


def _medium_scores(observation: Observation, key: Array) -> Array:
    cards = jnp.arange(NUM_CARDS)
    ranks = CARD_RANKS.astype(jnp.float32)
    position = observation.trick_position
    leading = position == 0
    led_card = jnp.maximum(observation.current_trick[0].astype(jnp.int32), 0)
    led_suit = CARD_SUITS[led_card]
    current_cards = observation.current_trick
    safe_current = jnp.maximum(current_cards.astype(jnp.int32), 0)
    played = current_cards >= 0
    same_suit = played & (CARD_SUITS[safe_current] == led_suit)
    current_high = jnp.max(jnp.where(same_suit, CARD_RANKS[safe_current], -1))
    follows = CARD_SUITS == led_suit
    losing = follows & (CARD_RANKS < current_high)
    can_follow = jnp.any(observation.action_mask & follows)
    can_lose = jnp.any(observation.action_mask & losing)

    lead_scores = -ranks
    forced_win_scores = -ranks
    duck_scores = jnp.where(losing, 100.0 + ranks, forced_win_scores)
    follow_scores = jnp.where(can_lose, duck_scores, forced_win_scores)
    discard_scores = ranks
    discard_scores += jnp.where(CARD_SUITS == HEARTS, 100.0, 0.0)
    discard_scores += jnp.where(cards == QUEEN_OF_SPADES, 250.0, 0.0)
    scores = jnp.where(
        leading,
        lead_scores,
        jnp.where(can_follow, follow_scores, discard_scores),
    )
    return scores + _random_tiebreak(key)


def _hard_scores(observation: Observation, key: Array) -> Array:
    scores = _medium_scores(observation, key)
    ranks = CARD_RANKS.astype(jnp.float32)
    position = observation.trick_position
    leading = position == 0

    total_taken = jnp.sum(observation.penalties)
    runner = jnp.argmax(observation.penalties)
    runner_points = observation.penalties[runner]
    moon_threat = (total_taken > 0) & (runner_points == total_taken)
    defending = moon_threat & (runner != observation.player)
    shooting = moon_threat & (runner == observation.player)

    led_card = jnp.maximum(observation.current_trick[0].astype(jnp.int32), 0)
    led_suit = CARD_SUITS[led_card]
    safe_current = jnp.maximum(observation.current_trick.astype(jnp.int32), 0)
    played = observation.current_trick >= 0
    same_suit = played & (CARD_SUITS[safe_current] == led_suit)
    current_high = jnp.max(jnp.where(same_suit, CARD_RANKS[safe_current], -1))
    wins_now = (CARD_SUITS == led_suit) & (CARD_RANKS > current_high)
    table_has_points = jnp.any(
        jnp.where(played, POINT_CARD_MASK[safe_current], False)
    )

    block_now = defending & (position == 3) & table_has_points & wins_now
    scores += jnp.where(block_now, 500.0 - ranks, 0.0)

    current_winner_offset = jnp.argmax(
        jnp.where(same_suit, CARD_RANKS[safe_current], -1)
    )
    current_winner = (observation.leader + current_winner_offset) % 4
    follows = CARD_SUITS == led_suit
    void_in_led = (~leading) & ~jnp.any(observation.action_mask & follows)
    spoil = defending & void_in_led & (current_winner != runner) & POINT_CARD_MASK
    scores += jnp.where(spoil, 400.0 + ranks, 0.0)

    pursue = shooting & (~leading) & table_has_points & wins_now
    scores += jnp.where(pursue, 350.0 + ranks, 0.0)
    scores += jnp.where(shooting & leading, 2.0 * ranks, 0.0)
    return scores


@dataclass(frozen=True)
class RulePolicy:
    """A selectable tactical policy with randomized tie-breaking."""

    difficulty: str = "medium"

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(
                f"unknown difficulty {self.difficulty!r}; available: {DIFFICULTIES}"
            )

    def __call__(self, observation: Observation, key: Array) -> Array:
        if self.difficulty == "easy":
            scores = _easy_scores(observation, key)
        elif self.difficulty == "medium":
            scores = _medium_scores(observation, key)
        else:
            scores = _hard_scores(observation, key)
        masked = jnp.where(observation.action_mask, scores, -jnp.inf)
        return jnp.argmax(masked).astype(jnp.int32)


def make_rule_policy(difficulty: str = "medium") -> RulePolicy:
    """Create a built-in policy by stable difficulty name."""

    return RulePolicy(difficulty=difficulty)
