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


def _base_scores(observation: Observation, key: Array) -> Array:
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


# Use a lower alert threshold while all recorded points are hearts.
MOON_ALERT_HEARTS_ONLY = 4
MOON_ALERT_WITH_QUEEN = 13

# Medium defends probabilistically; hard responds to every detected threat.
MOON_ALERTNESS = {"easy": 0.0, "medium": 0.5, "hard": 1.0}


def _moon_threat(observation: Observation) -> Array:
    """One opponent holds every point dealt so far, and enough to matter."""

    taken = observation.penalties.astype(jnp.float32)
    total = jnp.sum(taken)
    top = jnp.argmax(taken)
    queen_out = jnp.any(observation.trick_history == QUEEN_OF_SPADES) | jnp.any(
        observation.current_trick == QUEEN_OF_SPADES
    )
    alert = jnp.where(queen_out, MOON_ALERT_WITH_QUEEN, MOON_ALERT_HEARTS_ONLY)
    return (
        (total >= alert)
        & (taken[top] == total)
        & (top != observation.player.astype(jnp.int32))
    )


def _moon_guard_scores(
    observation: Observation, key: Array, scores: Array, alertness: float
) -> Array:
    """Prefer actions that can break an opponent's moon shot."""

    ranks = CARD_RANKS.astype(jnp.float32)
    current = observation.current_trick
    safe_current = jnp.clip(current.astype(jnp.int32), 0, NUM_CARDS - 1)
    played = current >= 0
    led_card = jnp.maximum(current[0].astype(jnp.int32), 0)
    led_suit = CARD_SUITS[led_card]
    follows = CARD_SUITS == led_suit
    same_suit = played & (CARD_SUITS[safe_current] == led_suit)
    current_high = jnp.max(jnp.where(same_suit, CARD_RANKS[safe_current], -1))
    beats_table = follows & (CARD_RANKS > current_high)
    table_has_points = jnp.any(jnp.where(played, POINT_CARD_MASK[safe_current], False))
    can_follow = jnp.any(observation.action_mask & follows)
    awake = jax.random.uniform(jax.random.fold_in(key, 7)) < alertness
    alarmed = _moon_threat(observation) & awake

    # Keep point cards when void so the shooter cannot collect them.
    hoard = alarmed & ~can_follow & POINT_CARD_MASK
    scores = scores + jnp.where(hoard, -500.0, 0.0)

    # Earlier seats may win a clean trick before another player adds points.
    breaking = (
        alarmed
        & (observation.trick_position > 0)
        & beats_table
        & (table_has_points | (observation.trick_position < 3))
    )
    return scores + jnp.where(breaking, 400.0 + ranks, 0.0)


def _spade_pressure_scores(
    observation: Observation, key: Array, alertness: float
) -> Array:
    scores = _moon_guard_scores(
        observation, key, _base_scores(observation, key), alertness
    )
    ranks = CARD_RANKS.astype(jnp.float32)
    history = observation.trick_history.reshape(-1)
    queen_seen = jnp.any(history == QUEEN_OF_SPADES) | jnp.any(
        observation.current_trick == QUEEN_OF_SPADES
    )
    pressure = (
        (observation.trick_position == 0)
        & (observation.trick_index >= 1)
        & (observation.trick_index <= 5)
        & ~queen_seen
        & ~observation.hand[QUEEN_OF_SPADES]
        & (CARD_SUITS == 2)
        & (CARD_RANKS <= 9)
    )
    has_pressure = jnp.any(observation.action_mask & pressure)
    scores += jnp.where(has_pressure & pressure, 250.0 + ranks, 0.0)
    return scores


def _hard_scores(observation: Observation, key: Array, alertness: float) -> Array:
    scores = _spade_pressure_scores(observation, key, alertness)
    ranks = CARD_RANKS.astype(jnp.float32)
    leading = observation.trick_position == 0
    current = observation.current_trick
    safe_current = jnp.clip(current.astype(jnp.int32), 0, NUM_CARDS - 1)
    current_played = current >= 0
    led_card = jnp.maximum(current[0].astype(jnp.int32), 0)
    led_suit = CARD_SUITS[led_card]
    follows = CARD_SUITS == led_suit
    same_suit = current_played & (CARD_SUITS[safe_current] == led_suit)
    current_high = jnp.max(jnp.where(same_suit, CARD_RANKS[safe_current], -1))
    losing = follows & (CARD_RANKS < current_high)
    can_lose = jnp.any(observation.action_mask & losing)
    table_has_points = jnp.any(
        jnp.where(current_played, POINT_CARD_MASK[safe_current], False)
    )
    scores += jnp.where((observation.trick_index == 0) & ~leading, 300.0 + ranks, 0.0)
    forced_clean = (
        (observation.trick_position == 3) & ~table_has_points & ~can_lose & follows
    )
    scores += jnp.where(forced_clean, 300.0 + ranks, 0.0)
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
        alertness = MOON_ALERTNESS[self.difficulty]
        if self.difficulty == "easy":
            scores = _base_scores(observation, key)
        elif self.difficulty == "medium":
            scores = _spade_pressure_scores(observation, key, alertness)
        else:
            scores = _hard_scores(observation, key, alertness)
        masked = jnp.where(observation.action_mask, scores, -jnp.inf)
        return jnp.argmax(masked).astype(jnp.int32)


def make_rule_policy(difficulty: str = "medium") -> RulePolicy:
    """Create a built-in policy by stable difficulty name."""

    return RulePolicy(difficulty=difficulty)
