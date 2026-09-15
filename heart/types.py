"""JAX-compatible environment data structures."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class State(NamedTuple):
    """Omniscient immutable state used only inside the environment."""

    hands: Array
    current_trick: Array
    trick_history: Array
    trick_winners: Array
    penalties: Array
    scores: Array
    moon_shooter: Array
    winner_mask: Array
    leader: Array
    active_player: Array
    trick_index: Array
    trick_position: Array
    hearts_broken: Array
    num_cards_played: Array
    terminated: Array


class Observation(NamedTuple):
    """Information legally available to one player."""

    player: Array
    hand: Array
    hand_sizes: Array
    current_trick: Array
    trick_history: Array
    trick_winners: Array
    penalties: Array
    scores: Array
    leader: Array
    trick_index: Array
    trick_position: Array
    hearts_broken: Array
    action_mask: Array


class Info(NamedTuple):
    """Diagnostics returned by a transition."""

    invalid_action: Array
    trick_completed: Array
    deal_completed: Array
    trick_winner: Array
    points_won: Array
    moon_shooter: Array
    winner_mask: Array

