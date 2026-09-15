"""Pure Hearts rules and state transitions."""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp
from jax import Array

from heart.cards import (
    CARD_RANKS,
    CARD_SUITS,
    CARDS_PER_PLAYER,
    HEART_MASK,
    HEARTS,
    NUM_CARDS,
    NUM_PLAYERS,
    POINT_CARD_MASK,
    QUEEN_OF_SPADES,
    SUIT_MASKS,
    TRICKS_PER_DEAL,
    TWO_OF_CLUBS,
)
from heart.config import SINGLE, SingleDealRules
from heart.types import Info, Observation, State


def _penalty_values(rules: SingleDealRules) -> Array:
    values = HEART_MASK.astype(jnp.int16)
    return values.at[QUEEN_OF_SPADES].set(rules.queen_of_spades_penalty)


def reset(key: Array, rules: SingleDealRules = SINGLE) -> tuple[State, Observation]:
    """Shuffle and deal one fixed-horizon game."""

    deck = jax.random.permutation(key, NUM_CARDS)
    owners = jnp.repeat(jnp.arange(NUM_PLAYERS), CARDS_PER_PLAYER)
    hands = jnp.zeros((NUM_PLAYERS, NUM_CARDS), dtype=jnp.bool_)
    hands = hands.at[owners, deck].set(True)
    opening_player = jnp.argmax(hands[:, TWO_OF_CLUBS]).astype(jnp.int32)

    state = State(
        hands=hands,
        current_trick=jnp.full((NUM_PLAYERS,), -1, dtype=jnp.int8),
        trick_history=jnp.full((TRICKS_PER_DEAL, NUM_PLAYERS), -1, dtype=jnp.int8),
        trick_winners=jnp.full((TRICKS_PER_DEAL,), -1, dtype=jnp.int8),
        penalties=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
        scores=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
        moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
        winner_mask=jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
        leader=opening_player,
        active_player=opening_player,
        trick_index=jnp.asarray(0, dtype=jnp.int32),
        trick_position=jnp.asarray(0, dtype=jnp.int32),
        hearts_broken=jnp.asarray(False),
        num_cards_played=jnp.asarray(0, dtype=jnp.int32),
        terminated=jnp.asarray(False),
    )
    return state, observe(state, opening_player, rules)


def legal_action_mask(state: State, rules: SingleDealRules = SINGLE) -> Array:
    """Return a 52-way mask for the current player's legal cards."""

    hand = state.hands[state.active_player]
    first_action = state.num_cards_played == 0
    leading = state.trick_position == 0

    led_card = jnp.maximum(state.current_trick[0].astype(jnp.int32), 0)
    led_suit = CARD_SUITS[led_card].astype(jnp.int32)
    following_cards = hand & SUIT_MASKS[led_suit]
    must_follow = (~leading) & jnp.any(following_cards)
    legal = jnp.where(must_follow, following_cards, hand)

    non_hearts = legal & ~HEART_MASK
    must_avoid_heart_lead = (
        leading
        & (~state.hearts_broken)
        & rules.require_hearts_broken
        & jnp.any(non_hearts)
    )
    legal = jnp.where(must_avoid_heart_lead, non_hearts, legal)

    non_points = legal & ~POINT_CARD_MASK
    must_avoid_first_points = (
        (state.trick_index == 0) & rules.forbid_first_trick_points & jnp.any(non_points)
    )
    legal = jnp.where(must_avoid_first_points, non_points, legal)

    forced_open = jnp.arange(NUM_CARDS) == TWO_OF_CLUBS
    legal = jnp.where(first_action, forced_open, legal)
    return legal & ~state.terminated


def observe(
    state: State,
    player: Array | int,
    rules: SingleDealRules = SINGLE,
) -> Observation:
    """Project omniscient state into a legal player observation."""

    if not isinstance(player, jax.core.Tracer):
        try:
            concrete_player = operator.index(player)
        except TypeError as error:
            raise TypeError("player must be a scalar integer") from error
        if not 0 <= concrete_player < NUM_PLAYERS:
            raise ValueError(f"player must be in [0, {NUM_PLAYERS})")

    raw_player = jnp.asarray(player)
    is_scalar_integer = (
        raw_player.ndim == 0
        and jnp.issubdtype(raw_player.dtype, jnp.integer)
        and not jnp.issubdtype(raw_player.dtype, jnp.bool_)
    )
    player = (
        raw_player.astype(jnp.int32)
        if is_scalar_integer
        else jnp.asarray(0, dtype=jnp.int32)
    )
    valid_player = (
        jnp.asarray(is_scalar_integer) & (player >= 0) & (player < NUM_PLAYERS)
    )
    safe_player = jnp.clip(player, 0, NUM_PLAYERS - 1)
    is_active = valid_player & (player == state.active_player)
    mask = jnp.where(is_active, legal_action_mask(state, rules), False)
    return Observation(
        player=player,
        hand=jnp.where(valid_player, state.hands[safe_player], False),
        hand_sizes=jnp.sum(state.hands, axis=1, dtype=jnp.int16),
        current_trick=state.current_trick,
        trick_history=state.trick_history,
        trick_winners=state.trick_winners,
        penalties=state.penalties,
        scores=state.scores,
        leader=state.leader,
        trick_index=state.trick_index,
        trick_position=state.trick_position,
        hearts_broken=state.hearts_broken,
        action_mask=mask,
    )


def settle_deal(
    penalties: Array,
    rules: SingleDealRules = SINGLE,
) -> tuple[Array, Array, Array, Array]:
    """Apply moon scoring and return scores, shooter, winners, and rewards."""

    total_points = 13 + rules.queen_of_spades_penalty
    candidate = jnp.argmax(penalties).astype(jnp.int32)
    shot_moon = rules.shooting_the_moon & (penalties[candidate] == total_points)
    moon_shooter = jnp.where(shot_moon, candidate, -1).astype(jnp.int8)
    moon_scores = (
        jnp.full((NUM_PLAYERS,), total_points, dtype=jnp.int16).at[candidate].set(0)
    )
    scores = jnp.where(shot_moon, moon_scores, penalties)
    winner_mask = scores == jnp.min(scores)
    moon_rewards = jnp.where(
        jnp.arange(NUM_PLAYERS) == candidate,
        0.0,
        -jnp.asarray(total_points, dtype=jnp.float32),
    )
    rewards = jnp.where(shot_moon, moon_rewards, relative_rewards(scores))
    return scores, moon_shooter, winner_mask, rewards


def relative_rewards(scores: Array, normalizer: float = 1.0) -> Array:
    """Return mathematically zero-sum opponent-relative rewards."""

    float_scores = scores.astype(jnp.float32)
    total = jnp.sum(float_scores, dtype=jnp.float32)
    rewards = ((total - float_scores) / (NUM_PLAYERS - 1) - float_scores) / normalizer
    return rewards


def _apply_legal_action(
    state: State,
    action: Array,
    rules: SingleDealRules,
) -> tuple[State, Array, Info]:
    player = state.active_player
    position = state.trick_position
    hands = state.hands.at[player, action].set(False)
    current_trick = state.current_trick.at[position].set(action.astype(jnp.int8))
    hearts_broken = state.hearts_broken | (CARD_SUITS[action] == HEARTS)
    trick_completed = position == NUM_PLAYERS - 1

    def continue_trick(_: None) -> tuple[State, Array, Info]:
        next_state = state._replace(
            hands=hands,
            current_trick=current_trick,
            active_player=(player + 1) % NUM_PLAYERS,
            trick_position=position + 1,
            hearts_broken=hearts_broken,
            num_cards_played=state.num_cards_played + 1,
        )
        info = Info(
            invalid_action=jnp.asarray(False),
            trick_completed=jnp.asarray(False),
            deal_completed=jnp.asarray(False),
            trick_winner=jnp.asarray(-1, dtype=jnp.int32),
            points_won=jnp.asarray(0, dtype=jnp.int16),
            moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
            winner_mask=jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
        )
        return next_state, jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32), info

    def finish_trick(_: None) -> tuple[State, Array, Info]:
        led_suit = CARD_SUITS[current_trick[0]]
        eligible = CARD_SUITS[current_trick] == led_suit
        ranks = jnp.where(eligible, CARD_RANKS[current_trick], -1)
        winner_offset = jnp.argmax(ranks).astype(jnp.int32)
        winner = (state.leader + winner_offset) % NUM_PLAYERS
        points = jnp.sum(_penalty_values(rules)[current_trick], dtype=jnp.int16)
        penalties = state.penalties.at[winner].add(points)
        deal_completed = state.trick_index == TRICKS_PER_DEAL - 1
        settled_scores, moon_shooter, winner_mask, terminal_rewards = settle_deal(
            penalties, rules
        )
        scores = jnp.where(deal_completed, settled_scores, penalties)

        next_state = state._replace(
            hands=hands,
            current_trick=jnp.full((NUM_PLAYERS,), -1, dtype=jnp.int8),
            trick_history=state.trick_history.at[state.trick_index].set(current_trick),
            trick_winners=state.trick_winners.at[state.trick_index].set(
                winner.astype(jnp.int8)
            ),
            penalties=penalties,
            scores=scores,
            moon_shooter=jnp.where(deal_completed, moon_shooter, -1).astype(jnp.int8),
            winner_mask=jnp.where(
                deal_completed,
                winner_mask,
                jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
            ),
            leader=winner,
            active_player=winner,
            trick_index=state.trick_index + 1,
            trick_position=jnp.asarray(0, dtype=jnp.int32),
            hearts_broken=hearts_broken,
            num_cards_played=state.num_cards_played + 1,
            terminated=deal_completed,
        )
        rewards = jnp.where(
            deal_completed,
            terminal_rewards,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
        )
        info = Info(
            invalid_action=jnp.asarray(False),
            trick_completed=jnp.asarray(True),
            deal_completed=deal_completed,
            trick_winner=winner,
            points_won=points,
            moon_shooter=jnp.where(deal_completed, moon_shooter, -1).astype(jnp.int8),
            winner_mask=jnp.where(
                deal_completed,
                winner_mask,
                jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
            ),
        )
        return next_state, rewards, info

    return jax.lax.cond(trick_completed, finish_trick, continue_trick, operand=None)


def step(
    state: State,
    action: Array | int,
    rules: SingleDealRules = SINGLE,
) -> tuple[State, Observation, Array, Array, Info]:
    """Play one card; invalid actions leave the state unchanged."""

    raw_action = jnp.asarray(action)
    is_scalar_integer = (
        raw_action.ndim == 0
        and jnp.issubdtype(raw_action.dtype, jnp.integer)
        and not jnp.issubdtype(raw_action.dtype, jnp.bool_)
    )
    if is_scalar_integer:
        in_bounds = (raw_action >= 0) & (raw_action < NUM_CARDS)
        action = raw_action.astype(jnp.int32)
    else:
        in_bounds = jnp.asarray(False)
        action = jnp.asarray(0, dtype=jnp.int32)
    safe_action = jnp.clip(action, 0, NUM_CARDS - 1)
    mask = legal_action_mask(state, rules)
    is_legal = in_bounds & mask[safe_action]

    def apply(_: None) -> tuple[State, Array, Info]:
        return _apply_legal_action(state, safe_action, rules)

    def reject(_: None) -> tuple[State, Array, Info]:
        info = Info(
            invalid_action=jnp.asarray(True),
            trick_completed=jnp.asarray(False),
            deal_completed=jnp.asarray(False),
            trick_winner=jnp.asarray(-1, dtype=jnp.int32),
            points_won=jnp.asarray(0, dtype=jnp.int16),
            moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
            winner_mask=jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
        )
        return state, jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32), info

    next_state, rewards, info = jax.lax.cond(is_legal, apply, reject, operand=None)
    observation = observe(next_state, next_state.active_player, rules)
    return next_state, observation, rewards, next_state.terminated, info


def step_unchecked(
    state: State,
    action: Array | int,
    rules: SingleDealRules = SINGLE,
) -> tuple[State, Observation, Array, Array, Info]:
    """Play an action already proven legal by the caller.

    This trusted rollout path skips bounds, dtype, ownership, and rule-mask
    validation. Passing anything except a scalar legal integer card ID has
    undefined behavior; use :func:`step` at untrusted API boundaries.
    """

    action = jnp.asarray(action, dtype=jnp.int32)
    next_state, rewards, info = _apply_legal_action(state, action, rules)
    observation = observe(next_state, next_state.active_player, rules)
    return next_state, observation, rewards, next_state.terminated, info
