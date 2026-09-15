"""JAX-native multi-deal classic Hearts rules."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from heart.cards import NUM_CARDS, NUM_PLAYERS, TWO_OF_CLUBS
from heart.config import SingleDealRules
from heart.rules import legal_action_mask, observe, relative_rewards
from heart.rules import reset as reset_deal
from heart.rules import step as step_deal
from heart.types import Observation, State

CLASSIC_V0 = "classic-v0"
PASS = 0
PLAY = 1
TERMINAL = 2
PASS_LEFT = 0
PASS_RIGHT = 1
PASS_ACROSS = 2
PASS_HOLD = 3
PASS_NAMES = ("left", "right", "across", "hold")
PASS_COMBINATIONS = jnp.asarray(
    tuple(combinations(range(13), 3)),
    dtype=jnp.int8,
)
NUM_PASS_ACTIONS = 286
MAX_CLASSIC_DEALS = 16
MAX_CLASSIC_CORE_STEPS = 880


@dataclass(frozen=True)
class ClassicRules:
    """Versioned standard match configuration."""

    target_score: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.target_score, bool) or not isinstance(
            self.target_score, Integral
        ):
            raise TypeError("target_score must be an integer")
        if self.target_score != 100:
            raise ValueError("classic-v0 fixes target_score at 100")

    @property
    def deal_rules(self) -> SingleDealRules:
        return SingleDealRules(
            queen_of_spades_penalty=13,
            shooting_the_moon=True,
        )


CLASSIC = ClassicRules()


class ClassicState(NamedTuple):
    """Omniscient fixed-shape state for a complete game to 100 points."""

    game: State
    rng_key: Array
    match_scores: Array
    deal_index: Array
    phase: Array
    active_player: Array
    pass_direction: Array
    pass_cards: Array
    cards_received: Array
    last_deal_scores: Array
    last_deal_rewards: Array
    last_deal_moon_shooter: Array
    last_trick_cards: Array
    last_trick_winner: Array
    deal_boundary: Array
    winner_mask: Array
    terminated: Array


class ClassicObservation(NamedTuple):
    """Private observation with separate pass and play action masks."""

    game: Observation
    match_scores: Array
    deal_index: Array
    phase: Array
    pass_direction: Array
    cards_passed: Array
    cards_received: Array
    pass_action_mask: Array
    play_action_mask: Array


class ClassicInfo(NamedTuple):
    """Diagnostics for one pass selection or card play."""

    invalid_action: Array
    trick_completed: Array
    deal_completed: Array
    match_completed: Array
    trick_winner: Array
    points_won: Array
    deal_scores: Array
    match_scores: Array
    moon_shooter: Array
    winner_mask: Array


def _empty_classic_info(
    state: ClassicState,
    *,
    invalid: bool = False,
) -> ClassicInfo:
    return ClassicInfo(
        invalid_action=jnp.asarray(invalid),
        trick_completed=jnp.asarray(False),
        deal_completed=jnp.asarray(False),
        match_completed=jnp.asarray(False),
        trick_winner=jnp.asarray(-1, dtype=jnp.int32),
        points_won=jnp.asarray(0, dtype=jnp.int16),
        deal_scores=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
        match_scores=state.match_scores,
        moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
        winner_mask=state.winner_mask,
    )


def _relative_rewards(scores: Array) -> Array:
    return relative_rewards(scores, normalizer=26.0)


def _pass_direction(deal_index: Array) -> Array:
    return jnp.mod(deal_index, 4).astype(jnp.int8)


def _new_deal_state(
    deal_key: Array,
    next_key: Array,
    match_scores: Array,
    deal_index: Array,
    rules: ClassicRules,
) -> ClassicState:
    game, _ = reset_deal(deal_key, rules.deal_rules)
    direction = _pass_direction(deal_index)
    holding = direction == PASS_HOLD
    phase = jnp.where(holding, PLAY, PASS).astype(jnp.int8)
    active = jnp.where(holding, game.active_player, 0).astype(jnp.int32)
    empty_cards = jnp.full((NUM_PLAYERS, 3), -1, dtype=jnp.int8)
    return ClassicState(
        game=game,
        rng_key=next_key,
        match_scores=match_scores,
        deal_index=deal_index.astype(jnp.int8),
        phase=phase,
        active_player=active,
        pass_direction=direction,
        pass_cards=empty_cards,
        cards_received=empty_cards,
        last_deal_scores=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
        last_deal_rewards=jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
        last_deal_moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
        last_trick_cards=jnp.full((NUM_PLAYERS,), -1, dtype=jnp.int8),
        last_trick_winner=jnp.asarray(-1, dtype=jnp.int8),
        deal_boundary=jnp.asarray(False),
        winner_mask=jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
        terminated=jnp.asarray(False),
    )


def reset(
    key: Array,
    rules: ClassicRules = CLASSIC,
) -> tuple[ClassicState, ClassicObservation]:
    """Start a deterministic classic match from one explicit key."""

    deal_key, next_key = jax.random.split(key)
    state = _new_deal_state(
        deal_key,
        next_key,
        jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
        jnp.asarray(0, dtype=jnp.int8),
        rules,
    )
    return state, observe_classic(state, state.active_player, rules)


def observe_classic(
    state: ClassicState,
    player: Array | int,
    rules: ClassicRules = CLASSIC,
) -> ClassicObservation:
    """Project classic state without exposing other players' pass choices."""

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
    game_observation = observe(state.game, raw_player, rules.deal_rules)
    is_actor = valid_player & (~state.terminated) & (player == state.active_player)
    pass_mask = jnp.full((NUM_PASS_ACTIONS,), is_actor & (state.phase == PASS))
    play_mask = jnp.where(
        is_actor & (state.phase == PLAY),
        game_observation.action_mask,
        False,
    )
    game_observation = game_observation._replace(action_mask=play_mask)
    safe_player = jnp.clip(player, 0, NUM_PLAYERS - 1)
    return ClassicObservation(
        game=game_observation,
        match_scores=state.match_scores,
        deal_index=state.deal_index,
        phase=state.phase,
        pass_direction=state.pass_direction,
        cards_passed=jnp.where(valid_player, state.pass_cards[safe_player], -1),
        cards_received=jnp.where(valid_player, state.cards_received[safe_player], -1),
        pass_action_mask=pass_mask,
        play_action_mask=play_mask,
    )


def _apply_pass(
    state: ClassicState,
    action: Array,
    rules: ClassicRules,
) -> tuple[ClassicState, Array, ClassicInfo]:
    player = state.active_player
    hand_cards = jnp.nonzero(
        state.game.hands[player],
        size=13,
        fill_value=-1,
    )[0]
    selected = hand_cards[PASS_COMBINATIONS[action]].astype(jnp.int8)
    pass_cards = state.pass_cards.at[player].set(selected)
    all_selected = player == NUM_PLAYERS - 1

    def wait(_: None) -> tuple[ClassicState, Array, ClassicInfo]:
        next_state = state._replace(
            active_player=player + 1,
            pass_cards=pass_cards,
            deal_boundary=jnp.asarray(False),
        )
        return (
            next_state,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            _empty_classic_info(next_state),
        )

    def exchange(_: None) -> tuple[ClassicState, Array, ClassicInfo]:
        players = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
        recipients = jnp.where(
            state.pass_direction == PASS_LEFT,
            (players + 1) % NUM_PLAYERS,
            jnp.where(
                state.pass_direction == PASS_RIGHT,
                (players - 1) % NUM_PLAYERS,
                (players + 2) % NUM_PLAYERS,
            ),
        )
        selected_mask = (
            jnp.zeros_like(state.game.hands)
            .at[players[:, None], pass_cards.astype(jnp.int32)]
            .set(True)
        )
        hands = state.game.hands & ~selected_mask
        hands = hands.at[recipients[:, None], pass_cards.astype(jnp.int32)].set(True)
        received = jnp.full_like(pass_cards, -1).at[recipients].set(pass_cards)
        opener = jnp.argmax(hands[:, TWO_OF_CLUBS]).astype(jnp.int32)
        game = state.game._replace(
            hands=hands,
            leader=opener,
            active_player=opener,
        )
        next_state = state._replace(
            game=game,
            phase=jnp.asarray(PLAY, dtype=jnp.int8),
            active_player=opener,
            pass_cards=pass_cards,
            cards_received=received,
            deal_boundary=jnp.asarray(False),
        )
        return (
            next_state,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            _empty_classic_info(next_state),
        )

    return jax.lax.cond(all_selected, exchange, wait, operand=None)


def _apply_play(
    state: ClassicState,
    action: Array,
    rules: ClassicRules,
) -> tuple[ClassicState, Array, ClassicInfo]:
    game, _, _, _, core_info = step_deal(state.game, action, rules.deal_rules)

    def continue_deal(_: None) -> tuple[ClassicState, Array, ClassicInfo]:
        next_state = state._replace(
            game=game,
            active_player=game.active_player,
            deal_boundary=jnp.asarray(False),
        )
        info = ClassicInfo(
            invalid_action=core_info.invalid_action,
            trick_completed=core_info.trick_completed,
            deal_completed=jnp.asarray(False),
            match_completed=jnp.asarray(False),
            trick_winner=core_info.trick_winner,
            points_won=core_info.points_won,
            deal_scores=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int16),
            match_scores=state.match_scores,
            moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
            winner_mask=state.winner_mask,
        )
        return next_state, jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32), info

    def finish_deal(_: None) -> tuple[ClassicState, Array, ClassicInfo]:
        deal_scores = game.scores.astype(jnp.int16)
        rewards = _relative_rewards(deal_scores)
        match_scores = state.match_scores + deal_scores
        match_completed = jnp.any(match_scores >= rules.target_score)
        winner_mask = (match_scores == jnp.min(match_scores)) & match_completed

        def finish_match(_: None) -> ClassicState:
            return state._replace(
                game=game,
                match_scores=match_scores,
                phase=jnp.asarray(TERMINAL, dtype=jnp.int8),
                active_player=game.active_player,
                last_deal_scores=deal_scores,
                last_deal_rewards=rewards,
                last_deal_moon_shooter=game.moon_shooter,
                last_trick_cards=game.trick_history[-1],
                last_trick_winner=core_info.trick_winner.astype(jnp.int8),
                deal_boundary=jnp.asarray(True),
                winner_mask=winner_mask,
                terminated=jnp.asarray(True),
            )

        def start_next(_: None) -> ClassicState:
            deal_key, next_key = jax.random.split(state.rng_key)
            fresh = _new_deal_state(
                deal_key,
                next_key,
                match_scores,
                state.deal_index + jnp.asarray(1, dtype=jnp.int8),
                rules,
            )
            return fresh._replace(
                last_deal_scores=deal_scores,
                last_deal_rewards=rewards,
                last_deal_moon_shooter=game.moon_shooter,
                last_trick_cards=game.trick_history[-1],
                last_trick_winner=core_info.trick_winner.astype(jnp.int8),
                deal_boundary=jnp.asarray(True),
            )

        next_state = jax.lax.cond(
            match_completed,
            finish_match,
            start_next,
            operand=None,
        )
        info = ClassicInfo(
            invalid_action=jnp.asarray(False),
            trick_completed=core_info.trick_completed,
            deal_completed=jnp.asarray(True),
            match_completed=match_completed,
            trick_winner=core_info.trick_winner,
            points_won=core_info.points_won,
            deal_scores=deal_scores,
            match_scores=match_scores,
            moon_shooter=game.moon_shooter,
            winner_mask=winner_mask,
        )
        return next_state, rewards, info

    return jax.lax.cond(game.terminated, finish_deal, continue_deal, operand=None)


def step(
    state: ClassicState,
    action: Array | int,
    rules: ClassicRules = CLASSIC,
) -> tuple[ClassicState, ClassicObservation, Array, Array, ClassicInfo]:
    """Apply one phase-dependent scalar action with fail-closed validation."""

    raw_action = jnp.asarray(action)
    is_integer = (
        raw_action.ndim == 0
        and jnp.issubdtype(raw_action.dtype, jnp.integer)
        and not jnp.issubdtype(raw_action.dtype, jnp.bool_)
    )
    action = (
        raw_action.astype(jnp.int32) if is_integer else jnp.asarray(0, dtype=jnp.int32)
    )
    pass_in_bounds = (action >= 0) & (action < NUM_PASS_ACTIONS)
    play_in_bounds = (action >= 0) & (action < NUM_CARDS)
    safe_card = jnp.clip(action, 0, NUM_CARDS - 1)
    play_legal = legal_action_mask(state.game, rules.deal_rules)[safe_card]
    valid = (
        jnp.asarray(is_integer)
        & ~state.terminated
        & jnp.where(
            state.phase == PASS,
            pass_in_bounds,
            (state.phase == PLAY) & play_in_bounds & play_legal,
        )
    )

    def apply(_: None):
        return jax.lax.cond(
            state.phase == PASS,
            lambda __: _apply_pass(state, action, rules),
            lambda __: _apply_play(state, action, rules),
            operand=None,
        )

    def reject(_: None):
        return (
            state,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            _empty_classic_info(state, invalid=True),
        )

    next_state, rewards, info = jax.lax.cond(valid, apply, reject, operand=None)
    observation = observe_classic(
        next_state,
        next_state.active_player,
        rules,
    )
    return next_state, observation, rewards, next_state.terminated, info


@dataclass(frozen=True)
class ClassicEnv:
    """Stateless handle for the complete classic-v0 match."""

    rules: ClassicRules = CLASSIC
    mode: str = CLASSIC_V0

    def reset(self, key: Array) -> tuple[ClassicState, ClassicObservation]:
        return reset(key, self.rules)

    def step(
        self,
        state: ClassicState,
        action: Array | int,
    ) -> tuple[ClassicState, ClassicObservation, Array, Array, ClassicInfo]:
        return step(state, action, self.rules)

    def observe(
        self,
        state: ClassicState,
        player: Array | int,
    ) -> ClassicObservation:
        return observe_classic(state, player, self.rules)


def make_classic(**rule_overrides: object) -> ClassicEnv:
    return ClassicEnv(rules=ClassicRules(**rule_overrides))
