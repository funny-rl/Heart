"""Fixed-13-action single-learner adapter over the authoritative game engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from heart.agents import RulePolicy, make_rule_policy
from heart.cards import NUM_CARDS, NUM_PLAYERS
from heart.env import SIMPLEST_V0, HeartEnv, make
from heart.types import Info, Observation, State

HAND_SIZE = 13


class SingleAgentState(NamedTuple):
    """JAX-compatible adapter state with stable initial-hand action slots."""

    game: State
    hand_cards: Array
    opponent_key: Array
    decisions: Array


class SingleAgentObservation(NamedTuple):
    """Private controlled-player observation plus a 13-slot action interface."""

    game: Observation
    hand_cards: Array
    action_mask: Array


def _empty_info() -> Info:
    return Info(
        invalid_action=jnp.asarray(False),
        trick_completed=jnp.asarray(False),
        deal_completed=jnp.asarray(False),
        trick_winner=jnp.asarray(-1, dtype=jnp.int32),
        points_won=jnp.asarray(0, dtype=jnp.int16),
        moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
        winner_mask=jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_),
    )


@dataclass(frozen=True)
class SingleAgentEnv:
    """One controlled player against three built-in rule policies.

    External actions index the controlled player's 13 initially dealt cards.
    The mapping remains fixed for the full deal; played or currently illegal
    slots are masked out. Opponents are advanced automatically until the
    controlled player acts again or the deal terminates.
    """

    core: HeartEnv
    controlled_player: int
    opponent_players: tuple[int, int, int]
    opponent_policies: tuple[RulePolicy, RulePolicy, RulePolicy]

    @property
    def num_actions(self) -> int:
        return HAND_SIZE

    def _observation(self, game: State, hand_cards: Array) -> SingleAgentObservation:
        observation = self.core.observe(game, self.controlled_player)
        safe_cards = jnp.clip(hand_cards.astype(jnp.int32), 0, NUM_CARDS - 1)
        action_mask = (hand_cards >= 0) & observation.action_mask[safe_cards]
        return SingleAgentObservation(
            game=observation,
            hand_cards=hand_cards,
            action_mask=action_mask,
        )

    def _opponent_action(self, observation: Observation, key: Array) -> Array:
        player_to_policy = [-1] * NUM_PLAYERS
        for index, player in enumerate(self.opponent_players):
            player_to_policy[player] = index
        policy_index = jnp.asarray(player_to_policy, dtype=jnp.int32)[
            observation.player
        ]
        branches = tuple(
            (lambda _, selected=policy: selected(observation, key))
            for policy in self.opponent_policies
        )
        return jax.lax.switch(policy_index, branches, operand=None)

    def _advance_opponents(
        self,
        game: State,
        key: Array,
        rewards: Array,
        info: Info,
    ) -> tuple[State, Array, Array, Info]:
        def should_advance(carry: tuple[State, Array, Array, Info]) -> Array:
            current, _, _, _ = carry
            return (~current.terminated) & (
                current.active_player != self.controlled_player
            )

        def play_opponent(
            carry: tuple[State, Array, Array, Info],
        ) -> tuple[State, Array, Array, Info]:
            current, policy_key, cumulative_rewards, _ = carry
            policy_key, action_key = jax.random.split(policy_key)
            observation = self.core.observe(current, current.active_player)
            action = self._opponent_action(observation, action_key)
            next_game, _, step_rewards, _, step_info = self.core.step(current, action)
            return (
                next_game,
                policy_key,
                cumulative_rewards + step_rewards,
                step_info,
            )

        return jax.lax.while_loop(
            should_advance,
            play_opponent,
            (game, key, rewards, info),
        )

    def reset(self, key: Array) -> tuple[SingleAgentState, SingleAgentObservation]:
        """Deal, autoplay earlier seats, and stop at the learner's first turn."""

        deal_key, opponent_key = jax.random.split(key)
        game, _ = self.core.reset(deal_key)
        hand_cards = jnp.nonzero(
            game.hands[self.controlled_player],
            size=HAND_SIZE,
            fill_value=-1,
        )[0].astype(jnp.int16)
        game, opponent_key, _, _ = self._advance_opponents(
            game,
            opponent_key,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            _empty_info(),
        )
        state = SingleAgentState(
            game=game,
            hand_cards=hand_cards,
            opponent_key=opponent_key,
            decisions=jnp.asarray(0, dtype=jnp.int8),
        )
        return state, self._observation(game, hand_cards)

    def step(
        self, state: SingleAgentState, action: Array | int
    ) -> tuple[
        SingleAgentState,
        SingleAgentObservation,
        Array,
        Array,
        Info,
    ]:
        """Play one stable hand slot, then autoplay to the next learner turn."""

        raw_action = jnp.asarray(action)
        is_scalar_integer = (
            raw_action.ndim == 0
            and jnp.issubdtype(raw_action.dtype, jnp.integer)
            and not jnp.issubdtype(raw_action.dtype, jnp.bool_)
        )
        slot = (
            raw_action.astype(jnp.int32)
            if is_scalar_integer
            else jnp.asarray(0, dtype=jnp.int32)
        )
        in_bounds = jnp.asarray(is_scalar_integer) & (slot >= 0) & (slot < HAND_SIZE)
        safe_slot = jnp.clip(slot, 0, HAND_SIZE - 1)
        observation = self._observation(state.game, state.hand_cards)
        is_legal = in_bounds & observation.action_mask[safe_slot]
        card = state.hand_cards[safe_slot].astype(jnp.int32)

        def apply(_: None):
            game, _, rewards, _, info = self.core.step(state.game, card)
            game, opponent_key, rewards, info = self._advance_opponents(
                game,
                state.opponent_key,
                rewards,
                info,
            )
            next_state = SingleAgentState(
                game=game,
                hand_cards=state.hand_cards,
                opponent_key=opponent_key,
                decisions=state.decisions + jnp.asarray(1, dtype=jnp.int8),
            )
            return (
                next_state,
                self._observation(game, state.hand_cards),
                rewards[self.controlled_player],
                game.terminated,
                info,
            )

        def reject(_: None):
            _, _, _, _, info = self.core.step(state.game, -1)
            return (
                state,
                observation,
                jnp.asarray(0.0, dtype=jnp.float32),
                state.game.terminated,
                info,
            )

        return jax.lax.cond(is_legal, apply, reject, operand=None)


def make_single_agent(
    mode: str = SIMPLEST_V0,
    *,
    controlled_player: int = 0,
    opponents: str | Sequence[str] = "medium",
    pass_opponents: str | Sequence[str] | None = None,
    play_opponents: str | Sequence[str] | None = None,
    **rule_overrides: object,
):
    """Create a 13-action learner-versus-rules environment.

    A three-item opponent sequence is assigned to the non-controlled player IDs
    in ascending order. A single difficulty name is broadcast to all three.
    """

    if mode == "classic-v0":
        from heart.classic_single_agent import make_classic_single_agent

        return make_classic_single_agent(
            controlled_player=controlled_player,
            pass_opponents=pass_opponents or opponents,
            play_opponents=play_opponents or opponents,
            **rule_overrides,
        )
    if pass_opponents is not None or play_opponents is not None:
        raise ValueError("phase-specific opponents require mode='classic-v0'")
    if isinstance(controlled_player, bool) or not isinstance(
        controlled_player, Integral
    ):
        raise TypeError("controlled_player must be an integer")
    controlled_player = int(controlled_player)
    if not 0 <= controlled_player < NUM_PLAYERS:
        raise ValueError(f"controlled_player must be in [0, {NUM_PLAYERS})")
    if isinstance(opponents, str):
        difficulties = (opponents,) * (NUM_PLAYERS - 1)
    else:
        difficulties = tuple(opponents)
        if len(difficulties) != NUM_PLAYERS - 1:
            raise ValueError("opponents must contain exactly three difficulties")
    policies = tuple(make_rule_policy(difficulty) for difficulty in difficulties)
    opponent_players = tuple(
        player for player in range(NUM_PLAYERS) if player != controlled_player
    )
    return SingleAgentEnv(
        core=make(mode, **rule_overrides),
        controlled_player=controlled_player,
        opponent_players=opponent_players,
        opponent_policies=policies,
    )
