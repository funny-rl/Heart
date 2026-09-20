"""One-learned-player adapter for a complete classic-v0 match."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from heart.agents import (
    RulePassPolicy,
    RulePolicy,
    make_rule_pass_policy,
    make_rule_policy,
)
from heart.cards import NUM_CARDS, NUM_PLAYERS
from heart.classic import (
    NUM_PASS_ACTIONS,
    PASS,
    PLAY,
    ClassicEnv,
    ClassicInfo,
    ClassicObservation,
    ClassicState,
    _empty_classic_info,
    make_classic,
)

MAX_DECISION_EVENTS = 7


class ClassicSingleAgentState(NamedTuple):
    match: ClassicState
    pass_hand_cards: Array
    play_hand_cards: Array
    opponent_key: Array
    decisions: Array


class ClassicSingleAgentObservation(NamedTuple):
    match: ClassicObservation
    pass_hand_cards: Array
    play_hand_cards: Array
    pass_action_mask: Array
    play_action_mask: Array


class ClassicSingleAgentInfo(NamedTuple):
    """Compact event trace and learning metadata for one decision."""

    core: ClassicInfo
    event_actions: Array
    event_players: Array
    event_phases: Array
    event_valid: Array
    event_count: Array
    deal_completed: Array
    match_completed: Array
    discount: Array


def _hand_cards(state: ClassicState, player: int) -> Array:
    return jnp.nonzero(
        state.game.hands[player],
        size=13,
        fill_value=-1,
    )[0].astype(jnp.int8)


@dataclass(frozen=True)
class ClassicSingleAgentEnv:
    core: ClassicEnv
    controlled_player: int
    opponent_players: tuple[int, int, int]
    pass_policies: tuple[RulePassPolicy, RulePassPolicy, RulePassPolicy]
    play_policies: tuple[RulePolicy, RulePolicy, RulePolicy]

    def _observation(
        self,
        match: ClassicState,
        pass_hand_cards: Array,
        play_hand_cards: Array,
    ) -> ClassicSingleAgentObservation:
        observation = self.core.observe(match, self.controlled_player)
        safe_cards = jnp.clip(
            play_hand_cards.astype(jnp.int32),
            0,
            NUM_CARDS - 1,
        )
        play_mask = (play_hand_cards >= 0) & observation.play_action_mask[safe_cards]
        return ClassicSingleAgentObservation(
            match=observation,
            pass_hand_cards=pass_hand_cards,
            play_hand_cards=play_hand_cards,
            pass_action_mask=observation.pass_action_mask,
            play_action_mask=play_mask,
        )

    def _policy_index(self, player: Array) -> Array:
        mapping = [-1] * NUM_PLAYERS
        for index, opponent in enumerate(self.opponent_players):
            mapping[opponent] = index
        return jnp.asarray(mapping, dtype=jnp.int32)[player]

    def _opponent_action(
        self,
        observation: ClassicObservation,
        key: Array,
    ) -> Array:
        policy_index = self._policy_index(observation.game.player)
        pass_branches = tuple(
            (lambda _, selected=policy: selected(observation, key))
            for policy in self.pass_policies
        )
        play_branches = tuple(
            (lambda _, selected=policy: selected(observation.game, key))
            for policy in self.play_policies
        )
        return jax.lax.cond(
            observation.phase == PASS,
            lambda _: jax.lax.switch(
                policy_index,
                pass_branches,
                operand=None,
            ),
            lambda _: jax.lax.switch(
                policy_index,
                play_branches,
                operand=None,
            ),
            operand=None,
        )

    def _sync_hands(
        self,
        before: ClassicState,
        after: ClassicState,
        pass_hand: Array,
        play_hand: Array,
    ) -> tuple[Array, Array]:
        current = _hand_cards(after, self.controlled_player)
        new_deal = after.deal_index != before.deal_index
        completed_pass = (before.phase == PASS) & (after.phase == PLAY) & ~new_deal
        # Both layouts are refreshed when a deal opens, and the play layout
        # again once passing has handed the cards over. `reset` refreshes both
        # whatever the opening phase is, so a deal that opens into passing used
        # to be the one case where the play layout still described the deal
        # before it.
        pass_hand = jnp.where(new_deal, current, pass_hand)
        play_hand = jnp.where(new_deal | completed_pass, current, play_hand)
        return pass_hand, play_hand

    def _advance(
        self,
        match: ClassicState,
        pass_hand: Array,
        play_hand: Array,
        key: Array,
        rewards: Array,
        info: ClassicInfo,
        event_actions: Array,
        event_players: Array,
        event_phases: Array,
        event_valid: Array,
        event_count: Array,
        deal_completed: Array,
        boundary_seen: Array,
    ):
        carry = (
            match,
            pass_hand,
            play_hand,
            key,
            rewards,
            info,
            event_actions,
            event_players,
            event_phases,
            event_valid,
            event_count,
            deal_completed,
            boundary_seen,
        )

        def condition(values):
            current = values[0]
            return (~current.terminated) & (
                current.active_player != self.controlled_player
            )

        def opponent_step(values):
            (
                current,
                current_pass,
                current_play,
                policy_key,
                total_reward,
                current_info,
                actions,
                players,
                phases,
                valid_events,
                count,
                completed,
                saw_boundary,
            ) = values
            policy_key, action_key = jax.random.split(policy_key)
            observation = self.core.observe(current, current.active_player)
            action = self._opponent_action(observation, action_key)
            actions = actions.at[count].set(action.astype(jnp.int16))
            players = players.at[count].set(current.active_player.astype(jnp.int8))
            phases = phases.at[count].set(current.phase.astype(jnp.int8))
            valid_events = valid_events.at[count].set(True)
            next_match, _, step_reward, _, next_info = self.core.step(
                current,
                action,
            )
            current_pass, current_play = self._sync_hands(
                current,
                next_match,
                current_pass,
                current_play,
            )
            preserved_info = jax.tree_util.tree_map(
                lambda old, new: jnp.where(completed, old, new),
                current_info,
                next_info,
            )
            return (
                next_match,
                current_pass,
                current_play,
                policy_key,
                total_reward + step_reward,
                preserved_info,
                actions,
                players,
                phases,
                valid_events,
                count + 1,
                completed | next_info.deal_completed,
                saw_boundary | next_match.deal_boundary,
            )

        result = jax.lax.while_loop(condition, opponent_step, carry)
        return (result[0]._replace(deal_boundary=result[12]), *result[1:12])

    def reset(
        self,
        key: Array,
    ) -> tuple[ClassicSingleAgentState, ClassicSingleAgentObservation]:
        match_key, opponent_key = jax.random.split(key)
        match, _ = self.core.reset(match_key)
        initial = _hand_cards(match, self.controlled_player)
        empty_actions = jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int16)
        empty_players = jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8)
        empty_phases = jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8)
        empty_valid = jnp.zeros((MAX_DECISION_EVENTS,), dtype=jnp.bool_)
        (
            match,
            pass_hand,
            play_hand,
            opponent_key,
            _,
            _,
            *_,
        ) = self._advance(
            match,
            initial,
            initial,
            opponent_key,
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            _empty_classic_info(match, invalid=True),
            empty_actions,
            empty_players,
            empty_phases,
            empty_valid,
            jnp.asarray(0, dtype=jnp.int8),
            jnp.asarray(False),
            jnp.asarray(False),
        )
        state = ClassicSingleAgentState(
            match=match,
            pass_hand_cards=pass_hand,
            play_hand_cards=play_hand,
            opponent_key=opponent_key,
            decisions=jnp.asarray(0, dtype=jnp.int16),
        )
        return state, self._observation(
            match,
            pass_hand,
            play_hand,
        )

    def step(
        self,
        state: ClassicSingleAgentState,
        action: Array | int,
    ):
        raw_action = jnp.asarray(action)
        is_integer = (
            raw_action.ndim == 0
            and jnp.issubdtype(
                raw_action.dtype,
                jnp.integer,
            )
            and not jnp.issubdtype(raw_action.dtype, jnp.bool_)
        )
        if is_integer:
            pass_in_bounds = (raw_action >= 0) & (raw_action < NUM_PASS_ACTIONS)
            play_in_bounds = (raw_action >= 0) & (raw_action < 13)
            action = raw_action.astype(jnp.int32)
        else:
            pass_in_bounds = jnp.asarray(False)
            play_in_bounds = jnp.asarray(False)
            action = jnp.asarray(0, dtype=jnp.int32)
        observation = self._observation(
            state.match,
            state.pass_hand_cards,
            state.play_hand_cards,
        )
        safe_pass = jnp.clip(action, 0, observation.pass_action_mask.size - 1)
        safe_play = jnp.clip(action, 0, 12)
        valid_pass = pass_in_bounds & observation.pass_action_mask[safe_pass]
        valid_play = play_in_bounds & observation.play_action_mask[safe_play]
        valid = jnp.asarray(is_integer) & jnp.where(
            state.match.phase == PASS,
            valid_pass,
            (state.match.phase == PLAY) & valid_play,
        )
        core_action = jnp.where(
            state.match.phase == PASS,
            action,
            state.play_hand_cards[safe_play].astype(jnp.int32),
        )

        def apply(_: None):
            input_match = state.match._replace(deal_boundary=jnp.asarray(False))
            event_actions = (
                jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int16)
                .at[0]
                .set(core_action.astype(jnp.int16))
            )
            event_players = (
                jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8)
                .at[0]
                .set(input_match.active_player.astype(jnp.int8))
            )
            event_phases = (
                jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8)
                .at[0]
                .set(input_match.phase)
            )
            event_valid = (
                jnp.zeros((MAX_DECISION_EVENTS,), dtype=jnp.bool_).at[0].set(True)
            )
            match, _, rewards, _, info = self.core.step(
                input_match,
                core_action,
            )
            pass_hand, play_hand = self._sync_hands(
                input_match,
                match,
                state.pass_hand_cards,
                state.play_hand_cards,
            )
            (
                match,
                pass_hand,
                play_hand,
                key,
                rewards,
                info,
                event_actions,
                event_players,
                event_phases,
                event_valid,
                event_count,
                deal_completed,
            ) = self._advance(
                match,
                pass_hand,
                play_hand,
                state.opponent_key,
                rewards,
                info,
                event_actions,
                event_players,
                event_phases,
                event_valid,
                jnp.asarray(1, dtype=jnp.int8),
                info.deal_completed,
                match.deal_boundary,
            )
            next_state = ClassicSingleAgentState(
                match=match,
                pass_hand_cards=pass_hand,
                play_hand_cards=play_hand,
                opponent_key=key,
                decisions=state.decisions + 1,
            )
            return (
                next_state,
                self._observation(match, pass_hand, play_hand),
                rewards[self.controlled_player],
                match.terminated,
                ClassicSingleAgentInfo(
                    core=info,
                    event_actions=event_actions,
                    event_players=event_players,
                    event_phases=event_phases,
                    event_valid=event_valid,
                    event_count=event_count,
                    deal_completed=deal_completed,
                    match_completed=match.terminated,
                    discount=jnp.where(match.terminated, 0.0, 1.0).astype(jnp.float32),
                ),
            )

        def reject(_: None):
            info = _empty_classic_info(state.match, invalid=True)
            empty_actions = jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int16)
            return (
                state,
                observation,
                jnp.asarray(0.0, dtype=jnp.float32),
                state.match.terminated,
                ClassicSingleAgentInfo(
                    core=info,
                    event_actions=empty_actions,
                    event_players=jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8),
                    event_phases=jnp.full((MAX_DECISION_EVENTS,), -1, dtype=jnp.int8),
                    event_valid=jnp.zeros((MAX_DECISION_EVENTS,), dtype=jnp.bool_),
                    event_count=jnp.asarray(0, dtype=jnp.int8),
                    deal_completed=jnp.asarray(False),
                    match_completed=state.match.terminated,
                    discount=jnp.where(state.match.terminated, 0.0, 1.0).astype(
                        jnp.float32
                    ),
                ),
            )

        return jax.lax.cond(valid, apply, reject, operand=None)

    def replay_events(
        self,
        state: ClassicSingleAgentState,
        info: ClassicSingleAgentInfo,
    ) -> tuple[ClassicState, ...]:
        """Reconstruct all render frames produced by one decision."""

        count = int(info.event_count)
        actions = info.event_actions.tolist()
        cursor = state.match._replace(deal_boundary=jnp.asarray(False))
        frames: list[ClassicState] = [state.match]
        for action in actions[:count]:
            cursor = self.core.step(cursor, int(action))[0]
            frames.append(cursor)
        return tuple(frames)


def _difficulties(value: str | Sequence[str]) -> tuple[str, str, str]:
    if isinstance(value, str):
        return (value, value, value)
    values = tuple(value)
    if len(values) != 3:
        raise ValueError("opponent difficulties must contain exactly three names")
    return values


def make_classic_single_agent(
    *,
    controlled_player: int = 0,
    pass_opponents: str | Sequence[str] = "medium",
    play_opponents: str | Sequence[str] = "medium",
    **rule_overrides: object,
) -> ClassicSingleAgentEnv:
    if isinstance(controlled_player, bool) or not isinstance(
        controlled_player, Integral
    ):
        raise TypeError("controlled_player must be an integer")
    controlled_player = int(controlled_player)
    if not 0 <= controlled_player < NUM_PLAYERS:
        raise ValueError(f"controlled_player must be in [0, {NUM_PLAYERS})")
    opponent_players = tuple(
        player for player in range(NUM_PLAYERS) if player != controlled_player
    )
    return ClassicSingleAgentEnv(
        core=make_classic(**rule_overrides),
        controlled_player=controlled_player,
        opponent_players=opponent_players,
        pass_policies=tuple(
            make_rule_pass_policy(name) for name in _difficulties(pass_opponents)
        ),
        play_policies=tuple(
            make_rule_policy(name) for name in _difficulties(play_opponents)
        ),
    )
