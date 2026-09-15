from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart


def _as_numpy(array):
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        array = jax.random.key_data(array)
    return np.asarray(array)


def _action(observation, phase):
    return jnp.where(
        phase == heart.PASS,
        jnp.argmax(observation.pass_action_mask),
        jnp.argmax(observation.play_action_mask),
    ).astype(jnp.int32)


@cache
def _completed_deals(controlled_player: int, count: int):
    env = heart.make_single_agent(
        "classic-v0",
        controlled_player=controlled_player,
        opponents="easy",
    )
    reset = jax.jit(env.reset)

    @jax.jit
    def advance(state, observation):
        return env.step(state, _action(observation, state.match.phase))

    state, observation = reset(jax.random.key(700 + controlled_player))
    jax.block_until_ready(state)
    deals = []
    decisions = 0
    while len(deals) < count:
        state, observation, reward, terminated, info = advance(state, observation)
        jax.block_until_ready(state)
        decisions += 1
        if bool(info.deal_completed):
            deals.append((decisions, state, reward, terminated, info))
            decisions = 0
    return env, tuple(deals)


@pytest.mark.parametrize("controlled_player", range(heart.NUM_PLAYERS))
def test_deal_boundary_is_sticky_for_every_controlled_seat(controlled_player):
    _, deals = _completed_deals(controlled_player, 1)
    decisions, state, _, terminated, info = deals[0]

    assert decisions == 14
    assert not bool(terminated)
    assert bool(info.deal_completed)
    assert bool(state.match.deal_boundary)
    assert float(info.discount) == 1.0
    assert int(info.event_count) == int(info.event_valid.sum())
    assert 1 <= int(info.event_count) <= heart.MAX_DECISION_EVENTS


def test_passing_deals_have_fourteen_decisions_and_hold_has_thirteen():
    _, deals = _completed_deals(0, 4)

    assert [deal[0] for deal in deals] == [14, 14, 14, 13]
    assert [int(deal[1].match.pass_direction) for deal in deals] == [1, 2, 3, 0]
    assert all(bool(deal[4].deal_completed) for deal in deals)


def test_replay_contains_initial_frame_plus_every_recorded_event():
    env = heart.make_single_agent(
        "classic-v0",
        controlled_player=2,
        opponents="easy",
    )
    state, observation = jax.jit(env.reset)(jax.random.key(811))
    action = _action(observation, state.match.phase)
    _, _, _, _, info = jax.jit(env.step)(state, action)
    jax.block_until_ready(info)

    frames = env.replay_events(state, info)

    assert len(frames) == int(info.event_count) + 1
    np.testing.assert_array_equal(
        np.asarray(info.event_valid),
        np.arange(heart.MAX_DECISION_EVENTS) < int(info.event_count),
    )
    for index in range(int(info.event_count)):
        assert int(frames[index].active_player) == int(info.event_players[index])
        assert int(frames[index].phase) == int(info.event_phases[index])


def test_jitted_vmapped_step_preserves_event_and_discount_contract():
    env = heart.make_single_agent(
        "classic-v0",
        controlled_player=0,
        opponents="easy",
    )
    keys = jax.random.split(jax.random.key(901), 4)
    states, observations = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(_action)(
        observations,
        states.match.phase,
    )

    next_states, _, rewards, terminated, infos = jax.jit(jax.vmap(env.step))(
        states,
        actions,
    )
    jax.block_until_ready(next_states)

    np.testing.assert_array_equal(np.asarray(next_states.decisions), np.ones(4))
    np.testing.assert_array_equal(
        np.asarray(infos.event_count),
        np.asarray(infos.event_valid).sum(axis=1),
    )
    assert np.asarray(infos.event_count >= 1).all()
    np.testing.assert_array_equal(np.asarray(infos.discount), np.ones(4))
    np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))
    assert not np.asarray(terminated).any()


def test_invalid_action_has_empty_trace_and_terminal_discount_is_zero():
    env = heart.make_single_agent(
        "classic-v0",
        controlled_player=0,
        opponents="easy",
    )
    state, _ = jax.jit(env.reset)(jax.random.key(1001))
    next_state, _, reward, terminated, info = jax.jit(env.step)(
        state,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    jax.block_until_ready(next_state)

    for before, after in zip(
        jax.tree.leaves(state),
        jax.tree.leaves(next_state),
        strict=True,
    ):
        np.testing.assert_array_equal(_as_numpy(after), _as_numpy(before))
    assert float(reward) == 0.0
    assert not bool(terminated)
    assert bool(info.core.invalid_action)
    assert int(info.event_count) == 0
    assert not np.asarray(info.event_valid).any()
    np.testing.assert_array_equal(np.asarray(info.event_actions), -np.ones(7))
    assert not bool(info.deal_completed)
    assert float(info.discount) == 1.0

    terminal = state._replace(
        match=state.match._replace(
            phase=jnp.asarray(heart.TERMINAL, dtype=jnp.int8),
            terminated=jnp.asarray(True),
        )
    )
    _, _, _, terminal_flag, terminal_info = jax.jit(env.step)(
        terminal,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    assert bool(terminal_flag)
    assert bool(terminal_info.match_completed)
    assert int(terminal_info.event_count) == 0
    assert float(terminal_info.discount) == 0.0
