from __future__ import annotations

import jax
import numpy as np
import pytest

import heart


@pytest.mark.parametrize("controlled_player", range(heart.NUM_PLAYERS))
def test_reset_stops_at_controlled_player_with_stable_thirteen_slots(
    controlled_player,
):
    env = heart.make_single_agent(
        controlled_player=controlled_player,
        opponents=("easy", "medium", "hard"),
    )
    state, observation = jax.jit(env.reset)(jax.random.key(501 + controlled_player))

    assert int(state.game.active_player) == controlled_player
    assert state.hand_cards.shape == (13,)
    assert observation.action_mask.shape == (13,)
    assert int(np.asarray(state.game.hands[controlled_player]).sum()) == 13
    np.testing.assert_array_equal(
        np.asarray(state.hand_cards),
        np.flatnonzero(np.asarray(state.game.hands[controlled_player])),
    )
    assert bool(observation.action_mask.any())


def test_thirteen_agent_decisions_complete_fifty_two_card_plays():
    env = heart.make_single_agent(controlled_player=2, opponents="medium")
    state, observation = env.reset(jax.random.key(511))
    initial_slots = np.asarray(state.hand_cards).copy()
    terminal_reward = 0.0

    for decision in range(13):
        slot = int(np.flatnonzero(np.asarray(observation.action_mask))[0])
        state, observation, reward, terminated, info = env.step(state, slot)
        assert not bool(info.invalid_action)
        assert int(state.decisions) == decision + 1
        np.testing.assert_array_equal(np.asarray(state.hand_cards), initial_slots)
        assert not bool(observation.action_mask[slot])
        terminal_reward = float(reward)
        assert bool(terminated) == (decision == 12)

    assert int(state.game.num_cards_played) == 52
    assert bool(state.game.terminated)
    scores = np.asarray(state.game.scores, dtype=np.float32)
    total_points = 13 + env.core.rules.queen_of_spades_penalty
    expected = ((scores.sum() - scores[2]) / 3 - scores[2]) / total_points
    assert terminal_reward == pytest.approx(float(expected))


@pytest.mark.parametrize("action", [-1, 13, False, 0.5])
def test_invalid_slot_fails_closed(action):
    env = heart.make_single_agent()
    state, observation = env.reset(jax.random.key(521))
    next_state, next_observation, reward, _, info = env.step(state, action)

    assert bool(info.invalid_action)
    assert float(reward) == 0.0
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(next_state)):
        if jax.dtypes.issubdtype(after.dtype, jax.dtypes.prng_key):
            before = jax.random.key_data(before)
            after = jax.random.key_data(after)
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    np.testing.assert_array_equal(
        np.asarray(next_observation.action_mask),
        np.asarray(observation.action_mask),
    )


def test_jitted_step_returns_to_learner_or_termination():
    env = heart.make_single_agent(controlled_player=3, opponents="easy")
    state, observation = jax.jit(env.reset)(jax.random.key(531))
    action = observation.action_mask.argmax().astype(np.int32)
    next_state, _, _, terminated, info = jax.jit(env.step)(state, action)

    assert not bool(info.invalid_action)
    assert bool(terminated) or int(next_state.game.active_player) == 3


def test_vmap_batches_independent_single_learner_games():
    env = heart.make_single_agent(controlled_player=1, opponents="easy")
    keys = jax.random.split(jax.random.key(535), 4)
    states, observations = jax.jit(jax.vmap(env.reset))(keys)
    actions = observations.action_mask.argmax(axis=-1).astype(np.int32)
    next_states, _, rewards, terminated, infos = jax.jit(jax.vmap(env.step))(
        states, actions
    )

    np.testing.assert_array_equal(np.asarray(next_states.decisions), np.ones(4))
    assert rewards.shape == (4,)
    assert terminated.shape == (4,)
    assert not np.asarray(infos.invalid_action).any()


@pytest.mark.parametrize("controlled_player", [-1, 4, True, 1.5])
def test_single_agent_configuration_rejects_invalid_player(controlled_player):
    error = TypeError if isinstance(controlled_player, (bool, float)) else ValueError
    with pytest.raises(error):
        heart.make_single_agent(controlled_player=controlled_player)


def test_single_agent_configuration_requires_three_opponents():
    with pytest.raises(ValueError, match="exactly three"):
        heart.make_single_agent(opponents=("easy", "hard"))
