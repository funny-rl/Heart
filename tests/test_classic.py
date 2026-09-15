from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart
from heart.classic import (
    NUM_PASS_ACTIONS,
    PASS,
    PASS_ACROSS,
    PASS_COMBINATIONS,
    PASS_HOLD,
    PASS_LEFT,
    PASS_RIGHT,
    PLAY,
    _new_deal_state,
)


def _assert_trees_equal(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected)
    ):
        if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            actual_leaf = jax.random.key_data(actual_leaf)
            expected_leaf = jax.random.key_data(expected_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf)
        )


def _finish_first_deal(env, state):
    def rollout(initial_state):
        initial_info = env.step(initial_state, -1)[4]
        carry = (
            initial_state,
            jnp.zeros((heart.NUM_PLAYERS,), dtype=jnp.float32),
            initial_info,
        )

        def body(_, values):
            current, _, _ = values
            observation = env.observe(current, current.active_player)
            action = jnp.where(
                current.phase == PASS,
                jnp.argmax(observation.pass_action_mask),
                jnp.argmax(observation.play_action_mask),
            )
            next_state, _, rewards, _, info = env.step(current, action)
            return next_state, rewards, info

        return jax.lax.fori_loop(0, 56, body, carry)

    return jax.jit(rollout)(state)


def test_pass_action_table_is_exactly_thirteen_choose_three():
    table = np.asarray(PASS_COMBINATIONS)
    expected = np.asarray(tuple(combinations(range(13), 3)), dtype=np.int8)
    assert NUM_PASS_ACTIONS == 286
    assert table.shape == (286, 3)
    np.testing.assert_array_equal(table, expected)
    assert np.all(table[:, 0] < table[:, 1])
    assert np.all(table[:, 1] < table[:, 2])


@pytest.mark.parametrize(
    ("direction", "recipient"),
    [
        (PASS_LEFT, lambda player: (player + 1) % 4),
        (PASS_RIGHT, lambda player: (player - 1) % 4),
        (PASS_ACROSS, lambda player: (player + 2) % 4),
    ],
)
def test_pass_exchange_preserves_deck_and_moves_cards_to_correct_seats(
    direction, recipient
):
    env = heart.make("classic-v0")
    state, _ = env.reset(jax.random.key(101 + direction))
    state = state._replace(pass_direction=jnp.asarray(direction, dtype=jnp.int8))
    original_hands = np.asarray(state.game.hands)
    original_opener = int(np.argmax(original_hands[:, heart.TWO_OF_CLUBS]))
    selected = []

    for player in range(4):
        hand_cards = np.flatnonzero(original_hands[player])
        chosen = hand_cards[np.asarray(PASS_COMBINATIONS[0])]
        selected.append(chosen)
        state, _, rewards, terminated, info = env.step(state, 0)
        assert not bool(info.invalid_action)
        assert not bool(terminated)
        np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))

    exchanged = np.asarray(state.game.hands)
    assert int(state.phase) == PLAY
    np.testing.assert_array_equal(exchanged.sum(axis=0), np.ones(52))
    np.testing.assert_array_equal(exchanged.sum(axis=1), np.full(4, 13))
    for source, cards in enumerate(selected):
        destination = recipient(source)
        np.testing.assert_array_equal(
            np.asarray(state.cards_received[destination]), cards
        )
        assert exchanged[destination, cards].all()
        assert not exchanged[source, cards].any()

    expected_opener = recipient(original_opener)
    assert int(state.active_player) == expected_opener
    assert int(state.game.leader) == expected_opener
    assert bool(state.game.hands[expected_opener, heart.TWO_OF_CLUBS])
    observation = env.observe(state, expected_opener)
    assert int(observation.play_action_mask.sum()) == 1
    assert bool(observation.play_action_mask[heart.TWO_OF_CLUBS])


def test_hold_deal_skips_pass_phase_and_starts_with_two_of_clubs_holder():
    env = heart.make("classic-v0")
    deal_key, next_key = jax.random.split(jax.random.key(111))
    state = _new_deal_state(
        deal_key,
        next_key,
        jnp.zeros((4,), dtype=jnp.int16),
        jnp.asarray(3, dtype=jnp.int8),
        env.rules,
    )
    opener = int(np.argmax(np.asarray(state.game.hands)[:, heart.TWO_OF_CLUBS]))
    observation = env.observe(state, opener)

    assert int(state.pass_direction) == PASS_HOLD
    assert int(state.phase) == PLAY
    assert int(state.active_player) == opener
    assert not bool(observation.pass_action_mask.any())
    assert int(observation.play_action_mask.sum()) == 1
    assert bool(observation.play_action_mask[heart.TWO_OF_CLUBS])


def test_first_deal_reward_and_boundary_state_are_consistent():
    env = heart.make("classic-v0")
    state, _ = env.reset(jax.random.key(121))
    state, rewards, info = _finish_first_deal(env, state)
    deal_scores = np.asarray(info.deal_scores)
    expected_rewards = ((deal_scores.sum() - deal_scores) / 3.0 - deal_scores) / 26.0

    assert bool(info.deal_completed)
    assert not bool(info.match_completed)
    assert not bool(state.terminated)
    assert bool(state.deal_boundary)
    assert int(state.deal_index) == 1
    assert int(state.pass_direction) == PASS_RIGHT
    assert int(state.phase) == PASS
    np.testing.assert_array_equal(np.asarray(state.match_scores), deal_scores)
    np.testing.assert_array_equal(np.asarray(state.last_deal_scores), deal_scores)
    np.testing.assert_allclose(np.asarray(rewards), expected_rewards, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state.last_deal_rewards), expected_rewards)
    assert np.isclose(float(np.asarray(rewards).sum()), 0.0)
    assert int(state.game.num_cards_played) == 0
    np.testing.assert_array_equal(np.asarray(state.game.hands).sum(axis=1), [13] * 4)

    next_state, *_ = env.step(state, 0)
    assert not bool(next_state.deal_boundary)


@pytest.mark.parametrize("player", [-1, 4])
def test_jitted_invalid_observer_fails_closed(player):
    env = heart.make("classic-v0")
    state, _ = env.reset(jax.random.key(131))
    private_cards = jnp.arange(12, dtype=jnp.int8).reshape(4, 3)
    state = state._replace(
        pass_cards=private_cards,
        cards_received=jnp.flip(private_cards, axis=0),
    )
    observation = jax.jit(env.observe)(state, jnp.asarray(player))

    assert not bool(observation.game.hand.any())
    assert not bool(observation.game.action_mask.any())
    assert np.all(np.asarray(observation.cards_passed) == -1)
    assert np.all(np.asarray(observation.cards_received) == -1)
    assert not bool(observation.pass_action_mask.any())
    assert not bool(observation.play_action_mask.any())


def test_classic_reset_and_phase_steps_are_jittable():
    env = heart.make("classic-v0")
    key = jax.random.key(141)
    eager_state, eager_observation = env.reset(key)
    compiled_state, compiled_observation = jax.jit(env.reset)(key)
    _assert_trees_equal(compiled_state, eager_state)
    _assert_trees_equal(compiled_observation, eager_observation)

    eager = env.step(eager_state, 0)
    compiled = jax.jit(env.step)(compiled_state, jnp.asarray(0))
    _assert_trees_equal(compiled, eager)
