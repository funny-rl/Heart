from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import heart
from heart.cards import HEART_MASK, POINT_CARD_MASK, SUIT_MASKS
from heart.rules import settle_deal


def choose_first_legal(observation):
    return jnp.argmax(observation.action_mask)


def test_simplest_v0_is_default_and_only_registered_mode():
    assert heart.make().mode == "simplest-v0"
    assert heart.make("simplest-v0").mode == "simplest-v0"
    assert heart.AVAILABLE_MODES == ("simplest-v0",)


def play_to_end(env, key):
    state, observation = env.reset(key)
    rewards = jnp.zeros((4,), dtype=jnp.float32)
    for step_index in range(52):
        action = choose_first_legal(observation)
        state, observation, rewards, terminated, info = env.step(state, action)
        assert not bool(info.invalid_action)
        if step_index < 51:
            np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))
    return state, observation, rewards, terminated


def test_reset_deals_every_card_once():
    env = heart.make()
    state, observation = env.reset(jax.random.key(7))
    assert state.hands.shape == (4, 52)
    np.testing.assert_array_equal(np.asarray(state.hands).sum(axis=0), np.ones(52))
    np.testing.assert_array_equal(np.asarray(state.hands).sum(axis=1), np.full(4, 13))
    assert int(observation.action_mask.sum()) == 1
    assert bool(observation.action_mask[heart.TWO_OF_CLUBS])


def test_full_deal_has_fixed_horizon_and_zero_sum_reward():
    env = heart.make()
    state, observation, rewards, terminated = play_to_end(env, jax.random.key(11))
    assert bool(terminated)
    assert int(state.num_cards_played) == 52
    assert int(state.trick_index) == 13
    assert int(state.hands.sum()) == 0
    assert int(state.penalties.sum()) == 18
    assert np.isclose(float(rewards.sum()), 0.0)
    assert not bool(observation.action_mask.any())
    assert np.all(np.asarray(state.trick_history) >= 0)


def test_illegal_action_does_not_change_state():
    env = heart.make()
    state, _ = env.reset(jax.random.key(3))
    next_state, _, rewards, _, info = env.step(state, heart.QUEEN_OF_SPADES)
    assert bool(info.invalid_action)
    assert int(next_state.num_cards_played) == 0
    np.testing.assert_array_equal(np.asarray(next_state.hands), np.asarray(state.hands))
    np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))


def test_follow_suit_mask_when_available():
    env = heart.make()
    state, observation = env.reset(jax.random.key(19))
    state, observation, *_ = env.step(state, heart.TWO_OF_CLUBS)
    hand = np.asarray(observation.hand)
    clubs = hand & np.asarray(SUIT_MASKS[0])
    if clubs.any():
        np.testing.assert_array_equal(np.asarray(observation.action_mask), clubs)


def test_first_trick_excludes_point_cards_when_possible():
    env = heart.make()
    state, observation = env.reset(jax.random.key(23))
    state, observation, *_ = env.step(state, heart.TWO_OF_CLUBS)
    if not np.asarray(observation.hand & SUIT_MASKS[0]).any():
        legal = np.asarray(observation.action_mask)
        assert not np.asarray(legal & POINT_CARD_MASK).any()


def test_hearts_cannot_lead_before_breaking_when_alternative_exists():
    env = heart.make()
    state, observation = env.reset(jax.random.key(29))
    for _ in range(52):
        if int(state.trick_position) == 0 and not bool(state.hearts_broken):
            hand = np.asarray(observation.hand)
            if np.asarray(hand & ~HEART_MASK).any():
                assert not np.asarray(observation.action_mask & HEART_MASK).any()
        action = choose_first_legal(observation)
        state, observation, _, terminated, _ = env.step(state, action)
        if bool(terminated):
            break


def test_jit_and_eager_step_match():
    env = heart.make()
    state, observation = env.reset(jax.random.key(31))
    action = choose_first_legal(observation)
    eager = env.step(state, action)
    compiled = jax.jit(env.step)(state, action)
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled)
    ):
        np.testing.assert_array_equal(np.asarray(eager_leaf), np.asarray(compiled_leaf))


def test_vmap_runs_independent_games():
    env = heart.make()
    keys = jax.random.split(jax.random.key(37), 8)
    states, observations = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.argmax(observations.action_mask, axis=-1)
    next_states, _, _, _, infos = jax.jit(jax.vmap(env.step))(states, actions)
    np.testing.assert_array_equal(np.asarray(next_states.num_cards_played), np.ones(8))
    assert not np.asarray(infos.invalid_action).any()


def test_shooting_the_moon_is_a_solo_win():
    scores, shooter, winners, rewards = settle_deal(jnp.asarray([18, 0, 0, 0]))
    np.testing.assert_array_equal(np.asarray(scores), np.asarray([0, 18, 18, 18]))
    assert int(shooter) == 0
    np.testing.assert_array_equal(
        np.asarray(winners), np.asarray([True, False, False, False])
    )
    np.testing.assert_allclose(
        np.asarray(rewards), np.asarray([18.0, -6.0, -6.0, -6.0])
    )
