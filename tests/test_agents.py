from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart
from heart.agents.rule_based import (
    MOON_ALERTNESS,
    _hard_scores,
    _spade_pressure_scores,
)

ALERTNESS = MOON_ALERTNESS["hard"]
from heart.cards import CLUBS, DIAMONDS, SPADES, card_id


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rule_policy_always_chooses_legal_cards(difficulty):
    env = heart.DealEnv()
    policy = heart.make_rule_policy(difficulty)
    step = jax.jit(env.step)
    key = jax.random.key(101)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)

    for _ in range(52):
        key, action_key = jax.random.split(key)
        action = policy(observation, action_key)
        assert bool(observation.action_mask[action])
        state, observation, _, terminated, info = step(state, action)
        assert not bool(info.invalid_action)
    assert bool(terminated)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rule_policy_is_jittable(difficulty):
    env = heart.DealEnv()
    policy = heart.make_rule_policy(difficulty)
    _, observation = env.reset(jax.random.key(103))
    eager = policy(observation, jax.random.key(104))
    compiled = jax.jit(policy)(observation, jax.random.key(104))
    assert int(eager) == int(compiled)


def test_unknown_rule_policy_rejected():
    with pytest.raises(ValueError):
        heart.make_rule_policy("expert")


def _pass_observation(cards, direction=heart.PASS_LEFT):
    env = heart.make("classic-v0")
    state, _ = env.reset(jax.random.key(201))
    hands = state.game.hands.at[0].set(False).at[0, jnp.asarray(cards)].set(True)
    state = state._replace(
        game=state.game._replace(hands=hands),
        active_player=jnp.asarray(0),
        pass_direction=jnp.asarray(direction, dtype=jnp.int8),
    )
    return env.observe(state, 0)


def _passed_cards(observation, difficulty):
    action = heart.make_rule_pass_policy(difficulty)(observation, jax.random.key(202))
    hand = np.flatnonzero(np.asarray(observation.game.hand))
    return set(hand[np.asarray(heart.PASS_COMBINATIONS[int(action)])].tolist())


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_tactical_pass_sends_exposed_queen_but_keeps_guarded_queen(difficulty):
    queen = heart.QUEEN_OF_SPADES
    exposed = [queen, card_id(SPADES, 0), card_id(SPADES, 1)] + [
        card_id(CLUBS, rank) for rank in range(10)
    ]
    guarded = (
        [queen]
        + [card_id(SPADES, rank) for rank in (0, 1, 2, 3)]
        + [card_id(CLUBS, rank) for rank in range(8)]
    )
    assert queen in _passed_cards(_pass_observation(exposed), difficulty)
    assert queen not in _passed_cards(_pass_observation(guarded), difficulty)


def test_medium_keeps_low_spades_as_high_spade_insurance():
    cards = [card_id(SPADES, rank) for rank in (0, 1, 11, 12)] + [
        card_id(CLUBS, rank) for rank in range(9)
    ]
    passed = _passed_cards(_pass_observation(cards, heart.PASS_RIGHT), "medium")
    assert card_id(SPADES, 0) not in passed
    assert card_id(SPADES, 1) not in passed
    assert card_id(SPADES, 11) in passed
    assert card_id(SPADES, 12) in passed


def test_hard_prefers_diamond_void_and_never_passes_two_of_clubs_for_club_void():
    cards = [card_id(DIAMONDS, 8), card_id(DIAMONDS, 9)] + [
        card_id(CLUBS, rank) for rank in range(11)
    ]
    passed = _passed_cards(_pass_observation(cards), "hard")
    assert {card_id(DIAMONDS, 8), card_id(DIAMONDS, 9)} <= passed
    assert heart.TWO_OF_CLUBS not in passed


def _play_observation(cards, *, trick_index, current_trick):
    env = heart.DealEnv()
    _, observation = env.reset(jax.random.key(301))
    hand = jnp.zeros(52, dtype=jnp.bool_).at[jnp.asarray(cards)].set(True)
    trick = jnp.asarray(current_trick, dtype=jnp.int8)
    return observation._replace(
        hand=hand,
        action_mask=hand,
        current_trick=trick,
        trick_index=jnp.asarray(trick_index, dtype=jnp.int8),
        trick_position=jnp.sum(trick >= 0).astype(jnp.int8),
        trick_history=jnp.full((13, 4), -1, dtype=jnp.int8),
    )


def test_medium_pressures_with_highest_safe_spade_while_queen_is_unseen():
    cards = [card_id(SPADES, 0), card_id(SPADES, 9), card_id(SPADES, 12)]
    observation = _play_observation(
        cards, trick_index=2, current_trick=[-1, -1, -1, -1]
    )
    action = heart.make_rule_policy("medium")(observation, jax.random.key(302))
    assert int(action) in {card_id(SPADES, 0), card_id(SPADES, 9)}


def test_hard_unloads_high_card_on_first_trick_when_not_leading():
    cards = [card_id(CLUBS, 0), card_id(CLUBS, 1), card_id(CLUBS, 12)]
    observation = _play_observation(
        cards,
        trick_index=0,
        current_trick=[card_id(CLUBS, 3), -1, -1, -1],
    )
    key = jax.random.key(303)
    delta = _hard_scores(observation, key, ALERTNESS) - _spade_pressure_scores(
        observation, key, ALERTNESS
    )
    np.testing.assert_allclose(
        np.asarray(delta)[cards], 300.0 + np.asarray([0.0, 1.0, 12.0])
    )


def test_hard_cashes_highest_forced_winner_when_last_and_trick_is_clean():
    cards = [card_id(CLUBS, 4), card_id(CLUBS, 12)]
    observation = _play_observation(
        cards,
        trick_index=4,
        current_trick=[card_id(CLUBS, 3), card_id(CLUBS, 1), card_id(CLUBS, 2), -1],
    )
    key = jax.random.key(304)
    delta = _hard_scores(observation, key, ALERTNESS) - _spade_pressure_scores(
        observation, key, ALERTNESS
    )
    np.testing.assert_allclose(
        np.asarray(delta)[cards], 300.0 + np.asarray([4.0, 12.0])
    )
