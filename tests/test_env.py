from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart
from heart.cards import HEART_MASK, POINT_CARD_MASK, SUIT_MASKS, card_id
from heart.rules import settle_deal


def choose_first_legal(observation):
    return jnp.argmax(observation.action_mask)


def test_simplest_v0_is_default_and_classic_is_registered():
    assert heart.make().mode == "simplest-v0"
    assert heart.make("simplest-v0").mode == "simplest-v0"
    assert heart.AVAILABLE_MODES == ("simplest-v0", "classic-v0")
    assert heart.make("classic-v0").mode == "classic-v0"


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


def test_full_deal_has_fixed_horizon_and_terminal_reward_contract():
    env = heart.make()
    state, observation, rewards, terminated = play_to_end(env, jax.random.key(11))
    assert bool(terminated)
    assert int(state.num_cards_played) == 52
    assert int(state.trick_index) == 13
    assert int(state.hands.sum()) == 0
    assert int(state.penalties.sum()) == 18
    if int(state.moon_shooter) < 0:
        assert np.isclose(float(rewards.sum()), 0.0, rtol=0.0, atol=1e-6)
    else:
        assert np.isclose(float(rewards.sum()), -3.0, rtol=0.0, atol=1e-6)
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


def state_with_active_hand(state, cards, **updates):
    active_player = updates.get("active_player", state.active_player)
    hands = jnp.zeros_like(state.hands).at[active_player, jnp.asarray(cards)].set(True)
    return state._replace(hands=hands, **updates)


def test_first_trick_excludes_point_cards_when_player_is_void_in_led_suit():
    env = heart.make()
    state, _ = env.reset(jax.random.key(23))
    diamond_card = card_id(1, 0)
    heart_card = card_id(3, 0)
    state = state_with_active_hand(
        state,
        [diamond_card, heart_card, heart.QUEEN_OF_SPADES],
        current_trick=state.current_trick.at[0].set(heart.TWO_OF_CLUBS),
        active_player=(state.leader + 1) % 4,
        trick_position=jnp.asarray(1, dtype=jnp.int32),
        num_cards_played=jnp.asarray(1, dtype=jnp.int32),
    )

    legal = np.asarray(env.legal_action_mask(state))
    assert legal[diamond_card]
    assert not np.asarray(legal & POINT_CARD_MASK).any()


def test_first_trick_allows_points_when_only_points_remain():
    env = heart.make()
    state, _ = env.reset(jax.random.key(24))
    heart_card = card_id(3, 0)
    state = state_with_active_hand(
        state,
        [heart_card, heart.QUEEN_OF_SPADES],
        current_trick=state.current_trick.at[0].set(heart.TWO_OF_CLUBS),
        active_player=(state.leader + 1) % 4,
        trick_position=jnp.asarray(1, dtype=jnp.int32),
        num_cards_played=jnp.asarray(1, dtype=jnp.int32),
    )

    legal = np.asarray(env.legal_action_mask(state))
    assert legal[heart_card]
    assert legal[heart.QUEEN_OF_SPADES]


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


def test_unchecked_step_matches_safe_step_for_legal_action():
    env = heart.make()
    state, observation = env.reset(jax.random.key(32))
    action = choose_first_legal(observation).astype(jnp.int32)
    safe = jax.jit(env.step)(state, action)
    trusted = jax.jit(env.step_unchecked)(state, action)
    for safe_leaf, trusted_leaf in zip(jax.tree.leaves(safe), jax.tree.leaves(trusted)):
        np.testing.assert_array_equal(np.asarray(safe_leaf), np.asarray(trusted_leaf))


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
    np.testing.assert_allclose(np.asarray(rewards), np.asarray([0.0, -1.0, -1.0, -1.0]))


def test_ordinary_reward_is_normalized_by_configured_total_points():
    rules = heart.make(queen_of_spades_penalty=13).rules
    scores, shooter, _, rewards = settle_deal(
        jnp.asarray([0, 5, 8, 13], dtype=jnp.int16), rules
    )
    assert int(shooter) == -1
    np.testing.assert_array_equal(np.asarray(scores), [0, 5, 8, 13])
    expected = np.asarray([26 / 3, 2, -2, -26 / 3], dtype=np.float32) / 26
    np.testing.assert_allclose(np.asarray(rewards), expected, rtol=1e-6)


def test_custom_queen_penalty_normalizes_moon_opponents_to_minus_one():
    rules = heart.make(queen_of_spades_penalty=9).rules
    _, shooter, _, rewards = settle_deal(
        jnp.asarray([0, 22, 0, 0], dtype=jnp.int16), rules
    )
    assert int(shooter) == 1
    np.testing.assert_array_equal(np.asarray(rewards), [-1.0, 0.0, -1.0, -1.0])


@pytest.mark.parametrize("value", [True, False, 1.5, "5"])
def test_queen_penalty_rejects_non_integer_values(value):
    with pytest.raises(TypeError):
        heart.make(queen_of_spades_penalty=value)


def test_queen_penalty_rejects_values_that_overflow_score_storage():
    with pytest.raises(ValueError):
        heart.make(queen_of_spades_penalty=32755)

    rules = heart.make(queen_of_spades_penalty=32754).rules
    scores, shooter, winners, _ = settle_deal(
        jnp.asarray([32767, 0, 0, 0], dtype=jnp.int16), rules
    )
    np.testing.assert_array_equal(np.asarray(scores), [0, 32767, 32767, 32767])
    assert int(shooter) == 0
    np.testing.assert_array_equal(np.asarray(winners), [True, False, False, False])


@pytest.mark.parametrize("player", [-1, 4, 99])
def test_observe_rejects_invalid_concrete_player(player):
    env = heart.make()
    state, _ = env.reset(jax.random.key(41))
    with pytest.raises(ValueError):
        env.observe(state, player)


@pytest.mark.parametrize("player", [-1, 4, 99])
def test_jitted_observe_fails_closed_for_invalid_traced_player(player):
    env = heart.make()
    state, _ = env.reset(jax.random.key(43))
    observation = jax.jit(env.observe)(state, jnp.asarray(player))
    assert not bool(observation.hand.any())
    assert not bool(observation.action_mask.any())


@pytest.mark.parametrize("player", [False, True, 0.0, 1.9])
def test_jitted_observe_fails_closed_for_non_integer_player(player):
    env = heart.make()
    state, _ = env.reset(jax.random.key(44))
    observation = jax.jit(env.observe)(state, jnp.asarray(player))
    assert not bool(observation.hand.any())
    assert not bool(observation.action_mask.any())


@pytest.mark.parametrize("action", [False, True, 0.0, 0.9])
def test_non_integer_actions_are_invalid_and_do_not_change_state(action):
    env = heart.make()
    state, _ = env.reset(jax.random.key(47))
    next_state, _, rewards, _, info = env.step(state, action)
    assert bool(info.invalid_action)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(next_state)):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))


@pytest.mark.parametrize(
    "action",
    [jnp.asarray([0]), jnp.asarray([0, 1]), jnp.asarray([[0]])],
)
def test_non_scalar_actions_are_invalid_and_do_not_change_state(action):
    env = heart.make()
    state, _ = env.reset(jax.random.key(77))

    next_state, _, rewards, _, info = jax.jit(env.step)(state, action)

    assert bool(info.invalid_action)
    np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(next_state)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_unbroken_hearts_may_be_led_only_when_no_nonheart_remains():
    env = heart.make()
    state, _ = env.reset(jax.random.key(49))
    diamond_card = card_id(1, 0)
    heart_card = card_id(3, 0)
    state = state_with_active_hand(
        state,
        [diamond_card, heart_card],
        leader=jnp.asarray(0, dtype=jnp.int32),
        active_player=jnp.asarray(0, dtype=jnp.int32),
        trick_index=jnp.asarray(1, dtype=jnp.int32),
        num_cards_played=jnp.asarray(4, dtype=jnp.int32),
    )
    legal = np.asarray(env.legal_action_mask(state))
    assert legal[diamond_card] and not legal[heart_card]

    state = state_with_active_hand(state, [heart_card])
    legal = np.asarray(env.legal_action_mask(state))
    assert legal[heart_card] and int(legal.sum()) == 1


def test_off_suit_high_card_cannot_win_and_points_go_to_trick_winner():
    env = heart.make()
    state, _ = env.reset(jax.random.key(51))
    heart_card = card_id(3, 0)
    trick = jnp.asarray(
        [card_id(0, 3), card_id(2, 12), card_id(0, 11), -1], dtype=jnp.int8
    )
    state = state_with_active_hand(
        state,
        [heart_card],
        current_trick=trick,
        leader=jnp.asarray(0, dtype=jnp.int32),
        active_player=jnp.asarray(3, dtype=jnp.int32),
        trick_index=jnp.asarray(1, dtype=jnp.int32),
        trick_position=jnp.asarray(3, dtype=jnp.int32),
        num_cards_played=jnp.asarray(7, dtype=jnp.int32),
    )
    next_state, _, _, _, info = env.step(state, heart_card)
    assert bool(info.trick_completed)
    assert int(info.trick_winner) == 2
    assert int(info.points_won) == 1
    assert int(next_state.leader) == 2
    assert int(next_state.active_player) == 2
    np.testing.assert_array_equal(np.asarray(next_state.penalties), [0, 0, 1, 0])


def test_step_after_termination_preserves_every_state_leaf():
    env = heart.make()
    state, _, _, terminated = play_to_end(env, jax.random.key(53))
    assert bool(terminated)
    next_state, observation, rewards, next_terminated, info = env.step(state, 0)
    assert bool(next_terminated)
    assert bool(info.invalid_action)
    assert not bool(observation.action_mask.any())
    np.testing.assert_array_equal(np.asarray(rewards), np.zeros(4))
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(next_state)):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
