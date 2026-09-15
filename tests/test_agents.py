from __future__ import annotations

import jax
import pytest

import heart


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rule_policy_always_chooses_legal_cards(difficulty):
    env = heart.make()
    policy = heart.make_rule_policy(difficulty)
    key = jax.random.key(101)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)

    for _ in range(52):
        key, action_key = jax.random.split(key)
        action = policy(observation, action_key)
        assert bool(observation.action_mask[action])
        state, observation, _, terminated, info = env.step(state, action)
        assert not bool(info.invalid_action)
    assert bool(terminated)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rule_policy_is_jittable(difficulty):
    env = heart.make()
    policy = heart.make_rule_policy(difficulty)
    _, observation = env.reset(jax.random.key(103))
    eager = policy(observation, jax.random.key(104))
    compiled = jax.jit(policy)(observation, jax.random.key(104))
    assert int(eager) == int(compiled)


def test_unknown_rule_policy_rejected():
    with pytest.raises(ValueError):
        heart.make_rule_policy("expert")
