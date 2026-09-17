"""Play and render a reproducible random legal deal."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import heart


def main() -> None:
    env = heart.DealEnv()
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)
    rewards = jnp.zeros((4,), dtype=jnp.float32)

    while not bool(state.terminated):
        key, action_key = jax.random.split(key)
        scores = jax.random.uniform(action_key, (heart.NUM_CARDS,))
        action = jnp.argmax(jnp.where(observation.action_mask, scores, -1.0))
        state, observation, rewards, _, info = env.step(state, action)
        if bool(info.invalid_action):
            raise RuntimeError("random policy produced an illegal action")

    print(heart.render_ansi(state))
    print("Rewards:", rewards)


if __name__ == "__main__":
    main()
