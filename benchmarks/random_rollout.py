"""Measure compiled batched random-policy environment throughput."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

import heart


def build_rollout(batch_size: int):
    env = heart.make()
    batch_reset = jax.vmap(env.reset)
    batch_step = jax.vmap(env.step)

    def rollout(key):
        reset_key, play_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        states, observations = batch_reset(reset_keys)
        step_keys = jax.random.split(play_key, 52)

        def play(carry, step_key):
            states, observations = carry
            scores = jax.random.uniform(step_key, (batch_size, heart.NUM_CARDS))
            actions = jnp.argmax(
                jnp.where(observations.action_mask, scores, -1.0), axis=-1
            )
            states, observations, rewards, terminated, infos = batch_step(
                states, actions
            )
            return (states, observations), (rewards, terminated, infos.invalid_action)

        return jax.lax.scan(play, (states, observations), step_keys)

    return jax.jit(rollout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    rollout = build_rollout(args.batch_size)
    key = jax.random.key(0)
    compiled = rollout(key)
    jax.block_until_ready(compiled)

    start = time.perf_counter()
    for _ in range(args.runs):
        key, run_key = jax.random.split(key)
        result = rollout(run_key)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start

    deals = args.batch_size * args.runs
    actions = deals * 52
    print(f"device: {jax.default_backend()}")
    print(f"deals/s: {deals / elapsed:,.0f}")
    print(f"card actions/s: {actions / elapsed:,.0f}")


if __name__ == "__main__":
    main()
