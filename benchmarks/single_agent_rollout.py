"""Measure batched 13-decision learner-versus-rules throughput."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

import heart


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_rollout(batch_size: int, opponents: str = "medium"):
    """Build one compiled batch of complete single-learner deals."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    env = heart.make_single_agent(opponents=opponents)
    batch_reset = jax.vmap(env.reset)
    batch_step = jax.vmap(env.step)

    def rollout(key):
        states, observations = batch_reset(jax.random.split(key, batch_size))

        def play(carry, _):
            current_states, current_observations = carry
            actions = jnp.argmax(current_observations.action_mask, axis=-1)
            next_states, next_observations, *_ = batch_step(current_states, actions)
            return (next_states, next_observations), None

        (states, _), _ = jax.lax.scan(
            play,
            (states, observations),
            xs=None,
            length=heart.HAND_SIZE,
        )
        return (
            jnp.sum(states.game.scores.astype(jnp.int32))
            + jnp.sum(states.game.num_cards_played)
            + jnp.sum(states.decisions.astype(jnp.int32))
        )

    return jax.jit(rollout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=_positive_int, default=4096)
    parser.add_argument("--runs", type=_positive_int, default=10)
    parser.add_argument(
        "--opponents",
        choices=("easy", "medium", "hard"),
        default="medium",
    )
    args = parser.parse_args()

    rollout = build_rollout(args.batch_size, args.opponents)
    key = jax.random.key(0)
    jax.block_until_ready(rollout(key))

    results = []
    start = time.perf_counter()
    for _ in range(args.runs):
        key, run_key = jax.random.split(key)
        results.append(rollout(run_key))
    jax.block_until_ready(results)
    elapsed = time.perf_counter() - start

    deals = args.batch_size * args.runs
    decisions = deals * heart.HAND_SIZE
    card_plays = deals * heart.NUM_CARDS
    print(f"mode: {heart.SIMPLEST_V0}")
    print("interface: single-agent-13")
    print(f"opponents: {args.opponents}")
    print(f"batch size: {args.batch_size:,}")
    print(f"runs: {args.runs:,}")
    print(f"device: {jax.default_backend()}")
    print(f"deals/s: {deals / elapsed:,.0f}")
    print(f"learner decisions/s: {decisions / elapsed:,.0f}")
    print(f"card plays/s: {card_plays / elapsed:,.0f}")


if __name__ == "__main__":
    main()
