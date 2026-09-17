"""Measure compiled batched environment and policy throughput."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

import heart

WORKLOADS = ("engine-only", "policy-inclusive")
POLICIES = ("random", "easy", "medium", "hard")
STEP_MODES = ("safe", "trusted")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_rollout(
    batch_size: int,
    workload: str = "policy-inclusive",
    policy: str = "random",
    step_mode: str = "safe",
):
    """Build one compiled 52-action rollout returning a scalar checksum."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; available: {WORKLOADS}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; available: {POLICIES}")
    if step_mode not in STEP_MODES:
        raise ValueError(f"unknown step mode {step_mode!r}; available: {STEP_MODES}")

    env = heart.DealEnv()
    batch_reset = jax.vmap(env.reset)
    step_fn = env.step if step_mode == "safe" else env.step_unchecked
    batch_step = jax.vmap(step_fn)
    rule_policy = None if policy == "random" else heart.make_rule_policy(policy)
    batch_policy = None if rule_policy is None else jax.vmap(rule_policy)

    def rollout(key):
        reset_key, play_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        states, observations = batch_reset(reset_keys)

        def play(carry, step_key):
            states, observations = carry
            if workload == "engine-only":
                actions = jnp.argmax(observations.action_mask, axis=-1)
            elif batch_policy is None:
                scores = jax.random.uniform(step_key, (batch_size, heart.NUM_CARDS))
                actions = jnp.argmax(
                    jnp.where(observations.action_mask, scores, -1.0), axis=-1
                )
            else:
                action_keys = jax.random.split(step_key, batch_size)
                actions = batch_policy(observations, action_keys)
            states, observations, *_ = batch_step(states, actions)
            return (states, observations), None

        step_keys = (
            None if workload == "engine-only" else jax.random.split(play_key, 52)
        )
        (states, _), _ = jax.lax.scan(
            play,
            (states, observations),
            step_keys,
            length=52,
        )
        # Keep a data-dependent result without returning the full trajectory.
        return (
            jnp.sum(states.trick_history.astype(jnp.int32))
            + jnp.sum(states.penalties.astype(jnp.int32))
            + jnp.sum(states.num_cards_played)
        )

    return jax.jit(rollout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=_positive_int, default=4096)
    parser.add_argument("--runs", type=_positive_int, default=10)
    parser.add_argument("--workload", choices=WORKLOADS, default="policy-inclusive")
    parser.add_argument("--policy", choices=POLICIES, default="random")
    parser.add_argument("--step-mode", choices=STEP_MODES, default="safe")
    args = parser.parse_args()

    rollout = build_rollout(args.batch_size, args.workload, args.policy, args.step_mode)
    key = jax.random.key(0)
    compiled = rollout(key)
    jax.block_until_ready(compiled)

    start = time.perf_counter()
    results = []
    for _ in range(args.runs):
        key, run_key = jax.random.split(key)
        results.append(rollout(run_key))
    jax.block_until_ready(results)
    elapsed = time.perf_counter() - start

    deals = args.batch_size * args.runs
    actions = deals * 52
    print("subject: deal core (heart.DealEnv)")
    print(f"workload: {args.workload}")
    print(f"step mode: {args.step_mode}")
    if args.workload == "policy-inclusive":
        print(f"policy: {args.policy}")
    print(f"batch size: {args.batch_size:,}")
    print(f"runs: {args.runs:,}")
    print(f"device: {jax.default_backend()}")
    print(f"deals/s: {deals / elapsed:,.0f}")
    print(f"card actions/s: {actions / elapsed:,.0f}")


if __name__ == "__main__":
    main()
