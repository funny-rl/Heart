"""Generate a compact full-observability GIF of one complete deal."""

from __future__ import annotations

import argparse

import jax

import heart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="heart-preview.gif")
    parser.add_argument("--viewer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-ms", type=int, default=95)
    parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    args = parser.parse_args()

    env = heart.make("simplest-v0")
    policy = heart.make_rule_policy(args.difficulty)
    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)
    states = [state]

    while not bool(state.terminated):
        key, action_key = jax.random.split(key)
        action = policy(observation, action_key)
        state, observation, *_ = env.step(state, action)
        states.append(state)

    output = heart.save_gif(
        states,
        args.output,
        viewer=args.viewer,
        duration_ms=args.duration_ms,
    )
    print(output)


if __name__ == "__main__":
    main()
