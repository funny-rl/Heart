"""Play one deal and save it as an interactive standalone HTML replay."""

from __future__ import annotations

import argparse

import jax

import heart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="heart-replay.html")
    parser.add_argument("--viewer", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    args = parser.parse_args()

    env = heart.make("simplest-v0")
    policy = heart.make_rule_policy(args.difficulty)
    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)
    states = [state]

    while not bool(state.terminated):
        key, action_key = jax.random.split(key)
        action = policy(observation, action_key)
        state, observation, *_ = env.step(state, action)
        states.append(state)

    output = heart.save_replay_html(states, args.output, args.viewer, args.fps)
    print(output)


if __name__ == "__main__":
    main()
