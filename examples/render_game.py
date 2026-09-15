"""Create a browser-viewable snapshot of a live deal."""

from __future__ import annotations

import argparse

import jax

import heart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="heart-game.html")
    parser.add_argument("--moves", type=int, default=18)
    parser.add_argument("--viewer", type=int, default=0)
    args = parser.parse_args()

    env = heart.make()
    policy = heart.make_rule_policy("medium")
    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)
    state, observation = env.reset(reset_key)

    for _ in range(min(max(args.moves, 0), 52)):
        if bool(state.terminated):
            break
        key, action_key = jax.random.split(key)
        action = policy(observation, action_key)
        state, observation, *_ = env.step(state, action)

    output = heart.save_html(state, args.output, viewer=args.viewer)
    print(output)


if __name__ == "__main__":
    main()
