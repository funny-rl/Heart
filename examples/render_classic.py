"""Render one classic-v0 deal and its transition to the next deal."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import heart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", default="classic-deal-transition.gif")
    parser.add_argument("--html", default="classic-deal-transition.html")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--viewer", type=int, default=0)
    args = parser.parse_args()

    env = heart.make("classic-v0")
    state, observation = env.reset(jax.random.key(args.seed))
    states = [state]

    while int(state.deal_index) == 0:
        mask = (
            observation.pass_action_mask
            if int(state.phase) == heart.PASS
            else observation.play_action_mask
        )
        action = jnp.argmax(mask)
        state, observation, *_ = env.step(state, action)
        states.append(state)

    gif = heart.save_classic_gif(states, args.gif, viewer=args.viewer)
    html = heart.save_classic_replay_html(states, args.html, viewer=args.viewer)
    print(gif)
    print(html)


if __name__ == "__main__":
    main()
