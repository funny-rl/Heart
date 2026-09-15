# HEART

HEART is a compact, accelerator-native environment for competitive multi-agent
reinforcement learning in the card game Hearts. The game engine is written as
pure JAX transformations, so the same implementation can run eagerly for
debugging or be composed with `jax.jit`, `jax.vmap`, and `jax.lax.scan` for
large batched rollouts.

The initial `simplest-v0` mode intentionally has a small, fixed horizon:

- four independent players and a standard 52-card deck;
- one deal of 13 tricks (exactly 52 card actions);
- no card passing;
- one penalty point per heart and five for the queen of spades;
- collecting every point card shoots the moon and wins the deal outright;
- the two of clubs opens the first trick;
- players must follow suit, and hearts cannot lead until broken;
- point cards cannot be discarded on the first trick when another legal card
  is available.

## Install

```bash
git clone <repository-url> HEART
cd HEART
python -m pip install -e ".[test]"
pytest
```

JAX chooses CPU, GPU, or TPU according to the installed JAX runtime. Follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) when
installing an accelerator-specific runtime.

## Play one random deal

```python
import jax
import jax.numpy as jnp

import heart

env = heart.make("simplest-v0")
state, observation = env.reset(jax.random.key(0))

while not bool(state.terminated):
    mask = observation.action_mask
    action = jnp.argmax(
        jnp.where(mask, jax.random.uniform(jax.random.key(int(state.num_cards_played)), (52,)), -1.0)
    )
    state, observation, rewards, terminated, info = env.step(state, action)

print(state.penalties)
print(state.scores)
print(rewards)
```

For a readable terminal view:

```python
print(heart.render_ansi(state, viewer=0))
```

For a standalone browser game table:

```python
heart.save_html(state, "heart-game.html", viewer=0)
```

Rendering is fully observable: every hand is always visible. The `viewer`
argument only identifies the local player and rotates that seat to the bottom;
pass `viewer=None` for a neutral spectator view. The HTML table highlights the
active player's legal cards, maps played cards to their seats, and shows captured
hearts, the queen of spades, winners, and moon shots. No rendering dependency
runs inside the compiled environment path.

To save a complete interactive match replay, retain the initial state and each
subsequent state, then write them as one portable HTML file:

```python
heart.save_replay_html(states, "heart-replay.html", viewer=0, fps=2.0)
```

The replay includes play/pause, frame stepping, a timeline, speed controls, and
keyboard navigation. It remains fully observable at every frame.

## Batch environments

The environment methods operate on one immutable state. Standard JAX
transformations provide batching and compilation without a separate vector
environment implementation:

```python
import jax
import heart

env = heart.make("simplest-v0")
keys = jax.random.split(jax.random.key(0), 4096)

batch_reset = jax.jit(jax.vmap(env.reset))
batch_step = jax.jit(jax.vmap(env.step))

states, observations = batch_reset(keys)
states, observations, rewards, terminated, infos = batch_step(states, actions)
```

Only the currently active player's action is supplied for each environment.
`observation.player` identifies that player, and `observation.action_mask`
contains the legal cards.

## API contract

```python
state, observation = env.reset(key)
state, observation, rewards, terminated, info = env.step(state, action)
```

- `action` is a card ID in `[0, 52)`.
- `rewards` has shape `(4,)` and is zero until the deal ends.
- terminal rewards compare each player's moon-adjusted final score with the
  other players' mean, so the four rewards sum to zero; all earlier rewards are
  zero.
- an illegal action leaves the state unchanged and sets
  `info.invalid_action`; callers should treat this as a policy error.
- `env.step` and `env.reset` have no hidden mutable state.

See [`examples/random_game.py`](examples/random_game.py) and
[`benchmarks/random_rollout.py`](benchmarks/random_rollout.py) for complete
examples.

## Development

```bash
python -m pip install -e ".[test]"
pytest
python examples/random_game.py
python examples/render_game.py --output heart-game.html
python examples/render_replay.py --output heart-replay.html
python benchmarks/random_rollout.py --batch-size 4096
```

Mode-specific behavior lives in immutable rule configurations. Future modes
can add standard queen scoring, passing, alternate moon scoring, or multi-deal
match play without changing the state-transition contract of `simplest-v0`.

## Built-in opponents

Rule policies share one JIT-compatible interface and always respect the action
mask:

```python
policy = heart.make_rule_policy("medium")  # easy, medium, or hard
action = policy(observation, key)
```

- `easy` plays a simple low-card policy;
- `medium` ducks under winning cards and discards dangerous point cards when
  void in the led suit;
- `hard` additionally reacts to a possible moon shooter and uses public state.

These policies are reproducible baselines rather than claims of optimal play.
Their tactics are intentionally isolated in `heart/agents/rule_based.py` so new
difficulty levels can be added without modifying the environment.
