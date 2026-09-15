# Rendering and replay contract

Rendering is a host-side inspection surface. It must not enter the compiled
training transition or enlarge the recurrent JAX state.

## Full observability

All renderers consume omniscient `State` or `ClassicState` values and reveal
every player's current hand. The `viewer` argument does not hide information:

- an integer `0`–`3` marks that player as local and places that seat at the
  bottom of the HTML table;
- `None` uses a neutral spectator view with player 0 at the bottom.

This contract is intended for debugging, evaluation, teaching, and replay.
Policies receive private observations rather than renderer input.

## `simplest-v0` snapshots

```python
text = heart.render_ansi(state, viewer=0)
page = heart.render_html(state, viewer=0)
path = heart.save_html(state, "heart-game.html", viewer=0)
```

The dependency-free HTML shows all hands, trick placement, active player, legal
cards, captured point cards, live penalties, final effective scores, rewards,
winners, and moon state. Immediately after a trick completes, it reconstructs
the completed four-card trick with original player seats; a newly started trick
takes precedence after its first card.

## Classic match snapshots and boundaries

```python
text = heart.render_classic_ansi(state, viewer=0)
page = heart.render_classic_html(state, viewer=0)
path = heart.save_classic_html(state, "classic-game.html", viewer=0)
```

Classic renderers add the cumulative 100-point scoreboard, current pass
direction, omniscient pass-flow information, deal rewards, moon status, and
match winners.

On a non-terminal `deal_boundary`, `state.game` is already the newly dealt next
hand. The renderer combines that hand with a completed-deal overlay sourced
from `last_deal_scores`, `last_deal_rewards`, and
`last_deal_moon_shooter`. A hold boundary announces immediate play rather than
a pass phase. The next valid core event clears the flag. The single-learner
adapter keeps it sticky when opponent autoplay crosses a boundary, so the next
learner-visible state cannot miss the completed deal.

At terminal completion, `game` remains the final completed deal and renderers
use the match-level `winner_mask`, including every player tied for the lowest
cumulative score.

## Interactive `simplest-v0` replay

Retain reset state followed by every successor state:

```python
states = [state]
while not bool(state.terminated):
    state, observation, *_ = env.step(state, action)
    states.append(state)

heart.save_replay_html(states, "heart-replay.html", viewer=0, fps=2.0)
```

A complete canonical replay contains 53 frames. The portable HTML embeds every
snapshot and offers play/pause, frame stepping, a timeline, speed selection,
and keyboard controls. `fps` must be finite and in `(0, 1000]`; the timer
interval is never below one millisecond.

## Compressed classic event replay

A classic learner decision may autoplay several opponents, so its before and
after states alone would skip visible core actions. The adapter records up to
`MAX_DECISION_EVENTS = 7` actions in `ClassicSingleAgentInfo`:

```python
before = state
state, observation, reward, terminated, info = env.step(state, action)
frames = env.replay_events(before, info)
```

`replay_events` returns `before.match` followed by one `ClassicState` for each
valid recorded event. It uses `event_count` and `event_valid` to ignore sentinel
padding. Expansion is deterministic and host-side; do not call it inside
`jax.jit`. The resulting frames can be sent to classic snapshot or GIF
renderers. They are not inputs to the `simplest-v0` HTML replay helper.

## Animated previews

Install Pillow support with `python -m pip install -e '.[render]'`, then use:

```python
heart.save_gif(states, "heart-preview.gif", viewer=0, duration_ms=95)
heart.save_classic_gif(classic_states, "classic-preview.gif", viewer=0)
```

GIF rendering is a host-only convenience. Output paths must end in `.gif`.
Non-16:9 sizes use a centered 960×540 virtual canvas with letterboxing rather
than distorting seat coordinates.

Generated output reveals every player's cards and may be large when many
matches are retained. Treat it as evaluation output, not a training artifact;
generated game and replay HTML files are ignored by Git.

HTML replay is the canonical inspectable output in 0.1.0. MP4 encoding and
pixel-level card-motion animation are not yet part of the public contract.
