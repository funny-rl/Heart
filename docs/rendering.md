# Rendering and replay contract

Rendering is a host-side inspection surface. It must not enter the compiled
training transition or enlarge the recurrent JAX state.

## Full observability

All renderers consume omniscient `State`. HTML and ANSI outputs reveal every
player's current hand. The `viewer` argument does not hide information:

- an integer `0`–`3` marks that player as local and places that seat at the
  bottom of the HTML table;
- `None` uses a neutral spectator view with player 0 at the bottom.

This contract is intended for debugging, evaluation, teaching, and replay.
Policies should receive player-private `Observation`, not renderer input.

## Snapshots

```python
text = heart.render_ansi(state, viewer=0)
page = heart.render_html(state, viewer=0)
path = heart.save_html(state, "heart-game.html", viewer=0)
```

The HTML is a complete dependency-free document. It shows all hands, current
trick placement, active player, legal cards for the active player, captured
hearts and Q♠, scores, winner state, and moon-shot state.

## Interactive replay

Retain a sequence containing reset state followed by every successor state:

```python
states = [state]
while not bool(state.terminated):
    state, observation, *_ = env.step(state, action)
    states.append(state)

heart.save_replay_html(states, "heart-replay.html", viewer=0, fps=2.0)
```

A complete canonical replay contains 53 frames. The portable HTML embeds every
snapshot and offers play/pause, previous/next frame, a range timeline, speed
selection, and space/arrow keyboard controls. `fps` must be positive and sets
the base frame rate.

Generated HTML can disclose every player's cards and may be large when many
matches are retained. Treat it as evaluation output, not a training artifact;
generated `heart-game*.html` and `heart-replay*.html` files are ignored by Git.

HTML replay is the canonical inspectable output in 0.1.0. MP4 encoding and
pixel-level card-motion animation are not yet part of the public contract.

## Animated preview

The optional Pillow renderer turns a short sequence of states into a looping
full-observability GIF suitable for a README or experiment index:

```python
heart.save_gif(states, "heart-preview.gif", viewer=0, duration_ms=95)
```

Install it with `python -m pip install -e '.[render]'`. GIF generation is a
host-only convenience surface and is not imported from Pillow until invoked.
