# Environment contract

## Construction and lifecycle

```python
env = heart.make("simplest-v0")
state, observation = env.reset(key)
state, observation, rewards, terminated, info = env.step(state, action)
```

`HeartEnv` is an immutable ruleset handle. Randomness enters through the reset
key and rule-policy keys; `step` has no hidden random source or mutable state.
One legal action plays one card, so every completed episode contains exactly 52
transitions.

`env.observe(state, player)` projects state for one player. Only the active
player receives a non-empty action mask. `env.legal_action_mask(state)` returns
the authoritative mask for the current actor.

## Card and action encoding

Actions are scalar integer card IDs in `[0, 52)`. Suits occupy contiguous
13-card blocks in this order:

| IDs | Suit | Rank order |
| --- | --- | --- |
| 0–12 | Clubs | 2, 3, ..., 10, J, Q, K, A |
| 13–25 | Diamonds | 2, 3, ..., 10, J, Q, K, A |
| 26–38 | Spades | 2, 3, ..., 10, J, Q, K, A |
| 39–51 | Hearts | 2, 3, ..., 10, J, Q, K, A |

Thus 2♣ is action `0` and Q♠ is action `36`. The action mask has shape `(52,)`
and boolean dtype.

## State

`State` is the omniscient fixed-shape environment PyTree.

| Field | Shape | Meaning |
| --- | --- | --- |
| `hands` | `(4, 52)` | Boolean ownership matrix |
| `current_trick` | `(4,)` | Cards in play order; `-1` marks unused positions |
| `trick_history` | `(13, 4)` | Completed tricks in play order |
| `trick_winners` | `(13,)` | Winner per completed trick; `-1` if unfinished |
| `penalties` | `(4,)` | Raw captured penalty points |
| `scores` | `(4,)` | Effective scores; moon-adjusted at termination |
| `moon_shooter` | scalar | Shooter ID or `-1` |
| `winner_mask` | `(4,)` | Terminal winner flags |
| `leader`, `active_player` | scalar | Trick leader and current actor |
| `trick_index`, `trick_position` | scalar | Completed tricks and cards in current trick |
| `hearts_broken` | scalar | Whether a heart has been played |
| `num_cards_played` | scalar | Legal actions applied so far |
| `terminated` | scalar | Deal completion flag |

Consumers should not depend on incidental integer widths; shapes, meanings,
and sentinel values are the public contract.

## Observation

`Observation` is player-private policy input. It includes the selected player's
hand, all hand sizes, current and completed tricks, winners, scores, turn
metadata, and that player's action mask. It does **not** expose opponent hands.

This differs deliberately from rendering: renderers consume omniscient `State`
and show every hand for inspection, while agents should consume `Observation`.

## Transition and diagnostics

`rewards` has shape `(4,)` and is all zero before termination. `Info` reports
invalid actions, trick/deal completion, the latest trick winner and points,
moon shooter, and terminal winner mask.

An out-of-range or masked action fails closed: state is unchanged,
`info.invalid_action` is true, and rewards remain zero. Calling `step` again
after termination is therefore also invalid.

The API is designed for `jax.jit`, `jax.vmap`, and `jax.lax.scan`. Host-side
Python conversion of traced values, rendering, file I/O, and logging must stay
outside compiled rollout functions.
