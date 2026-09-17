# Environment contract

## Construction and lifecycle

The published environment is the complete match:

```python
env = heart.make("classic-v0")     # the default, and the only mode
state, observation = env.reset(key)
state, observation, rewards, terminated, info = env.step(state, action)
```

The deal core it runs on has its own handle, for testing and measuring the
transition on its own rather than as a second environment:

```python
deal = heart.DealEnv()             # 52 card plays, no passing, no match score
state, observation = deal.reset(key)
```

Both handles are immutable. Randomness enters through explicit reset and policy
keys; `step` has no hidden random source or mutable state. A deal ends after
exactly 52 card-play transitions. The classic core spans complete deals:
passing deals contain four pass selections plus 52 card plays (56 events), and
hold deals contain 52 card plays.

`env.observe(state, player)` projects private policy input. Only the current
actor receives a non-empty mask for the current phase.

## Card and action encoding

Card-play actions are scalar integer IDs in `[0, 52)`. Suits occupy contiguous
13-card blocks:

| IDs   | Suit     | Rank order                |
| ----- | -------- | ------------------------- |
| 0–12  | Clubs    | 2, 3, ..., 10, J, Q, K, A |
| 13–25 | Diamonds | 2, 3, ..., 10, J, Q, K, A |
| 26–38 | Spades   | 2, 3, ..., 10, J, Q, K, A |
| 39–51 | Hearts   | 2, 3, ..., 10, J, Q, K, A |

Thus 2♣ is action `0` and Q♠ is action `36`.

A deal exposes one boolean `(52,)` action mask. `classic-v0` exposes a
phase-specific `(286,)` pass mask and `(52,)` play mask. The 286 pass actions
are all `13C3` unordered triples of slots in the acting player's sorted
13-card hand. On hold deals the pass phase is skipped.

## Single-learner adapters

For one fixed deal against three built-in opponents:

```python
env = heart.make_single_agent(
    controlled_player=0,
    opponents=("easy", "medium", "hard"),
)
state, observation = env.reset(key)
state, observation, reward, terminated, info = env.step(state, slot)
```

This adapter has a fixed 13-action interface. Slot `i` refers to the same card
from the controlled player's initial sorted hand; played and currently illegal
slots are masked. Opponents autoplay until the learner acts again. The external
episode has exactly 13 decisions while the core applies all 52 card plays.

For a complete classic match:

```python
env = heart.make_single_agent(
    "classic-v0",
    controlled_player=0,
    pass_opponents="medium",
    play_opponents="hard",
)
```

A passing deal exposes one 286-way pass choice and 13 card choices, exactly 14
learner decisions. A hold deal exposes exactly 13 card choices. Opponent events
are automatically advanced and recorded in `ClassicSingleAgentInfo`.

## Deal state and observation

`State` is the omniscient fixed-shape deal PyTree.

| Field                           | Shape     | Meaning                                          |
| ------------------------------- | --------- | ------------------------------------------------ |
| `hands`                         | `(4, 52)` | Boolean ownership matrix                         |
| `current_trick`                 | `(4,)`    | Cards in play order; `-1` marks unused positions |
| `trick_history`                 | `(13, 4)` | Completed tricks in play order                   |
| `trick_winners`                 | `(13,)`   | Winner per completed trick; `-1` if unfinished   |
| `penalties`                     | `(4,)`    | Raw captured penalty points                      |
| `scores`                        | `(4,)`    | Effective scores; moon-adjusted at termination   |
| `moon_shooter`                  | scalar    | Shooter ID or `-1`                               |
| `winner_mask`                   | `(4,)`    | Terminal winner flags                            |
| `leader`, `active_player`       | scalar    | Trick leader and current actor                   |
| `trick_index`, `trick_position` | scalar    | Completed tricks and cards in current trick      |
| `hearts_broken`                 | scalar    | Whether a heart has been played                  |
| `num_cards_played`              | scalar    | Legal actions applied so far                     |
| `terminated`                    | scalar    | Deal completion flag                             |

`Observation` contains the selected player's hand, all hand sizes, public
tricks and winners, scores, turn metadata, and that player's action mask. It
does not expose opponent hands. `rewards` has shape `(4,)` and is zero before
the final card; `Info` reports invalid actions, trick/deal completion, points,
moon shooter, and terminal winners.

## `classic-v0` state and observation

`ClassicState` wraps the current deal in `game` and adds cumulative
`match_scores`, zero-based `deal_index`, phase, active player, pass direction,
pass bookkeeping, previous-deal summary, boundary flag, match winner mask, and
terminal flag. `ClassicObservation` contains a private core observation plus
cumulative scores and the separate pass and play masks. It exposes only the
selected player's passed and received cards.

`ClassicInfo` distinguishes trick, deal, and match completion. Core rewards are
zero inside a deal and are emitted at its boundary:

```text
reward_i = (mean(other effective deal scores) - own effective deal score) / 26
```

On a non-terminal boundary the next deal is already present in `state.game`.
`last_deal_scores`, `last_deal_rewards`, and `last_deal_moon_shooter` retain the
completed result. See the full [`classic-v0` contract](classic.md).

## Compressed classic learner events

One learner decision can contain several core events. `ClassicSingleAgentInfo`
stores fixed-size `event_actions`, `event_players`, `event_phases`, and
`event_valid` arrays together with `event_count`. The capacity is
`MAX_DECISION_EVENTS = 7`; invalid padding is excluded by the prefix mask.

Given the pre-decision state and returned info, `env.replay_events` expands the
trace into the initial `ClassicState` plus one state per valid event. This is a host-side replay operation, not a JIT transition.

`info.discount` is 1 before match termination and 0 at termination. Because
rewards are sparse at deal boundaries and learner transitions contain different
numbers of core events, episodic training should normally use `gamma=1`.

## Validation and diagnostics

An out-of-range or masked action fails closed: state is unchanged,
`info.invalid_action` is true, and reward is zero. Post-terminal actions are
also invalid. Float and boolean actions are not silently cast to action IDs.

Concrete player IDs passed to `observe` must be scalar integers in `[0, 4)` or
the call raises `TypeError`/`ValueError`. A dynamically traced invalid ID or
non-integer dtype cannot raise inside ordinary JIT; it fails closed without
revealing a hand, pass choice, received cards, or action mask.

Consumers should not depend on incidental integer widths. Documented shapes,
meanings, sentinel values, and phase behavior are the compatibility contract.

## Trusted rollout path

`heart.DealEnv` provides `step_unchecked(state, action)` for compiled code
that selected a scalar integer card ID from the authoritative mask. It skips
dtype, bounds, ownership, and legality validation but preserves the transition
and next observation. Behavior is undefined if its precondition is violated.
The classic core currently exposes only the safe phase-dependent `step`.

Both APIs are designed for `jax.jit`, `jax.vmap`, and JAX control flow.
Rendering, Python conversion, file I/O, and `replay_events` must stay outside
compiled rollout functions.
