# `classic-v0` rules and learning contract

`classic-v0` is HEART's complete four-player Hearts match. It keeps the same
JAX-native card-play rules as the deal core, restores standard Q♠ scoring and
passing, and carries scores across deals until the 100-point boundary.

## Match rules

Four independent players receive 13 cards from a uniformly shuffled 52-card
deck. Each heart is 1 penalty point and Q♠ is 13, for 26 raw points per deal.
The player holding 2♣ opens after passing. Following suit, first-trick point
restrictions, hearts-broken leading, trick resolution, and shooting the moon
follow the same precedence described in the [deal rules](rules.md).

A moon shooter receives 0 effective points for that deal and every opponent
receives 26. Effective deal scores are added to `match_scores`.

The match is checked only after a complete deal. Once any player has at least
100 points, every player tied for the lowest cumulative score is a winner. This
shared-lowest result is an intentional `classic-v0` project rule: the match does
not add tie-break deals.

## Passing

Passing rotates by zero-based `deal_index`:

| `deal_index % 4` | Direction | Recipient of cards from player `p` |
| ---------------- | --------- | ---------------------------------- |
| 0                | left      | `(p + 1) % 4`                      |
| 1                | right     | `(p - 1) % 4`                      |
| 2                | across    | `(p + 2) % 4`                      |
| 3                | hold      | no cards move                      |

On a passing deal, each player makes one scalar selection before card play.
The action space contains all `13C3 = 286` unordered triples of slots in that
player's sorted 13-card hand. The exchange is simultaneous after all four
selections; each player still owns exactly 13 cards afterward. Passing can move
2♣, so its new owner becomes the opening player.

The core event count for one deal is therefore:

- 56 events on left, right, and across deals: four pass selections plus 52 card
  plays;
- 52 events on hold deals: 52 card plays and no pass selection.

`ClassicObservation.pass_action_mask` has shape `(286,)` and
`play_action_mask` has shape `(52,)`. Exactly the mask for the current phase and
actor is enabled. Other players' pass choices and received cards remain private
in policy observations; the omniscient renderer view (`viewer=None`)
deliberately reveals them, while a seat view does not.

## Reward and discounting

Core rewards have shape `(4,)`. Every non-boundary event returns zero. When a
deal completes, let `d_i` be player `i`'s moon-adjusted effective deal score:

```text
reward_i = -d_i
```

The reward is the negative effective penalty score and is emitted after every
deal, including the terminal deal. It is not normalized or made zero-sum. A
moon shot therefore returns `0` for the shooter and `-26` for every opponent.
Match termination and cumulative winners are reported separately through
`ClassicInfo` and `ClassicState`. Reward normalization belongs in a training
pipeline rather than in the game contract.

For episodic training, `gamma = 1` is recommended. The environment already
provides sparse deal-boundary rewards, and discounting by engine events or by
single-learner decisions would otherwise make the same deal depend on seat and
the number of automatically played opponents. The single-learner adapter
returns `info.discount = 1` before match termination and `0` at termination.

## Single-learner decisions

Create the adapter with:

```python
env = heart.make_single_agent(
    "classic-v0",
    controlled_player=0,
    pass_opponents="medium",
    play_opponents="hard",
)
```

The adapter automatically advances the other three seats until the controlled
player acts again. A passing deal has exactly 14 learner decisions: one
286-way pass decision and 13 card-play decisions. A hold deal has exactly 13
card-play decisions. `pass_hand_cards` and `play_hand_cards` expose the stable
slot-to-card mappings needed to interpret the phase-specific masks.

One learner decision can contain several underlying core events. The returned
`ClassicSingleAgentInfo` stores a fixed-size compressed trace:

| Field           | Meaning                                      |
| --------------- | -------------------------------------------- |
| `event_actions` | Core pass-combination IDs or global card IDs |
| `event_players` | Acting player for each core event            |
| `event_phases`  | `PASS` or `PLAY` for each event              |
| `event_valid`   | Valid prefix mask for the fixed-size arrays  |
| `event_count`   | Number of valid recorded events              |

The trace capacity is `MAX_DECISION_EVENTS = 7`; unused entries use sentinel
values and have `event_valid=False`. It is learning metadata, not a sequence of
full states, so it keeps recurrent batches compact.

For host-side replay, retain the state from immediately before `env.step` and
expand the returned trace:

```python
before = state
state, observation, reward, terminated, info = env.step(state, action)
frames = env.replay_events(before, info)
```

`replay_events` deterministically reapplies the recorded core actions and
returns the initial frame plus one frame per valid event. Call it outside
`jax.jit`; it converts the compact trace into Python render states.

## Deal boundaries and rendering

On a non-terminal deal boundary, `ClassicState.game` already contains the newly
dealt next hand while these fields retain the completed result:

- `deal_boundary=True`;
- `last_deal_scores` and `last_deal_rewards`;
- `last_deal_moon_shooter`;
- `last_trick_cards` and `last_trick_winner`;
- updated `match_scores` and next `deal_index`/`pass_direction`.

The next valid core event clears `deal_boundary`. The single-learner adapter
keeps the flag sticky when opponent autoplay crosses a deal boundary, so the
learner cannot miss it. At terminal completion, the final deal remains in
`game`, `phase=TERMINAL`, and `winner_mask` describes the match winners.

Use `render_classic_ansi`, `render_classic_html`, `render_classic_gif_frame`,
or `save_classic_replay_html` for these states. Boundary renderers pause on the
completed-deal overlay, reproduce the last trick from its compact snapshot,
and announce the next direction before the following event reveals the pass
screen. A hold boundary explicitly says that play begins immediately without
passing.
