<div align="center">

# 🏆 HEART Leaderboard

**Submit a policy as a computation graph. Any architecture, no code on the host.**

[Standings](#standings) · [Contract](#the-contract) · [Observation](#what-a-policy-sees) · [Budgets](#budgets) · [Build an entry](#build-an-entry) · [Submit](#sending-it-in) · [Ranking](#how-entries-are-ranked)

</div>

---

## Standings

<!-- standings:start -->
<div align="center">

### 🥇 &nbsp; RL (TA) &nbsp; · &nbsp; **5.21** penalty points per deal

<sub>8,192 complete matches to 100 points · 83,942 deals · rotating seats · shared deals</sub>

</div>

| | Entry | Penalty / deal | vs field | Won | Last | Elo | Matches |
| :--: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 🥇 | **RL (TA)** | **5.205** &nbsp;<sub>± 0.031</sub> | −1.54 | 40.6 % | 7.1 % | **1618** &nbsp;<sub>± 2</sub> | 5,460 |
| 🥈 | **PSRO (TA)** | **5.308** &nbsp;<sub>± 0.030</sub> | −1.44 | 38.4 % | 7.6 % | **1609** &nbsp;<sub>± 1</sub> | 5,440 |
| 🥉 | rule-hard | 6.083 &nbsp;<sub>± 0.034</sub> | −0.66 | 27.5 % | 15.1 % | 1541 &nbsp;<sub>± 2</sub> | 5,495 |
| 4 | rule-medium | 6.495 &nbsp;<sub>± 0.036</sub> | −0.25 | 23.0 % | 20.0 % | 1510 &nbsp;<sub>± 2</sub> | 5,421 |
| 5 | rule-easy | 6.604 &nbsp;<sub>± 0.035</sub> | −0.14 | 21.8 % | 21.1 % | 1500 &nbsp;<sub>± 2</sub> | 5,509 |
| 6 | baseline | 10.774 &nbsp;<sub>± 0.039</sub> | +4.03 | 1.5 % | 80.0 % | 1222 &nbsp;<sub>± 1</sub> | 5,443 |

<div align="center">
<sub>

Updated 2026-09-18 · field average **6.74** points per deal

</sub>
</div>
<!-- standings:end -->

> **Read it honestly.** `(TA)` marks the two policies this project trained; they
> are here as a yardstick, not as contenders. `baseline` is the packaged floor —
> it plays the cheapest legal card and nothing else — and a weak seat absorbs
> points, which lifts everyone else at the table. The figures move when the field
> moves, so compare entries within one run rather than across runs.

<sub>Every entry above, the packaged policies included, went through this same
contract: the same observation, the same action space, the same checks. A
reference policy that could not itself be submitted would not be a fair
yardstick.</sub>

### What the numbers mean

| Column | Meaning |
| --- | --- |
| **Deal** | one hand: thirteen tricks, dealt and played out. A deal hands out **26 penalty points** — one per heart, thirteen for the queen of spades — so four seats average **6.5** each. |
| **Match** | a `classic-v0` game, played until someone reaches **100 points**. That takes about eleven deals, and the lowest cumulative score wins. Matches here are played to the end, so the endgame near 100 counts. |
| **Penalty / deal** | average points this entry took per deal. **Lower is better** — this is what Hearts itself counts, not a reward this environment invented. |
| **vs field** | the same figure minus the field's average. Negative means the entry takes fewer points than the table around it. |
| **Won** | share of **complete matches** whose final cumulative score was the lowest at the table — the game's own definition of winning, not a per-deal count. Ties count for everyone tied, so the column sums above 100 %. |
| **Last** | share of **complete matches** whose final cumulative score was the **highest** — again per match to 100 points, not per deal. A policy can win often and still collapse often; these two columns separate steady play from streaky play. |
| **Elo** | fitted from every match's six seat-versus-seat pairings, won by the lower final score. It says who beats whom; the points column says by how much. |
| **Matches** | complete matches this entry sat in — the sample each figure is computed over. Each runs about eleven deals. |

A field averages more than 6.5 because a shot moon pays 78 points instead of 26.

---

A leaderboard has to run strangers' policies. Weights alone would force everyone
into one architecture; a Python plugin would hand each submitter the host
process. An entry here is neither: it is a policy **exported to a portable
computation graph**.

| | Any architecture | Runs submitter code | Checkable before running |
| --- | :---: | :---: | :---: |
| Weights only | no | no | needs hand-written validation |
| **Graph (this contract)** | **yes** | **no** | **signature, size, compute** |
| Python plugin | yes | **yes** | only inside a sandbox |

The host deserialises the graph, checks it, and runs it inside XLA. `jax.export`
refuses to serialise host callbacks at all, so a graph has no route to the file
system, the network, or the host process.

## The contract

```text
(observation: ClassicSingleAgentObservation)
    -> (pass_logits: float32[b, 286], play_logits: float32[b, 13])
```

**This is the environment's own interface, not a second one.** An entry receives
exactly the observation `heart.make_classic_single_agent` gives a learner, and
chooses from exactly the actions that learner may take. Nothing is re-encoded on
the way in, so an entry sees what the packaged policies see — and a packaged
policy is itself a valid entry.

Every leaf of the observation carries a leading **symbolic** batch dimension, so
the host picks the batch. `heart.contest.observation_signature()` is that
signature and `heart.contest.sample_observation()` is one unbatched example.

Three things follow from the interface, and they are easy to miss:

- **The host masks and then takes the argmax.** Illegal actions are set to
  negative infinity before the choice, so an entry may leave them unscored. Only
  the ordering of legal logits matters; their scale does not.
- **An entry is deterministic.** The contract passes no random key, so a policy
  is a fixed function of the observation and cannot play a mixed strategy.
  Variety across games comes from the deals, not from the entry.
- **Both heads are always called.** During the passing phase the play logits are
  ignored and vice versa, so an entry may return anything for the head that is
  not in use — but it must return the right shape.

## What a policy sees

The observation is `ClassicSingleAgentObservation`: the match view plus the
seat's own hand, laid out as thirteen slots.

| Field | What it holds |
| --- | --- |
| `match.game` | the deal: hands as seen, the current trick, trick history and winners, penalties, who is to act, whether hearts are broken |
| `match.match_scores` | cumulative scores, the figures the hundred-point rule reads |
| `match.deal_index`, `match.phase`, `match.pass_direction` | where the match is |
| `match.cards_passed`, `match.cards_received` | what this seat gave and got |
| `pass_hand_cards`, `play_hand_cards` | the seat's thirteen slots, ascending, `-1` for a hole |
| `pass_action_mask`, `play_action_mask` | which of the 286 triples and thirteen slots the rules allow |

**Hand slots hold still for a deal.** A card that is played leaves its slot
behind and the mask hides it, so slot 4 means the same card on every turn of a
deal. `play_hand_cards` is refreshed once passing has handed the cards over —
or, in a deal that skips passing, when the deal opens. While a deal is still
being passed it therefore still holds the **previous** deal's layout, which
costs nothing, since the play head is ignored until the passing phase ends.
Read the hand you are about to play from `play_hand_cards`, and the hand you
are passing from `pass_hand_cards`.

The play head chooses a **slot**, not a card, which is the environment's own
action space — and it lets a set-equivariant policy answer per card without
learning anything about position.

## Budgets

| Limit | Value | Why |
| --- | ---: | --- |
| Serialised size | 8 MiB | a graph this large is not a policy |
| Compute | 5,000,000 flops per decision | one seat must not starve the league |
| Platforms | must cover the host's | an entry that cannot run cannot be ranked |
| Outputs | finite | a non-finite logit decides nothing |

Compute is read from the **compiled graph's cost analysis** and divided by the
batch, so it does not depend on the machine that measures it.

## Build an entry

```python
import jax.numpy as jnp
from pathlib import Path
from heart.contest import export_policy

def policy(observation):                  # any architecture you like
    hand = observation.play_hand_cards.astype(jnp.float32)
    hidden = jnp.tanh(hand @ first_weights)
    return hidden @ pass_weights, hidden @ play_weights

Path("entry.bin").write_bytes(export_policy(policy))
```

`export_policy` exports for the platforms this machine can run, which is what
the host checks against. Serialising needs `flatbuffers`:
`pip install 'heart-marl[contest]'`.

### Check it before you send it

```python
from heart.contest import load_submission

entry = load_submission(Path("entry.bin").read_bytes(), "my-entry")
print(entry.flops_per_decision, entry.size_bytes)
```

`load_submission` raises `SubmissionError` with the reason. The host runs
exactly this check, so an entry that loads here will not be turned away for any
reason listed above.

### A worked entry

`heart.contest.baseline_blob()` returns a complete, valid entry — play the
cheapest legal card, pass the three highest. It is the floor the standings are
measured against and the shortest example of the contract in use.
`submissions/heart-marl/` is that entry, laid out the way a submission looks.

## Sending it in

Entries arrive as a pull request. **The directory is your team** — name it after
your GitHub handle — and it holds the graph you exported plus a one-line
manifest:

```
submissions/
└── your-github-handle/
    ├── entry.bin     # what you exported above
    └── entry.toml
```

```toml
name = "clever-passer"        # what the standings call it; optional, defaults to the directory
description = "one line: what the policy does"
```

1. Fork the repository and create a branch.
2. Add — or edit — `submissions/<your-handle>/`.
3. Run `python submissions/validate.py`, the same check CI runs.
4. Open the pull request.

CI validates the entry on the pull request, so a malformed file is refused
before a human looks at it. After a merge the league is re-run and the standings
above are updated.

### Submitting again replaces what you had

A team holds one entry, and that is a property of where the file lives rather
than a rule anyone has to enforce: your second submission writes to the same
`submissions/<your-handle>/entry.bin`, so it **replaces** the first. Git records
it as a modification, the league re-runs, and your row moves. There is no second
row to retire.

You may rename the entry whenever you like — `name` is only what the standings
print — but two teams cannot show the same name, since a table nobody can read
is worse than an awkward name.

A team directory is 2 to 39 lowercase letters, digits or hyphens, matching the
shape of a GitHub handle.

## How entries are ranked

Entries meet in sampled four-seat line-ups. **Seats rotate** and every round
**deals from a shared seed**, so entries are compared on the same cards rather
than on their luck.

The ranking figure is **penalty points per deal, lower being better**. That is
the game's own score. A reward is this environment's normalisation of those
points — zero-sum, divided by the deal's total — and ranking on it would make
the standings depend on a modelling choice rather than on Hearts. Two entries
whose intervals overlap **share a rank** rather than being ordered by noise.

**Elo** is reported beside it. Every table is read as its six seat-versus-seat
pairings, won by the lower score, and fitted to a rating; sequential Elo would
depend on the order games happened to run, so the fit repeats over the whole
record. Its spread comes from resampling the pairings.

Sharing a rank is not a formality. At roughly 6,500 deals per entry the error on
points is ±0.05, and separating entries that differ by 0.1 needs that order of
sample. A field measured over a few hundred deals will mostly tie, and it should
say so.

```python
from heart.contest import run_league

for standing in run_league(entries, lineups=2048, rounds=8, seed=0):
    print(standing.rank, standing.name, standing.points, standing.elo)
```
