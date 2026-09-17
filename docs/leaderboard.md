<div align="center">

# 🏆 HEART Leaderboard

**Submit a policy as a computation graph. Any architecture, no code on the host.**

[Standings](#standings) · [Contract](#the-contract) · [Observation](#what-a-policy-sees) · [Budgets](#budgets) · [Build an entry](#build-an-entry) · [Ranking](#how-entries-are-ranked)

</div>

---

## Standings

<div align="center">

### 🥇 &nbsp; PSRO &nbsp; · &nbsp; **6.20** penalty points per deal

<sub>8,192 complete matches to 100 points · 363,900 deals · rotating seats · shared deals</sub>

</div>

| | Entry | Penalty / deal | vs field | Won | Last | Elo | Matches | Deals |
|:--:|---|--:|--:|--:|--:|--:|--:|--:|
| 🥇 | **PSRO** | **6.197** <sub>± 0.029</sub> | **−0.61** | **33.2 %** | **16.6 %** | **1548** <sub>± 2</sub> | 6,532 | 72,377 |
| 🥈 | **RL** | **6.442** <sub>± 0.029</sub> | −0.36 | 29.0 % | 18.2 % | **1532** <sub>± 2</sub> | 6,564 | 72,901 |
| 🥉 | rule-hard | 6.806 <sub>± 0.031</sub> | +0.00 | 25.2 % | 26.1 % | 1496 <sub>± 2</sub> | 6,618 | 73,687 |
| 4 | rule-medium | 7.210 <sub>± 0.032</sub> | +0.41 | 21.4 % | 31.8 % | 1469 <sub>± 2</sub> | 6,520 | 72,429 |
| 5 | rule-easy | 7.358 <sub>± 0.032</sub> | +0.55 | 18.5 % | 33.1 % | 1456 <sub>± 1</sub> | 6,534 | 72,506 |

<div align="center">
<sub>

`RL` is recurrent PPO trained alone; `PSRO` is the best response a population
method produced against a field containing `RL` and the rule tiers.

</sub>
</div>

### What the numbers mean

| Column | Meaning |
| --- | --- |
| **Deal** | one hand: thirteen tricks, dealt and played out. A deal hands out **26 penalty points** — one per heart, thirteen for the queen of spades — so four seats average **6.5** each. |
| **Match** | a `classic-v0` game, played until someone reaches **100 points**. That takes about eleven deals, and the lowest cumulative score wins. Matches here are played to the end, so the endgame near 100 counts. |
| **Penalty / deal** | average points this entry took per deal. **Lower is better** — this is what Hearts itself counts, not a reward this environment invented. |
| **vs field** | the same figure minus the field's average of **6.80**. Negative means the entry takes fewer points than the table around it. |
| **Won** | share of **complete matches** whose final cumulative score was the lowest at the table — the game's own definition of winning, not a per-deal count. Ties count for everyone tied, so the column sums above 100 %. |
| **Last** | share of **complete matches** whose final cumulative score was the **highest** — again per match to 100 points, not per deal. A policy can win often and still collapse often; these two columns separate steady play from streaky play. |
| **Elo** | fitted from every match's six seat-versus-seat pairings, won by the lower final score. It says who beats whom; the points column says by how much. |
| **Matches / Deals** | the sample each figure is computed over. |

The field averages 6.80 rather than 6.5 because a shot moon pays 78 points
instead of 26. Every rank is separated — the closest pair, `rule-medium` to
`rule-easy`, differs by 0.15 against a combined error of 0.045 — and the rule
tiers land in their designed order, which is a check on the measurement as much
as on them.

`PSRO` and `RL` are close on points but further apart on **Last**: 16.6 % versus
18.2 %. The population method's edge is as much in avoiding disasters as in
winning outright.

> **Read it honestly.** This field is close to what `PSRO` trained against, so
> the league flatters it. On a different measurement — each policy seated
> against three copies of one rule tier — `RL` finishes ahead. Both numbers are
> real and they answer different questions: how a policy fares *in this field*,
> and how it generalises to opponents it never met. A leaderboard answers only
> the first.

<sub>The five entries above are this project's own policies. They read richer
observations than this contract offers and so cannot be submitted through it;
they are ranked by the same statistics and are here to show a populated table.
Submitted entries will appear alongside them.</sub>

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
(observations: float32[b, 228])
    -> (pass_logits: float32[b, 286], play_logits: float32[b, 52])
```

`b` must be exported as a **symbolic** dimension so the host picks the batch.
Everything inside the graph is yours.

Three things follow from the signature, and they are easy to miss:

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

Every segment is `float32` and **seat-relative**: index 0 is the seat to act, 1
the seat to its left, and so on, so a policy never learns absolute seats.

| Segment | Width | Meaning |
| --- | ---: | --- |
| `hand` | 52 | cards the seat holds |
| `played` | 52 | cards already gone this deal |
| `table` | 52 | cards lying in the current trick |
| `legal` | 52 | cards the rules allow right now |
| `position` | 4 | how many seats played before this one |
| `taken_points` | 4 | deal penalties so far, ÷ 26 |
| `match_scores` | 4 | cumulative match scores, ÷ 100 |
| `pass_direction` | 4 | left, right, across, hold |
| `phase` | 2 | passing, playing |
| `flags` | 2 | hearts broken, queen of spades gone |
| **total** | **228** | `heart.contest.OBSERVATION_DIM` |

`heart.contest.OFFSETS` gives each segment's exact slice, and
`heart.contest.encode_observation(state, player)` is the encoder the league
itself runs — there is no second implementation to drift from.

## Budgets

| Limit | Value | Why |
| --- | ---: | --- |
| Serialised size | 8 MiB | a graph this large is not a policy |
| Compute | 5,000,000 flops per decision | one seat must not starve the league |
| Outputs | finite | a non-finite logit decides nothing |

Compute is read from the **compiled graph's cost analysis** and divided by the
batch, so it does not depend on the machine that measures it. For scale:

| Entry | Flops per decision | Verdict |
| --- | ---: | :---: |
| the packaged baseline | 1,599 | accepted |
| one 32-unit hidden layer | 54,336 | accepted |
| twenty 228-wide layers | 2,337,456 | accepted |
| forty 512-wide layers | 21,551,104 | **refused** |
| sixty 1024-wide layers | 126,988,288 | **refused** |

## Build an entry

```python
import jax, jax.numpy as jnp
from pathlib import Path
from jax import export
from heart.contest import OBSERVATION_DIM

def policy(observations):                 # any architecture you like
    hidden = jnp.tanh(observations @ first_weights)
    return hidden @ pass_weights, hidden @ play_weights

batch = export.symbolic_shape("b")[0]
signature = jax.ShapeDtypeStruct((batch, OBSERVATION_DIM), jnp.float32)
Path("entry.bin").write_bytes(export.export(jax.jit(policy))(signature).serialize())
```

Serialising needs `flatbuffers`: `pip install 'heart-marl[contest]'`.

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
