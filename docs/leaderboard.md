<div align="center">

# 🏆 HEART Leaderboard

**Submit a policy as a computation graph. Any architecture, no code on the host.**

[Standings](#standings) · [Contract](#the-contract) · [Observation](#what-a-policy-sees) · [Budgets](#budgets) · [Build an entry](#build-an-entry) · [Ranking](#how-entries-are-ranked)

</div>

---

## Standings

<div align="center">

### 🥇 &nbsp; PSRO &nbsp; — &nbsp; `1531` Elo

<sub>8,192 tables · rotating seats · shared deals · every rank separated</sub>

</div>

| | Entry | Elo | | Deal reward | Seats |
|:--:|---|--:|:--|--:|--:|
| 🥇 | **PSRO** | **1531** <sub>± 2</sub> | `███████████` | **+0.0308** <sub>± 0.0022</sub> | 6,532 |
| 🥈 | **RL** | **1516** <sub>± 2</sub> | `█████████` | **+0.0148** <sub>± 0.0021</sub> | 6,564 |
| 🥉 | rule-hard | 1499 <sub>± 2</sub> | `██████` | −0.0007 <sub>± 0.0023</sub> | 6,618 |
| 4 | rule-medium | 1482 <sub>± 2</sub> | `███` | −0.0186 <sub>± 0.0023</sub> | 6,520 |
| 5 | rule-easy | 1471 <sub>± 2</sub> | `█` | −0.0263 <sub>± 0.0023</sub> | 6,534 |

<div align="center">
<sub>

`RL` — recurrent PPO trained on its own &nbsp;•&nbsp; `PSRO` — best response to a population containing `RL` and the rule tiers

</sub>
</div>

Rewards sum to **−0.0001**, as a zero-sum field must. Every rank is separated:
the closest pair, `rule-medium` to `rule-easy`, differs by 0.008 against a
combined error of 0.003.

> **Read it honestly.** This field is close to what `PSRO` trained against, so
> the league flatters it. On a different measurement — each policy seated
> against three copies of one rule tier — `RL` finishes ahead of `PSRO` by about
> 0.01 on all three tiers. Both numbers are real and they answer different
> questions: how a policy fares *in this field*, and how it generalises to
> opponents it never met. A leaderboard answers only the first.

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
The host masks logits against the rules before drawing an action, so an entry
may leave illegal actions unscored. Everything inside the graph is yours.

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

Two figures are reported, because they answer different questions:

- **Deal reward** — the mean normalized zero-sum reward per deal. It is the
  quantity the environment optimises and has the smaller variance.
- **Elo** — every table is read as its six seat-versus-seat pairings and fitted
  to a rating. Sequential Elo would depend on the order games happened to run,
  so the fit repeats to convergence over the whole record; the spread comes from
  resampling the pairings.

Entries whose intervals overlap **share a rank** rather than being ordered by
noise. That is not a formality: at 3,000 seats per entry the error is ±0.003 and
most of a five-entry field ties. Resolving differences of 0.01 takes on the
order of 10,000 seats each.

```python
from heart.contest import run_league

for standing in run_league(entries, lineups=2048, rounds=8, seed=0):
    print(standing.rank, standing.name, standing.elo, standing.reward)
```
