# Leaderboard submissions

An entry is a policy exported as a portable computation graph. The host never
imports submitter code: it reads the graph, checks it against the contract
below, and runs it inside XLA. A graph cannot open a file, reach the network, or
call back into the host process — `jax.export` refuses to serialise host
callbacks at all — so a submission is data, not a program the host trusts.

## The contract

```text
(observations: float32[b, 228])
    -> (pass_logits: float32[b, 286], play_logits: float32[b, 52])
```

`b` must be exported as a **symbolic** dimension so the host chooses the batch.
Logits are masked by the host before an action is drawn, so an entry may leave
illegal actions unpenalised. Any architecture is allowed; only this signature
and the budgets below are fixed.

### What an observation holds

Every segment is `float32` and seat-relative: index 0 is the seat to act, 1 the
seat to its left, and so on, so a policy never learns absolute seats.

| Segment | Width | Meaning |
| ------- | ----- | ------- |
| `hand` | 52 | cards the seat holds |
| `played` | 52 | cards already gone this deal |
| `table` | 52 | cards in the current trick |
| `legal` | 52 | cards the rules allow right now |
| `position` | 4 | how many seats have played before this one |
| `taken_points` | 4 | deal penalties so far, divided by 26 |
| `match_scores` | 4 | cumulative match scores, divided by 100 |
| `pass_direction` | 4 | left, right, across, hold |
| `phase` | 2 | passing, playing |
| `flags` | 2 | hearts broken, queen of spades already played |

`heart.contest.OFFSETS` gives the exact slice of each segment, and
`heart.contest.encode_observation(state, player)` is the encoder the league
itself uses.

## Budgets

| Limit | Value | Why |
| ----- | ----- | --- |
| Serialised size | 8 MiB | a graph this large is not a policy |
| Compute | 5,000,000 flops per decision | one seat must not starve the league |
| Outputs | finite | a non-finite logit decides nothing |

Compute is measured from the compiled graph's cost analysis, divided by the
batch, so it does not depend on the machine. For scale: a 32-unit hidden layer
costs about 54,000 flops per decision, forty 512-wide layers costs about
21,500,000 and is refused.

## Preparing an entry

```python
import jax, jax.numpy as jnp
from jax import export
from heart.contest import OBSERVATION_DIM

def policy(observations):           # any architecture you like
    hidden = jnp.tanh(observations @ weights_one)
    return hidden @ weights_pass, hidden @ weights_play

batch, = export.symbolic_shape("b")
signature = jax.ShapeDtypeStruct((batch, OBSERVATION_DIM), jnp.float32)
blob = export.export(jax.jit(policy))(signature).serialize()
Path("entry.bin").write_bytes(blob)
```

Serialising needs `flatbuffers`, which comes with the `contest` extra:
`pip install 'heart-marl[contest]'`.

## Checking an entry before you send it

```python
from heart.contest import load_submission

entry = load_submission(Path("entry.bin").read_bytes(), "my-entry")
print(entry.flops_per_decision, entry.size_bytes)
```

`load_submission` raises `SubmissionError` with the reason when an entry is
refused, and the host runs exactly this check. An entry that loads here will not
be turned away for the reasons above.

## Ranking

Entries meet in sampled four-seat line-ups. Seats rotate and every line-up is
dealt from a shared seed, so entries are compared on the same deals rather than
on their luck. The ranking metric is the mean normalized deal reward against the
field, reported with its standard error; two entries whose intervals overlap
share a rank rather than being ordered by noise.
