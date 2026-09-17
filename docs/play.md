# Human play

`heart.play` puts a person in one seat of a `classic-v0` match and policies in
the other three. It serves a small local page; the environment remains the
authority for every action.

## Running

```bash
python -m heart                                  # three medium opponents
python -m heart --seat easy --seat medium --seat hard
python -m heart --human-seat 2 --port 8080 --seed 7
```

The command prints the URL to open. `--seat` is given once per non-human seat in
seat order, or once to apply to all of them.

## Seat specifications

A seat is either a packaged rule tier — `easy`, `medium`, `hard` — or a plugin
reference `package.module:factory(argument)`. The factory is imported and called
with the argument string, and must return a `SeatPolicy`:

```python
def __call__(self, state: ClassicState, player: int, key: jax.Array) -> int
```

The return value is a card id during `PLAY` and a `PASS_COMBINATIONS` row index
during `PASS`. Because the hook is a plain import path, learned policies stay
outside this distribution and bring their own dependencies.

## HTTP contract

| Route | Method | Body | Returns |
| ----- | ------ | ---- | ------- |
| `/` | GET | — | The play page |
| `/state` | GET | — | Snapshot |
| `/action` | POST | `{"slots": [a, b, c]}` or `{"card": id}` | Snapshot |
| `/new` | POST | `{}` | Snapshot of a fresh match |

A snapshot carries `view` (a standalone HTML rendering from the person's seat),
`phase`, `your_turn`, `finished`, `seat`, `scores`, `deal`, `log`, the `hand` as
`{slot, card, name, rank, suit, red}` entries — enough for the page to draw a
card face — and `legal` card ids during `PLAY`. Finished matches add `winners`.

Rejected actions answer `400` and leave the match untouched: an illegal card, a
malformed pass selection, and acting out of turn are all refused by the same
masks the compiled path uses. The server keeps one match and serialises
requests, so it is a single-player local tool, not a multi-seat lobby.

## Latency

Interactive play steps the environment one action at a time, so `heart.play`
JIT-compiles the transition, the observation, and the rule policies once. The
first action absorbs that compile; later actions cost single-digit milliseconds.
