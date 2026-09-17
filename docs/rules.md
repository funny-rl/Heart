# Deal rules and reward

This document is the normative semantic contract for one deal: the trick
semantics `classic-v0` runs on, and what `heart.rules` implements. The match
layer above it — passing, the running score, the hundred-point end — is in
[`classic.md`](classic.md).

## Deal and turn order

Reset uniformly permutes all 52 card IDs and deals 13 cards to each of four
players. The player holding 2♣ opens. Play then proceeds clockwise. A trick is
won by the highest-ranked card of the led suit, and its winner leads next.

No passing phase exists. Termination occurs after the fourth card of trick 13.

## Legal-action precedence

The current hand is filtered in this order:

1. Follow the led suit when at least one such card is held.
2. On a lead before hearts are broken, remove hearts when a non-heart remains.
3. During the first trick, remove all point cards when a non-point legal card
   remains.
4. On the first action of the deal, force 2♣.

A heart played at any position breaks hearts. If every otherwise legal card is
a heart or, on trick one, a point card, the restriction does not create an empty
mask.

## Penalties and winners

Each captured heart is 1 raw penalty point. Captured Q♠ is 5. The normal deal
therefore distributes 18 total raw points. The player or tied players with the
lowest effective score win.

If one player captures all 18 raw points, shooting the moon applies:

- the shooter receives effective score 0;
- every opponent receives effective score 18;
- only the shooter is marked as winner.

`penalties` retains raw captured points; terminal `scores` contains the effective
moon-adjusted result.

## Reward

All non-terminal transitions return `[0, 0, 0, 0]`. For an ordinary deal at
termination:

```text
r_i = (sum(scores_j for j != i) / 3 - scores_i) / 18
```

The divisor is the configured total point value (`13 + queen_of_spades_penalty`),
so it changes consistently for rule overrides. This ordinary-deal reward is
zero-sum and preserves lower-is-better score ordering. A moon shot instead uses
a dedicated terminal signal: the shooter receives `0`, while each opponent
receives `-1`.

## Configuration boundary

`heart.make(**overrides)` and `heart.DealEnv(SingleDealRules(**overrides))`
permit immutable rule-field overrides for controlled experiments. An overridden
environment is not a canonical result and must report its full configuration.
New published semantics should receive a new versioned environment ID rather
than silently changing this contract.

`queen_of_spades_penalty` must be a non-boolean integer from 0 through 32754.
The upper bound ensures that 13 heart points plus Q♠ remain representable by the
current signed `int16` penalty and score arrays.
