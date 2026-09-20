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

Each captured heart is 1 raw penalty point. Captured Q♠ is 13. The normal deal
therefore distributes 26 total raw points. The player or tied players with the
lowest effective score win.

If one player captures all 26 raw points, shooting the moon applies:

- the shooter receives effective score 0;
- every opponent receives effective score 26;
- only the shooter is marked as winner.

`penalties` retains raw captured points; terminal `scores` contains the effective
moon-adjusted result.

## Reward

All non-terminal transitions return `[0, 0, 0, 0]`. At termination:

```text
r_i = -scores_i
```

The reward preserves the game's effective penalty points without normalization;
only the sign changes so larger is better. A moon shot gives the shooter `0`
and each opponent the negative configured total point value.

## Configuration boundary

`classic-v0` is fixed to the standard rules above. For controlled single-deal
experiments only, `heart.DealEnv(SingleDealRules(**overrides))` permits immutable
deal-rule overrides. An overridden deal is not a canonical result and must
report its full configuration. New published semantics should receive a new
versioned environment ID rather than silently changing this contract.

`queen_of_spades_penalty` must be a non-boolean integer from 0 through 32754.
The upper bound ensures that 13 heart points plus Q♠ remain representable by the
current signed `int16` penalty and score arrays.
