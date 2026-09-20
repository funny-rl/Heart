# Changelog

All notable user-facing changes to HEART are recorded here. Package versions
follow Semantic Versioning, while environment IDs version game semantics.

## [0.1.0] - 2026-09-20

### Environment

- JAX-native `classic-v0` matches to 100 points with standard 13-point Q♠,
  cumulative scoring, shooting the moon, shared-lowest winners, and the
  left/right/across/hold passing cycle.
- Fixed-shape immutable state, private player observations, legal-action masks,
  explicit PRNG keys, and `jit`/`vmap` compatible transitions.
- A 286-action pass interface and stable 13-slot play interface for single-agent
  learning against configurable rule-based opponents.
- Deal-boundary rewards equal to each player's negative effective penalty score.
- A direct `DealEnv` handle for testing and benchmarking the 52-play deal core,
  using the same standard deal rules as `classic-v0`.
- Safe public transitions plus `step_unchecked` for actions already selected from
  the legal mask.

### Policies and evaluation

- Easy, medium, and hard JIT-compatible policies for passing and card play.
- Moon-shot defense in the medium and hard play policies.
- Batched rollout benchmarks with synchronized safe and trusted execution modes.

### Rendering and play

- ANSI, HTML, and GIF rendering from either a player seat or an omniscient view.
- Portable HTML match replays with timeline and speed controls.
- A browser-based human-play server with running scores, deal progress, paced
  opponent turns, and rule or plugin-supplied seat policies.
- Checked-in GIF and HTML previews of a complete deal transition.

### Leaderboard

- Portable `jax.export` policy submissions using the environment single-agent
  observation and its pass and play action spaces.
- Submission validation for signatures, finite outputs, platform support, file
  size, and per-decision compute limits.
- Complete-match league evaluation with shared deals, rotating seats, penalty
  points per deal, match win and last-place rates, Elo, and uncertainty.
- Pull-request submission layout, replacement rules, validation script, and CI
  checks.

### Project

- Python 3.10–3.12 support, wheel and source-distribution builds, installed-wheel
  smoke tests, Ruff checks, and CPU test coverage in CI.
- Public documentation for rules, observations, rendering, human play,
  reproducibility, releases, leaderboard scoring, and policy submission.
