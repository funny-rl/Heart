# Changelog

All notable user-facing changes to HEART are recorded here. The project follows
[Semantic Versioning](https://semver.org/) for package releases; environment IDs
separately version game semantics.

## [0.1.0] - 2026-09-15

### Added

- A checked-in 57-frame classic deal-transition GIF and interactive HTML replay
  covering left passing, all 52 card plays, scoring, and the next-deal boundary.
- JAX-native `classic-v0` complete matches to 100 points with 13-point Q♠,
  cumulative scoring, intentional shared-lowest winners, moon scoring, and the
  left/right/across/hold passing cycle over 286 three-card combinations.
- A classic single-learner adapter with 14 decisions on passing deals and 13 on
  hold deals, sparse normalized deal rewards, discount metadata, compressed
  core-event traces, and deterministic `replay_events` expansion.
- Full-observability classic ANSI, HTML, and GIF rendering with cumulative
  scoreboards, pass flow, sticky deal-boundary summaries, and match winners.
- JAX-native four-player `simplest-v0` environment with a fixed 52-action deal.
- Follow-suit, 2♣ opening, first-trick point, and hearts-broken action masks.
- Five-point Q♠, terminal zero-sum rewards, and solo-win moon-shot scoring.
- A fixed 13-action, 13-decision single-learner adapter against configurable
  easy, medium, and hard rule opponents.
- Batched `jit`/`vmap` rollout support and a throughput harness.
- Easy, medium, and hard JIT-compatible rule-policy baselines.
- Full-observability ANSI and HTML snapshots with live penalties, final scores,
  terminal rewards, and complete four-card trick boundaries.
- Portable interactive HTML match replays with timeline and speed controls.
- Optional full-observability Pillow renderer for compact animated GIF previews.
- Strict observer, action, and configurable-penalty validation at public API
  boundaries.
- Opt-in `step_unchecked` for trusted mask-selected compiled rollouts.
- Separate engine/policy and safe/trusted benchmark modes with synchronized
  minimal outputs.
- Adversarial rule, API-boundary, rendering, and benchmark regression tests.
- SHA-pinned CI actions, Ruff enforcement, and installed-wheel full-deal and
  HTML replay smoke tests.
- Public contracts, contribution policies, citation metadata, and CI workflow.

### Fixed

- Guarantee an exact float32 zero sum for terminal relative rewards.
- Fail-closed handling for non-scalar actions at the `simplest-v0` boundary.
- Preservation of deal-completion diagnostics across classic opponent autoplay.
- CI development dependencies, memory-safe test sharding, and installed-wheel
  smoke coverage for both classic core and single-learner interfaces.
- Consistent NumPy integer player IDs and explicit empty opponent validation.
