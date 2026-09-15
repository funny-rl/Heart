# Changelog

All notable user-facing changes to HEART are recorded here. The project follows
[Semantic Versioning](https://semver.org/) for package releases; environment IDs
separately version game semantics.

## [Unreleased]

### Fixed

- Reject invalid observer IDs, non-integer actions, and unsafe Q♠ penalty
  overrides instead of leaking or silently coercing values.
- Show the completed fourth card at every trick boundary in ANSI, HTML, and GIF
  rendering, including the terminal frame.
- Enforce GIF output format, centered arbitrary-size layout, and finite replay
  frame rates.

### Added

- Opt-in `step_unchecked` for trusted mask-selected compiled rollouts.
- Separate engine/policy and safe/trusted benchmark modes with synchronized
  minimal outputs.
- Adversarial rule, API-boundary, rendering, and benchmark regression tests.
- SHA-pinned CI actions, Ruff enforcement, and installed-wheel full-deal and
  HTML replay smoke tests.

## [0.1.0] - 2026-09-15

### Added

- JAX-native four-player `simplest-v0` environment with a fixed 52-action deal.
- Follow-suit, 2♣ opening, first-trick point, and hearts-broken action masks.
- Five-point Q♠, terminal zero-sum rewards, and solo-win moon-shot scoring.
- Batched `jit`/`vmap` rollout support and a throughput harness.
- Easy, medium, and hard JIT-compatible rule-policy baselines.
- Full-observability ANSI and HTML snapshots.
- Portable interactive HTML match replays with timeline and speed controls.
- Optional full-observability Pillow renderer for compact animated GIF previews.
- Public contracts, contribution policies, citation metadata, and CI workflow.
