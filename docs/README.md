# HEART documentation

This directory contains the normative public contracts for HEART. The README
is an entry point; when implementation details and prose disagree, treat the
versioned tests and the documents below as the compatibility evidence that must
be reconciled in the same change.

| Area                  | Owner                                     | Document                               |
| --------------------- | ----------------------------------------- | -------------------------------------- |
| Environment API       | `heart/env.py`, `heart/types.py`          | [Environment contract](environment.md) |
| Game semantics        | `heart/rules.py`, `heart/config.py`       | [deal rules](rules.md)                 |
| Human inspection      | `heart/render.py`, `heart/html_render.py` | [Rendering and replay](rendering.md)   |
| Human play            | `heart/play.py`                           | [Human play](play.md)                  |
| Leaderboard entries   | `heart/contest.py`                        | [Leaderboard](leaderboard.md)          |
| Experimental evidence | examples and benchmarks                   | [Reproducibility](reproducibility.md)  |
| Distribution          | `pyproject.toml`, CI                      | [Release policy](release.md)           |

Public API or semantic changes must update their owning implementation, focused
tests, this documentation, and [CHANGELOG.md](../CHANGELOG.md) together.
