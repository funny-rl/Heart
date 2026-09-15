# Contributing to HEART

HEART accepts changes when game semantics, numerical behavior, and performance
consequences can be reviewed independently. Keep a pull request small enough
that its rationale and evidence remain understandable together.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development setup

Use Python 3.10 or newer:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Core imports and environment execution must not require rendering frameworks,
browsers, network access, or external datasets.

## Environment change contract

Before modifying rules, inspect the current implementation, its focused tests,
and the owning document under `docs/`. In the pull request:

1. identify behavior retained and why it remains sound;
2. identify behavior changed and give the correctness, API, speed, memory, or
   compilation reason;
3. state whether action, observation, reward, termination, or environment-ID
   compatibility changes;
4. distinguish measurements from design assumptions and policy heuristics.

Intentional semantic changes to a published environment require a new
versioned ID. Do not silently change `simplest-v0`.

Keep rendering, serialization, diagnostics, and file I/O outside the lean JAX
transition. Prefer fixed-shape PyTrees, explicit keys, mask-safe array logic,
and fail-closed invalid actions. Do not combine a semantic change with an
unrelated refactor.

`step_unchecked` is a trusted performance API. Changes to it require
safe-versus-trusted output-equivalence tests for legal actions and benchmark
receipts that isolate validation overhead.

## Tests and performance

Rules changes need focused normal, boundary, and adversarial tests. At minimum,
run:

```bash
python -m pytest
python -m compileall -q heart
python examples/random_game.py
python examples/render_replay.py --output heart-replay.html
```

For hot-path performance claims, report synchronized before/after results,
output equivalence, commit IDs, JAX versions, device/backend, batch size,
warm-up handling, and the exact command. Never commit generated replays,
caches, environments, or benchmark dumps.

## Documentation and pull requests

Update public API, mode, schema, replay, configuration, and default changes in
the same pull request. Add a changelog entry for user-visible behavior.

A pull request should explain motivation, compatibility impact, validation,
and performance impact. Link relevant issues and disclose AI-assisted work when
its review context would be useful. Security findings follow
[SECURITY.md](SECURITY.md), not a public issue.

Unless explicitly stated otherwise, intentionally submitted contributions are
provided under the Apache-2.0 license.
