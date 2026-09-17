# Release and compatibility policy

HEART follows semantic package versions and versioned environment IDs as two
separate compatibility signals.

## Environment IDs

An environment ID such as `classic-v0` names game semantics. Bug fixes that
restore its documented contract may retain the ID. Intentional changes to
legality, observations, scoring, rewards, termination, or action encoding need
a new environment version.

Pre-1.0 Python APIs may change, but incompatible changes must be documented in
[CHANGELOG.md](../CHANGELOG.md) and tested. Silent semantic drift is not
acceptable even during research preview.

## Release gate

A release candidate must have:

1. a clean source tree and green tests on Python 3.10–3.12;
2. successful wheel and source-distribution builds;
3. import and one-deal smoke tests from the built wheel;
4. documentation matching the registered modes and exported API;
5. snapshot and complete-replay generation checks;
6. an updated changelog and citation version;
7. one source commit shared by the annotated tag and GitHub Release.

PyPI publication, GitHub Release creation, and performance claims must not be
presented as complete before their corresponding artifacts exist.

## Deprecation

When practical, deprecated Python names should warn for at least one minor
release before removal. Environment IDs remain constructible for the lifetime
of a major release unless a correctness or security issue makes that unsafe.
