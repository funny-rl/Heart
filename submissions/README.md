# Submissions

**The directory is your team.** Name it after your GitHub handle; it holds the
policy you exported and a one-line manifest:

```
submissions/
└── your-github-handle/
    ├── entry.bin        # the exported graph
    ├── entry.toml
    └── validation.json  # local export report; optional in the pull request
```

```toml
name = "clever-passer"        # what the standings call it; optional
description = "one line: what the policy does"
```

Both values are single-line strings. `name` is at most 64 characters,
`description` is at most 200, and `entry.toml` is at most 16 KiB.

Open a pull request that adds — or edits — that directory. CI validates the
entry on the pull request, so a broken file fails before a human looks at it.

**Submitting again replaces what you had.** A second entry writes to the same
path, so Git records a modification rather than a new row, and the standings are
re-run after the merge. Two teams may not show the same `name`.

A team directory is your 1-to-39-character lowercase GitHub handle: letters,
digits, and single internal hyphens. Check yours with
`python submissions/validate.py`.

A trainer can create the directory in one host-side call after its final
checkpoint:

```python
heart.save_submission(
    policy,
    "submissions/your-github-handle",
    name="clever-passer",
    description="one line: what the policy does",
)
```

This export step is separate from the compiled training update. It writes
`entry.bin`, `entry.toml`, and a local `validation.json` report.
