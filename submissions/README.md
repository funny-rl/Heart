# Submissions

**The directory is your GitHub identity.** Name it after your GitHub handle; it
holds the policy you exported and a one-line manifest. Public standings use this
handle, alongside evaluation results, rather than a policy alias or real name.

```
submissions/
└── your-github-handle/
    ├── entry.bin        # the exported graph
    ├── entry.toml
    └── validation.json  # local export report; optional in the pull request
```

```toml
name = "clever-passer"        # optional policy alias; not the leaderboard identity
description = "one line: what the policy does"
```

Both values are single-line strings. `name` is at most 64 characters,
`description` is at most 200, and `entry.toml` is at most 16 KiB. Policy aliases
may be shared by different handles.

Open a pull request that adds — or edits — that directory. CI validates the
entry on the pull request, so a broken file fails before a human looks at it.
This repository and its pull requests are public, including submitted graphs,
metadata, and PR descriptions. Do not include your name or student number;
provide any required course identity separately to the teaching staff.

**Submitting again replaces what you had.** A second entry writes to the same
path, so Git records a modification rather than a new row, and the standings are
re-run after the merge. The GitHub handle remains your public identifier.

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
