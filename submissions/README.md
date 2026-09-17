# Submissions

**The directory is your team.** Name it after your GitHub handle; it holds the
policy you exported and a one-line manifest:

```
submissions/
└── your-github-handle/
    ├── entry.bin     # the exported graph
    └── entry.toml
```

```toml
name = "clever-passer"        # what the standings call it; optional
description = "one line: what the policy does"
```

Open a pull request that adds — or edits — that directory. CI validates the
entry on the pull request, so a broken file fails before a human looks at it.

**Submitting again replaces what you had.** A second entry writes to the same
path, so Git records a modification rather than a new row, and the standings are
re-run after the merge. Two teams may not show the same `name`.

A team directory is 2 to 39 lowercase letters, digits or hyphens. Check yours
with `python submissions/validate.py`.
