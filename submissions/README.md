# Submissions

One directory per entry, holding the exported policy and who it belongs to:

```
submissions/
└── your-entry-name/
    ├── entry.bin     # the exported graph
    └── entry.toml    # who made it, and what it is
```

Open a pull request that adds exactly that. CI validates the file on the pull
request, so a broken entry fails before a human looks at it.

`entry.toml`:

```toml
name = "your-entry-name"      # must match the directory
author = "your github handle"
description = "one line: what the policy does"
```

Names are lowercase letters, digits and hyphens. Replacing your own entry is a
pull request that edits it; the standings are re-run after a merge.
