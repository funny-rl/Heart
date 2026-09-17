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

**A team holds one entry.** The `author` field identifies it, compared without
case or a leading `@`, and a pull request adding a second entry under the same
author is refused. To improve yours, edit the directory you already have —
replace `entry.bin` and open a pull request. The standings are re-run after a
merge.

Names are lowercase letters, digits and hyphens, 2 to 32 characters.
