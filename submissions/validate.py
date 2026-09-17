"""Validate every entry in this directory, or the ones a pull request touched.

Run with no arguments to check them all:

    python submissions/validate.py

CI passes the changed paths so a pull request only pays for its own entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

from heart.contest import SubmissionError, load_submission

ROOT = Path(__file__).parent
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")


def manifest(directory: Path) -> dict:
    """Read an entry's manifest, or an empty one if it cannot be read."""

    path = directory / "entry.toml"
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return {}


def team(declared: dict) -> str:
    """The name an entry is owned by, compared case- and @-insensitively."""

    return str(declared.get("author", "")).strip().lstrip("@").casefold()


def duplicates(directories: list[Path]) -> list[str]:
    """One entry per team: a second one has to replace the first, not join it."""

    owners: dict[str, list[str]] = {}
    for directory in directories:
        owner = team(manifest(directory))
        if owner:
            owners.setdefault(owner, []).append(directory.name)
    return [
        f"{owner} has {len(entries)} entries ({', '.join(sorted(entries))});"
        " a team may hold one, so replace it rather than adding another"
        for owner, entries in sorted(owners.items())
        if len(entries) > 1
    ]


def check(directory: Path) -> list[str]:
    problems = []
    blob = directory / "entry.bin"
    meta = directory / "entry.toml"
    if not blob.exists():
        problems.append(f"{directory.name}: no entry.bin")
    if not meta.exists():
        problems.append(f"{directory.name}: no entry.toml")
    if problems:
        return problems

    if not NAME.match(directory.name):
        problems.append(
            f"{directory.name}: a name is 2-32 lowercase letters, digits or hyphens"
        )
    try:
        declared = tomllib.loads(meta.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return problems + [f"{directory.name}: entry.toml is not readable: {error}"]
    for field in ("name", "author", "description"):
        if not str(declared.get(field, "")).strip():
            problems.append(f"{directory.name}: entry.toml needs a {field}")
    if declared.get("name") != directory.name:
        problems.append(
            f"{directory.name}: entry.toml names '{declared.get('name')}' instead"
        )

    try:
        entry = load_submission(blob.read_bytes(), directory.name)
    except SubmissionError as error:
        return problems + [f"{directory.name}: {error}"]
    print(
        f"  {directory.name}: {entry.size_bytes:,} bytes, "
        f"{entry.flops_per_decision:,.0f} flops per decision"
    )
    return problems


def main(argv: list[str]) -> int:
    if argv:
        directories = sorted(
            {
                ROOT / Path(item).parts[1]
                for item in argv
                if Path(item).parts[:1] == ("submissions",)
                and len(Path(item).parts) > 2
            }
        )
    else:
        directories = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if not directories:
        print("no entries to check")
        return 0
    print(f"checking {len(directories)} entr{'y' if len(directories) == 1 else 'ies'}")
    problems = [problem for directory in directories for problem in check(directory)]
    # One entry per team is a property of the whole directory, not of the
    # entries a pull request happens to touch.
    problems.extend(duplicates(sorted(p for p in ROOT.iterdir() if p.is_dir())))
    for problem in problems:
        print(f"  refused - {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
