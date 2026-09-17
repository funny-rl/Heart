"""Validate the leaderboard entries, or the ones a pull request touched.

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
TEAM = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}$")


def check(directory: Path) -> list[str]:
    """Check one team's entry. The directory is the team, so a second
    submission edits this same path and replaces what was there."""

    problems = []
    blob = directory / "entry.bin"
    meta = directory / "entry.toml"
    if not blob.exists():
        problems.append(f"{directory.name}: no entry.bin")
    if not meta.exists():
        problems.append(f"{directory.name}: no entry.toml")
    if problems:
        return problems

    if not TEAM.match(directory.name):
        problems.append(
            f"{directory.name}: a team directory is 2-39 lowercase letters, digits"
            " or hyphens, matching your GitHub handle"
        )
    try:
        declared = tomllib.loads(meta.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return problems + [f"{directory.name}: entry.toml is not readable: {error}"]
    if not str(declared.get("description", "")).strip():
        problems.append(f"{directory.name}: entry.toml needs a description")
    display = str(declared.get("name", directory.name)).strip()
    if not display:
        problems.append(f"{directory.name}: entry.toml has an empty name")

    try:
        entry = load_submission(blob.read_bytes(), display or directory.name)
    except SubmissionError as error:
        return problems + [f"{directory.name}: {error}"]
    print(
        f"  {directory.name} ({display}): {entry.size_bytes:,} bytes, "
        f"{entry.flops_per_decision:,.0f} flops per decision"
    )
    return problems


def clashing_names(directories: list[Path]) -> list[str]:
    """Two teams showing the same name would be unreadable in the standings."""

    shown: dict[str, list[str]] = {}
    for directory in directories:
        meta = directory / "entry.toml"
        if not meta.exists():
            continue
        try:
            declared = tomllib.loads(meta.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue
        display = str(declared.get("name", directory.name)).strip().casefold()
        if display:
            shown.setdefault(display, []).append(directory.name)
    return [
        f"the name '{name}' is used by {', '.join(sorted(teams))}; pick distinct names"
        for name, teams in sorted(shown.items())
        if len(teams) > 1
    ]


def main(argv: list[str]) -> int:
    everything = sorted(path for path in ROOT.iterdir() if path.is_dir())
    if argv:
        touched = {
            ROOT / Path(item).parts[1]
            for item in argv
            if Path(item).parts[:1] == ("submissions",) and len(Path(item).parts) > 2
        }
        directories = sorted(touched & set(everything))
    else:
        directories = everything
    if not directories:
        print("no entries to check")
        return 0

    print(f"checking {len(directories)} entr{'y' if len(directories) == 1 else 'ies'}")
    problems = [problem for directory in directories for problem in check(directory)]
    problems.extend(clashing_names(everything))
    for problem in problems:
        print(f"  refused - {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
