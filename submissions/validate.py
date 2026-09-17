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
    for problem in problems:
        print(f"  refused - {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
