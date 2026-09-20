"""Validate the leaderboard entries, or the ones a pull request touched.

Run with no arguments to check them all:

    python submissions/validate.py

Pass changed repository paths to check only the affected entry directories.
CI calls the script without paths and therefore validates every entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from heart.contest import MAX_SUBMISSION_BYTES, SubmissionError, load_submission

ROOT = Path(__file__).parent
TEAM = re.compile(r"^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,38}$")
MAX_METADATA_BYTES = 16 * 1024


def _one_line(value: object, field: str, limit: int) -> tuple[str, str | None]:
    if not isinstance(value, str):
        return "", f"entry.toml {field} must be a string"
    stripped = value.strip()
    if not stripped:
        return "", f"entry.toml has an empty {field}"
    if "\n" in value or "\r" in value:
        return "", f"entry.toml {field} must fit on one line"
    if len(stripped) > limit:
        return "", f"entry.toml {field} must be at most {limit} characters"
    return stripped, None


def check(directory: Path) -> list[str]:
    """Check one team's entry. The directory is the team, so a second
    submission edits this same path and replaces what was there."""

    if directory.is_symlink():
        return [f"{directory.name}: team directory must not be a symbolic link"]
    problems = []
    blob = directory / "entry.bin"
    meta = directory / "entry.toml"
    if not blob.exists():
        problems.append(f"{directory.name}: no entry.bin")
    if not meta.exists():
        problems.append(f"{directory.name}: no entry.toml")
    if problems:
        return problems

    if blob.is_symlink() or meta.is_symlink():
        return [f"{directory.name}: entry files must not be symbolic links"]
    if not blob.is_file() or not meta.is_file():
        return [f"{directory.name}: entry files must be regular files"]

    if blob.stat().st_size > MAX_SUBMISSION_BYTES:
        problems.append(
            f"{directory.name}: entry.bin is over {MAX_SUBMISSION_BYTES} bytes"
        )
    if meta.stat().st_size > MAX_METADATA_BYTES:
        problems.append(
            f"{directory.name}: entry.toml is over {MAX_METADATA_BYTES} bytes"
        )
    if problems:
        return problems

    if not TEAM.match(directory.name):
        problems.append(
            f"{directory.name}: a team directory is a 1-39 character lowercase"
            " GitHub handle with only alphanumerics and single internal hyphens"
        )
    try:
        declared = tomllib.loads(meta.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        return problems + [f"{directory.name}: entry.toml is not readable: {error}"]
    _, description_problem = _one_line(declared.get("description"), "description", 200)
    if description_problem:
        problems.append(f"{directory.name}: {description_problem}")
    display, name_problem = _one_line(declared.get("name", directory.name), "name", 64)
    if name_problem:
        problems.append(f"{directory.name}: {name_problem}")
    if problems:
        return problems

    try:
        entry = load_submission(blob.read_bytes(), display)
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
        if (
            not meta.is_file()
            or meta.is_symlink()
            or meta.stat().st_size > MAX_METADATA_BYTES
        ):
            continue
        try:
            declared = tomllib.loads(meta.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
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
    everything = sorted(
        path for path in ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"
    )
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
