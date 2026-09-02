#!/usr/bin/env python3
"""Select the test files impacted by the current diff.

The chain is gerenuk -> ty-find -> pytest:

1. ``gerenuk changed-symbols`` reports which Python symbols the working tree
   changed against a base ref (git only -- no type checker, no environment).
2. ``tyf refs --tests`` asks the ty-find daemon which test files reference each
   of those symbols.
3. The union of those test files, plus any test file that changed itself, is
   what pytest needs to run.

Prints one test path per line on stdout. Exits 10 to mean "selection is not
trustworthy, run the whole suite"; ``scripts/pre-commit.sh`` treats any
non-zero exit that way, so a crash here degrades to a full run rather than to
a silent gap in coverage. Selection is deliberately conservative -- over-
selecting costs seconds, under-selecting lets a break through the commit gate.

The base ref comes from ``GERENUK_BASE``; the commit hook sets it to ``HEAD``
so the selection covers the commit being made rather than the whole branch.
Left unset, gerenuk's own default (``origin/main``) applies.

Run it directly with ``uv run poe impacted-tests``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# A change to any of these invalidates symbol-level selection entirely: shared
# fixtures and dependency pins are reachable from tests that never name a
# changed symbol.
BROAD_CHANGE_NAMES = frozenset({"conftest.py", "pyproject.toml", "uv.lock"})

RUN_EVERYTHING = 10

GERENUK = os.environ.get("GERENUK", "gerenuk")
# ty-find ships its binary as `tyf`; GERENUK_TYF is the env var gerenuk itself
# reads, so pointing it once redirects both tools.
TYF = os.environ.get("GERENUK_TYF", "tyf")


class SelectionFailed(RuntimeError):
    """Raised when a tool in the chain could not answer; caller runs everything."""


def _note(message: str) -> None:
    sys.stderr.write(f"impacted-tests: {message}\n")


def _run(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd, input=stdin, capture_output=True, text=True, check=False
    )


def changed_symbols(base: str | None) -> dict:
    """Ask gerenuk what the working tree changed, as parsed JSON."""
    cmd = [GERENUK, "changed-symbols", "--format", "json"]
    if base:
        cmd += ["--base", base]
    proc = _run(cmd)
    # gerenuk exits 1 when it *has* findings, so only 2+ is a real failure.
    if proc.returncode > 1:
        _note(f"gerenuk exited {proc.returncode}: {proc.stderr.strip()}")
        raise SelectionFailed
    return json.loads(proc.stdout)


def test_files_referencing(symbols: list[str], root: Path) -> set[str]:
    """Ask tyf which test files reference any of these symbol names."""
    if not symbols:
        return set()
    # gerenuk reports "module.path:name"; tyf queries take the bare name.
    names = sorted({symbol.rsplit(":", 1)[-1] for symbol in symbols})
    proc = _run(
        [
            TYF,
            "--format",
            "json",
            "refs",
            "--stdin",
            "--tests",
            "--references-limit",
            "0",
        ],
        stdin="\n".join(names) + "\n",
    )
    if proc.returncode != 0:
        _note(f"tyf exited {proc.returncode}: {proc.stderr.strip()}")
        raise SelectionFailed
    payload = json.loads(proc.stdout)
    # tyf returns a bare object for a single query and a list for several.
    results = payload if isinstance(payload, list) else [payload]
    found: set[str] = set()
    for result in results:
        for ref in result.get("test_references", []):
            path = Path(ref["file"])
            found.add(str(path.relative_to(root) if path.is_absolute() else path))
    return found


def test_files_importing(modules: list[str], root: Path) -> set[str]:
    """Test files that name a module changed at module level.

    A module-level edit (imports, constants, decorators) has no single owning
    symbol for tyf to trace, so fall back to asking git which test files
    mention the module at all. The patterns are deliberately loose -- they
    match the module named in a string or a comment, not just an import -- so
    that a test reaching a module indirectly (importlib, a fixture path, a
    subprocess invocation) is still picked up. Over-selecting costs seconds.
    """
    found: set[str] = set()
    for module in modules:
        tail = module.rsplit(".", 1)[-1]
        for pattern in (module, rf"\b{tail}\."):
            proc = _run(
                ["git", "-C", str(root), "grep", "-l", "-E", pattern, "--", "tests/"]
            )
            found.update(line for line in proc.stdout.splitlines() if line.strip())
    return found


def select(root: Path, base: str | None) -> list[str]:
    """Return the impacted test paths, or raise SelectionFailed to run all."""
    report = changed_symbols(base)

    if report.get("errors"):
        _note(f"gerenuk reported {report['errors']}")
        raise SelectionFailed

    broad = [
        path
        for path in report.get("non_python_changes", [])
        if Path(path).name in BROAD_CHANGE_NAMES
    ]
    if broad:
        _note(f"broad change ({', '.join(broad)})")
        raise SelectionFailed

    symbols = [entry["symbol"] for entry in report.get("changed_symbols", [])]
    modules = list(report.get("module_level_changes", []))

    selected = set(report.get("test_files_changed", []))
    selected |= test_files_referencing(symbols, root)
    selected |= test_files_importing(modules, root)

    # Source changed but nothing traced back to a test: assume the mapping
    # missed something rather than that the change is untested.
    if (symbols or modules) and not selected:
        _note("no test mapped to the change")
        raise SelectionFailed

    return sorted(selected)


def main() -> int:
    top_level = _run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    root = Path(top_level or ".").resolve()

    try:
        selected = select(root, os.environ.get("GERENUK_BASE") or None)
    except (SelectionFailed, json.JSONDecodeError, ValueError) as exc:
        if not isinstance(exc, SelectionFailed):
            _note(f"unreadable tool output: {exc}")
        return RUN_EVERYTHING

    sys.stdout.write("".join(f"{path}\n" for path in selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
