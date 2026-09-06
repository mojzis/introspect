"""``introspy guide``: the opt-in pitch, served verbatim from the package.

The page is prose in ``src/introspect/guide.md``; the site publishes the same
file through a snippet include. These tests hold the page to its contract so an
edit to the prose cannot silently break the command or the promises the aesop
tools page makes about it.
"""

import re
import tempfile
from pathlib import Path

from typer.main import get_group
from typer.testing import CliRunner

from introspect.cli import app
from introspect.guide import GUIDE_PATH, guide_text

from .test_cli import _patch_cli_paths

runner = CliRunner()

# A guide that grows past a screenful stops being read. Cut it, don't raise this.
LINE_CAP = 60

_FENCE = re.compile(r"^```")
_INLINE = re.compile(r"`([^`]+)`")


def _invocations(text: str) -> list[list[str]]:
    """Every ``introspy ...`` the page shows, as argv: inline spans and every
    line of a fenced ``bash`` block."""
    commands: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if _FENCE.match(line):
            fence = None if fence is not None else line[3:].strip()
            continue
        if fence == "bash" and line.strip():
            commands.append(line.strip())
        elif fence is None:
            commands.extend(_INLINE.findall(line))
    return [
        cmd.split()
        for cmd in commands
        if cmd == "introspy" or cmd.startswith("introspy ")
    ]


def test_guide_prints_the_page_verbatim_and_exits_clean(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _patch_cli_paths(monkeypatch, tmp)

        result = runner.invoke(app, ["guide"])

        assert result.exit_code == 0, result.output
        assert result.stdout == guide_text(), "the CLI serves the page byte for byte"
        assert not db_path.exists(), (
            "guide needs no logs and must not build the database"
        )


def test_guide_page_ships_in_the_package() -> None:
    assert GUIDE_PATH.is_file(), f"{GUIDE_PATH} must be inside the installed package"
    assert Path("src/introspect").resolve() in GUIDE_PATH.resolve().parents


def test_guide_page_fits_the_line_cap_and_is_plain_ascii() -> None:
    text = guide_text()
    lines = text.rstrip("\n").splitlines()
    assert 10 < len(lines) <= LINE_CAP, f"{len(lines)} lines; cap is {LINE_CAP}"
    offender = next((c for c in text if not c.isascii()), None)
    assert offender is None, (
        f"guides are piped and captured, so they stay ASCII: {offender!r}"
    )


def test_guide_page_ends_with_a_single_next_line() -> None:
    text = guide_text()
    assert text.endswith("\n"), (
        "a page without a trailing newline lands the prompt mid-line"
    )
    lines = text.rstrip("\n").splitlines()
    assert lines[-1].startswith("next: run "), lines[-1]
    assert sum(line.startswith("next: run ") for line in lines) == 1


def test_every_command_the_guide_shows_exists() -> None:
    commands = set(get_group(app).commands)
    shown = _invocations(guide_text())
    assert shown, "the guide should show at least one introspy command"
    for argv in shown:
        if len(argv) == 1:
            continue  # a bare `introspy` is the entry point, not a command
        assert argv[1] in commands, f"{' '.join(argv)!r} names a command the CLI lacks"


def test_guide_makes_the_three_promises_of_the_pitch() -> None:
    """The todo that created this page: what it records, what it costs, and
    the two commands to start with."""
    text = guide_text()
    assert "## What it records" in text
    assert "## What it costs" in text
    assert "introspy stats" in text
    assert "introspy query" in text
    assert "INTROSPECT_VERSION_CHECK=off" in text, (
        "the one network request must be named"
    )
