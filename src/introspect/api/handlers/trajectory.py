"""Trajectory frame — one session's tool-call sequence as a glyph strip.

Renders a session as a left-to-right strip of tiles (one per tool call, in
order), split into locate / implement / verify phases, with per-session
mechanism metrics.  It's the view that exposes *behavior* the token/cost
totals hide: reflexive grep, re-reading hot files, a locate phase that
thrashed before the first edit.

Two granularities share one data pass (``build_trajectory_context``):

* **category** — coarse buckets (Read / Edit / Write / search / test / git /
  pkg / agent / web / mcp / bash), one glyph per call.
* **detail** — same tiles, but Bash calls are labelled by their two-word
  prefix (``git status``, ``uv run``), so you see *what* ran, not just the
  bucket.  Non-Bash tiles show the file basename or tool name.

Adapted from the tyftest ``harness/trajectory.py`` experiment emitter; the
run-layout / condition machinery is dropped since this is a single-session
view.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from ._helpers import path_basename, safe_json

if TYPE_CHECKING:
    import duckdb

VALID_VIEWS = ("category", "detail")
DEFAULT_VIEW = "category"

# category key -> (glyph, color, label).  Colors echo the tokenscape / house
# palette; glyphs are the terminal-strip mnemonics from the tyftest emitter.
CATEGORIES: dict[str, tuple[str, str, str]] = {
    "read": ("R", "#3fa63f", "Read"),
    "edit": ("E", "#b24bd8", "Edit"),
    "write": ("W", "#8e30b0", "Write"),
    "search": ("S", "#c8a000", "search"),
    "test": ("v", "#3f6ad6", "test"),
    "git": ("G", "#7a8a99", "git"),
    "pkg": ("P", "#d2691e", "pkg"),
    "task": ("A", "#00b4b4", "agent"),
    "web": ("N", "#2ca089", "web"),
    "mcp": ("M", "#c05a7d", "mcp"),
    "bash": ("b", "#888888", "bash"),
    "other": ("·", "#bbbbbb", "other"),
}

# Ordered Bash sub-classification — first match wins, so a command that both
# greps and runs pytest counts as ``test`` (primary intent).  ``poe test`` /
# ``poe check`` are matched by the test rule before the generic ``poe`` in pkg.
BASH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "test",
        re.compile(
            r"\b(pytest|jest|vitest|go test|cargo test|tox)\b"
            r"|poe (test|check)|npm (run )?test"
        ),
    ),
    ("search", re.compile(r"\b(grep|rg|ripgrep|ag|fd|find|tyf|biston)\b")),
    ("git", re.compile(r"\bgit\b")),
    (
        "pkg",
        re.compile(
            r"\b(uv|pip|pipx|poetry|poe|npm|yarn|pnpm|cargo|paru|pacman|apt|brew)\b"
        ),
    ),
]

# non-Bash tool name -> category
_TOOL_CAT: dict[str, str] = {
    "Read": "read",
    "Edit": "edit",
    "MultiEdit": "edit",
    "Write": "write",
    "NotebookEdit": "write",
    "Task": "task",
    "Agent": "task",
    "WebFetch": "web",
    "WebSearch": "web",
}

# tools that count as "edits" for the locate-phase boundary
_EDIT_TOOLS = {"edit", "write"}


def _classify(tool_name: str, cmd: str) -> str:
    """Map a tool call to a coarse category key."""
    if tool_name == "Bash":
        for cat, pat in BASH_PATTERNS:
            if pat.search(cmd):
                return cat
        return "bash"
    if tool_name in _TOOL_CAT:
        return _TOOL_CAT[tool_name]
    if tool_name.startswith("mcp__"):
        return "mcp"
    return "other"


def _prefix(cmd: str) -> str:
    """First two words of a shell command (the bash.py grouping, in Python)."""
    words = cmd.split()
    if len(words) >= 2:  # noqa: PLR2004
        return f"{words[0]} {words[1]}"
    return words[0] if words else "(empty)"


def _detail_label(tool_name: str, cat: str, inp: dict) -> str:
    """Fine-grained tile label for the detail view."""
    if tool_name == "Bash":
        return _prefix(str(inp.get("command", "")))
    if cat in ("read", "edit", "write"):
        path = inp.get("file_path") or inp.get("notebook_path")
        return path_basename(path) if path else tool_name
    if tool_name.startswith("mcp__"):
        return tool_name[len("mcp__") :]
    return tool_name


def _tooltip(tool_name: str, inp: dict) -> str:
    """Full hover text for a tile (native ``title`` attribute)."""
    if tool_name == "Bash":
        return str(inp.get("command", "")).strip() or "Bash"
    path = inp.get("file_path") or inp.get("notebook_path")
    if path:
        return f"{tool_name}: {path}"
    if tool_name.startswith("mcp__"):
        return tool_name
    return tool_name


def build_trajectory_context(
    db: duckdb.DuckDBPyConnection, session_id: str, view: str = DEFAULT_VIEW
) -> dict:
    """Build the trajectory-frame context for one session.

    Returns ``{"has_data": False}`` when the session made no tool calls.
    """
    view = view if view in VALID_VIEWS else DEFAULT_VIEW
    rows = db.execute(
        """
        SELECT tool_name, tool_input, is_error
        FROM tool_calls
        WHERE session_id = ?
        ORDER BY called_at, tool_use_id
        """,
        [session_id],
    ).fetchall()

    if not rows:
        return {"has_data": False, "view": view}

    calls: list[dict] = []
    reads: list[str] = []
    first_edit: int | None = None
    first_test: int | None = None

    for i, (tool_name, tool_input, is_error) in enumerate(rows):
        inp = safe_json(tool_input)
        cmd = str(inp.get("command", "")) if tool_name == "Bash" else ""
        cat = _classify(tool_name, cmd)
        glyph, color, label = CATEGORIES[cat]
        if cat in _EDIT_TOOLS and first_edit is None:
            first_edit = i
        if cat == "test" and first_test is None:
            first_test = i
        if cat == "read":
            path = inp.get("file_path")
            if path:
                reads.append(str(path))
        calls.append(
            {
                "cat": cat,
                "glyph": glyph,
                "color": color,
                "label": label,
                "detail": _detail_label(tool_name, cat, inp),
                "tip": _tooltip(tool_name, inp),
                # is_error is DuckDB JSON text: 'true'/'false'/NULL, not a bool
                "is_error": str(is_error).lower() == "true",
            }
        )

    # phase boundaries: locate [0, edit), implement [edit, test), verify [test, end)
    locate_len = first_edit if first_edit is not None else len(calls)
    implement_start = first_edit if first_edit is not None else len(calls)
    verify_start = first_test if first_test is not None else len(calls)
    # test before first edit -> no implement band
    verify_start = max(verify_start, implement_start)

    reread_counts = Counter(reads)
    first_nav = next((c["cat"] for c in calls if c["cat"] in ("search", "read")), "—")

    metrics = {
        "total": len(calls),
        "locate_len": locate_len,
        "distinct_files": len(reread_counts),
        "max_rereads": max(reread_counts.values(), default=0),
        "search": sum(1 for c in calls if c["cat"] == "search"),
        "edits": sum(1 for c in calls if c["cat"] in _EDIT_TOOLS),
        "first_nav": first_nav,
    }

    # legend: only the categories that actually appear, in canonical order
    present = {c["cat"] for c in calls}
    legend = [
        {"glyph": g, "color": col, "label": lab}
        for key, (g, col, lab) in CATEGORIES.items()
        if key in present
    ]

    return {
        "has_data": True,
        "view": view,
        "calls": calls,
        "metrics": metrics,
        "legend": legend,
        "phases": {
            "locate_end": implement_start,
            "implement_end": verify_start,
        },
    }
