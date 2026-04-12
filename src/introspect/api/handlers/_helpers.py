"""Shared helpers, constants, and template setup for route handlers."""

import logging
import re
from pathlib import Path

import duckdb
from fastapi import Request
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE_DEFAULT = 50
SESSIONS_PAGE_SIZES = [25, 50, 100, 200]

# Allowed sort columns for sessions page
SESSIONS_SORT_COLS = {
    "started_at": "ls.started_at",
    "duration": "ls.duration",
    "user_msgs": "ls.user_messages",
    "asst_msgs": "ls.assistant_messages",
    "tool_calls": "tc.tool_count",
    "model": "ls.model",
    "project": "ls.project",
    "branch": "ls.git_branch",
    "title": "fp.first_prompt",
}
SESSIONS_SORT_DEFAULT = "started_at"

RAW_PER_PAGE = 20

_XML_TAG_RE = re.compile(r"<[^>]+>")


def clean_title(raw: str) -> str:
    """Strip all XML-style tags from session titles."""
    return _XML_TAG_RE.sub("", raw).strip()


def parent(request: Request) -> str:
    """Return the base template: full page for normal requests, partial for HTMX."""
    if request.headers.get("HX-Request"):
        return "partial.html"
    return "base.html"


def conn(request: Request):
    """Get the DuckDB connection from request state."""
    return request.state.conn


DEFAULT_PAGE_SIZE = 50

# Reusable SQL fragment for per-session tool counts.
TOOL_COUNTS_SUBQUERY = """(
    SELECT session_id, COUNT(*) AS tool_count
    FROM tool_calls GROUP BY session_id
) tc"""

# Built-in / meta commands that don't reflect real work — hidden from the UI.
OBVIOUS_COMMANDS: frozenset[str] = frozenset(
    {
        "/clear",
        "/compact",
        "/config",
        "/cost",
        "/doctor",
        "/exit",
        "/fast",
        "/help",
        "/init",
        "/listen",
        "/login",
        "/logout",
        "/model",
        "/quit",
        "/status",
        "/terminal-setup",
        "/vim",
    }
)

OBVIOUS_COMMANDS_SQL = "(" + ", ".join(f"'{c}'" for c in sorted(OBVIOUS_COMMANDS)) + ")"

COMMAND_LIST_SUBQUERY = (
    "(SELECT session_id,"  # noqa: S608
    " string_agg(DISTINCT command, ', ' ORDER BY command) AS commands"
    " FROM message_commands"
    f" WHERE command NOT IN {OBVIOUS_COMMANDS_SQL}"
    " GROUP BY session_id) cmd"
)

TOOL_COUNTS_WITH_ERRORS_SUBQUERY = """(
    SELECT session_id,
           COUNT(*) AS tool_count,
           COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
    FROM tool_calls GROUP BY session_id
) tc"""


def format_duration(total_seconds: float) -> str:
    """Format seconds as M:SS string."""
    secs = int(total_seconds)
    return f"{secs // 60}:{secs % 60:02d}"


# Columns selected by SESSION_INFO_SELECT (positional).
SESSION_INFO_SELECT = """
    ls.session_id,
    ls.started_at,
    ls.ended_at,
    ls.duration,
    ls.user_messages,
    ls.assistant_messages,
    ls.model,
    ls.project,
    ls.git_branch,
    fp.first_prompt,
    COALESCE(tc.tool_count, 0) AS tool_count,
    cmd.commands
"""

SESSION_INFO_JOINS = f"""
    LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
    LEFT JOIN {TOOL_COUNTS_SUBQUERY} ON ls.session_id = tc.session_id
    LEFT JOIN {COMMAND_LIST_SUBQUERY} ON ls.session_id = cmd.session_id
"""

_EMPTY_SESSION_INFO: dict[str, object] = {
    "date": "",
    "start_time": "",
    "end_time": "",
    "duration": "",
    "user_msgs": 0,
    "asst_msgs": 0,
    "model": "",
    "project": "",
    "branch": "",
    "title": "",
    "tool_count": 0,
    "commands": "",
}


def session_row_to_dict(row: tuple) -> dict:
    """Convert a SESSION_INFO_SELECT row to a template-friendly dict."""
    (
        session_id,
        started_at,
        ended_at,
        duration,
        user_msgs,
        asst_msgs,
        model,
        project,
        git_branch,
        first_prompt,
        tool_count,
        commands,
    ) = row
    dur_str = format_duration(duration.total_seconds()) if duration else ""
    return {
        "id": session_id,
        "date": str(started_at)[5:10] if started_at else "",
        "start_time": str(started_at)[11:16] if started_at else "",
        "end_time": str(ended_at)[11:16] if ended_at else "",
        "duration": dur_str,
        "user_msgs": user_msgs or 0,
        "asst_msgs": asst_msgs or 0,
        "model": model or "",
        "project": project or "",
        "branch": git_branch or "",
        "title": clean_title(first_prompt or "")[:120],
        "tool_count": tool_count or 0,
        "commands": commands or "",
    }


def fetch_token_usage(
    db: duckdb.DuckDBPyConnection,
    *,
    session_id: str | None = None,
    include_cache: bool = False,
) -> tuple | None:
    """Fetch token usage sums. Returns None on error.

    Without session_id: returns (input_tokens, output_tokens).
    With include_cache: appends (cache_creation_tokens, cache_read_tokens).
    """
    cache_cols = ""
    if include_cache:
        cache_cols = """,
            SUM(CAST(json_extract(
                message, '$.usage.cache_creation_input_tokens'
            ) AS BIGINT)),
            SUM(CAST(json_extract(
                message, '$.usage.cache_read_input_tokens'
            ) AS BIGINT))"""

    session_filter = ""
    params: list[str] = []
    if session_id is not None:
        session_filter = "AND session_id = ?"
        params.append(session_id)

    try:
        return db.execute(
            f"""
            SELECT
                SUM(CAST(json_extract(message, '$.usage.input_tokens') AS BIGINT)),
                SUM(CAST(json_extract(message, '$.usage.output_tokens') AS BIGINT))
                {cache_cols}
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract(message, '$.usage.input_tokens') IS NOT NULL
              {session_filter}
        """,  # noqa: S608
            params,
        ).fetchone()
    except Exception:
        log.debug("token usage query failed", exc_info=True)
        return None
