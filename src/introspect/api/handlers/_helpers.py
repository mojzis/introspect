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
    "files_read": "fr_agg.files_read",
    "files_edited": "fw_agg.files_edited",
    "files_read_only": "fr_agg.files_read_only",
    "files_outside": "fr_agg.files_outside",
}
SESSIONS_SORT_DEFAULT = "started_at"

RAW_PER_PAGE = 20

_XML_TAG_RE = re.compile(r"<[^>]+>")
# <command-message> duplicates the <command-name> for slash-command / skill
# invocations (e.g. "<command-name>marimo-pair</command-name>"
# "<command-message>/marimo-pair</command-message>"), so drop it entirely
# instead of leaving the repeated name in the title.
_COMMAND_MESSAGE_RE = re.compile(r"<command-message>.*?</command-message>", re.DOTALL)


def clean_title(raw: str) -> str:
    """Strip all XML-style tags from session titles."""
    without_msg = _COMMAND_MESSAGE_RE.sub("", raw)
    # Replace tags with a space so adjacent block contents don't run together
    # (e.g. "<command-name>commit</command-name><command-args>fix</command-args>"
    # becomes "commit fix", not "commitfix"); then collapse whitespace runs.
    detagged = _XML_TAG_RE.sub(" ", without_msg)
    return " ".join(detagged.split())


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

# Reusable SQL fragments for per-session file metrics
# (backed by file_reads / file_writes views).
FILE_READS_SUBQUERY = """(
    SELECT
        fr.session_id,
        COUNT(DISTINCT fr.file_path) AS files_read,
        COUNT(DISTINCT fr.file_path) FILTER (
            WHERE fr.file_path NOT IN (
                SELECT DISTINCT fw.file_path FROM file_writes fw
                WHERE fw.session_id = fr.session_id
            )
        ) AS files_read_only,
        COUNT(DISTINCT fr.file_path) FILTER (
            WHERE NOT starts_with(fr.file_path, COALESCE(ls.cwd, ''))
        ) AS files_outside
    FROM file_reads fr
    JOIN logical_sessions ls ON fr.session_id = ls.session_id
    GROUP BY fr.session_id
) fr_agg"""

FILE_WRITES_SUBQUERY = """(
    SELECT session_id, COUNT(DISTINCT file_path) AS files_edited
    FROM file_writes GROUP BY session_id
) fw_agg"""

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
    COALESCE(fr_agg.files_read, 0) AS files_read,
    COALESCE(fw_agg.files_edited, 0) AS files_edited,
    COALESCE(fr_agg.files_read_only, 0) AS files_read_only,
    COALESCE(fr_agg.files_outside, 0) AS files_outside,
    cmd.commands
"""

SESSION_INFO_JOINS = f"""
    LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
    LEFT JOIN {TOOL_COUNTS_SUBQUERY} ON ls.session_id = tc.session_id
    LEFT JOIN {FILE_READS_SUBQUERY} ON ls.session_id = fr_agg.session_id
    LEFT JOIN {FILE_WRITES_SUBQUERY} ON ls.session_id = fw_agg.session_id
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
    "files_read": 0,
    "files_edited": 0,
    "files_read_only": 0,
    "files_outside": 0,
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
        files_read,
        files_edited,
        files_read_only,
        files_outside,
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
        "files_read": files_read or 0,
        "files_edited": files_edited or 0,
        "files_read_only": files_read_only or 0,
        "files_outside": files_outside or 0,
        "commands": commands or "",
    }


def escape_ilike(s: str) -> str:
    """Escape ILIKE special characters so they match literally."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def fetch_distinct_projects(
    db: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Return sorted list of distinct project names."""
    rows = db.execute("""
        SELECT DISTINCT project
        FROM logical_sessions
        WHERE project IS NOT NULL
        ORDER BY project
    """).fetchall()
    return [r[0] for r in rows]


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
