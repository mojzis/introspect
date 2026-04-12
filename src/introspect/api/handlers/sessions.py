"""Session-related route handlers."""

import json
import math
import re

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.search import ensure_search_corpus, fts_available

from ._helpers import (
    OBVIOUS_COMMANDS_SQL,
    SESSION_INFO_JOINS,
    SESSION_INFO_SELECT,
    SESSIONS_PAGE_SIZES,
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_COLS,
    SESSIONS_SORT_DEFAULT,
    conn,
    fetch_token_usage,
    parent,
    session_row_to_dict,
    templates,
)

_MESSAGE_HARD_CAP = 5000
_THINKING_PREVIEW_MAX = 200
_TOOL_HINT_MAX = 120
_COMMAND_NAME_RE = re.compile(r"<command-name>([^<]+)</command-name>")

# Per-tool preferred input keys for the one-line collapsed summary.
# First matching key in order wins; missing tools fall back to the first value.
_TOOL_HINT_KEYS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "MultiEdit": ("file_path",),
    "Write": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Bash": ("command",),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "Task": ("description", "subagent_type"),
    "Agent": ("description", "subagent_type"),
    "TaskCreate": ("subject",),
    "TaskUpdate": ("taskId", "status"),
    "Skill": ("skill", "args"),
    "ScheduleWakeup": ("reason",),
}


_SECONDS_PER_MINUTE = 60
_MIN_SENTENCE_LEN = 20


def _format_exec_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), _SECONDS_PER_MINUTE)
    return f"{mins}m {secs}s"


def _cap(value: str | None, limit: int = _MESSAGE_HARD_CAP) -> str:
    """Cap long strings with a visible truncation marker."""
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n… [truncated]"


def _pretty_tool_input(raw: str | None) -> str:
    """Pretty-print a tool input string. Falls back to the raw value on failure."""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return _cap(raw)
    try:
        return _cap(json.dumps(parsed, indent=2, ensure_ascii=False))
    except (TypeError, ValueError):
        return _cap(raw)


def _slash_command_label(text: str | None) -> str:
    """Extract the command name from a <command-name>…</command-name> wrapper."""
    if not text:
        return ""
    match = _COMMAND_NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip().splitlines()[0][:80]


def _thinking_preview(text: str | None) -> tuple[str, bool]:
    """Return (preview, has_more) for a thinking block.

    Preview is the first sentence if reasonably short, otherwise a hard cap.
    """
    if not text:
        return "", False
    stripped = text.strip()
    first_period = stripped.find(". ")
    if _MIN_SENTENCE_LEN < first_period < _THINKING_PREVIEW_MAX:
        preview = stripped[: first_period + 1]
    else:
        preview = stripped[:_THINKING_PREVIEW_MAX]
    has_more = len(stripped) > len(preview)
    return preview, has_more


def _tool_hint(tool_name: str, raw_input: str | None) -> str:
    """One-line summary of a tool call's input for the collapsed view."""
    if not raw_input:
        return ""
    try:
        parsed = json.loads(raw_input)
    except (ValueError, TypeError):
        return _single_line(raw_input, _TOOL_HINT_MAX)
    if not isinstance(parsed, dict):
        return _single_line(str(parsed), _TOOL_HINT_MAX)
    keys = _TOOL_HINT_KEYS.get(tool_name, ())
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return _single_line(str(parsed[key]), _TOOL_HINT_MAX)
    for value in parsed.values():
        if value not in (None, ""):
            return _single_line(str(value), _TOOL_HINT_MAX)
    return ""


def _single_line(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _coerce_bool(value: object) -> bool:
    """Normalize DuckDB/JSON boolean-ish values to a Python bool."""
    if value is None or value is False:
        return False
    if value is True:
        return True
    return str(value).strip().lower() == "true"


async def sessions(  # noqa: PLR0913
    request: Request,
    page: int,
    page_size: int,
    sort: str,
    order: str,
    model: str,
    project: str,
    branch: str,
    command: str,
    q: str,
) -> HTMLResponse:
    """Paginated session list with filtering and sorting."""
    db = conn(request)

    # Clamp page_size to allowed values
    if page_size not in SESSIONS_PAGE_SIZES:
        page_size = SESSIONS_PER_PAGE_DEFAULT

    # Build WHERE clause from filters
    where_clauses: list[str] = []
    params: list[str | int] = []
    if model.strip():
        where_clauses.append("ls.model = ?")
        params.append(model.strip())
    if project.strip():
        where_clauses.append("ls.project = ?")
        params.append(project.strip())
    if branch.strip():
        where_clauses.append("ls.git_branch = ?")
        params.append(branch.strip())
    if command.strip():
        where_clauses.append(
            "EXISTS (SELECT 1 FROM message_commands mc"
            " WHERE mc.session_id = ls.session_id AND mc.command = ?)"
        )
        params.append(command.strip())
    search_query = q.strip()
    if search_query:
        ensure_search_corpus(db)
        if fts_available(db):
            where_clauses.append(
                "EXISTS (SELECT 1 FROM (SELECT *, "
                "fts_main_search_corpus.match_bm25(rowid, ?) AS score "
                "FROM search_corpus) sc "
                "WHERE sc.session_id = ls.session_id AND sc.score IS NOT NULL)"
            )
            params.append(search_query)
        else:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM search_corpus sc"
                " WHERE sc.session_id = ls.session_id"
                " AND sc.content_text ILIKE ?)"
            )
            params.append(f"%{search_query}%")

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Count with filters
    total = db.execute(
        f"SELECT COUNT(*) FROM logical_sessions ls {where}",  # noqa: S608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    # Resolve sort column
    default_col = SESSIONS_SORT_COLS[SESSIONS_SORT_DEFAULT]
    sort_col = SESSIONS_SORT_COLS.get(sort, default_col)
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"
    nulls = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    rows = db.execute(
        f"""
        SELECT {SESSION_INFO_SELECT}
        FROM logical_sessions ls
        {SESSION_INFO_JOINS}
        {where}
        ORDER BY {sort_col} {sort_dir} {nulls}
        LIMIT ? OFFSET ?
    """,  # noqa: S608
        [*params, page_size, offset],
    ).fetchall()

    session_list = [session_row_to_dict(row) for row in rows]

    # Get distinct values for filter dropdowns
    models = db.execute("""
        SELECT DISTINCT model FROM logical_sessions
        WHERE model IS NOT NULL ORDER BY model
    """).fetchall()
    projects = db.execute("""
        SELECT DISTINCT project AS proj
        FROM logical_sessions
        WHERE project IS NOT NULL
        ORDER BY proj
    """).fetchall()
    branches = db.execute("""
        SELECT DISTINCT git_branch FROM logical_sessions
        WHERE git_branch IS NOT NULL ORDER BY git_branch
    """).fetchall()
    commands_list = db.execute(f"""
        SELECT DISTINCT command FROM message_commands
        WHERE command NOT IN {OBVIOUS_COMMANDS_SQL}
        ORDER BY command
    """).fetchall()  # noqa: S608

    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "parent": parent(request),
            "sessions": session_list,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_size": page_size,
            "page_sizes": SESSIONS_PAGE_SIZES,
            "sort": sort,
            "order": order.lower(),
            "filter_model": model,
            "filter_project": project,
            "filter_branch": branch,
            "filter_command": command,
            "filter_q": search_query,
            "models": [r[0] for r in models],
            "projects": [r[0] for r in projects],
            "branches": [r[0] for r in branches],
            "commands_list": [r[0] for r in commands_list],
        },
    )


async def session_detail(request: Request, session_id: str) -> HTMLResponse:
    """Full session detail with all messages."""
    db = conn(request)

    session_info = db.execute(
        """
        SELECT session_id, started_at, ended_at, duration,
               user_messages, assistant_messages, model, cwd, git_branch
        FROM logical_sessions
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    token_usage = fetch_token_usage(db, session_id=session_id, include_cache=True)

    # Tool call summary
    tool_summary = db.execute(
        """
        SELECT
            COUNT(*) AS total_calls,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_calls,
            MODE(tool_name) AS most_used_tool
        FROM tool_calls
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    # File metrics
    session_cwd = session_info[7] if session_info else ""
    file_metrics = db.execute(
        """
        SELECT
            COUNT(DISTINCT CASE WHEN tc.tool_name = 'Read'
                THEN json_extract_string(tc.tool_input, '$.file_path')
                END) AS files_read,
            COUNT(DISTINCT CASE WHEN tc.tool_name
                IN ('Edit', 'Write', 'MultiEdit', 'NotebookEdit')
                THEN COALESCE(
                    json_extract_string(tc.tool_input, '$.file_path'),
                    json_extract_string(tc.tool_input, '$.notebook_path')
                ) END) AS files_edited,
            COUNT(DISTINCT CASE WHEN tc.tool_name = 'Read'
                AND NOT starts_with(
                    COALESCE(json_extract_string(tc.tool_input, '$.file_path'), ''),
                    ?)
                AND json_extract_string(tc.tool_input, '$.file_path') IS NOT NULL
                THEN json_extract_string(tc.tool_input, '$.file_path')
                END) AS files_outside
        FROM tool_calls tc
        WHERE tc.session_id = ?
    """,
        [session_cwd or "", session_id],
    ).fetchone()

    # One row per classified content block, with paired tool results already
    # joined in from the tool_calls view (so agent_tool_call rows know their
    # result text, error flag, and execution time).
    cur = db.execute(
        """
        SELECT
            e.timestamp,
            e.kind,
            e.is_sidechain,
            e.text,
            e.thinking_text,
            e.tool_name,
            e.tool_input,
            e.tool_use_id,
            tc.tool_use_result,
            tc.is_error,
            tc.execution_time,
        FROM session_messages_enriched e
        LEFT JOIN tool_calls tc ON tc.tool_use_id = e.tool_use_id
        WHERE e.session_id = ?
          AND e.kind <> 'tool_result'
        ORDER BY e.timestamp ASC, e.block_idx ASC
        """,
        [session_id],
    )
    col_names = [d[0] for d in cur.description]
    parsed_messages = []
    for row in cur.fetchall():
        rec = dict(zip(col_names, row, strict=True))
        kind = rec["kind"]
        exec_secs = (
            rec["execution_time"].total_seconds()
            if rec["execution_time"] is not None
            else None
        )
        thinking_preview, thinking_has_more = _thinking_preview(rec["thinking_text"])
        parsed_messages.append(
            {
                "timestamp": str(rec["timestamp"])[:19] if rec["timestamp"] else "",
                "kind": kind,
                "is_sidechain": bool(rec["is_sidechain"]),
                "text": _cap(rec["text"]),
                "thinking_text": _cap(rec["thinking_text"]),
                "thinking_preview": thinking_preview,
                "thinking_has_more": thinking_has_more,
                "command_label": (
                    _slash_command_label(rec["text"]) if kind == "slash_command" else ""
                ),
                "tool_name": rec["tool_name"] or "",
                "tool_hint": _tool_hint(rec["tool_name"] or "", rec["tool_input"]),
                "tool_input": _pretty_tool_input(rec["tool_input"]),
                "tool_result": _cap(rec["tool_use_result"]),
                "is_error": _coerce_bool(rec["is_error"]),
                "exec_time": (
                    _format_exec_time(exec_secs) if exec_secs is not None else ""
                ),
                "tool_use_id": rec["tool_use_id"] or "",
            }
        )

    return templates.TemplateResponse(
        request,
        "session_detail.html",
        {
            "parent": parent(request),
            "session": session_info,
            "session_id": session_id,
            "messages": parsed_messages,
            "token_usage": token_usage,
            "tool_summary": tool_summary,
            "file_metrics": file_metrics,
        },
    )
