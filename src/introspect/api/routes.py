"""Route handlers for introspect web UI."""

import json
import math
import re
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from introspect.search import ensure_search_corpus, fts_search

router = APIRouter()

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE_DEFAULT = 50
SESSIONS_PAGE_SIZES = [25, 50, 100, 200]

# Allowed sort columns for sessions page
_SESSIONS_SORT_COLS = {
    "started_at": "ls.started_at",
    "duration": "ls.duration",
    "user_msgs": "ls.user_messages",
    "asst_msgs": "ls.assistant_messages",
    "model": "ls.model",
    "project": "ls.cwd",
    "branch": "ls.git_branch",
}
_SESSIONS_SORT_DEFAULT = "started_at"

_XML_TAG_PREFIX_RE = re.compile(r"^<[^>]+>")


def _clean_title(raw: str) -> str:
    """Strip leading XML-style tags from session titles."""
    return _XML_TAG_PREFIX_RE.sub("", raw).strip()


def _parent(request: Request) -> str:
    """Return the base template: full page for normal requests, partial for HTMX."""
    if request.headers.get("HX-Request"):
        return "partial.html"
    return "base.html"


def _parse_content_block(block) -> dict:  # noqa: PLR0911
    """Parse a single content block from a message."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)[:500]}

    block_type = block.get("type", "text")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "tool_use":
        input_str = block.get("input", "")
        if isinstance(input_str, dict):
            input_str = json.dumps(input_str, indent=2)
        return {
            "type": "tool_use",
            "name": block.get("name", ""),
            "tool_use_id": block.get("id", ""),
            "input": str(input_str)[:500],
        }
    if block_type == "tool_result":
        result_content = block.get("content", "")
        if isinstance(result_content, list):
            result_content = json.dumps(result_content)
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", ""),
            "content": str(result_content)[:500],
            "is_error": block.get("is_error", False),
        }
    if block_type == "thinking":
        return {"type": "thinking", "text": block.get("thinking", "")}
    return {"type": "text", "text": str(block)[:500]}


def _conn(request: Request):
    """Get the DuckDB connection from request state."""
    return request.state.conn


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard with session count, recent sessions, and quick stats."""
    conn = _conn(request)

    session_count = conn.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    tool_count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    failed_count = conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE is_error = 'true'"
    ).fetchone()[0]

    recent_sessions = conn.execute("""
        SELECT session_id, started_at, duration, user_messages,
               assistant_messages, model, cwd
        FROM logical_sessions
        ORDER BY started_at DESC
        LIMIT 5
    """).fetchall()

    top_tools = conn.execute("""
        SELECT tool_name, COUNT(*) AS cnt
        FROM tool_calls
        GROUP BY tool_name
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "parent": _parent(request),
            "session_count": session_count,
            "tool_count": tool_count,
            "failed_count": failed_count,
            "recent_sessions": recent_sessions,
            "top_tools": top_tools,
        },
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions(  # noqa: PLR0913
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(SESSIONS_PER_PAGE_DEFAULT, ge=1, le=500),
    sort: str = Query(_SESSIONS_SORT_DEFAULT),
    order: str = Query("desc"),
    model: str = Query("", alias="model"),
    project: str = Query("", alias="project"),
    branch: str = Query("", alias="branch"),
):
    """Paginated session list with filtering and sorting."""
    conn = _conn(request)

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
        where_clauses.append("ls.cwd LIKE ?")
        params.append(f"%/{project.strip()}")
    if branch.strip():
        where_clauses.append("ls.git_branch = ?")
        params.append(branch.strip())

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Count with filters
    total = conn.execute(
        f"SELECT COUNT(*) FROM logical_sessions ls {where}",  # nosec B608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    # Resolve sort column
    default_col = _SESSIONS_SORT_COLS[_SESSIONS_SORT_DEFAULT]
    sort_col = _SESSIONS_SORT_COLS.get(sort, default_col)
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"
    nulls = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    rows = conn.execute(
        f"""
        SELECT
            ls.session_id,
            ls.started_at,
            ls.ended_at,
            ls.duration,
            ls.user_messages,
            ls.assistant_messages,
            ls.model,
            ls.cwd,
            ls.git_branch,
            fp.first_prompt
        FROM logical_sessions ls
        LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
        {where}
        ORDER BY {sort_col} {sort_dir} {nulls}
        LIMIT ? OFFSET ?
    """,  # nosec B608
        [*params, page_size, offset],
    ).fetchall()

    session_list = []
    for row in rows:
        (
            session_id,
            started_at,
            ended_at,
            duration,
            user_msgs,
            asst_msgs,
            _model,
            cwd,
            git_branch,
            first_prompt,
        ) = row
        dur_str = ""
        if duration:
            total_secs = int(duration.total_seconds())
            dur_str = f"{total_secs // 60}:{total_secs % 60:02d}"
        proj = ""
        if cwd:
            proj = cwd.rstrip("/").rsplit("/", 1)[-1]
        session_list.append(
            {
                "id": session_id,
                "date": str(started_at)[5:10] if started_at else "",
                "start_time": str(started_at)[11:16] if started_at else "",
                "end_time": str(ended_at)[11:16] if ended_at else "",
                "duration": dur_str,
                "user_msgs": user_msgs or 0,
                "asst_msgs": asst_msgs or 0,
                "model": _model or "",
                "project": proj,
                "branch": git_branch or "",
                "title": _clean_title(first_prompt or "")[:120],
            }
        )

    # Get distinct values for filter dropdowns
    models = conn.execute("""
        SELECT DISTINCT model FROM logical_sessions
        WHERE model IS NOT NULL ORDER BY model
    """).fetchall()
    projects = conn.execute("""
        SELECT DISTINCT split_part(rtrim(cwd, '/'), '/', -1) AS proj
        FROM logical_sessions
        WHERE cwd IS NOT NULL
        ORDER BY proj
    """).fetchall()
    branches = conn.execute("""
        SELECT DISTINCT git_branch FROM logical_sessions
        WHERE git_branch IS NOT NULL ORDER BY git_branch
    """).fetchall()

    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "parent": _parent(request),
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
            "models": [r[0] for r in models],
            "projects": [r[0] for r in projects],
            "branches": [r[0] for r in branches],
        },
    )


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    """Full session detail with all messages."""
    conn = _conn(request)

    session_info = conn.execute(
        """
        SELECT session_id, started_at, ended_at, duration,
               user_messages, assistant_messages, model, cwd, git_branch
        FROM logical_sessions
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    messages = conn.execute(
        """
        SELECT timestamp, type, role, message, uuid
        FROM raw_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """,
        [session_id],
    ).fetchall()

    parsed_messages = []
    for msg in messages:
        timestamp, msg_type, role, message_json, uuid = msg
        content_blocks = []

        try:
            msg_data = (
                json.loads(message_json)
                if isinstance(message_json, str)
                else message_json
            )
        except (json.JSONDecodeError, TypeError):
            msg_data = {}

        raw_content = msg_data.get("content", "")

        if isinstance(raw_content, str):
            content_blocks.append({"type": "text", "text": raw_content})
        elif isinstance(raw_content, list):
            content_blocks = [_parse_content_block(block) for block in raw_content]

        parsed_messages.append(
            {
                "timestamp": str(timestamp)[:19] if timestamp else "",
                "type": msg_type,
                "role": role or msg_type,
                "content_blocks": content_blocks,
                "uuid": uuid,
            }
        )

    return templates.TemplateResponse(
        request,
        "session_detail.html",
        {
            "parent": _parent(request),
            "session": session_info,
            "session_id": session_id,
            "messages": parsed_messages,
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Query("", alias="q")):
    """Search results with snippets."""
    conn = _conn(request)
    results = []

    if q.strip():
        ensure_search_corpus(conn)
        results = fts_search(conn, q, 50)

    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "parent": _parent(request),
            "query": q,
            "results": results,
        },
    )


@router.get("/tools", response_class=HTMLResponse)
async def tools(
    request: Request,
    failed: bool = Query(False),
    name: str = Query("", alias="name"),
):
    """Tool call stats with filtering (non-MCP tools only)."""
    conn = _conn(request)

    # Base filter: exclude MCP tools (they have their own page)
    where_clauses = ["tool_name NOT LIKE 'mcp__%'"]
    params: list[str | int] = []
    if failed:
        where_clauses.append("is_error = 'true'")
    if name.strip():
        where_clauses.append("tool_name = ?")
        params.append(name.strip())

    where = "WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(
        f"""
        SELECT
            tc.session_id,
            tc.called_at,
            tc.tool_name,
            tc.is_error,
            LEFT(tc.tool_input, 200) AS input_preview,
            tc.execution_time,
            fp.first_prompt
        FROM tool_calls tc
        LEFT JOIN session_titles fp ON tc.session_id = fp.session_id
        {where}
        ORDER BY tc.called_at DESC
        LIMIT 100
    """,  # nosec B608
        params,
    ).fetchall()

    # Get tool names with counts for filter buttons (non-MCP only)
    tool_names = conn.execute("""
        SELECT tool_name, COUNT(*) AS cnt
        FROM tool_calls
        WHERE tool_name IS NOT NULL
          AND tool_name NOT LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    # Stats summary
    stats = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_total
        FROM tool_calls
        {where}
    """,  # nosec B608
        params,
    ).fetchone()

    return templates.TemplateResponse(
        request,
        "tools.html",
        {
            "parent": _parent(request),
            "tool_calls": rows,
            "tool_names": tool_names,
            "filter_failed": failed,
            "filter_name": name,
            "stats": stats,
        },
    )


RAW_PER_PAGE = 20


@router.get("/raw", response_class=HTMLResponse)
async def raw_data(
    request: Request,
    page: int = Query(1, ge=1),
    session: str = Query("", alias="session"),
    record_type: str = Query("", alias="type"),
):
    """Raw unfiltered JSONL records with all fields."""
    conn = _conn(request)

    where_clauses: list[str] = []
    params: list[str] = []
    if session.strip():
        where_clauses.append("CAST(sessionId AS VARCHAR) LIKE ?")
        params.append(f"{session.strip()}%")
    if record_type.strip():
        where_clauses.append("type = ?")
        params.append(record_type.strip())

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM raw_data {where}",  # nosec B608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / RAW_PER_PAGE))
    offset = (page - 1) * RAW_PER_PAGE

    result = conn.execute(
        f"SELECT * FROM raw_data {where} LIMIT {RAW_PER_PAGE} OFFSET {offset}",  # nosec B608
        params,
    )
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()

    # Get distinct types for filter dropdown
    record_types = conn.execute(
        "SELECT DISTINCT type FROM raw_data WHERE type IS NOT NULL ORDER BY type"
    ).fetchall()

    # Build records as list of {column, value, is_json, preview} dicts
    preview_len = 100
    records = []
    for row in rows:
        fields = []
        for col, val in zip(columns, row, strict=True):
            if val is None:
                continue
            val_str = str(val).strip()
            # Try to pretty-print JSON objects/arrays
            is_json = False
            if isinstance(val, (dict, list)):
                try:
                    val_str = json.dumps(val, indent=2, ensure_ascii=False)
                    is_json = True
                except (TypeError, ValueError):
                    pass
            elif isinstance(val, str) and val.strip()[:1] in ("{", "["):
                try:
                    parsed = json.loads(val)
                    val_str = json.dumps(parsed, indent=2, ensure_ascii=False)
                    is_json = True
                except (json.JSONDecodeError, ValueError):
                    pass
            long = len(val_str) > preview_len
            preview = val_str[:preview_len] + "..." if long else val_str
            fields.append(
                {
                    "column": col,
                    "value": val_str,
                    "preview": preview,
                    "long": long,
                    "is_json": is_json,
                }
            )
        records.append(fields)

    return templates.TemplateResponse(
        request,
        "raw.html",
        {
            "parent": _parent(request),
            "records": records,
            "columns": columns,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "filter_session": session,
            "filter_type": record_type,
            "record_types": [r[0] for r in record_types],
        },
    )


@router.get("/mcps", response_class=HTMLResponse)
async def mcps(
    request: Request,
    server: str = Query("", alias="server"),
    command: str = Query("", alias="command"),
    failed: bool = Query(False),
):
    """MCP tool analysis with server/command breakdown."""
    conn = _conn(request)

    # --- Server overview ---
    mcp_servers = conn.execute("""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            COUNT(*) AS cnt,
            COUNT(DISTINCT split_part(tool_name, '__', 3)) AS command_count,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY server_name
        ORDER BY cnt DESC
    """).fetchall()

    # --- Commands for selected server (or all) ---
    cmd_where = ["tool_name LIKE 'mcp__%'"]
    mcp_params: list[str | int] = []
    if server.strip():
        cmd_where.append("split_part(tool_name, '__', 2) = ?")
        mcp_params.append(server.strip())

    mcp_commands = conn.execute(
        f"""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            split_part(tool_name, '__', 3) AS command_name,
            COUNT(*) AS cnt,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
        FROM tool_calls
        WHERE {" AND ".join(cmd_where)}
        GROUP BY server_name, command_name
        ORDER BY cnt DESC
    """,  # nosec B608
        mcp_params,
    ).fetchall()

    # --- Filtered call list ---
    list_where = ["tool_name LIKE 'mcp__%'"]
    list_params: list[str | int] = []
    if server.strip():
        list_where.append("split_part(tc.tool_name, '__', 2) = ?")
        list_params.append(server.strip())
    if command.strip():
        list_where.append("split_part(tc.tool_name, '__', 3) = ?")
        list_params.append(command.strip())
    if failed:
        list_where.append("tc.is_error = 'true'")

    rows = conn.execute(
        f"""
        SELECT
            tc.session_id,
            tc.called_at,
            split_part(tc.tool_name, '__', 2) AS server_name,
            split_part(tc.tool_name, '__', 3) AS command_name,
            tc.is_error,
            LEFT(tc.tool_input, 200) AS input_preview,
            tc.execution_time,
            fp.first_prompt
        FROM tool_calls tc
        LEFT JOIN session_titles fp ON tc.session_id = fp.session_id
        WHERE {" AND ".join(list_where)}
        ORDER BY tc.called_at DESC
        LIMIT 100
    """,  # nosec B608
        list_params,
    ).fetchall()

    mcp_stats = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_total
        FROM tool_calls tc
        WHERE {" AND ".join(list_where)}
    """,  # nosec B608
        list_params,
    ).fetchone()

    return templates.TemplateResponse(
        request,
        "mcps.html",
        {
            "parent": _parent(request),
            "mcp_servers": mcp_servers,
            "mcp_commands": mcp_commands,
            "tool_calls": rows,
            "filter_server": server,
            "filter_command": command,
            "filter_failed": failed,
            "stats": mcp_stats,
        },
    )


@router.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    """Stats and insights page."""
    conn = _conn(request)

    # Summary metrics
    summary = conn.execute("""
        SELECT
            COUNT(*) AS total_sessions,
            MIN(started_at) AS earliest_session,
            SUM(user_messages) AS total_user_messages,
            SUM(user_messages + assistant_messages) AS total_turns
        FROM logical_sessions
    """).fetchone()
    total_sessions = summary[0]
    earliest_session = summary[1]
    total_user_messages = summary[2] or 0
    total_turns = summary[3] or 0

    tool_summary = conn.execute("""
        SELECT
            COUNT(*) AS total_tool_calls,
            COUNT(*) FILTER (WHERE is_error = 'true') AS total_failed
        FROM tool_calls
    """).fetchone()
    total_tool_calls = tool_summary[0]
    total_failed = tool_summary[1]

    # Session duration distribution
    duration_buckets = conn.execute("""
        SELECT
            bucket,
            COUNT(*) AS cnt
        FROM (
            SELECT
                CASE
                    WHEN duration < INTERVAL '1 minute' THEN '< 1 min'
                    WHEN duration < INTERVAL '5 minutes' THEN '1-5 min'
                    WHEN duration < INTERVAL '15 minutes' THEN '5-15 min'
                    WHEN duration < INTERVAL '30 minutes' THEN '15-30 min'
                    ELSE '30+ min'
                END AS bucket,
                CASE
                    WHEN duration < INTERVAL '1 minute' THEN 1
                    WHEN duration < INTERVAL '5 minutes' THEN 2
                    WHEN duration < INTERVAL '15 minutes' THEN 3
                    WHEN duration < INTERVAL '30 minutes' THEN 4
                    ELSE 5
                END AS sort_order
            FROM logical_sessions
        ) sub
        GROUP BY bucket, sort_order
        ORDER BY sort_order
    """).fetchall()

    # Turns per session distribution
    turns_buckets = conn.execute("""
        SELECT
            bucket,
            COUNT(*) AS cnt
        FROM (
            SELECT
                CASE
                    WHEN (user_messages + assistant_messages) <= 2 THEN '1-2 turns'
                    WHEN (user_messages + assistant_messages) <= 5 THEN '3-5 turns'
                    WHEN (user_messages + assistant_messages) <= 10 THEN '6-10 turns'
                    WHEN (user_messages + assistant_messages) <= 20 THEN '11-20 turns'
                    WHEN (user_messages + assistant_messages) <= 50 THEN '21-50 turns'
                    WHEN (user_messages + assistant_messages) <= 100 THEN '51-100 turns'
                    ELSE '100+ turns'
                END AS bucket,
                CASE
                    WHEN (user_messages + assistant_messages) <= 2 THEN 1
                    WHEN (user_messages + assistant_messages) <= 5 THEN 2
                    WHEN (user_messages + assistant_messages) <= 10 THEN 3
                    WHEN (user_messages + assistant_messages) <= 20 THEN 4
                    WHEN (user_messages + assistant_messages) <= 50 THEN 5
                    WHEN (user_messages + assistant_messages) <= 100 THEN 6
                    ELSE 7
                END AS sort_order
            FROM logical_sessions
        ) sub
        GROUP BY bucket, sort_order
        ORDER BY sort_order
    """).fetchall()

    # Tool usage breakdown (non-MCP tools only, with percentage)
    tool_breakdown = conn.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate,
            100.0 * COUNT(*) / (SELECT COUNT(*) FROM tool_calls) AS pct_of_total
        FROM tool_calls
        WHERE tool_name IS NOT NULL
          AND tool_name NOT LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    # MCP usage: servers and their commands
    mcp_breakdown = conn.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate,
            100.0 * COUNT(*) / (SELECT COUNT(*) FROM tool_calls) AS pct_of_total
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    mcp_servers = conn.execute("""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            COUNT(*) AS cnt,
            COUNT(DISTINCT split_part(tool_name, '__', 3)) AS command_count
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY server_name
        ORDER BY cnt DESC
    """).fetchall()

    # Bash command breakdown
    bash_breakdown = conn.execute("""
        SELECT
            split_part(
                trim(json_extract_string(tool_input, '$.command')),
                ' ', 1
            ) AS first_word,
            COUNT(*) AS cnt
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND json_extract_string(tool_input, '$.command') IS NOT NULL
        GROUP BY first_word
        ORDER BY cnt DESC
    """).fetchall()

    bash_two_words = conn.execute("""
        WITH cmds AS (
            SELECT trim(
                json_extract_string(tool_input, '$.command')
            ) AS cmd
            FROM tool_calls
            WHERE tool_name = 'Bash'
              AND json_extract_string(
                  tool_input, '$.command'
              ) IS NOT NULL
        )
        SELECT
            CASE
                WHEN array_length(
                    string_split(cmd, ' ')
                ) >= 2
                THEN split_part(cmd, ' ', 1)
                    || ' '
                    || split_part(cmd, ' ', 2)
                ELSE cmd
            END AS first_two_words,
            COUNT(*) AS cnt
        FROM cmds
        GROUP BY first_two_words
        ORDER BY cnt DESC
        LIMIT 30
    """).fetchall()

    bash_chained = conn.execute("""
        SELECT
            COUNT(*) FILTER (
                WHERE json_extract_string(tool_input, '$.command') LIKE '%&&%'
            ) AS chained_count,
            COUNT(*) AS total_bash
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND json_extract_string(tool_input, '$.command') IS NOT NULL
    """).fetchone()

    # Longest sessions top 15 with all metrics
    longest_sessions = conn.execute("""
        SELECT
            ls.session_id, ls.started_at, ls.duration, ls.model,
            ls.user_messages, ls.assistant_messages,
            (ls.user_messages + ls.assistant_messages) AS total_turns,
            COALESCE(tc.tool_count, 0) AS tool_count,
            COALESCE(tc.failed_count, 0) AS failed_count
        FROM logical_sessions ls
        LEFT JOIN (
            SELECT
                session_id,
                COUNT(*) AS tool_count,
                COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
            FROM tool_calls
            GROUP BY session_id
        ) tc ON ls.session_id = tc.session_id
        ORDER BY ls.duration DESC
        LIMIT 15
    """).fetchall()

    # Most tool calls sessions top 15 with all metrics
    most_tools_sessions = conn.execute("""
        SELECT
            ls.session_id,
            COALESCE(tc.tool_count, 0) AS tool_count,
            COALESCE(tc.failed_count, 0) AS failed_count,
            ls.started_at, ls.duration, ls.model,
            ls.user_messages, ls.assistant_messages,
            (ls.user_messages + ls.assistant_messages) AS total_turns
        FROM logical_sessions ls
        LEFT JOIN (
            SELECT
                session_id,
                COUNT(*) AS tool_count,
                COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
            FROM tool_calls
            GROUP BY session_id
        ) tc ON ls.session_id = tc.session_id
        ORDER BY tc.tool_count DESC NULLS LAST
        LIMIT 15
    """).fetchall()

    # Sessions per day
    sessions_per_day = conn.execute("""
        SELECT
            CAST(started_at AS DATE) AS day,
            COUNT(*) AS cnt
        FROM logical_sessions
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
    """).fetchall()

    # Token usage summary (best effort)
    try:
        token_usage = conn.execute("""
            SELECT
                SUM(CAST(json_extract(message, '$.usage.input_tokens') AS BIGINT)),
                SUM(CAST(json_extract(message, '$.usage.output_tokens') AS BIGINT))
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract(message, '$.usage.input_tokens') IS NOT NULL
        """).fetchone()
    except Exception:
        token_usage = None

    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "parent": _parent(request),
            "total_sessions": total_sessions,
            "total_tool_calls": total_tool_calls,
            "total_failed": total_failed,
            "total_user_messages": total_user_messages,
            "total_turns": total_turns,
            "earliest_session": earliest_session,
            "duration_buckets": duration_buckets,
            "turns_buckets": turns_buckets,
            "tool_breakdown": tool_breakdown,
            "mcp_breakdown": mcp_breakdown,
            "mcp_servers": mcp_servers,
            "bash_breakdown": bash_breakdown,
            "bash_two_words": bash_two_words,
            "bash_chained": bash_chained,
            "longest_sessions": longest_sessions,
            "most_tools_sessions": most_tools_sessions,
            "sessions_per_day": sessions_per_day,
            "token_usage": token_usage,
        },
    )
