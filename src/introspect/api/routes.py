"""Route handlers for introspect web UI."""

import json
import math
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from introspect.search import build_search_corpus, fts_search

router = APIRouter()

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE = 20


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
async def sessions(request: Request, page: int = Query(1, ge=1)):
    """Paginated session list."""
    conn = _conn(request)

    total = conn.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    total_pages = max(1, math.ceil(total / SESSIONS_PER_PAGE))
    offset = (page - 1) * SESSIONS_PER_PAGE

    rows = conn.execute(
        """
        SELECT session_id, started_at, ended_at, duration,
               user_messages, assistant_messages, model, cwd
        FROM logical_sessions
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """,
        [SESSIONS_PER_PAGE, offset],
    ).fetchall()

    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "parent": _parent(request),
            "sessions": rows,
            "page": page,
            "total_pages": total_pages,
            "total": total,
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
        # Auto-build corpus if needed
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'search_corpus' AND table_type = 'BASE TABLE'
        """).fetchall()
        if not tables:
            build_search_corpus(conn)

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
    """Tool call stats with filtering."""
    conn = _conn(request)

    where_clauses = []
    params: list[str | int] = []
    if failed:
        where_clauses.append("is_error = 'true'")
    if name.strip():
        where_clauses.append("tool_name = ?")
        params.append(name.strip())

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    rows = conn.execute(
        f"""
        SELECT
            session_id,
            called_at,
            tool_name,
            is_error,
            LEFT(tool_input, 200) AS input_preview,
            execution_time
        FROM tool_calls
        {where}
        ORDER BY called_at DESC
        LIMIT 100
    """,  # nosec B608
        params,
    ).fetchall()

    # Get tool name list for filter dropdown
    tool_names = conn.execute("""
        SELECT DISTINCT tool_name
        FROM tool_calls
        WHERE tool_name IS NOT NULL
        ORDER BY tool_name
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
            "tool_names": [t[0] for t in tool_names],
            "filter_failed": failed,
            "filter_name": name,
            "stats": stats,
        },
    )


@router.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    """Stats and insights page."""
    conn = _conn(request)

    total_sessions = conn.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    total_tool_calls = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    total_failed = conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE is_error = 'true'"
    ).fetchone()[0]

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

    # Tool usage breakdown
    tool_breakdown = conn.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM tool_calls
        WHERE tool_name IS NOT NULL
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    # Longest sessions top 5
    longest_sessions = conn.execute("""
        SELECT session_id, started_at, duration, model
        FROM logical_sessions
        ORDER BY duration DESC
        LIMIT 5
    """).fetchall()

    # Most tool calls sessions top 5
    most_tools_sessions = conn.execute("""
        SELECT
            session_id,
            COUNT(*) AS tool_count,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
        FROM tool_calls
        GROUP BY session_id
        ORDER BY tool_count DESC
        LIMIT 5
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
            "duration_buckets": duration_buckets,
            "tool_breakdown": tool_breakdown,
            "longest_sessions": longest_sessions,
            "most_tools_sessions": most_tools_sessions,
            "sessions_per_day": sessions_per_day,
            "token_usage": token_usage,
        },
    )
