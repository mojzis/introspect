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
        "dashboard.html",
        {
            "request": request,
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
        "sessions.html",
        {
            "request": request,
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
        "session_detail.html",
        {
            "request": request,
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
        "search.html",
        {
            "request": request,
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
    """,
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
    """,
        params,
    ).fetchone()

    return templates.TemplateResponse(
        "tools.html",
        {
            "request": request,
            "tool_calls": rows,
            "tool_names": [t[0] for t in tool_names],
            "filter_failed": failed,
            "filter_name": name,
            "stats": stats,
        },
    )
