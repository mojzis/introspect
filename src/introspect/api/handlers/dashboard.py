"""Dashboard route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import conn, parent, templates


async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard with session count, recent sessions, and quick stats."""
    db = conn(request)

    session_count = db.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    tool_count = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    failed_count = db.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE is_error = 'true'"
    ).fetchone()[0]

    recent_sessions = db.execute("""
        SELECT session_id, started_at, duration, user_messages,
               assistant_messages, model, cwd
        FROM logical_sessions
        ORDER BY started_at DESC
        LIMIT 5
    """).fetchall()

    top_tools = db.execute("""
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
            "parent": parent(request),
            "session_count": session_count,
            "tool_count": tool_count,
            "failed_count": failed_count,
            "recent_sessions": recent_sessions,
            "top_tools": top_tools,
        },
    )
