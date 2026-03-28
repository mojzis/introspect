"""Dashboard route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import (
    clean_title,
    conn,
    fetch_token_usage,
    format_duration,
    parent,
    templates,
)


async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard with session count, recent sessions, and quick stats."""
    db = conn(request)

    # Single scan of logical_sessions for all summary metrics
    summary = db.execute("""
        SELECT
            COUNT(*),
            COUNT(DISTINCT project)
                FILTER (WHERE project IS NOT NULL),
            AVG(EXTRACT(EPOCH FROM duration))
                FILTER (WHERE duration IS NOT NULL),
            COUNT(*) FILTER (
                WHERE CAST(started_at AS DATE) = CURRENT_DATE
            ),
            COUNT(*) FILTER (
                WHERE started_at >= date_trunc('week', CURRENT_DATE)
            )
        FROM logical_sessions
    """).fetchone()
    session_count = summary[0]
    project_count = summary[1] or 0
    avg_duration_str = format_duration(summary[2] or 0)
    activity_today = summary[3] or 0
    activity_week = summary[4] or 0

    # Single scan of tool_calls for counts
    tool_stats = db.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE is_error = 'true')
        FROM tool_calls
    """).fetchone()
    tool_count = tool_stats[0]
    failed_count = tool_stats[1]

    success_rate = (
        round(100 * (1 - failed_count / tool_count), 1) if tool_count > 0 else 100.0
    )

    token_usage = fetch_token_usage(db)

    # Recent sessions with titles
    recent_sessions = db.execute("""
        SELECT ls.session_id, ls.started_at, ls.duration, ls.user_messages,
               ls.assistant_messages, ls.model, ls.cwd, fp.first_prompt
        FROM logical_sessions ls
        LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
        ORDER BY ls.started_at DESC
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
            "success_rate": success_rate,
            "token_usage": token_usage,
            "project_count": project_count,
            "avg_duration": avg_duration_str,
            "activity_today": activity_today,
            "activity_week": activity_week,
            "recent_sessions": recent_sessions,
            "top_tools": top_tools,
            "clean_title": clean_title,
        },
    )
