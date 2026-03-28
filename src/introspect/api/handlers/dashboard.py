"""Dashboard route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import clean_title, conn, parent, templates


async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard with session count, recent sessions, and quick stats."""
    db = conn(request)

    session_count = db.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    tool_count = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    failed_count = db.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE is_error = 'true'"
    ).fetchone()[0]

    success_rate = (
        round(100 * (1 - failed_count / tool_count), 1) if tool_count > 0 else 100.0
    )

    # Token usage (best effort)
    try:
        token_usage = db.execute("""
            SELECT
                SUM(CAST(json_extract(message, '$.usage.input_tokens') AS BIGINT)),
                SUM(CAST(json_extract(message, '$.usage.output_tokens') AS BIGINT))
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract(message, '$.usage.input_tokens') IS NOT NULL
        """).fetchone()
    except Exception:
        token_usage = None

    # Active projects and avg duration
    extras = db.execute("""
        SELECT
            COUNT(DISTINCT split_part(rtrim(cwd, '/'), '/', -1))
                FILTER (WHERE cwd IS NOT NULL),
            AVG(EXTRACT(EPOCH FROM duration))
                FILTER (WHERE duration IS NOT NULL)
        FROM logical_sessions
    """).fetchone()
    project_count = extras[0] or 0
    avg_secs = extras[1] or 0
    avg_duration_str = f"{int(avg_secs) // 60}:{int(avg_secs) % 60:02d}"

    # Today / this week activity
    activity = db.execute("""
        SELECT
            COUNT(*) FILTER (
                WHERE CAST(started_at AS DATE) = CURRENT_DATE
            ) AS today,
            COUNT(*) FILTER (
                WHERE started_at >= date_trunc('week', CURRENT_DATE)
            ) AS this_week
        FROM logical_sessions
    """).fetchone()
    activity_today = activity[0] or 0
    activity_week = activity[1] or 0

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
