"""Tools route handler."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import (
    DEFAULT_PAGE_SIZE,
    clean_title,
    conn,
    escape_ilike,
    fetch_distinct_projects,
    parent,
    templates,
)


async def tools(  # noqa: PLR0913
    request: Request,
    failed: bool,
    name: str,
    session: str,
    project: str,
    q: str,
    page: int = 1,
) -> HTMLResponse:
    """Tool call stats with filtering (non-MCP tools only)."""
    db = conn(request)
    session = session.strip()
    project = project.strip()
    q = q.strip()

    # Base filter: exclude MCP tools (they have their own page)
    where_clauses = ["tc.tool_name NOT LIKE 'mcp__%'"]
    params: list[str | int] = []
    if failed:
        where_clauses.append("tc.is_error = 'true'")
    if name.strip():
        where_clauses.append("tc.tool_name = ?")
        params.append(name.strip())
    if session:
        where_clauses.append("tc.session_id = ?")
        params.append(session)
    if project:
        where_clauses.append("ls.project = ?")
        params.append(project)
    if q:
        escaped = f"%{escape_ilike(q)}%"
        where_clauses.append(
            "(json_extract_string(tc.tool_input, '$.description')"
            " ILIKE ? OR tc.tool_input ILIKE ?)"
        )
        params.extend([escaped, escaped])

    where = "WHERE " + " AND ".join(where_clauses)
    joins = (
        "LEFT JOIN session_titles fp USING (session_id)"
        " LEFT JOIN logical_sessions ls"
        " ON tc.session_id = ls.session_id"
    )

    # Stats summary (also gives total count for pagination)
    stats = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE tc.is_error = 'true') AS failed_total
        FROM tool_calls tc
        {joins}
        {where}
    """,  # noqa: S608
        params,
    ).fetchone()

    total_pages = max(1, math.ceil(stats[0] / DEFAULT_PAGE_SIZE))
    offset = (page - 1) * DEFAULT_PAGE_SIZE

    rows = db.execute(
        f"""
        SELECT
            tc.session_id,
            strftime(tc.called_at, '%b %d %H:%M') AS called_at_fmt,
            tc.tool_name,
            tc.is_error,
            json_extract_string(tc.tool_input, '$.description') AS description,
            LEFT(tc.tool_input, 200) AS input_preview,
            tc.execution_time,
            fp.first_prompt,
            tc.tool_use_id,
            ls.project
        FROM tool_calls tc
        {joins}
        {where}
        ORDER BY tc.called_at DESC
        LIMIT ? OFFSET ?
    """,  # noqa: S608
        [*params, DEFAULT_PAGE_SIZE, offset],
    ).fetchall()

    # Get tool names with counts and success rate for filter buttons (non-MCP only)
    tn_where = ["tool_name IS NOT NULL", "tool_name NOT LIKE 'mcp__%'"]
    tn_params: list[str] = []
    if session:
        tn_where.append("session_id = ?")
        tn_params.append(session)
    tool_names = db.execute(
        f"""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM tool_calls
        WHERE {" AND ".join(tn_where)}
        GROUP BY tool_name
        ORDER BY cnt DESC
    """,  # noqa: S608
        tn_params,
    ).fetchall()

    return templates.TemplateResponse(
        request,
        "tools.html",
        {
            "parent": parent(request),
            "tool_calls": rows,
            "tool_names": tool_names,
            "filter_failed": failed,
            "filter_name": name,
            "filter_session": session,
            "filter_project": project,
            "filter_q": q,
            "projects": fetch_distinct_projects(db),
            "stats": stats,
            "page": page,
            "total_pages": total_pages,
            "clean_title": clean_title,
        },
    )
