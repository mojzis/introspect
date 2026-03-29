"""Tools route handler."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import DEFAULT_PAGE_SIZE, conn, parent, templates


async def tools(
    request: Request,
    failed: bool,
    name: str,
    session: str,
    project: str,
    branch: str,
    page: int = 1,
) -> HTMLResponse:
    """Tool call stats with filtering (non-MCP tools only)."""
    db = conn(request)
    session = session.strip()
    project = project.strip()
    branch = branch.strip()

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
    if branch:
        where_clauses.append("ls.git_branch = ?")
        params.append(branch)

    where = "WHERE " + " AND ".join(where_clauses)
    join = "LEFT JOIN logical_sessions ls ON tc.session_id = ls.session_id"

    # Stats summary (also gives total count for pagination)
    stats = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE tc.is_error = 'true') AS failed_total
        FROM tool_calls tc
        {join}
        {where}
    """,  # nosec B608
        params,
    ).fetchone()

    total_pages = max(1, math.ceil(stats[0] / DEFAULT_PAGE_SIZE))
    offset = (page - 1) * DEFAULT_PAGE_SIZE

    rows = db.execute(
        f"""
        SELECT
            tc.session_id,
            tc.called_at,
            tc.tool_name,
            tc.is_error,
            LEFT(tc.tool_input, 2000) AS input_preview,
            tc.execution_time,
            fp.first_prompt,
            ls.project,
            ls.git_branch
        FROM tool_calls tc
        LEFT JOIN session_titles fp USING (session_id)
        LEFT JOIN logical_sessions ls ON tc.session_id = ls.session_id
        {where}
        ORDER BY tc.called_at DESC
        LIMIT ? OFFSET ?
    """,  # nosec B608
        [*params, DEFAULT_PAGE_SIZE, offset],
    ).fetchall()

    # Get tool names with counts and success rate for filter buttons (non-MCP only)
    tn_where = ["tc.tool_name IS NOT NULL", "tc.tool_name NOT LIKE 'mcp__%'"]
    tn_params: list[str] = []
    if session:
        tn_where.append("tc.session_id = ?")
        tn_params.append(session)
    if project:
        tn_where.append("ls.project = ?")
        tn_params.append(project)
    if branch:
        tn_where.append("ls.git_branch = ?")
        tn_params.append(branch)
    tool_names = db.execute(
        f"""
        SELECT
            tc.tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE tc.is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM tool_calls tc
        LEFT JOIN logical_sessions ls ON tc.session_id = ls.session_id
        WHERE {" AND ".join(tn_where)}
        GROUP BY tc.tool_name
        ORDER BY cnt DESC
    """,  # nosec B608
        tn_params,
    ).fetchall()

    # Distinct projects and branches for filter dropdowns
    projects = [
        r[0]
        for r in db.execute("""
        SELECT DISTINCT project FROM logical_sessions
        WHERE project IS NOT NULL ORDER BY project
    """).fetchall()
    ]
    branches = [
        r[0]
        for r in db.execute("""
        SELECT DISTINCT git_branch FROM logical_sessions
        WHERE git_branch IS NOT NULL ORDER BY git_branch
    """).fetchall()
    ]

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
            "filter_branch": branch,
            "projects": projects,
            "branches": branches,
            "stats": stats,
            "page": page,
            "total_pages": total_pages,
        },
    )
