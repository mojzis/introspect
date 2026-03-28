"""Tools route handler."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import conn, parent, templates

TOOLS_PAGE_SIZE = 50


async def tools(
    request: Request, failed: bool, name: str, page: int = 1
) -> HTMLResponse:
    """Tool call stats with filtering (non-MCP tools only)."""
    db = conn(request)

    # Base filter: exclude MCP tools (they have their own page)
    where_clauses = ["tool_name NOT LIKE 'mcp__%'"]
    params: list[str | int] = []
    if failed:
        where_clauses.append("is_error = 'true'")
    if name.strip():
        where_clauses.append("tool_name = ?")
        params.append(name.strip())

    where = "WHERE " + " AND ".join(where_clauses)

    # Stats summary (also gives total count for pagination)
    stats = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_total
        FROM tool_calls
        {where}
    """,  # nosec B608
        params,
    ).fetchone()

    total_pages = max(1, math.ceil(stats[0] / TOOLS_PAGE_SIZE))
    offset = (page - 1) * TOOLS_PAGE_SIZE

    rows = db.execute(
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
        LIMIT ? OFFSET ?
    """,  # nosec B608
        [*params, TOOLS_PAGE_SIZE, offset],
    ).fetchall()

    # Get tool names with counts and success rate for filter buttons (non-MCP only)
    tool_names = db.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM tool_calls
        WHERE tool_name IS NOT NULL
          AND tool_name NOT LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    return templates.TemplateResponse(
        request,
        "tools.html",
        {
            "parent": parent(request),
            "tool_calls": rows,
            "tool_names": tool_names,
            "filter_failed": failed,
            "filter_name": name,
            "stats": stats,
            "page": page,
            "total_pages": total_pages,
        },
    )
