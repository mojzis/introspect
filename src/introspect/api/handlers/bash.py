"""Bash commands route handler."""

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

_MAX_PREFIXES = 40


def _cmd_expr(alias: str = "") -> str:
    """SQL expression extracting the trimmed command from tool_input."""
    col = f"{alias}tool_input" if alias else "tool_input"
    return f"trim(json_extract_string({col}, '$.command'))"


def _prefix_expr(alias: str = "") -> str:
    """SQL expression grouping commands by first 2 words.

    *alias* should include the trailing dot, e.g. ``"tc."``.
    """
    cmd = _cmd_expr(alias)
    return (
        f"CASE WHEN array_length(string_split({cmd}, ' ')) >= 2"
        f" THEN split_part({cmd}, ' ', 1)"
        f" || ' ' || split_part({cmd}, ' ', 2)"
        f" ELSE {cmd} END"
    )


def _base_where(alias: str = "") -> tuple[str, ...]:
    """Base WHERE clauses for bash tool calls.

    *alias* should include the trailing dot, e.g. ``"tc."``.
    """
    return (
        f"{alias}tool_name = 'Bash'",
        f"{_cmd_expr(alias)} IS NOT NULL",
    )


async def bash(  # noqa: PLR0913
    request: Request,
    prefix: str,
    session: str,
    project: str,
    q: str,
    failed: bool,
    page: int = 1,
) -> HTMLResponse:
    """Bash command analytics with filtering."""
    db = conn(request)
    session = session.strip()
    prefix = prefix.strip()
    project = project.strip()
    q = q.strip()

    where_clauses = list(_base_where("tc."))
    params: list[str | int] = []
    if failed:
        where_clauses.append("tc.is_error = 'true'")
    if prefix:
        where_clauses.append(f"({_prefix_expr('tc.')}) = ?")
        params.append(prefix)
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
            f" ILIKE ? OR {_cmd_expr('tc.')} ILIKE ?)"
        )
        params.extend([escaped, escaped])

    where = "WHERE " + " AND ".join(where_clauses)
    joins = (
        "LEFT JOIN session_titles fp USING (session_id)"
        " LEFT JOIN logical_sessions ls"
        " ON tc.session_id = ls.session_id"
    )

    # Stats summary
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

    # Paginated rows
    rows = db.execute(
        f"""
        SELECT
            tc.session_id,
            strftime(tc.called_at, '%b %d %H:%M') AS called_at_fmt,
            {_cmd_expr("tc.")} AS command,
            tc.is_error,
            json_extract_string(tc.tool_input, '$.description') AS description,
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

    # Prefix buttons: command grouping by first 2 words
    pfx_where = list(_base_where())
    pfx_params: list[str] = []
    if session:
        pfx_where.append("session_id = ?")
        pfx_params.append(session)

    prefixes = db.execute(
        f"""
        SELECT
            {_prefix_expr()} AS prefix,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM tool_calls
        WHERE {" AND ".join(pfx_where)}
        GROUP BY prefix ORDER BY cnt DESC LIMIT {_MAX_PREFIXES}
    """,  # noqa: S608
        pfx_params,
    ).fetchall()

    return templates.TemplateResponse(
        request,
        "bash.html",
        {
            "parent": parent(request),
            "bash_calls": rows,
            "prefixes": prefixes,
            "filter_failed": failed,
            "filter_prefix": prefix,
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
