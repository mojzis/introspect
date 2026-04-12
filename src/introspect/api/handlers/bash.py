"""Bash commands route handler."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import DEFAULT_PAGE_SIZE, clean_title, conn, parent, templates


async def bash(
    request: Request, prefix: str, session: str, failed: bool, page: int = 1
) -> HTMLResponse:
    """Bash command analytics with filtering by prefix, session, and status."""
    db = conn(request)
    session = session.strip()
    prefix = prefix.strip()

    where_clauses = [
        "tool_name = 'Bash'",
        "json_extract_string(tool_input, '$.command') IS NOT NULL",
    ]
    params: list[str | int] = []
    if failed:
        where_clauses.append("is_error = 'true'")
    if prefix:
        where_clauses.append(
            "(CASE WHEN array_length(string_split("
            "trim(json_extract_string(tool_input, '$.command')), ' ')) >= 2"
            " THEN split_part(trim(json_extract_string(tool_input, '$.command')),"
            " ' ', 1) || ' ' || split_part(trim(json_extract_string(tool_input,"
            " '$.command')), ' ', 2)"
            " ELSE trim(json_extract_string(tool_input, '$.command'))"
            " END) = ?"
        )
        params.append(prefix)
    if session:
        where_clauses.append("session_id = ?")
        params.append(session)

    where = "WHERE " + " AND ".join(where_clauses)

    # Stats summary
    stats = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_total
        FROM tool_calls
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
            tc.called_at,
            json_extract_string(tc.tool_input, '$.command') AS command,
            tc.is_error,
            json_extract_string(tc.tool_input, '$.description') AS description,
            tc.execution_time,
            fp.first_prompt
        FROM tool_calls tc
        LEFT JOIN session_titles fp USING (session_id)
        {where}
        ORDER BY tc.called_at DESC
        LIMIT ? OFFSET ?
    """,  # noqa: S608
        [*params, DEFAULT_PAGE_SIZE, offset],
    ).fetchall()

    # Prefix buttons: command grouping by first 2 words
    pfx_where = [
        "tool_name = 'Bash'",
        "json_extract_string(tool_input, '$.command') IS NOT NULL",
    ]
    pfx_params: list[str] = []
    if session:
        pfx_where.append("session_id = ?")
        pfx_params.append(session)

    prefixes = db.execute(
        f"""
        WITH cmds AS (
            SELECT trim(json_extract_string(tool_input, '$.command')) AS cmd,
                   is_error
            FROM tool_calls
            WHERE {" AND ".join(pfx_where)}
        )
        SELECT
            CASE WHEN array_length(string_split(cmd, ' ')) >= 2
                 THEN split_part(cmd, ' ', 1) || ' ' || split_part(cmd, ' ', 2)
                 ELSE cmd
            END AS prefix,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate
        FROM cmds
        GROUP BY prefix ORDER BY cnt DESC LIMIT 40
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
            "stats": stats,
            "page": page,
            "total_pages": total_pages,
            "clean_title": clean_title,
        },
    )
