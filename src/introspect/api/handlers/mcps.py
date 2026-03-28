"""MCP tools route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import conn, parent, templates


async def mcps(
    request: Request,
    server: str,
    command: str,
    failed: bool,
) -> HTMLResponse:
    """MCP tool analysis with server/command breakdown."""
    db = conn(request)
    server_name = server.strip()
    command_name = command.strip()

    # --- Server overview ---
    mcp_servers = db.execute("""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            COUNT(*) AS cnt,
            COUNT(DISTINCT split_part(tool_name, '__', 3)) AS command_count,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY server_name
        ORDER BY cnt DESC
    """).fetchall()

    # --- Commands for selected server (or all) ---
    cmd_where = ["tool_name LIKE 'mcp__%'"]
    mcp_params: list[str | int] = []
    if server_name:
        cmd_where.append("split_part(tool_name, '__', 2) = ?")
        mcp_params.append(server_name)

    mcp_commands = db.execute(
        f"""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            split_part(tool_name, '__', 3) AS command_name,
            COUNT(*) AS cnt,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
        FROM tool_calls
        WHERE {" AND ".join(cmd_where)}
        GROUP BY server_name, command_name
        ORDER BY cnt DESC
    """,  # nosec B608
        mcp_params,
    ).fetchall()

    # --- Filtered call list ---
    list_where = ["tool_name LIKE 'mcp__%'"]
    list_params: list[str | int] = []
    if server_name:
        list_where.append("split_part(tc.tool_name, '__', 2) = ?")
        list_params.append(server_name)
    if command_name:
        list_where.append("split_part(tc.tool_name, '__', 3) = ?")
        list_params.append(command_name)
    if failed:
        list_where.append("tc.is_error = 'true'")

    rows = db.execute(
        f"""
        SELECT
            tc.session_id,
            tc.called_at,
            split_part(tc.tool_name, '__', 2) AS server_name,
            split_part(tc.tool_name, '__', 3) AS command_name,
            tc.is_error,
            LEFT(tc.tool_input, 200) AS input_preview,
            tc.execution_time,
            fp.first_prompt
        FROM tool_calls tc
        LEFT JOIN session_titles fp ON tc.session_id = fp.session_id
        WHERE {" AND ".join(list_where)}
        ORDER BY tc.called_at DESC
        LIMIT 100
    """,  # nosec B608
        list_params,
    ).fetchall()

    mcp_stats = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_total
        FROM tool_calls tc
        WHERE {" AND ".join(list_where)}
    """,  # nosec B608
        list_params,
    ).fetchone()

    return templates.TemplateResponse(
        request,
        "mcps.html",
        {
            "parent": parent(request),
            "mcp_servers": mcp_servers,
            "mcp_commands": mcp_commands,
            "tool_calls": rows,
            "filter_server": server,
            "filter_command": command,
            "filter_failed": failed,
            "stats": mcp_stats,
        },
    )
