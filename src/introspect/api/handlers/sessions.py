"""Session-related route handlers."""

import json
import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import (
    COMMAND_LIST_SUBQUERY,
    SESSIONS_PAGE_SIZES,
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_COLS,
    SESSIONS_SORT_DEFAULT,
    TOOL_COUNTS_SUBQUERY,
    clean_title,
    conn,
    fetch_token_usage,
    parent,
    parse_content_block,
    templates,
)


async def sessions(  # noqa: PLR0913
    request: Request,
    page: int,
    page_size: int,
    sort: str,
    order: str,
    model: str,
    project: str,
    branch: str,
    command: str,
) -> HTMLResponse:
    """Paginated session list with filtering and sorting."""
    db = conn(request)

    # Clamp page_size to allowed values
    if page_size not in SESSIONS_PAGE_SIZES:
        page_size = SESSIONS_PER_PAGE_DEFAULT

    # Build WHERE clause from filters
    where_clauses: list[str] = []
    params: list[str | int] = []
    if model.strip():
        where_clauses.append("ls.model = ?")
        params.append(model.strip())
    if project.strip():
        where_clauses.append("ls.cwd LIKE ?")
        params.append(f"%/{project.strip()}")
    if branch.strip():
        where_clauses.append("ls.git_branch = ?")
        params.append(branch.strip())
    if command.strip():
        where_clauses.append(
            "EXISTS (SELECT 1 FROM message_commands mc"
            " WHERE mc.session_id = ls.session_id AND mc.command = ?)"
        )
        params.append(command.strip())

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Count with filters
    total = db.execute(
        f"SELECT COUNT(*) FROM logical_sessions ls {where}",  # nosec B608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    # Resolve sort column
    default_col = SESSIONS_SORT_COLS[SESSIONS_SORT_DEFAULT]
    sort_col = SESSIONS_SORT_COLS.get(sort, default_col)
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"
    nulls = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    rows = db.execute(
        f"""
        SELECT
            ls.session_id,
            ls.started_at,
            ls.ended_at,
            ls.duration,
            ls.user_messages,
            ls.assistant_messages,
            ls.model,
            ls.cwd,
            ls.git_branch,
            fp.first_prompt,
            COALESCE(tc.tool_count, 0) AS tool_count,
            cmd.commands
        FROM logical_sessions ls
        LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
        LEFT JOIN {TOOL_COUNTS_SUBQUERY} ON ls.session_id = tc.session_id
        LEFT JOIN {COMMAND_LIST_SUBQUERY} ON ls.session_id = cmd.session_id
        {where}
        ORDER BY {sort_col} {sort_dir} {nulls}
        LIMIT ? OFFSET ?
    """,  # nosec B608
        [*params, page_size, offset],
    ).fetchall()

    session_list = []
    for row in rows:
        (
            session_id,
            started_at,
            ended_at,
            duration,
            user_msgs,
            asst_msgs,
            _model,
            cwd,
            git_branch,
            first_prompt,
            tool_count,
            commands,
        ) = row
        dur_str = ""
        if duration:
            total_secs = int(duration.total_seconds())
            dur_str = f"{total_secs // 60}:{total_secs % 60:02d}"
        proj = ""
        if cwd:
            proj = cwd.rstrip("/").rsplit("/", 1)[-1]
        session_list.append(
            {
                "id": session_id,
                "date": str(started_at)[5:10] if started_at else "",
                "start_time": str(started_at)[11:16] if started_at else "",
                "end_time": str(ended_at)[11:16] if ended_at else "",
                "duration": dur_str,
                "user_msgs": user_msgs or 0,
                "asst_msgs": asst_msgs or 0,
                "model": _model or "",
                "project": proj,
                "branch": git_branch or "",
                "title": clean_title(first_prompt or "")[:120],
                "tool_count": tool_count or 0,
                "commands": commands or "",
            }
        )

    # Get distinct values for filter dropdowns
    models = db.execute("""
        SELECT DISTINCT model FROM logical_sessions
        WHERE model IS NOT NULL ORDER BY model
    """).fetchall()
    projects = db.execute("""
        SELECT DISTINCT split_part(rtrim(cwd, '/'), '/', -1) AS proj
        FROM logical_sessions
        WHERE cwd IS NOT NULL
        ORDER BY proj
    """).fetchall()
    branches = db.execute("""
        SELECT DISTINCT git_branch FROM logical_sessions
        WHERE git_branch IS NOT NULL ORDER BY git_branch
    """).fetchall()
    commands_list = db.execute("""
        SELECT DISTINCT command FROM message_commands
        ORDER BY command
    """).fetchall()

    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "parent": parent(request),
            "sessions": session_list,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_size": page_size,
            "page_sizes": SESSIONS_PAGE_SIZES,
            "sort": sort,
            "order": order.lower(),
            "filter_model": model,
            "filter_project": project,
            "filter_branch": branch,
            "filter_command": command,
            "models": [r[0] for r in models],
            "projects": [r[0] for r in projects],
            "branches": [r[0] for r in branches],
            "commands_list": [r[0] for r in commands_list],
        },
    )


async def session_detail(request: Request, session_id: str) -> HTMLResponse:
    """Full session detail with all messages."""
    db = conn(request)

    session_info = db.execute(
        """
        SELECT session_id, started_at, ended_at, duration,
               user_messages, assistant_messages, model, cwd, git_branch
        FROM logical_sessions
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    token_usage = fetch_token_usage(db, session_id=session_id, include_cache=True)

    # Tool call summary
    tool_summary = db.execute(
        """
        SELECT
            COUNT(*) AS total_calls,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_calls,
            MODE(tool_name) AS most_used_tool
        FROM tool_calls
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    messages = db.execute(
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
            content_blocks = [parse_content_block(block) for block in raw_content]

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
        request,
        "session_detail.html",
        {
            "parent": parent(request),
            "session": session_info,
            "session_id": session_id,
            "messages": parsed_messages,
            "token_usage": token_usage,
            "tool_summary": tool_summary,
        },
    )
