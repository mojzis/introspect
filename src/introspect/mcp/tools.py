"""MCP tool definitions for introspect."""

from __future__ import annotations

from introspect.db import get_connection
from introspect.mcp.server import mcp
from introspect.search import build_search_corpus, fts_search


@mcp.tool()
def search_conversations(query: str, limit: int = 10) -> str:
    """Full-text search across conversation logs.

    Returns session summaries with matching snippets.
    """
    conn = get_connection()
    try:
        # Ensure search corpus exists
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'search_corpus' AND table_type = 'BASE TABLE'
        """).fetchall()
        if not tables:
            build_search_corpus(conn)

        results = fts_search(conn, query, limit)
        if not results:
            return "No results found."

        lines: list[str] = []
        for session_id, timestamp, role, snippet, score in results:
            ts = str(timestamp)[:19] if timestamp else "?"
            score_str = f"{score:.4f}" if score is not None else "?"
            lines.append(
                f"[{ts}] session={session_id} role={role} "
                f"score={score_str}\n  {snippet}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def get_session(session_id: str) -> str:
    """Get full session content by session ID.

    Returns all messages as structured data.
    """
    conn = get_connection()
    try:
        # Session metadata
        meta = conn.execute(
            """
            SELECT
                session_id, started_at, ended_at, duration,
                user_messages, assistant_messages, model, cwd, git_branch
            FROM logical_sessions
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if not meta:
            return f"Session '{session_id}' not found."

        lines: list[str] = [
            f"Session: {meta[0]}",
            f"Started: {meta[1]}",
            f"Ended: {meta[2]}",
            f"Duration: {meta[3]}",
            f"User messages: {meta[4]}",
            f"Assistant messages: {meta[5]}",
            f"Model: {meta[6]}",
            f"CWD: {meta[7]}",
            f"Branch: {meta[8]}",
            "",
            "--- Messages ---",
        ]

        turns = conn.execute(
            """
            SELECT turn_order, type, content_text
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY turn_order
            """,
            [session_id],
        ).fetchall()

        for turn_order, msg_type, content in turns:
            label = "User" if msg_type == "user" else "Assistant"
            text = (content or "")[:500]
            lines.append(f"\n[{turn_order}] {label}:\n{text}")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def recent_sessions(n: int = 10) -> str:
    """List the most recent N sessions with metadata."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                session_id, started_at, ended_at, duration,
                user_messages, assistant_messages, model, cwd, git_branch
            FROM logical_sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [n],
        ).fetchall()

        if not rows:
            return "No sessions found."

        lines: list[str] = []
        for row in rows:
            started = str(row[1])[:19] if row[1] else "?"
            duration = str(row[3]) if row[3] else "?"
            lines.append(
                f"session={row[0]}\n"
                f"  started={started} duration={duration}\n"
                f"  user_msgs={row[4]} asst_msgs={row[5]}\n"
                f"  model={row[6]} cwd={row[7]} branch={row[8]}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def tool_failures(command_prefix: str = "", limit: int = 20) -> str:
    """List failed tool calls, optionally filtered by tool name prefix."""
    conn = get_connection()
    try:
        if command_prefix:
            rows = conn.execute(
                """
                SELECT
                    session_id, called_at, tool_name,
                    LEFT(tool_input, 200) AS input_preview,
                    LEFT(tool_use_result::VARCHAR, 200) AS result_preview,
                    execution_time
                FROM tool_calls
                WHERE is_error = 'true'
                  AND tool_name LIKE ? || '%'
                ORDER BY called_at DESC
                LIMIT ?
                """,
                [command_prefix, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    session_id, called_at, tool_name,
                    LEFT(tool_input, 200) AS input_preview,
                    LEFT(tool_use_result::VARCHAR, 200) AS result_preview,
                    execution_time
                FROM tool_calls
                WHERE is_error = 'true'
                ORDER BY called_at DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        if not rows:
            return "No failed tool calls found."

        lines: list[str] = []
        for row in rows:
            called = str(row[1])[:19] if row[1] else "?"
            lines.append(
                f"[{called}] session={row[0]} tool={row[2]} exec_time={row[5]}\n"
                f"  input: {row[3]}\n"
                f"  result: {row[4]}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()
