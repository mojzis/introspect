"""MCP tool definitions for introspect."""

from __future__ import annotations

import re
from datetime import datetime

import duckdb

from introspect.db import DEFAULT_DB_PATH, get_read_connection
from introspect.search import ensure_search_corpus, fts_search

_VALID_ROLES = {"user", "assistant"}

# Max characters per cell in run_sql output; long values (e.g. tool_input JSON
# blobs) are truncated so one wide row doesn't blow up the response.
_SQL_CELL_MAX = 200
# Hard cap on run_sql rows regardless of caller's `limit` argument.
_SQL_ROW_CAP = 500


def search_conversations(
    query: str,
    limit: int = 10,
    offset: int = 0,
    cwd_prefix: str = "",
    role: str = "",
    since: str = "",
    session_id: str = "",
    require_all: bool = False,
) -> str:
    """Full-text search across conversation logs (BM25 when available).

    Returns session summaries with context-windowed snippets.

    Optional filters (empty string disables each filter — FastMCP exposes
    these as plain strings rather than Optional for tool-schema simplicity):
      - ``cwd_prefix``: match sessions whose working directory starts with this
        string (e.g. ``/home/matous/git/logogame`` or just ``/home/matous/git``).
      - ``role``: ``'user'`` or ``'assistant'``.
      - ``since``: ISO date/timestamp (e.g. ``'2026-04-01'``); matches messages
        at or after this point.
      - ``session_id``: restrict to a single session.
      - ``require_all``: multi-word queries must match ALL terms (AND mode).
      - ``offset``: skip N results — use with ``limit`` for pagination.
    """
    if role and role not in _VALID_ROLES:
        return f"Error: role must be one of {sorted(_VALID_ROLES)} (got {role!r})."
    if since:
        try:
            datetime.fromisoformat(since)
        except ValueError as exc:
            return f"Error: invalid 'since' (expected ISO date/timestamp): {exc}"

    conn = get_read_connection()
    try:
        ensure_search_corpus(conn)

        results = fts_search(
            conn,
            query,
            limit=limit,
            offset=offset,
            cwd_prefix=cwd_prefix or None,
            role=role or None,
            since=since or None,
            session_id=session_id or None,
            require_all=require_all,
        )
        if not results:
            return "No results found."

        lines: list[str] = []
        for sid, timestamp, msg_role, cwd, snippet, score in results:
            ts = str(timestamp)[:19] if timestamp else "?"
            score_str = f"{score:.4f}" if score is not None else "?"
            lines.append(
                f"[{ts}] session={sid} role={msg_role} "
                f"cwd={cwd or '?'} score={score_str}\n  {snippet}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


def get_session(session_id: str) -> str:
    """Get full session content by session ID.

    Returns all messages as structured data.
    """
    conn = get_read_connection()
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


def recent_sessions(n: int = 10) -> str:
    """List the most recent N sessions with metadata."""
    conn = get_read_connection()
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


_SQL_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_COMMENT_LINE = re.compile(r"--[^\n]*")
# Only single-quoted strings are SQL literals; double-quoted tokens are
# identifiers and must not be rewritten by the validator.
_SQL_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_SQL_ALLOWED_FIRST_KEYWORDS = {"select", "with"}


def _validate_read_only_sql(sql: str) -> str | None:
    """Return an error message if `sql` is not a safe read-only query.

    This is the PRIMARY guard — do not weaken it assuming the connection is
    read-only. ``run_sql`` opens a fresh ``read_only=True`` connection as a
    defense-in-depth backstop, but even that permits some side-effecting
    statements (e.g. ``COPY ... TO '/file'`` can write outside the DB).
    The "first keyword must be SELECT or WITH" check blocks ATTACH, INSTALL,
    LOAD, PRAGMA, COPY, INSERT, UPDATE, DELETE, DROP, CREATE, CALL, etc.
    """
    stripped = _SQL_COMMENT_BLOCK.sub(" ", sql)
    stripped = _SQL_COMMENT_LINE.sub(" ", stripped)
    # Replace string-literal contents before scanning so a `;` or keyword
    # inside a literal doesn't trip the multi-statement / first-keyword
    # checks. Double-quoted identifiers are intentionally preserved.
    scan = _SQL_STRING_LITERAL.sub("''", stripped)
    scan = scan.strip().rstrip(";").strip()
    if not scan:
        return "SQL is empty."
    if ";" in scan:
        return "Multiple statements are not allowed."
    first_word = scan.split(None, 1)[0].lower()
    if first_word not in _SQL_ALLOWED_FIRST_KEYWORDS:
        return f"Only SELECT / WITH queries are allowed (got: {first_word!r})."
    return None


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    """Format a result set as an aligned text table with truncated cells."""
    if not rows:
        return f"(0 rows)\ncolumns: {', '.join(columns)}"

    def cell(value: object) -> str:
        text = "NULL" if value is None else str(value)
        text = text.replace("\n", " ").replace("\r", " ")
        if len(text) > _SQL_CELL_MAX:
            text = text[: _SQL_CELL_MAX - 1] + "…"
        return text

    str_rows = [[cell(v) for v in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    header = render(columns)
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(render(r) for r in str_rows)
    return f"{header}\n{sep}\n{body}\n({len(rows)} rows)"


def run_sql(sql: str, limit: int = 100) -> str:
    """Execute a read-only SELECT / WITH query against the introspect DB.

    Only single SELECT or WITH statements are permitted; write operations,
    ATTACH, PRAGMA, INSTALL, LOAD, COPY, and multi-statement scripts are
    rejected. Use `describe_schema` to discover available views and columns.

    Results are capped at `limit` rows (max 500). The cap is pushed into
    the query planner as an outer LIMIT so DuckDB doesn't materialize more
    than the cap before the tool fetches rows. Long cell values are
    truncated. Returns an aligned text table.
    """
    error = _validate_read_only_sql(sql)
    if error:
        return f"Error: {error}"

    capped_limit = max(1, min(limit, _SQL_ROW_CAP))

    # Fresh strict read-only connection — do NOT route through
    # get_read_connection(), which silently falls back to a writable
    # connection when the materialized DB file is missing.
    if not DEFAULT_DB_PATH.exists():
        return (
            f"Error: materialized DB not found at {DEFAULT_DB_PATH}. "
            "Start `introspect serve` once to materialize views."
        )
    try:
        conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    except duckdb.Error as exc:
        return f"Error opening DB ({type(exc).__name__}): {exc}"

    # Wrap the user query so the row cap is applied by the planner, not
    # just by fetchmany. Safe because the inner SQL has already passed the
    # read-only validator and `capped_limit` is a clamped int.
    inner = sql.strip().rstrip(";").strip()
    wrapped = f"SELECT * FROM ({inner}) AS _introspect_q LIMIT {capped_limit}"

    try:
        try:
            cursor = conn.execute(wrapped)
        except duckdb.Error as exc:
            return f"SQL error ({type(exc).__name__}): {exc}"
        columns = [d[0] for d in (cursor.description or [])]
        rows = cursor.fetchall()
        return _format_rows(columns, rows)
    finally:
        conn.close()


def describe_schema() -> str:
    """List views/tables available to `run_sql` with their columns.

    Returns a compact listing grouped by table, pulled from the attached
    DuckDB's information_schema. Use this before writing a `run_sql` query
    to discover column names and types.
    """
    conn = get_read_connection()
    try:
        rows = conn.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No tables or views found."

    by_table: dict[str, list[str]] = {}
    for table_name, column_name, data_type in rows:
        by_table.setdefault(table_name, []).append(f"{column_name} {data_type}")

    # Surface the views a caller most often wants first; everything else
    # follows alphabetically.
    priority = [
        "logical_sessions",
        "tool_calls",
        "conversation_turns",
        "session_titles",
        "search_corpus",
    ]
    ordered: list[str] = [name for name in priority if name in by_table]
    ordered.extend(name for name in sorted(by_table) if name not in ordered)

    lines: list[str] = []
    for table_name in ordered:
        lines.append(f"{table_name}:")
        for col in by_table[table_name]:
            lines.append(f"  {col}")
        lines.append("")
    return "\n".join(lines).rstrip()


def tool_failures(command_prefix: str = "", limit: int = 20) -> str:
    """List failed tool calls, optionally filtered by tool name prefix."""
    conn = get_read_connection()
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
