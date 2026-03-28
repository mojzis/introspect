"""DuckDB database initialization and view management."""

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path.home() / ".introspect" / "introspect.duckdb"
DEFAULT_JSONL_GLOB = str(Path.home() / ".claude" / "projects" / "**" / "*.jsonl")


def get_read_connection(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
) -> duckdb.DuckDBPyConnection:
    """Open materialized DB read-only, falling back to lazy views."""
    if db_path.exists():
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'raw_messages' AND table_type = 'BASE TABLE'"
            ).fetchall()
            if tables:
                return conn
            conn.close()
        except Exception:  # nosec B110
            pass
    return get_connection(db_path, jsonl_glob)


def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with views created."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    _create_views(conn, jsonl_glob)
    return conn


def materialize_views(
    conn: duckdb.DuckDBPyConnection,
    jsonl_glob: str,
    days: int = 0,
) -> None:
    """Materialize raw data into tables for fast querying.

    Args:
        conn: DuckDB connection.
        jsonl_glob: Glob pattern for JSONL files.
        days: Number of days of history to load. 0 means no limit.
    """
    day_filter = ""
    and_day_filter = ""
    if days > 0:
        day_filter = (
            f"WHERE timestamp::TIMESTAMP >= CURRENT_TIMESTAMP - INTERVAL '{days} days'"
        )
        and_day_filter = (
            f"AND timestamp::TIMESTAMP >= CURRENT_TIMESTAMP - INTERVAL '{days} days'"
        )

    # Drop everything to avoid table/view name conflicts
    for name in (
        "sessions",
        "conversation_turns",
        "tool_calls",
        "logical_sessions",
        "raw_messages",
        "raw_data",
        "search_corpus",
    ):
        conn.execute(f"DROP VIEW IF EXISTS {name}")  # nosec B608
        conn.execute(f"DROP TABLE IF EXISTS {name}")  # nosec B608

    conn.execute(f"""
        CREATE TABLE raw_data AS
        SELECT *
        FROM read_json_auto(
            '{jsonl_glob}',
            filename=true,
            format='newline_delimited',
            union_by_name=true,
            ignore_errors=true
        )
        {day_filter}
    """)  # nosec B608

    conn.execute(f"""
        CREATE TABLE raw_messages AS
        SELECT
            filename AS file_path,
            type,
            timestamp::TIMESTAMP AS timestamp,
            sessionId AS session_id,
            uuid,
            parentUuid AS parent_uuid,
            isSidechain AS is_sidechain,
            cwd,
            version,
            entrypoint,
            gitBranch AS git_branch,
            json_extract_string(message, '$.role') AS role,
            json_extract_string(message, '$.model') AS model,
            message,
            toolUseResult AS tool_use_result,
        FROM read_json_auto(
            '{jsonl_glob}',
            filename=true,
            format='newline_delimited',
            union_by_name=true,
            ignore_errors=true
        )
        WHERE type IN ('user', 'assistant')
        {and_day_filter}
    """)  # nosec B608

    _create_derived_views(conn)


def _create_views(conn: duckdb.DuckDBPyConnection, jsonl_glob: str) -> None:
    """Create lazy views over JSONL files."""
    # Raw data: completely unfiltered JSONL — every field, every row
    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_data AS
        SELECT *
        FROM read_json_auto(
            '{jsonl_glob}',
            filename=true,
            format='newline_delimited',
            union_by_name=true,
            ignore_errors=true
        )
    """)  # nosec B608

    # Raw messages: all JSONL lines with parsed fields
    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_messages AS
        SELECT
            filename AS file_path,
            type,
            timestamp::TIMESTAMP AS timestamp,
            sessionId AS session_id,
            uuid,
            parentUuid AS parent_uuid,
            isSidechain AS is_sidechain,
            cwd,
            version,
            entrypoint,
            gitBranch AS git_branch,
            json_extract_string(message, '$.role') AS role,
            json_extract_string(message, '$.model') AS model,
            message,
            toolUseResult AS tool_use_result,
        FROM read_json_auto(
            '{jsonl_glob}',
            filename=true,
            format='newline_delimited',
            union_by_name=true,
            ignore_errors=true
        )
        WHERE type IN ('user', 'assistant')
    """)  # nosec B608

    _create_derived_views(conn)


def _create_derived_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create views that depend on raw_messages (works with both tables and views)."""
    # Logical sessions: one row per session with summary stats
    conn.execute("""
        CREATE OR REPLACE VIEW logical_sessions AS
        SELECT
            session_id,
            MIN(timestamp) AS started_at,
            MAX(timestamp) AS ended_at,
            age(MAX(timestamp), MIN(timestamp)) AS duration,
            COUNT(*) FILTER (
                WHERE type = 'user'
                AND role = 'user'
                AND json_extract_string(
                    message, '$.content[0].type'
                ) IS DISTINCT FROM 'tool_result'
            ) AS user_messages,
            COUNT(*) FILTER (WHERE type = 'assistant') AS assistant_messages,
            ANY_VALUE(model) AS model,
            ANY_VALUE(cwd) AS cwd,
            ANY_VALUE(git_branch) AS git_branch,
            ANY_VALUE(entrypoint) AS entrypoint,
        FROM raw_messages
        GROUP BY session_id
    """)

    # Tool calls: assistant tool_use content blocks joined with results
    conn.execute("""
        CREATE OR REPLACE VIEW tool_calls AS
        WITH uses AS (
            SELECT
                session_id,
                timestamp AS called_at,
                uuid AS assistant_uuid,
                json_extract_string(message, '$.content[0].type') AS content_type,
                json_extract_string(message, '$.content[0].name') AS tool_name,
                json_extract_string(message, '$.content[0].id') AS tool_use_id,
                json_extract_string(message, '$.content[0].input') AS tool_input,
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract_string(message, '$.content[0].type') = 'tool_use'
        ),
        results AS (
            SELECT
                json_extract_string(message, '$.content[0].tool_use_id') AS tool_use_id,
                json_extract(message, '$.content[0].is_error') AS is_error,
                tool_use_result,
                timestamp AS result_at,
            FROM raw_messages
            WHERE type = 'user'
              AND json_extract_string(message, '$.content[0].type') = 'tool_result'
        )
        SELECT
            u.session_id,
            u.called_at,
            u.tool_name,
            u.tool_use_id,
            u.tool_input,
            r.is_error,
            r.tool_use_result,
            r.result_at,
            age(r.result_at, u.called_at) AS execution_time,
        FROM uses u
        LEFT JOIN results r ON u.tool_use_id = r.tool_use_id
    """)

    # Conversation turns: human/assistant pairs
    conn.execute("""
        CREATE OR REPLACE VIEW conversation_turns AS
        WITH ordered AS (
            SELECT
                session_id,
                timestamp,
                type,
                role,
                uuid,
                parent_uuid,
                message,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id ORDER BY timestamp
                ) AS turn_order,
            FROM raw_messages
            WHERE (type = 'user' AND role = 'user')
               OR (type = 'assistant'
                   AND json_extract_string(message, '$.content[0].type') = 'text')
        )
        SELECT
            session_id,
            timestamp,
            type,
            role,
            uuid,
            turn_order,
            CASE
                WHEN type = 'user'
                THEN json_extract_string(message, '$.content')
                ELSE json_extract_string(message, '$.content[0].text')
            END AS content_text,
        FROM ordered
    """)

    # Convenience alias: "sessions" → logical_sessions
    conn.execute("""
        CREATE OR REPLACE VIEW sessions AS
        SELECT * FROM logical_sessions
    """)
