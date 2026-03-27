"""DuckDB database initialization and view management."""

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path.home() / ".introspect" / "introspect.duckdb"
DEFAULT_JSONL_GLOB = str(Path.home() / ".claude" / "projects" / "**" / "*.jsonl")


def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with views created."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    _create_views(conn, jsonl_glob)
    return conn


def _create_views(conn: duckdb.DuckDBPyConnection, jsonl_glob: str) -> None:
    """Create lazy views over JSONL files."""
    # Raw messages: all JSONL lines with parsed fields
    raw_messages_sql = f"""
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
    """  # nosec B608
    conn.execute(raw_messages_sql)

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
