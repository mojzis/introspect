"""DuckDB database initialization and view management."""

import contextlib
from pathlib import Path

import duckdb

from introspect.projects import resolve_project_map

DEFAULT_DB_PATH = Path.home() / ".introspect" / "introspect.duckdb"
DEFAULT_JSONL_GLOB = str(Path.home() / ".claude" / "projects" / "**" / "*.jsonl")

_READ_JSON_OPTS = (
    "filename=true, format='newline_delimited', union_by_name=true, ignore_errors=true"
)

_RAW_MESSAGES_COLUMNS = """
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
"""


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
        except duckdb.Error:
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
    *,
    resolve_projects: bool = True,
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
        "session_titles",
        "conversation_turns",
        "tool_calls",
        "logical_sessions",
        "project_map",
        "raw_messages",
        "raw_data",
        "search_corpus",
    ):
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP VIEW IF EXISTS {name}")  # nosec B608
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP TABLE IF EXISTS {name}")  # nosec B608

    _read = f"read_json_auto('{jsonl_glob}', {_READ_JSON_OPTS})"

    conn.execute(f"""
        CREATE TABLE raw_data AS
        SELECT * FROM {_read}
        {day_filter}
    """)  # nosec B608

    conn.execute(f"""
        CREATE TABLE raw_messages AS
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {_read}
        WHERE type IN ('user', 'assistant')
        {and_day_filter}
    """)  # nosec B608

    # Add indexes for common query patterns
    conn.execute("CREATE INDEX idx_rm_session ON raw_messages(session_id)")
    conn.execute("CREATE INDEX idx_rm_type ON raw_messages(type)")
    conn.execute("CREATE INDEX idx_rm_timestamp ON raw_messages(timestamp)")

    _build_project_map(conn, resolve_projects=resolve_projects)
    _create_derived_views(conn)


def _build_project_map(
    conn: duckdb.DuckDBPyConnection, *, resolve_projects: bool = True
) -> None:
    """Build the project_map table mapping cwd → canonical project."""
    conn.execute("""
        CREATE TABLE project_map (
            cwd VARCHAR PRIMARY KEY,
            canonical_path VARCHAR NOT NULL,
            project_name VARCHAR NOT NULL
        )
    """)

    rows = conn.execute(
        "SELECT DISTINCT cwd FROM raw_messages WHERE cwd IS NOT NULL"
    ).fetchall()
    cwds = [r[0] for r in rows]
    if not cwds:
        return

    mapping = (
        resolve_project_map(cwds)
        if resolve_projects
        else dict(zip(cwds, cwds, strict=True))
    )

    rows = [
        (cwd, canonical, canonical.rstrip("/").rsplit("/", 1)[-1])
        for cwd, canonical in mapping.items()
    ]
    conn.executemany("INSERT INTO project_map VALUES (?, ?, ?)", rows)


def _create_views(conn: duckdb.DuckDBPyConnection, jsonl_glob: str) -> None:
    """Create lazy views over JSONL files."""
    _read = f"read_json_auto('{jsonl_glob}', {_READ_JSON_OPTS})"

    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_data AS
        SELECT * FROM {_read}
    """)  # nosec B608

    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_messages AS
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {_read}
        WHERE type IN ('user', 'assistant')
    """)  # nosec B608

    # Empty project_map so the JOIN in logical_sessions works in lazy mode
    conn.execute("""
        CREATE TABLE IF NOT EXISTS project_map (
            cwd VARCHAR PRIMARY KEY,
            canonical_path VARCHAR NOT NULL,
            project_name VARCHAR NOT NULL
        )
    """)

    _create_derived_views(conn)


def _create_derived_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create views that depend on raw_messages (works with both tables and views)."""
    # Logical sessions: one row per session with summary stats
    conn.execute("""
        CREATE OR REPLACE VIEW logical_sessions AS
        SELECT
            rm.session_id,
            MIN(rm.timestamp) AS started_at,
            MAX(rm.timestamp) AS ended_at,
            age(MAX(rm.timestamp), MIN(rm.timestamp)) AS duration,
            COUNT(*) FILTER (
                WHERE rm.type = 'user'
                AND rm.role = 'user'
                AND json_extract_string(
                    rm.message, '$.content[0].type'
                ) IS DISTINCT FROM 'tool_result'
            ) AS user_messages,
            COUNT(*) FILTER (WHERE rm.type = 'assistant') AS assistant_messages,
            ANY_VALUE(rm.model) AS model,
            ANY_VALUE(rm.cwd) AS cwd,
            COALESCE(
                ANY_VALUE(pm.project_name),
                split_part(rtrim(ANY_VALUE(rm.cwd), '/'), '/', -1)
            ) AS project,
            ANY_VALUE(rm.git_branch) AS git_branch,
            ANY_VALUE(rm.entrypoint) AS entrypoint,
        FROM raw_messages rm
        LEFT JOIN project_map pm ON rm.cwd = pm.cwd
        GROUP BY rm.session_id
    """)

    # Tool calls: assistant tool_use content blocks joined with results.
    # Unnests all content blocks so multi-tool messages are captured.
    conn.execute("""
        CREATE OR REPLACE VIEW tool_calls AS
        WITH uses AS (
            SELECT
                m.session_id,
                m.timestamp AS called_at,
                m.uuid AS assistant_uuid,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].name'
                ) AS tool_name,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].id'
                ) AS tool_use_id,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].input'
                ) AS tool_input,
            FROM raw_messages m,
                 generate_series(
                     0,
                     CAST(json_array_length(
                         json_extract(m.message, '$.content')
                     ) - 1 AS BIGINT)
                 ) AS i(idx)
            WHERE m.type = 'assistant'
              AND json_array_length(json_extract(m.message, '$.content')) > 0
              AND json_extract_string(
                  m.message, '$.content[' || i.idx || '].type'
              ) = 'tool_use'
        ),
        results AS (
            SELECT
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].tool_use_id'
                ) AS tool_use_id,
                json_extract(
                    m.message, '$.content[' || i.idx || '].is_error'
                ) AS is_error,
                m.tool_use_result,
                m.timestamp AS result_at,
            FROM raw_messages m,
                 generate_series(
                     0,
                     CAST(json_array_length(
                         json_extract(m.message, '$.content')
                     ) - 1 AS BIGINT)
                 ) AS i(idx)
            WHERE m.type = 'user'
              AND json_array_length(json_extract(m.message, '$.content')) > 0
              AND json_extract_string(
                  m.message, '$.content[' || i.idx || '].type'
              ) = 'tool_result'
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

    # Session titles: first meaningful user prompt per session
    conn.execute("""
        CREATE OR REPLACE VIEW session_titles AS
        SELECT session_id, first_prompt FROM (
            SELECT
                session_id,
                COALESCE(
                    json_extract_string(message, '$.content[0].text'),
                    json_extract_string(message, '$.content')
                ) AS first_prompt,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id ORDER BY timestamp
                ) AS rn
            FROM raw_messages
            WHERE type = 'user' AND role = 'user'
              AND json_extract_string(
                  message, '$.content[0].type'
              ) IS DISTINCT FROM 'tool_result'
              AND COALESCE(
                  json_extract_string(message, '$.content[0].text'),
                  json_extract_string(message, '$.content'),
                  ''
              ) NOT LIKE '/clear%'
              AND COALESCE(
                  json_extract_string(message, '$.content[0].text'),
                  json_extract_string(message, '$.content'),
                  ''
              ) NOT LIKE '<command-name>/clear%'
              AND COALESCE(
                  json_extract_string(message, '$.content[0].text'),
                  json_extract_string(message, '$.content'),
                  ''
              ) NOT LIKE '<local-command-caveat>%'
        ) sub WHERE rn = 1
    """)
