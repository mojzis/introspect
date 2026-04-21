"""DuckDB database initialization and view management."""

import contextlib
from pathlib import Path

import duckdb

from introspect.projects import resolve_project_map

DEFAULT_DB_PATH = Path.home() / ".introspect" / "introspect.duckdb"
DEFAULT_JSONL_GLOB = str(Path.home() / ".claude" / "projects" / "**" / "*.jsonl")


class DatabaseLockedError(duckdb.IOException):
    """Raised when the DuckDB database is locked by another process.

    Inherits from ``duckdb.IOException`` so existing ``except duckdb.IOException``
    handlers continue to catch it.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        super().__init__(
            f"Another Introspect process is using the database at {db_path}."
        )


# DuckDB's Python bindings do not expose a lock-specific exception class,
# so we classify by the error message. Both markers have been stable across
# DuckDB 0.9 through 1.x; if DuckDB renames them we'll fall back to the raw
# ``IOException`` (ugly but correct).
_LOCK_ERROR_MARKERS = ("Conflicting lock", "Could not set lock")


def _is_lock_error(exc: duckdb.IOException) -> bool:
    """Return True if a DuckDB IOException indicates a lock conflict."""
    msg = str(exc)
    return any(marker in msg for marker in _LOCK_ERROR_MARKERS)


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
    """Get a DuckDB connection with views created.

    Raises:
        DatabaseLockedError: if another process holds a write lock on the DB.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    _create_views(conn, jsonl_glob)
    return conn


def connect_writable(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a writable DuckDB connection, translating lock conflicts.

    Raises:
        DatabaseLockedError: if another process holds a write lock on the DB.
    """
    try:
        return duckdb.connect(str(db_path))
    except duckdb.IOException as e:
        if _is_lock_error(e):
            raise DatabaseLockedError(db_path) from e
        raise


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
        "message_commands",
        "session_titles",
        "conversation_turns",
        "session_messages_enriched",
        "assistant_message_costs",
        "file_reads",
        "file_writes",
        "tool_calls",
        "logical_sessions",
        "project_map",
        "raw_messages",
        "raw_data",
        "search_corpus",
    ):
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP TABLE IF EXISTS {name}")

    _read = f"read_json_auto('{jsonl_glob}', {_READ_JSON_OPTS})"

    conn.execute(f"""
        CREATE TABLE raw_data AS
        SELECT * FROM {_read}
        {day_filter}
    """)  # noqa: S608

    conn.execute(f"""
        CREATE TABLE raw_messages AS
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {_read}
        WHERE type IN ('user', 'assistant')
        {and_day_filter}
    """)  # noqa: S608

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
    """)  # noqa: S608

    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_messages AS
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {_read}
        WHERE type IN ('user', 'assistant')
    """)  # noqa: S608

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

    # Per-assistant-message token usage, deduplicated by message.id.
    #
    # raw_messages can contain duplicate API responses with different `uuid`
    # values but identical `message.id` (the same Anthropic API response
    # logged more than once — observed at up to 17 copies).  Naive SUM
    # aggregations over raw_messages therefore over-count tokens (and cost)
    # by 2-17x.  Every cost/token computation should read from this view.
    #
    # `DISTINCT ON` is supported by DuckDB; the ORDER BY in the same SELECT
    # makes "earliest copy wins" deterministic.
    conn.execute("""
        CREATE OR REPLACE VIEW assistant_message_costs AS
        SELECT DISTINCT ON (json_extract_string(message, '$.id'))
            session_id,
            uuid,
            parent_uuid,
            timestamp,
            is_sidechain,
            model,
            json_extract_string(message, '$.id') AS message_id,
            COALESCE(CAST(json_extract(
                message, '$.usage.input_tokens'
            ) AS BIGINT), 0) AS input_tokens,
            COALESCE(CAST(json_extract(
                message, '$.usage.output_tokens'
            ) AS BIGINT), 0) AS output_tokens,
            COALESCE(CAST(json_extract(
                message, '$.usage.cache_read_input_tokens'
            ) AS BIGINT), 0) AS cache_read_tokens,
            COALESCE(CAST(json_extract(
                message, '$.usage.cache_creation_input_tokens'
            ) AS BIGINT), 0) AS cache_creation_tokens,
            COALESCE(CAST(json_extract(
                message,
                '$.usage.cache_creation.ephemeral_5m_input_tokens'
            ) AS BIGINT), 0) AS cache_creation_5m,
            COALESCE(CAST(json_extract(
                message,
                '$.usage.cache_creation.ephemeral_1h_input_tokens'
            ) AS BIGINT), 0) AS cache_creation_1h
        FROM raw_messages
        WHERE type = 'assistant'
          AND json_extract_string(message, '$.id') IS NOT NULL
        ORDER BY json_extract_string(message, '$.id'), timestamp
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

    # Enriched per-block view: one row per content block, classified into a
    # 'kind' that the session detail page dispatches on. Unnests content blocks
    # so an assistant message with text + thinking + 2 tool_use yields 4 rows.
    conn.execute("""
        CREATE OR REPLACE VIEW session_messages_enriched AS
        WITH blocks AS (
            SELECT
                m.session_id,
                m.uuid,
                m.parent_uuid,
                m.timestamp,
                m.type,
                m.role,
                m.is_sidechain,
                m.model,
                i.idx AS block_idx,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].type'
                ) AS block_type,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].text'
                ) AS block_text,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].thinking'
                ) AS block_thinking,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].name'
                ) AS block_tool_name,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].id'
                ) AS block_tool_use_id,
                json_extract_string(
                    m.message, '$.content[' || i.idx || '].input'
                ) AS block_tool_input,
            FROM raw_messages m,
                 generate_series(
                     0,
                     CAST(json_array_length(
                         json_extract(m.message, '$.content')
                     ) - 1 AS BIGINT)
                 ) AS i(idx)
            WHERE json_array_length(json_extract(m.message, '$.content')) > 0
        ),
        string_content AS (
            -- User messages where content is a plain string (not an array).
            SELECT
                session_id,
                uuid,
                parent_uuid,
                timestamp,
                type,
                role,
                is_sidechain,
                model,
                0 AS block_idx,
                'text' AS block_type,
                json_extract_string(message, '$.content') AS block_text,
                NULL AS block_thinking,
                NULL AS block_tool_name,
                NULL AS block_tool_use_id,
                NULL AS block_tool_input,
            FROM raw_messages
            -- Slash commands historically arrive as array content; the
            -- string-content branch handles older/alternate user prompts.
            WHERE type IN ('user', 'assistant')
              AND json_type(json_extract(message, '$.content')) = 'VARCHAR'
        ),
        unified AS (
            SELECT * FROM blocks
            UNION ALL
            SELECT * FROM string_content
        )
        SELECT
            session_id,
            uuid,
            parent_uuid,
            timestamp,
            block_idx,
            is_sidechain,
            model,
            CASE
                WHEN block_type = 'thinking' THEN 'agent_thinking'
                WHEN block_type = 'tool_use' THEN 'agent_tool_call'
                WHEN block_type = 'tool_result' THEN 'tool_result'
                WHEN type = 'assistant' THEN 'agent_text'
                WHEN type = 'user' AND role = 'user' AND (
                        COALESCE(block_text, '') LIKE '<command-name>%'
                        OR COALESCE(block_text, '') LIKE '<local-command-%'
                    ) THEN 'slash_command'
                -- Sidechain user messages are the prompt the main agent
                -- passed to the subagent via Task/Agent, NOT human input.
                WHEN type = 'user' AND role = 'user' AND is_sidechain
                    THEN 'subagent_prompt'
                WHEN type = 'user' AND role = 'user' THEN 'human_prompt'
                ELSE 'agent_text'
            END AS kind,
            block_text AS text,
            block_thinking AS thinking_text,
            block_tool_name AS tool_name,
            block_tool_use_id AS tool_use_id,
            block_tool_input AS tool_input,
        FROM unified
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

    # Commands: extract <command-name>...</command-name> tags from user messages.
    conn.execute("""
        CREATE OR REPLACE VIEW message_commands AS
        WITH msg_text AS (
            SELECT session_id, uuid, timestamp,
                   regexp_extract_all(
                       COALESCE(
                           json_extract_string(message, '$.content[0].text'),
                           json_extract_string(message, '$.content')
                       ),
                       '<command-name>([^<]+)</command-name>', 1
                   ) AS cmds
            FROM raw_messages
            WHERE type = 'user' AND role = 'user'
        )
        SELECT session_id, uuid, timestamp,
               unnest(cmds) AS command
        FROM msg_text
        WHERE len(cmds) > 0
    """)

    # File reads: one row per Read tool call with extracted file_path.
    conn.execute("""
        CREATE OR REPLACE VIEW file_reads AS
        SELECT
            tc.session_id,
            tc.tool_use_id,
            tc.called_at,
            json_extract_string(tc.tool_input, '$.file_path') AS file_path
        FROM tool_calls tc
        WHERE tc.tool_name = 'Read'
          AND json_extract_string(tc.tool_input, '$.file_path') IS NOT NULL
    """)

    # File writes: one row per write tool call with extracted file_path.
    conn.execute("""
        CREATE OR REPLACE VIEW file_writes AS
        SELECT
            tc.session_id,
            tc.tool_use_id,
            tc.called_at,
            tc.tool_name,
            COALESCE(
                json_extract_string(tc.tool_input, '$.file_path'),
                json_extract_string(tc.tool_input, '$.notebook_path')
            ) AS file_path
        FROM tool_calls tc
        WHERE tc.tool_name IN ('Edit', 'Write', 'MultiEdit', 'NotebookEdit')
          AND COALESCE(
              json_extract_string(tc.tool_input, '$.file_path'),
              json_extract_string(tc.tool_input, '$.notebook_path')
          ) IS NOT NULL
    """)
