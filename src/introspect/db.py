"""DuckDB database initialization and view management."""

import contextlib
import glob
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from introspect.projects import resolve_project_map
from introspect.search import build_search_corpus
from introspect.sql_fragments import (
    COMMAND_LIST_SUBQUERY,
    FILE_READS_SUBQUERY,
    FILE_WRITES_SUBQUERY,
    SESSION_COST_SUBQUERY,
    TOOL_COUNTS_SUBQUERY,
)

log = logging.getLogger(__name__)

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


# ``maximum_object_size`` is sized to the current load set rather than pinned at
# a fixed constant. DuckDB's JSON reader allocates a per-scan-thread buffer of
# ``maximum_object_size * 2`` before reading any data, so a large constant
# multiplied by the thread count is what OOM-kills tight machines (e.g. a 15GB
# WSL VM). A JSONL line can't exceed its file, so we bound the buffer by the
# largest file (plus slack), clamped to a sane floor/ceiling.
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * _BYTES_PER_MB
_MIN_OBJECT_SIZE_BYTES = 32 * _BYTES_PER_MB  # floor (also DuckDB's is 16MB)
_MAX_OBJECT_SIZE_BYTES = 512 * _BYTES_PER_MB  # ceiling — fits any realistic message
_OBJECT_SIZE_SLACK_BYTES = 1 * _BYTES_PER_MB  # +1MB over the largest file
_MEMORY_LIMIT_HEADROOM = 256 * _BYTES_PER_MB  # slack above scan buffers

_ENV_MAX_OBJECT_SIZE_MB = "INTROSPECT_MAX_OBJECT_SIZE_MB"
_ENV_THREADS = "INTROSPECT_THREADS"


def _read_json_opts(max_object_size: int) -> str:
    """Build the shared ``read_json_auto`` option string for ``max_object_size``."""
    return (
        f"filename=true, format='newline_delimited', union_by_name=true, "
        f"ignore_errors=true, maximum_object_size={max_object_size}"
    )


def _quote_sql_string(value: str) -> str:
    """Escape a Python string for use as a DuckDB SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def _jsonl_read_expr(source: str | list[str], max_object_size: int) -> str:
    """Build a ``read_json_auto(...)`` expression.

    ``source`` may be a glob string (fast path) or an explicit list of file
    paths (used by the per-file fallback that excludes unparseable files).
    ``max_object_size`` bounds DuckDB's per-line JSON buffer.
    """
    opts = _read_json_opts(max_object_size)
    if isinstance(source, str):
        return f"read_json_auto({_quote_sql_string(source)}, {opts})"
    quoted = ", ".join(_quote_sql_string(p) for p in source)
    return f"read_json_auto([{quoted}], {opts})"


def _env_int(name: str, *, minimum: int) -> int | None:
    """Read an ``INTROSPECT_*`` integer override, or None if unset/invalid.

    Invalid values log a WARNING and fall back to the computed value, mirroring
    the ``INTROSPECT_REFRESH_WINDOW`` handling in ``api/main.py``.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid %s=%r; ignoring and using computed value.", name, raw)
        return None
    if value < minimum:
        log.warning(
            "Invalid %s=%r (must be >= %d); ignoring and using computed value.",
            name,
            raw,
            minimum,
        )
        return None
    return value


def _largest_file_size(files: list[str]) -> int:
    """Return the size of the largest file, skipping unreadable ones."""
    largest = 0
    for path in files:
        try:
            largest = max(largest, os.path.getsize(path))  # noqa: PTH202
        except OSError:
            continue
    return largest


def _clamp_object_size(largest: int) -> int:
    """Clamp the largest-file size (+slack) to the [floor, ceiling] object size."""
    if largest <= 0:
        return _MIN_OBJECT_SIZE_BYTES
    return max(
        _MIN_OBJECT_SIZE_BYTES,
        min(largest + _OBJECT_SIZE_SLACK_BYTES, _MAX_OBJECT_SIZE_BYTES),
    )


def _object_size_for(largest: int) -> int:
    """Resolve ``maximum_object_size`` from the largest file or the env override."""
    override_mb = _env_int(_ENV_MAX_OBJECT_SIZE_MB, minimum=1)
    if override_mb is not None:
        return override_mb * _BYTES_PER_MB
    return _clamp_object_size(largest)


def _resolve_object_size(files: list[str]) -> int:
    """Resolve ``maximum_object_size`` for a glob's file set (lazy-view path)."""
    return _object_size_for(_largest_file_size(files))


@dataclass(frozen=True)
class LoadPlan:
    """RAM-aware settings for a materialization load."""

    max_object_size: int
    threads: int | None  # None => leave DuckDB's default (== cpu_count)
    memory_limit_bytes: int | None  # None => leave DuckDB's default
    file_count: int
    total_size: int
    largest_size: int
    largest_file: str | None
    available_ram: int | None
    tight: bool  # even threads=1 needs more buffer than the budget


def _available_ram() -> int | None:
    """Return available RAM in bytes, or None if psutil is unavailable.

    Falls back to None (no cap — current behavior) if psutil can't be imported
    or the query fails for any reason.
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _plan_load(files: list[str]) -> LoadPlan:
    """Compute object size, thread cap, and memory limit for the load set.

    Buffer budget is 50% of available RAM; each scan thread needs
    ``2 * max_object_size``, so threads are capped to fit the budget (floor 1,
    ceiling cpu_count). ``memory_limit`` is 60% of available RAM (DuckDB's
    default is 80% of *total*, wrong inside a RAM-shared WSL VM), but floored at
    the scan-buffer requirement plus headroom so our own cap can never reject
    the buffers we're about to allocate — on a genuinely tight machine we want
    DuckDB to try and let the OS cope, not fail deterministically. Env overrides
    (``INTROSPECT_MAX_OBJECT_SIZE_MB`` / ``INTROSPECT_THREADS``) win.
    """
    total = 0
    largest = 0
    largest_path: str | None = None
    for path in files:
        try:
            size = os.path.getsize(path)  # noqa: PTH202
        except OSError:
            continue
        total += size
        if size > largest:
            largest, largest_path = size, path
    object_size = _object_size_for(largest)

    cpu = os.cpu_count() or 1
    ram = _available_ram()
    per_thread = 2 * object_size
    threads: int | None = None
    memory_limit: int | None = None
    tight = False
    if ram is not None:
        budget = ram // 2
        n = max(1, min(budget // per_thread, cpu))
        threads = n if n < cpu else None  # only SET when below the default
        tight = per_thread > budget

    override_threads = _env_int(_ENV_THREADS, minimum=1)
    if override_threads is not None:
        threads = override_threads

    if ram is not None:
        effective_threads = threads if threads is not None else cpu
        buffer_need = effective_threads * per_thread
        memory_limit = max(int(ram * 0.6), buffer_need + _MEMORY_LIMIT_HEADROOM)

    return LoadPlan(
        max_object_size=object_size,
        threads=threads,
        memory_limit_bytes=memory_limit,
        file_count=len(files),
        total_size=total,
        largest_size=largest,
        largest_file=largest_path,
        available_ram=ram,
        tight=tight,
    )


def _fmt_bytes(n: int | None) -> str:
    """Human-readable byte size for log lines (``None`` → ``"default"``)."""
    if n is None:
        return "default"
    if n >= _BYTES_PER_GB:
        return f"{n / _BYTES_PER_GB:.1f}GB"
    return f"{n / _BYTES_PER_MB:.0f}MB"


def _apply_load_settings(conn: duckdb.DuckDBPyConnection, plan: LoadPlan) -> None:
    """Apply RAM-aware thread/memory settings to a writer connection.

    Materialize-only — never call this on a read-only connection.
    """
    if plan.threads is not None:
        conn.execute(f"SET threads = {plan.threads}")
    if plan.memory_limit_bytes is not None:
        conn.execute(f"SET memory_limit = '{plan.memory_limit_bytes}B'")
    # Every derived view sorts explicitly by timestamp, so insertion order is
    # never relied upon; dropping the guarantee lets DuckDB parallelize and
    # spill the load more freely (per the DuckDB out-of-memory guide).
    conn.execute("SET preserve_insertion_order = false")
    temp_dir = Path.home() / ".introspect" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    conn.execute(f"SET temp_directory = {_quote_sql_string(str(temp_dir))}")


def _filter_parseable_files(files: list[str]) -> list[str]:
    """Return ``files`` minus those that raise a hard read error.

    Probes each file with a COUNT(*) so any DuckDB error that aborts the
    bulk load (size limit, missing file, permission error) surfaces here.
    Per-line malformed JSON is still swallowed by ``ignore_errors=true`` in
    the read options — that's intentional: the goal is to keep partial
    files (a few corrupt lines among many good ones) rather than drop them
    wholesale. Each probe bounds ``maximum_object_size`` by that file's own
    size, since a line can't be larger than its file.

    Cost is O(files): one in-memory full scan per input, so this is the
    slow path used only after a bulk-load failure.
    """
    if files:
        log.warning(
            "Probing %d JSONL file(s) individually after bulk-load failure; "
            "this can take a while.",
            len(files),
        )
    parseable: list[str] = []
    probe = duckdb.connect(":memory:")
    try:
        for path in files:
            try:
                object_size = _clamp_object_size(os.path.getsize(path))  # noqa: PTH202
            except OSError as exc:
                log.warning("Skipping unreadable JSONL file %s: %s", path, exc)
                continue
            sql = f"SELECT COUNT(*) FROM {_jsonl_read_expr(path, object_size)}"  # noqa: S608
            try:
                probe.execute(sql).fetchone()
            except duckdb.Error as exc:
                log.warning("Skipping unparseable JSONL file %s: %s", path, exc)
                continue
            parseable.append(path)
    finally:
        probe.close()
    return parseable


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
            if _has_materialized_raw_messages(conn):
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
        "session_stats",
        "message_commands",
        "session_titles",
        "conversation_turns",
        "session_context_loads",
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
        "materialize_meta",
    ):
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP TABLE IF EXISTS {name}")

    # Size the load to the machine: object size follows the largest file, and
    # thread/memory caps keep DuckDB's per-thread JSON buffers within a RAM
    # budget. Applied here because every materialize path funnels through this
    # writer connection.
    files = sorted(glob.glob(jsonl_glob, recursive=True))  # noqa: PTH207
    plan = _plan_load(files)
    _apply_load_settings(conn, plan)
    log.info(
        "materialize: %d file(s), %s total, largest %s (%s), object_size=%s, "
        "threads=%s, memory_limit=%s",
        plan.file_count,
        _fmt_bytes(plan.total_size),
        _fmt_bytes(plan.largest_size),
        plan.largest_file or "-",
        _fmt_bytes(plan.max_object_size),
        plan.threads if plan.threads is not None else (os.cpu_count() or 1),
        _fmt_bytes(plan.memory_limit_bytes),
    )
    if plan.tight:
        log.warning(
            "Low memory for JSONL load: largest file %s needs ~%s of scan buffer "
            "but only ~%s is available; proceeding, but the OS may kill the "
            "process. Consider excluding or trimming that file.",
            plan.largest_file or "-",
            _fmt_bytes(2 * plan.max_object_size),
            _fmt_bytes(plan.available_ram),
        )

    # ``read_json_auto`` raises IOException when no files match the glob, so
    # an empty Claude home (fresh install, sandboxed CI) needs explicit empty
    # stubs to keep the rest of the pipeline working without special-casing
    # every consumer.
    if files:
        _load_raw_tables(
            conn, jsonl_glob, day_filter, and_day_filter, plan.max_object_size
        )
    else:
        _create_empty_raw_tables(conn)

    # Add indexes for common query patterns
    conn.execute("CREATE INDEX idx_rm_session ON raw_messages(session_id)")
    conn.execute("CREATE INDEX idx_rm_type ON raw_messages(type)")
    conn.execute("CREATE INDEX idx_rm_timestamp ON raw_messages(timestamp)")

    _build_project_map(conn, resolve_projects=resolve_projects)
    _create_derived_views(conn, materialize=True)
    _create_indexes(conn, _DERIVED_INDEXES)
    _create_session_stats(conn, materialize=True)
    _create_indexes(conn, _SESSION_STATS_INDEXES)
    _record_materialized_at(conn)


# Empty-stub schema used when the JSONL glob matches nothing. The column types
# and names mirror what ``read_json_auto`` produces over the real records, so
# downstream views compile and execute (returning zero rows) unchanged.
_EMPTY_RAW_DATA_SQL = """
    CREATE TABLE raw_data AS
    SELECT
        NULL::VARCHAR AS filename,
        NULL::VARCHAR AS type,
        NULL::VARCHAR AS timestamp,
        NULL::VARCHAR AS sessionId,
        NULL::VARCHAR AS uuid,
        NULL::VARCHAR AS parentUuid,
        NULL::BOOLEAN AS isSidechain,
        NULL::VARCHAR AS cwd,
        NULL::VARCHAR AS version,
        NULL::VARCHAR AS entrypoint,
        NULL::VARCHAR AS gitBranch,
        NULL::JSON AS message,
        NULL::JSON AS toolUseResult,
        NULL::JSON AS attachment,
    WHERE FALSE
"""

_EMPTY_RAW_MESSAGES_SQL = """
    CREATE TABLE raw_messages AS
    SELECT
        NULL::VARCHAR AS file_path,
        NULL::VARCHAR AS type,
        NULL::TIMESTAMP AS timestamp,
        NULL::VARCHAR AS session_id,
        NULL::VARCHAR AS uuid,
        NULL::VARCHAR AS parent_uuid,
        NULL::BOOLEAN AS is_sidechain,
        NULL::VARCHAR AS cwd,
        NULL::VARCHAR AS version,
        NULL::VARCHAR AS entrypoint,
        NULL::VARCHAR AS git_branch,
        NULL::VARCHAR AS role,
        NULL::VARCHAR AS model,
        NULL::JSON AS message,
        NULL::JSON AS tool_use_result,
    WHERE FALSE
"""


def _create_empty_raw_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create empty ``raw_data`` and ``raw_messages`` tables matching the
    schemas produced by ``read_json_auto`` over real Claude Code logs."""
    conn.execute(_EMPTY_RAW_DATA_SQL)
    conn.execute(_EMPTY_RAW_MESSAGES_SQL)


def _record_materialized_at(conn: duckdb.DuckDBPyConnection) -> None:
    """Stamp the DB with the current materialization timestamp (UTC)."""
    conn.execute("""
        CREATE TABLE materialize_meta (
            materialized_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("INSERT INTO materialize_meta VALUES (?)", [datetime.now(UTC)])


def read_last_materialized(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    """Return the timestamp recorded by the most recent ``materialize_views``.

    Returns ``None`` when the DB has no ``materialize_meta`` table (older
    builds, lazy-view connections, or partially-built DBs).
    """
    has_meta = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'materialize_meta' AND table_type = 'BASE TABLE'"
    ).fetchone()
    if not has_meta:
        return None
    # ``materialize_meta`` holds exactly one row by construction —
    # ``materialize_views`` drops and recreates the table on every call.
    row = conn.execute("SELECT materialized_at FROM materialize_meta").fetchone()
    return row[0] if row else None


def ensure_materialized(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
    *,
    days: int = 0,
    resolve_projects: bool = True,
) -> datetime | None:
    """Make sure the on-disk DB has materialized tables; build them if not.

    Returns the ``materialized_at`` timestamp recorded in the DB, or ``None``
    if the DB is materialized but predates the ``materialize_meta`` table.

    Reuses an existing materialized DB when present (cheap read-only probe).
    Only acquires a write lock when a build is actually needed, so this stays
    safe to call from CLI commands while a server holds the read DB.

    Raises:
        DatabaseLockedError: if a build is needed but another process holds
            the write lock.
    """
    if db_path.exists():
        try:
            with contextlib.closing(
                duckdb.connect(str(db_path), read_only=True)
            ) as probe:
                if _has_materialized_raw_messages(probe):
                    return read_last_materialized(probe)
        except duckdb.Error:
            # Fall through to rebuild — a corrupt or incompatible file will
            # be replaced by the writable connection below.
            pass

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    try:
        materialize_views(conn, jsonl_glob, days, resolve_projects=resolve_projects)
        build_search_corpus(conn)
        return read_last_materialized(conn)
    finally:
        conn.close()


def _column_exists(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """True when ``table`` exposes ``column`` (base table or view)."""
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?",
        [table, column],
    ).fetchone()
    return row is not None


def _has_materialized_raw_messages(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when ``raw_messages`` exists as a base table in ``conn``."""
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'raw_messages' AND table_type = 'BASE TABLE'"
    ).fetchone()
    return row is not None


def _load_raw_tables(
    conn: duckdb.DuckDBPyConnection,
    jsonl_glob: str,
    day_filter: str,
    and_day_filter: str,
    max_object_size: int,
) -> None:
    """Create ``raw_data`` and ``raw_messages`` from the JSONL glob.

    First tries the bulk read (fast path). On any DuckDB error from
    ``read_json_auto`` (e.g. an oversized JSON object that exceeds
    ``maximum_object_size`` even after we've raised it, or a corrupted file),
    probes each file individually, drops the unparseable ones, and retries
    with the surviving list so a single bad file can't take down the whole
    load.
    """
    try:
        _create_raw_tables(
            conn, jsonl_glob, day_filter, and_day_filter, max_object_size
        )
    except duckdb.Error as exc:
        log.warning(
            "Bulk JSONL load failed (%s); retrying per-file to skip unparseable files.",
            exc,
        )
        bulk_exc = exc
    else:
        return

    # The first CREATE TABLE may have succeeded before the second failed —
    # drop both so the retry starts from a clean slate.
    for name in ("raw_messages", "raw_data"):
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP TABLE IF EXISTS {name}")

    files = sorted(glob.glob(jsonl_glob, recursive=True))  # noqa: PTH207
    parseable = _filter_parseable_files(files)
    if not parseable:
        # Surface a ``duckdb.Error`` subclass so existing catch sites
        # (mcp/tools.py, api handlers) handle this uniformly with other
        # load failures. Chain ``bulk_exc`` so the original cause is kept.
        msg = (
            f"No parseable JSONL files under {jsonl_glob!r}; "
            f"all {len(files)} candidate file(s) failed to load."
        )
        raise duckdb.IOException(msg) from bulk_exc

    _create_raw_tables(conn, parseable, day_filter, and_day_filter, max_object_size)


def _create_raw_tables(
    conn: duckdb.DuckDBPyConnection,
    source: str | list[str],
    day_filter: str,
    and_day_filter: str,
    max_object_size: int,
) -> None:
    """Issue the two CREATE TABLE statements for ``raw_data``/``raw_messages``."""
    read_expr = _jsonl_read_expr(source, max_object_size)

    conn.execute(f"""
        CREATE TABLE raw_data AS
        SELECT * FROM {read_expr}
        {day_filter}
    """)  # noqa: S608

    conn.execute(f"""
        CREATE TABLE raw_messages AS
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {read_expr}
        WHERE type IN ('user', 'assistant')
        {and_day_filter}
    """)  # noqa: S608

    # ``read_json_auto`` only emits an ``attachment`` column when some record
    # carries one; logs without attachment records (or narrow test fixtures)
    # would otherwise leave ``session_context_loads`` unable to bind it.
    conn.execute("ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS attachment JSON")


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
    object_size = _resolve_object_size(sorted(glob.glob(jsonl_glob, recursive=True)))  # noqa: PTH207
    _read = _jsonl_read_expr(jsonl_glob, object_size)

    conn.execute(f"""
        CREATE OR REPLACE VIEW raw_data AS
        SELECT * FROM {_read}
    """)  # noqa: S608

    # Guarantee an ``attachment`` column so ``session_context_loads`` binds even
    # when no record carries one (see the ALTER in ``_create_raw_tables``).
    if not _column_exists(conn, "raw_data", "attachment"):
        conn.execute(f"""
            CREATE OR REPLACE VIEW raw_data AS
            SELECT *, NULL::JSON AS attachment FROM {_read}
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
    _create_session_stats(conn, materialize=False)


def _create_relation(
    conn: duckdb.DuckDBPyConnection, name: str, body: str, *, materialize: bool
) -> None:
    """Create ``name`` as a TABLE (materialized) or VIEW (lazy) over ``body``.

    Single dispatch point for both ``_create_derived_views`` and
    ``_create_session_stats`` so the materialized/lazy semantics can't drift
    between callers.
    """
    if materialize:
        conn.execute(f"CREATE TABLE {name} AS {body}")
    else:
        conn.execute(f"CREATE OR REPLACE VIEW {name} AS {body}")


def _create_derived_views(
    conn: duckdb.DuckDBPyConnection, *, materialize: bool = False
) -> None:
    """Create derived structures over raw_messages.

    When ``materialize=True`` they are created as TABLEs (suitable for the
    on-disk DB built by the background refresh).  When False (the lazy path
    used by ``_create_views``) they are created as VIEWs.  The SELECT bodies
    are identical in both modes.
    """

    def _make(name: str, body: str) -> None:
        _create_relation(conn, name, body, materialize=materialize)

    # Logical sessions: one row per session with summary stats
    _make(
        "logical_sessions",
        """
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
        """,
    )

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
    _make(
        "assistant_message_costs",
        """
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
        """,
    )

    # Tool calls: assistant tool_use content blocks joined with results.
    # Unnests all content blocks so multi-tool messages are captured.
    _make(
        "tool_calls",
        """
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
        """,
    )

    # Enriched per-block view: one row per content block, classified into a
    # 'kind' that the session detail page dispatches on. Unnests content blocks
    # so an assistant message with text + thinking + 2 tool_use yields 4 rows.
    _make(
        "session_messages_enriched",
        """
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
        """,
    )

    # Conversation turns: human/assistant pairs
    _make(
        "conversation_turns",
        """
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
        """,
    )

    # Session titles: first meaningful user prompt per session
    _make(
        "session_titles",
        """
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
        """,
    )

    # Commands: extract <command-name>...</command-name> tags from user messages.
    _make(
        "message_commands",
        """
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
        """,
    )

    # Harness-injected context loads: one row per auto-loaded context item.
    #
    # These live in ``type='attachment'`` records — which every view over
    # ``raw_messages`` drops (it filters to type IN ('user','assistant')) — so
    # this reads ``raw_data`` directly and json-extracts the ``attachment``
    # map.  Only the subtypes that represent *context the harness fed the model*
    # are kept (CLAUDE.md/rules auto-load, ``@``-file expansions, the skill
    # menu, MCP instruction blocks, hook output); listing/reminder chatter
    # (output_style, total_tokens_reminder, task_reminder, …) is dropped.
    #
    # Note: the *root* project CLAUDE.md and global ~/.claude/CLAUDE.md arrive
    # inline as a ``<system-reminder>`` block in the first user message, not as
    # an attachment, so they are not captured here; ``nested_memory`` covers
    # nested CLAUDE.md / ``.claude/rules/*`` files loaded on directory entry.
    _make(
        "session_context_loads",
        """
        WITH att AS (
            SELECT
                sessionId AS session_id,
                timestamp::TIMESTAMP AS timestamp,
                CAST(attachment AS JSON) AS a,
                json_extract_string(
                    CAST(attachment AS JSON), '$.type'
                ) AS atype
            FROM raw_data
            WHERE type = 'attachment'
        )
        SELECT
            session_id,
            timestamp,
            CASE atype
                WHEN 'nested_memory' THEN 'claude_md'
                WHEN 'file' THEN 'file_ref'
                WHEN 'skill_listing' THEN 'skill_listing'
                WHEN 'mcp_instructions_delta' THEN 'mcp'
                WHEN 'hook_success' THEN 'hook'
                WHEN 'hook_non_blocking_error' THEN 'hook'
            END AS load_kind,
            CASE atype
                WHEN 'nested_memory' THEN COALESCE(
                    json_extract_string(a, '$.displayPath'),
                    json_extract_string(a, '$.path')
                )
                WHEN 'file' THEN COALESCE(
                    json_extract_string(a, '$.displayPath'),
                    json_extract_string(a, '$.filename')
                )
                WHEN 'mcp_instructions_delta' THEN array_to_string(
                    CAST(json_extract(a, '$.addedNames') AS VARCHAR[]), ', '
                )
                WHEN 'hook_success' THEN json_extract_string(a, '$.hookName')
                WHEN 'hook_non_blocking_error'
                    THEN json_extract_string(a, '$.hookName')
            END AS name,
            CASE atype
                WHEN 'mcp_instructions_delta'
                    THEN length(json_extract_string(a, '$.addedBlocks'))
                ELSE length(json_extract_string(a, '$.content'))
            END AS char_len
        FROM att
        WHERE atype IN (
            'nested_memory', 'file', 'skill_listing',
            'mcp_instructions_delta', 'hook_success', 'hook_non_blocking_error'
        )
        """,
    )

    # File reads: one row per Read tool call with extracted file_path.
    _make(
        "file_reads",
        """
        SELECT
            tc.session_id,
            tc.tool_use_id,
            tc.called_at,
            json_extract_string(tc.tool_input, '$.file_path') AS file_path
        FROM tool_calls tc
        WHERE tc.tool_name = 'Read'
          AND json_extract_string(tc.tool_input, '$.file_path') IS NOT NULL
        """,
    )

    # File writes: one row per write tool call with extracted file_path.
    _make(
        "file_writes",
        """
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
        """,
    )


_DERIVED_INDEXES = (
    "CREATE INDEX idx_lsess_started ON logical_sessions(started_at)",
    "CREATE INDEX idx_lsess_project ON logical_sessions(project)",
    "CREATE INDEX idx_lsess_model ON logical_sessions(model)",
    "CREATE INDEX idx_lsess_branch ON logical_sessions(git_branch)",
    "CREATE INDEX idx_tcalls_session ON tool_calls(session_id)",
    "CREATE INDEX idx_tcalls_tooluseid ON tool_calls(tool_use_id)",
    "CREATE INDEX idx_amc_session ON assistant_message_costs(session_id)",
    "CREATE INDEX idx_amc_uuid ON assistant_message_costs(uuid)",
    "CREATE INDEX idx_sme_session ON session_messages_enriched(session_id)",
    "CREATE INDEX idx_sme_tooluseid ON session_messages_enriched(tool_use_id)",
    "CREATE INDEX idx_freads_session ON file_reads(session_id)",
    "CREATE INDEX idx_fwrites_session ON file_writes(session_id)",
    "CREATE INDEX idx_mcmds_session ON message_commands(session_id)",
    "CREATE INDEX idx_mcmds_command ON message_commands(command)",
    "CREATE INDEX idx_sctxloads_session ON session_context_loads(session_id)",
    "CREATE INDEX idx_sctxloads_kind ON session_context_loads(load_kind)",
    "CREATE INDEX idx_stitles_session ON session_titles(session_id)",
)


_SESSION_STATS_INDEXES = (
    "CREATE INDEX idx_sstats_started ON session_stats(started_at)",
    "CREATE INDEX idx_sstats_project ON session_stats(project)",
    "CREATE INDEX idx_sstats_model ON session_stats(model)",
    "CREATE INDEX idx_sstats_branch ON session_stats(git_branch)",
    "CREATE INDEX idx_sstats_cost ON session_stats(cost_usd)",
)


def _create_indexes(
    conn: duckdb.DuckDBPyConnection, statements: tuple[str, ...]
) -> None:
    """Execute a tuple of CREATE INDEX statements in order."""
    for stmt in statements:
        conn.execute(stmt)


# SELECT body shared by the table and view forms of ``session_stats``.
# Splices in five module-level SQL fragments (no user input); safe by
# construction.
_SESSION_STATS_BODY = f"""
    SELECT
        ls.session_id,
        ls.started_at,
        ls.ended_at,
        ls.duration,
        ls.user_messages,
        ls.assistant_messages,
        ls.model,
        ls.cwd,
        ls.project,
        ls.git_branch,
        ls.entrypoint,
        COALESCE(tc.tool_count, 0) AS tool_count,
        COALESCE(fr_agg.files_read, 0) AS files_read,
        COALESCE(fw_agg.files_edited, 0) AS files_edited,
        COALESCE(fr_agg.files_read_only, 0) AS files_read_only,
        COALESCE(fr_agg.files_outside, 0) AS files_outside,
        fp.first_prompt,
        cmd.commands,
        sc.cost_usd
    FROM logical_sessions ls
    LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
    LEFT JOIN {TOOL_COUNTS_SUBQUERY} ON ls.session_id = tc.session_id
    LEFT JOIN {FILE_READS_SUBQUERY} ON ls.session_id = fr_agg.session_id
    LEFT JOIN {FILE_WRITES_SUBQUERY} ON ls.session_id = fw_agg.session_id
    LEFT JOIN {COMMAND_LIST_SUBQUERY} ON ls.session_id = cmd.session_id
    LEFT JOIN {SESSION_COST_SUBQUERY} ON ls.session_id = sc.session_id
"""  # noqa: S608


def _create_session_stats(
    conn: duckdb.DuckDBPyConnection, *, materialize: bool
) -> None:
    """Create the ``session_stats`` rollup as a TABLE or VIEW.

    The SELECT body is shared so the listing-page query is identical in both
    materialized and lazy modes.
    """
    _create_relation(
        conn, "session_stats", _SESSION_STATS_BODY, materialize=materialize
    )
