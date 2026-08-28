"""DuckDB database initialization and view management."""

import contextlib
import glob
import json
import logging
import os
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import duckdb

from introspect.cache_ttl import (
    CACHE_REQUESTS_BODY,
    SESSION_CACHE_TTL_BODY,
    TTL_OBSERVED_SQL,
)
from introspect.codex import transcode_rollout_with_metadata
from introspect.pricing import LONG_CONTEXT_REQUEST_SQL
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
DEFAULT_CODEX_GLOB = str(Path.home() / ".codex" / "sessions" / "**" / "*.jsonl")


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


# DuckDB's default ``maximum_object_size`` is 16MB. Some Claude Code tool
# results (Read of large files, big diffs) exceed that and abort the entire
# load. 512MB comfortably fits any realistic message.
_MAX_TOOL_RESULT_SIZE_BYTES = 512 * 1024 * 1024

_READ_JSON_OPTS = (
    f"filename=true, format='newline_delimited', union_by_name=true, "
    f"ignore_errors=true, maximum_object_size={_MAX_TOOL_RESULT_SIZE_BYTES}"
)


def _quote_sql_string(value: str) -> str:
    """Escape a Python string for use as a DuckDB SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def _jsonl_read_expr(source: str | list[str]) -> str:
    """Build a ``read_json_auto(...)`` expression.

    ``source`` may be a glob string (fast path) or an explicit list of file
    paths (used by the per-file fallback that excludes unparseable files).
    """
    if isinstance(source, str):
        return f"read_json_auto({_quote_sql_string(source)}, {_READ_JSON_OPTS})"
    quoted = ", ".join(_quote_sql_string(p) for p in source)
    return f"read_json_auto([{quoted}], {_READ_JSON_OPTS})"


def _filter_parseable_files(files: list[str]) -> list[str]:
    """Return ``files`` minus those that raise a hard read error.

    Probes each file with a COUNT(*) so any DuckDB error that aborts the
    bulk load (size limit, missing file, permission error) surfaces here.
    Per-line malformed JSON is still swallowed by ``ignore_errors=true`` in
    ``_READ_JSON_OPTS`` — that's intentional: the goal is to keep partial
    files (a few corrupt lines among many good ones) rather than drop them
    wholesale.

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
            sql = f"SELECT COUNT(*) FROM {_jsonl_read_expr(path)}"  # noqa: S608
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
    'anthropic' AS provider,
    'claude-code' AS harness,
"""


# Single ordered source of truth for the ``raw_messages`` column set, keyed
# by name: (Codex struct type, final ``raw_messages`` SQL type). Everything
# below that needs the column list — the Codex struct binding, the
# ``$1``-bound select, and the empty-stub select — is generated from this so
# a future column addition lands in one place. Union with the Claude side
# (``_RAW_MESSAGES_COLUMNS`` above) happens ``BY NAME``, so its order need
# not match this one — only the names.
_RAW_MESSAGE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("file_path", "VARCHAR", "VARCHAR"),
    ("type", "VARCHAR", "VARCHAR"),
    ("timestamp", "VARCHAR", "TIMESTAMP"),
    ("session_id", "VARCHAR", "VARCHAR"),
    ("uuid", "VARCHAR", "VARCHAR"),
    ("parent_uuid", "VARCHAR", "VARCHAR"),
    ("is_sidechain", "BOOLEAN", "BOOLEAN"),
    ("cwd", "VARCHAR", "VARCHAR"),
    ("version", "VARCHAR", "VARCHAR"),
    ("entrypoint", "VARCHAR", "VARCHAR"),
    ("git_branch", "VARCHAR", "VARCHAR"),
    ("role", "VARCHAR", "VARCHAR"),
    ("model", "VARCHAR", "VARCHAR"),
    ("message", "VARCHAR", "JSON"),
    ("tool_use_result", "VARCHAR", "JSON"),
    ("provider", "VARCHAR", "VARCHAR"),
    ("harness", "VARCHAR", "VARCHAR"),
)

_RAW_MESSAGES_COLUMN_NAMES = tuple(name for name, _, _ in _RAW_MESSAGE_COLUMNS)

# Struct type for binding Python-side Codex rows (see ``codex.transcode_rollout``)
# as a DuckDB relation via ``unnest($1::..., recursive := true)``.
_CODEX_ROW_STRUCT = (
    "STRUCT(\n    "
    + ",\n    ".join(
        f"{name} {struct_type}" for name, struct_type, _ in _RAW_MESSAGE_COLUMNS
    )
    + "\n)[]"
)


def _transcode_codex_file(
    path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Transcode one Codex rollout file into ``raw_messages``-shaped rows.

    ``message``/``tool_use_result`` are JSON-encoded here (rather than left
    as native Python objects) because ``tool_use_result`` can be a bare
    string (raw shell output) — casting a Python ``str`` straight to DuckDB's
    ``JSON`` type re-parses it *as* JSON and blows up on unquoted text.
    Round-tripping through ``json.dumps`` first guarantees valid JSON text
    for the ``::JSON`` cast in ``_codex_select_sql``. ``None`` stays ``None``
    so ``tool_use_result IS NULL`` still behaves like the Claude side.
    """
    rows, metadata = transcode_rollout_with_metadata(path)
    for row in rows:
        row["message"] = json.dumps(row["message"])
        tool_use_result = row["tool_use_result"]
        row["tool_use_result"] = (
            None if tool_use_result is None else json.dumps(tool_use_result)
        )
    return rows, metadata


def _codex_select_sql(day_filter: str = "") -> str:
    """SELECT producing ``raw_messages``-shaped rows from a ``$1``-bound Codex
    row list. Caller must pass ``[rows]`` as the ``execute`` params."""
    cols = ",\n            ".join(
        f"{name}::{final_type} AS {name}"
        for name, _, final_type in _RAW_MESSAGE_COLUMNS
    )
    return f"""
        SELECT
            {cols}
        FROM (SELECT unnest($1::{_CODEX_ROW_STRUCT}, recursive := true))
        {day_filter}
    """  # noqa: S608


def _codex_insert_sql(day_filter: str = "") -> str:
    """Insert Codex rows, dropping repeated native response items.

    Codex can replay a parent transcript into a nested rollout.  The rollout
    transcoder gives every emitted row a per-file synthetic ``uuid``, so the
    native OpenAI response-item id in ``message.id`` is the only stable
    identity available across files.  Rows without a native id are retained:
    they include user events and synthesized enrichment rows.
    """
    columns = ",\n            ".join(_RAW_MESSAGES_COLUMN_NAMES)
    existing_native_id = "json_extract_string(existing.message, '$.id')"
    return f"""
        INSERT INTO codex_raw_messages
        WITH deduplicated AS ({_codex_incoming_sql(day_filter)})
        SELECT {columns}
        FROM deduplicated AS candidate
        WHERE candidate.native_id IS NULL
           OR candidate.native_id = candidate.uuid
           OR NOT EXISTS (
                SELECT 1
                FROM codex_raw_messages AS existing
                WHERE existing.session_id = candidate.session_id
                  AND {existing_native_id} = candidate.native_id
                  AND {existing_native_id} != existing.uuid
           )
    """  # noqa: S608


def _codex_incoming_sql(day_filter: str = "") -> str:
    """Select one deterministic winner per native response id in one file."""
    columns = ",\n                ".join(_RAW_MESSAGES_COLUMN_NAMES)
    native_id = "json_extract_string(message, '$.id')"
    return f"""
        SELECT * EXCLUDE (row_number)
        FROM (
            SELECT
                incoming.*,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id, native_id
                    ORDER BY timestamp, uuid
                ) AS row_number
            FROM (
                SELECT
                    {columns},
                    {native_id} AS native_id
                FROM ({_codex_select_sql(day_filter)})
            ) AS incoming
        )
        WHERE native_id IS NULL
           OR native_id = uuid
           OR row_number = 1
    """  # noqa: S608


def _codex_delete_later_copies_sql(day_filter: str = "") -> str:
    """Delete stored native-id rows beaten by an earlier incoming copy."""
    existing_native_id = "json_extract_string(existing.message, '$.id')"
    return f"""
        WITH deduplicated AS ({_codex_incoming_sql(day_filter)})
        DELETE FROM codex_raw_messages AS existing
        USING (
            SELECT session_id, native_id, timestamp, uuid
            FROM deduplicated
            WHERE native_id IS NOT NULL AND native_id != uuid
        ) AS candidate
        WHERE existing.session_id = candidate.session_id
          AND {existing_native_id} = candidate.native_id
          AND {existing_native_id} != existing.uuid
          AND (
              existing.timestamp > candidate.timestamp
              OR (
                  existing.timestamp = candidate.timestamp
                  AND existing.uuid > candidate.uuid
              )
          )
    """  # noqa: S608


# Empty-stub select matching the ``raw_messages`` schema, generated from the
# same column table so it can never drift from ``_codex_select_sql``'s output
# types. Used both as the empty Claude-side stub and as ``codex_raw_messages``'s
# starting schema before any rollout file is inserted.
_EMPTY_RAW_MESSAGES_SELECT = (
    "SELECT\n        "
    + ",\n        ".join(
        f"NULL::{final_type} AS {name}" for name, _, final_type in _RAW_MESSAGE_COLUMNS
    )
    + "\n    WHERE FALSE"
)

_CODEX_SESSION_METADATA_COLUMNS = (
    "session_id VARCHAR",
    "title VARCHAR",
    "agent_path VARCHAR",
    "agent_nickname VARCHAR",
    "parent_thread_id VARCHAR",
)
_CODEX_SESSION_METADATA_STRUCT = (
    "STRUCT(" + ", ".join(_CODEX_SESSION_METADATA_COLUMNS) + ")[]"
)


def _merge_codex_session_metadata(
    metadata: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep the first nonempty display value per session and metadata field."""
    merged: dict[str, dict[str, str]] = {}
    for record in metadata:
        session_id = record["session_id"]
        current = merged.setdefault(
            session_id,
            {
                "session_id": session_id,
                "title": "",
                "agent_path": "",
                "agent_nickname": "",
                "parent_thread_id": "",
            },
        )
        for field in ("title", "agent_path", "agent_nickname", "parent_thread_id"):
            if not current[field] and record[field]:
                current[field] = record[field]
    return list(merged.values())


def _create_codex_session_metadata_table(
    conn: duckdb.DuckDBPyConnection, metadata: list[dict[str, str]]
) -> None:
    """Materialize one Codex display-metadata row per logical session."""
    columns = ", ".join(_CODEX_SESSION_METADATA_COLUMNS)
    conn.execute(f"CREATE TABLE codex_session_metadata ({columns})")
    if not metadata:
        return
    merged = _merge_codex_session_metadata(metadata)
    conn.execute(
        f"""
        INSERT INTO codex_session_metadata
        SELECT
            session_id,
            NULLIF(title, '') AS title,
            NULLIF(agent_path, '') AS agent_path,
            NULLIF(agent_nickname, '') AS agent_nickname,
            NULLIF(parent_thread_id, '') AS parent_thread_id
        FROM (SELECT unnest($1::{_CODEX_SESSION_METADATA_STRUCT}, recursive := true))
        """,  # noqa: S608
        [merged],
    )


def _create_codex_raw_messages_table(
    conn: duckdb.DuckDBPyConnection,
    codex_glob: str | Sequence[str] | None,
    day_filter: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Materialize ``codex_raw_messages`` once, streaming one rollout file at
    a time so peak memory is one file's rows rather than the whole corpus.

    Empty (zero rows) when ``codex_glob`` is ``None`` (Codex not requested)
    or matches nothing (missing ``~/.codex/sessions``) — the ``UNION ALL BY
    NAME`` against it elsewhere is then a silent no-op.
    """
    conn.execute(f"CREATE TABLE codex_raw_messages AS {_EMPTY_RAW_MESSAGES_SELECT}")
    if codex_glob is None:
        _create_codex_session_metadata_table(conn, [])
        return
    insert_sql = _codex_insert_sql(day_filter)
    delete_later_copies_sql = _codex_delete_later_copies_sql(day_filter)
    metadata: list[dict[str, str]] = []
    paths = (
        sorted(codex_glob)
        if not isinstance(codex_glob, str)
        else sorted(glob.glob(codex_glob, recursive=True))  # noqa: PTH207
    )
    total = len(paths)
    completed = 0
    for path in paths:
        try:
            rows, file_metadata = _transcode_codex_file(path)
            metadata.extend(file_metadata)
            if rows:
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute(delete_later_copies_sql, [rows])
                    conn.execute(insert_sql, [rows])
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
        except Exception:
            log.warning(
                "Skipping unparseable Codex rollout file %s", path, exc_info=True
            )
            continue
        finally:
            completed += 1
            if progress is not None:
                progress(completed, total)
    _create_codex_session_metadata_table(conn, metadata)


# Columns DuckDB sometimes infers as native UUID (when every sampled value
# happens to look like one) rather than VARCHAR. Codex ids are never
# UUID-shaped (e.g. ``"<session-uuid>:14"``), so unioning the two sides
# without normalizing these columns can raise a UUID-cast error depending on
# what a given Claude corpus/day-window happens to contain.
_UUID_PRONE_COLUMNS = frozenset({"session_id", "uuid", "parent_uuid"})


def _claude_select_for_union(claude_select: str) -> str:
    """Wrap a Claude ``raw_messages`` SELECT so id-like columns are always
    VARCHAR, matching the Codex side's types for ``UNION ALL BY NAME``."""
    cols = ", ".join(
        f"{name}::VARCHAR AS {name}" if name in _UUID_PRONE_COLUMNS else name
        for name in _RAW_MESSAGES_COLUMN_NAMES
    )
    return f"SELECT {cols} FROM ({claude_select})"  # noqa: S608


def _has_codex_rows(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when ``codex_raw_messages`` (already materialized) has any rows."""
    row = conn.execute("SELECT EXISTS(SELECT 1 FROM codex_raw_messages)").fetchone()
    return bool(row and row[0])


def _create_raw_messages_union(
    conn: duckdb.DuckDBPyConnection, claude_select: str, *, relation: str = "TABLE"
) -> str:
    """Build the ``CREATE TABLE|VIEW raw_messages AS ... UNION ALL BY NAME
    SELECT * FROM codex_raw_messages`` statement.

    Only pays the id-columns-to-VARCHAR cast (``_claude_select_for_union``)
    when ``codex_raw_messages`` actually has rows to union — an empty
    Codex table (the common case) leaves the Claude side's native column
    types (e.g. DuckDB's UUID inference) untouched, which downstream code
    (``assistant_message_costs`` consumers) relies on.
    """
    select = (
        _claude_select_for_union(claude_select)
        if _has_codex_rows(conn)
        else claude_select
    )
    verb = "CREATE TABLE" if relation == "TABLE" else "CREATE OR REPLACE VIEW"
    return f"""
        {verb} raw_messages AS
        {select}
        UNION ALL BY NAME
        SELECT * FROM codex_raw_messages
    """  # noqa: S608


# --- Hardened read connection -------------------------------------------------
#
# The engine configuration below — not the SQL text validator in
# ``introspect.sql_query`` — is the primary boundary for the ad-hoc SQL
# surface (``POST /api/query`` and the MCP ``run_sql`` tool). See
# ``docs/security.md`` for the threat model.

# Startup-only options: DuckDB accepts these in ``connect(config=...)`` but
# ``memory_limit`` / ``threads`` can also be re-SET later, which is exactly why
# ``lock_configuration`` is issued last.
_MEMORY_LIMIT_CEILING_BYTES = 2 * 1024**3
_MEMORY_LIMIT_FLOOR_BYTES = 256 * 1024**2
_MEMORY_LIMIT_FRACTION = 0.25
# Fallback when the OS won't tell us how much RAM is free.
_MEMORY_LIMIT_FALLBACK_BYTES = 512 * 1024**2
_MAX_TEMP_DIRECTORY_SIZE = "1GB"

# Access settings, issued in this order after ``LOAD fts`` and before
# ``lock_configuration``. Every one of these takes something *away*; a
# performance knob does not belong in this tuple (``preserve_insertion_order``
# is a connect-time option in ``_read_config`` for exactly that reason).
_HARDENING_SETTINGS: tuple[tuple[str, str], ...] = (
    # Blocks read_csv / read_text / read_blob / read_json / glob / COPY TO /
    # ATTACH / httpfs — every filesystem and network escape from a SELECT.
    ("enable_external_access", "false"),
    ("allow_community_extensions", "false"),
    ("allow_unsigned_extensions", "false"),
    ("autoinstall_known_extensions", "false"),
    ("autoload_known_extensions", "false"),
    ("allow_persistent_secrets", "false"),
)


def _available_ram_bytes() -> int | None:
    """Best-effort "RAM we could use right now", or None if unknowable.

    Prefers Linux's ``MemAvailable`` (accounts for reclaimable page cache);
    falls back to ``SC_AVPHYS_PAGES`` where ``sysconf`` exposes it (macOS,
    BSDs). Returns None rather than guessing so the caller can apply an
    explicit fallback.
    """
    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _default_memory_limit() -> str:
    """DuckDB ``memory_limit`` string: ~25% of available RAM, capped at 2GB."""
    available = _available_ram_bytes()
    raw = (
        int(available * _MEMORY_LIMIT_FRACTION)
        if available
        else _MEMORY_LIMIT_FALLBACK_BYTES
    )
    clamped = max(_MEMORY_LIMIT_FLOOR_BYTES, min(raw, _MEMORY_LIMIT_CEILING_BYTES))
    return f"{clamped // (1024 * 1024)}MB"


def _default_threads() -> int:
    """At most half the cores, never fewer than one."""
    cores = os.cpu_count() or 1
    return max(1, cores // 2)


# Mirrors the value type ``duckdb.connect(config=...)`` accepts. Spelled out
# so the memoized dict is assignable to it — a bare ``dict[str, str]`` is not,
# because the parameter's value type is a union and dicts are invariant.
DuckDBConfig = dict[str, "str | bool | int | float | list[str]"]


def _memory_limit_setting() -> str:
    """``INTROSPECT_DB_MEMORY_LIMIT`` if it parses, else the derived default.

    An unusable value warns and falls back rather than propagating: this
    config is applied in the request middleware, so a typo would otherwise
    turn every page into a 500 with nothing naming the variable.
    """
    override = os.environ.get("INTROSPECT_DB_MEMORY_LIMIT", "").strip()
    if not override:
        return _default_memory_limit()
    try:
        with contextlib.closing(
            duckdb.connect(":memory:", config={"memory_limit": override})
        ):
            return override
    except duckdb.Error as exc:
        log.warning(
            "Ignoring INTROSPECT_DB_MEMORY_LIMIT=%r (%s); using %s.",
            override,
            exc,
            default := _default_memory_limit(),
        )
        return default


def _threads_setting() -> str:
    """``INTROSPECT_DB_THREADS`` if it is a positive integer, else the default."""
    override = os.environ.get("INTROSPECT_DB_THREADS", "").strip()
    if not override:
        return str(_default_threads())
    try:
        if (value := int(override)) > 0:
            return str(value)
    except ValueError:
        pass
    log.warning(
        "Ignoring INTROSPECT_DB_THREADS=%r (want a positive integer); using %d.",
        override,
        default := _default_threads(),
    )
    return str(default)


@cache
def _read_config(db_path_str: str) -> DuckDBConfig:
    """Frozen ``connect(config=...)`` mapping for one database path.

    Memoized because DuckDB refuses a second connection to a file whose
    existing instance was opened with a *different* configuration
    (``ConnectionException: Can't open a connection to same database file
    with a different configuration than existing connections``). The
    memory limit is derived from live RAM, so recomputing it per request
    would eventually produce a different string and break every concurrent
    caller.

    Pure: :func:`connect_read_hardened` creates the temp directory, so the
    cached value has no filesystem side effect to repeat or skip.
    """
    return {
        "memory_limit": _memory_limit_setting(),
        "threads": _threads_setting(),
        "max_temp_directory_size": _MAX_TEMP_DIRECTORY_SIZE,
        # Not a hardening setting — a memory one. DuckDB keeps input order for
        # un-``ORDER BY``-ed results by default, which costs memory the read
        # path has no use for: every relation the UI shows is explicitly
        # ordered. Set at connect time rather than in _HARDENING_SETTINGS so
        # that tuple stays "things this connection may not do".
        "preserve_insertion_order": "false",
        "temp_directory": _temp_directory(db_path_str),
    }


def _temp_directory(db_path_str: str) -> str:
    """Where DuckDB may spill, next to the database file itself."""
    return str(Path(db_path_str).parent / "duckdb-tmp")


# Set once a network ``INSTALL fts`` has been tried and failed. Connections
# are per-request and DuckDB drops an instance once its last connection
# closes, so ``_load_fts`` runs again on the next request — without this flag
# a machine with no extension and no network would pay the DNS timeout
# (~80s) on *every* page load, not once.
_fts_install_failed: list[bool] = [False]


def _try_load_fts(conn: duckdb.DuckDBPyConnection) -> bool:
    """``LOAD fts`` if the extension is already on disk. Never touches network."""
    try:
        conn.execute("LOAD fts")
    except duckdb.Error:
        return False
    return True


def _load_fts(conn: duckdb.DuckDBPyConnection) -> None:
    """Load the FTS extension while external access is still permitted.

    Verified on DuckDB 1.5.3: once ``enable_external_access=false`` is set,
    ``LOAD fts`` on an instance that has *not* already loaded it fails with
    ``PermissionException: Loading external extensions is disabled through
    configuration``, and ``INSTALL fts`` fails with ``PermissionException:
    Cannot access directory ".../.duckdb/extensions/..."``. Loading it first
    is therefore mandatory, not merely an optimisation — and re-``LOAD``ing an
    already-loaded extension after the lock is a harmless no-op, which is what
    keeps :func:`introspect.search.fts_available` working.

    Best effort: a machine with no FTS extension and no network simply falls
    back to the ILIKE search path, and only pays for discovering that once
    per process (see :data:`_fts_install_failed`).
    """
    if _try_load_fts(conn) or _fts_install_failed[0]:
        return
    # Not on disk — fetch it. Still permitted at this point, which is the
    # whole reason this runs before the SETs below.
    try:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
    except duckdb.Error as exc:
        _fts_install_failed[0] = True
        log.debug("FTS extension unavailable; search falls back to ILIKE: %s", exc)


def _configuration_locked(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when this DuckDB instance has already been hardened."""
    try:
        row = conn.execute("SELECT current_setting('lock_configuration')").fetchone()
    except duckdb.Error:
        return False
    return bool(row and row[0])


# Guards the read-modify-write between "is this instance locked?" and the SETs
# that lock it. Settings are instance-global, so two threads racing to harden
# the same freshly-opened instance would otherwise have the loser hit
# ``InvalidInputException: Cannot change configuration option "..." - the
# configuration has been locked``.
_harden_lock = threading.Lock()


def connect_read_hardened(
    db_path: Path = DEFAULT_DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """Open ``db_path`` read-only with the engine locked down.

    This is the *only* place the codebase opens a read-only connection to the
    main database file; ``tests/e2e/test_sql_hardening.py`` greps for
    violations. Routing everything through one factory is also a correctness
    requirement, not just tidiness: DuckDB keys its instance cache by path and
    rejects a second connection carrying a different ``config``, so a caller
    that opened the file bare would break every hardened caller (and vice
    versa).

    Order of operations, each step depending on the previous one:

    1. ``connect(config=...)`` applies the resource caps (memory, threads,
       temp-directory size and location).
    2. ``LOAD fts`` — must precede step 3; see :func:`_load_fts`.
    3. The ``_HARDENING_SETTINGS`` SETs cut off filesystem, network and
       extension access.
    4. ``lock_configuration`` makes steps 1-3 irreversible for the lifetime
       of the instance, so a ``SET``/``PRAGMA`` inside a query can't undo them.

    Idempotent: settings are instance-global and concurrent callers join the
    cached instance, so an already-locked instance skips straight to returning
    the connection. Re-issuing the SETs there would raise
    ``InvalidInputException: Cannot change configuration option "..." - the
    configuration has been locked``.
    """
    config = _read_config(str(db_path))
    with contextlib.suppress(OSError):
        Path(str(config["temp_directory"])).mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=True, config=config)
    try:
        with _harden_lock:
            if not _configuration_locked(conn):
                _load_fts(conn)
                for name, value in _HARDENING_SETTINGS:
                    conn.execute(f"SET {name} = {value}")
                conn.execute("SET lock_configuration = true")
    except duckdb.Error:
        conn.close()
        raise
    return conn


def configured_memory_limit(db_path: Path = DEFAULT_DB_PATH) -> str:
    """The ``memory_limit`` a hardened connection to ``db_path`` runs under.

    Exposed so an out-of-memory error can name the actual budget instead of
    telling the caller to go look it up.
    """
    return str(_read_config(str(db_path))["memory_limit"])


def get_read_connection(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
    codex_glob: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open materialized DB read-only, falling back to lazy views.

    The read-only branch goes through :func:`connect_read_hardened`, so every
    caller that lands on a materialized DB gets the locked-down engine. The
    fallback branch builds lazy views over the JSONL files and therefore
    *needs* filesystem access — it cannot be hardened, and is only reached
    when there is no materialized DB to read.
    """
    needs_upgrade = False
    if db_path.exists():
        try:
            conn = connect_read_hardened(db_path)
            if _has_materialized_schema(conn):
                return conn
            needs_upgrade = _has_materialized_raw_table(conn)
            conn.close()
        except duckdb.Error:
            pass
    if needs_upgrade:
        ensure_materialized(db_path, jsonl_glob, codex_glob=codex_glob)
        return connect_read_hardened(db_path)
    return get_connection(db_path, jsonl_glob, codex_glob)


def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
    jsonl_glob: str = DEFAULT_JSONL_GLOB,
    codex_glob: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with views created.

    Raises:
        DatabaseLockedError: if another process holds a write lock on the DB.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    _create_views(conn, jsonl_glob, codex_glob)
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


def materialize_views(  # noqa: PLR0913
    conn: duckdb.DuckDBPyConnection,
    jsonl_glob: str,
    days: int = 0,
    *,
    resolve_projects: bool = True,
    codex_glob: str | None = None,
    jsonl_candidates: Sequence[str] | None = None,
    codex_candidates: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    phase: Callable[[str], None] | None = None,
) -> None:
    """Materialize raw data into tables for fast querying.

    Args:
        conn: DuckDB connection.
        jsonl_glob: Glob pattern for JSONL files.
        days: Number of days of history to load. 0 means no limit.
        codex_glob: Glob pattern for Codex rollout JSONL files. ``None``
            (the default) skips Codex entirely — ``raw_messages`` is built
            from Claude data only, unchanged from before this parameter
            existed. A glob matching nothing is a silent no-op.
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
        "session_cache_ttl",
        "cache_requests",
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
        "codex_raw_messages",
        "codex_session_metadata",
        "search_corpus",
        "materialize_meta",
    ):
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        with contextlib.suppress(duckdb.CatalogException):
            conn.execute(f"DROP TABLE IF EXISTS {name}")

    # Materialized once here regardless of ``codex_glob`` — empty (zero rows)
    # when Codex wasn't requested or its glob matches nothing, so every
    # ``raw_messages`` construction below can union it in unconditionally.
    codex_source: str | Sequence[str] | None = (
        codex_candidates if codex_candidates is not None else codex_glob
    )
    _create_codex_raw_messages_table(conn, codex_source, day_filter, progress)

    # ``read_json_auto`` raises IOException when no files match the glob, so
    # an empty Claude home (fresh install, sandboxed CI) needs explicit empty
    # stubs to keep the rest of the pipeline working without special-casing
    # every consumer.
    jsonl_source: str | list[str] = (
        list(jsonl_candidates) if jsonl_candidates is not None else jsonl_glob
    )
    has_claude = (
        bool(jsonl_source)
        if not isinstance(jsonl_source, str)
        else next(glob.iglob(jsonl_source, recursive=True), None) is not None  # noqa: PTH207
    )
    if has_claude:
        _load_raw_tables(conn, jsonl_source, day_filter, and_day_filter)
    else:
        _create_empty_raw_tables(conn)

    # Add indexes for common query patterns
    conn.execute("CREATE INDEX idx_rm_session ON raw_messages(session_id)")
    conn.execute("CREATE INDEX idx_rm_type ON raw_messages(type)")
    conn.execute("CREATE INDEX idx_rm_timestamp ON raw_messages(timestamp)")

    _build_project_map(conn, resolve_projects=resolve_projects)
    if phase is not None:
        phase("derived")
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


def _create_empty_raw_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create empty ``raw_data`` and ``raw_messages`` tables matching the
    schemas produced by ``read_json_auto`` over real Claude Code logs.

    Codex data (if any — see ``_create_codex_raw_messages_table``) is unioned
    in even though the Claude side is empty, so a fresh Claude install with
    an existing Codex history still surfaces Codex sessions.
    """
    conn.execute(_EMPTY_RAW_DATA_SQL)
    conn.execute(_create_raw_messages_union(conn, _EMPTY_RAW_MESSAGES_SELECT))


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
    codex_glob: str | None = None,
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
            with contextlib.closing(connect_read_hardened(db_path)) as probe:
                if _has_materialized_schema(probe):
                    return read_last_materialized(probe)
        except duckdb.Error:
            # Fall through to rebuild — a corrupt or incompatible file will
            # be replaced by the writable connection below.
            pass

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    try:
        materialize_views(
            conn,
            jsonl_glob,
            days,
            resolve_projects=resolve_projects,
            codex_glob=codex_glob,
        )
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


def _has_materialized_schema(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when the materialized schema has the required support relations.

    ``raw_messages`` alone used to identify a usable on-disk database. That
    lets databases made before cache-TTL, Codex-title, search-corpus, and
    materialization metadata support pass the probe. Rebuild those databases
    on their next CLI use rather than treating an incomplete schema as
    current.
    """
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name IN "
        "('raw_messages', 'cache_requests', 'codex_session_metadata', "
        "'search_corpus', 'materialize_meta') "
        "AND table_type = 'BASE TABLE' "
        "GROUP BY table_type HAVING COUNT(*) = 5"
    ).fetchone()
    return row is not None


def has_compatible_materialized_db(db_path: Path) -> bool:
    """Return whether ``db_path`` is a complete database safe to publish.

    A warm database is only useful during progressive startup when all base
    relations needed by the current derived-view pipeline are present.  Probe
    it through the same hardened read factory used by requests so a stale or
    corrupt file is never published as the warm snapshot.
    """
    if not db_path.exists():
        return False
    try:
        with contextlib.closing(connect_read_hardened(db_path)) as conn:
            return _has_materialized_schema(conn)
    except duckdb.Error:
        return False


def _has_materialized_raw_table(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when ``raw_messages`` is a table from an older materialization."""
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'raw_messages' AND table_type = 'BASE TABLE'"
    ).fetchone()
    return row is not None


def _load_raw_tables(
    conn: duckdb.DuckDBPyConnection,
    jsonl_glob: str | list[str],
    day_filter: str,
    and_day_filter: str,
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
        _create_raw_tables(conn, jsonl_glob, day_filter, and_day_filter)
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

    files = (
        sorted(jsonl_glob)
        if not isinstance(jsonl_glob, str)
        else sorted(glob.glob(jsonl_glob, recursive=True))  # noqa: PTH207
    )
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

    _create_raw_tables(conn, parseable, day_filter, and_day_filter)


def _create_raw_tables(
    conn: duckdb.DuckDBPyConnection,
    source: str | list[str],
    day_filter: str,
    and_day_filter: str,
) -> None:
    """Issue the two CREATE TABLE statements for ``raw_data``/``raw_messages``."""
    read_expr = _jsonl_read_expr(source)

    conn.execute(f"""
        CREATE TABLE raw_data AS
        SELECT * FROM {read_expr}
        {day_filter}
    """)  # noqa: S608

    claude_select = f"""
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {read_expr}
        WHERE type IN ('user', 'assistant')
        {and_day_filter}
    """  # noqa: S608

    conn.execute(_create_raw_messages_union(conn, claude_select))

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


def _create_views(
    conn: duckdb.DuckDBPyConnection,
    jsonl_glob: str,
    codex_glob: str | None = None,
) -> None:
    """Create lazy views over JSONL files."""
    _read = _jsonl_read_expr(jsonl_glob)

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

    claude_select = f"""
        SELECT {_RAW_MESSAGES_COLUMNS}
        FROM {_read}
        WHERE type IN ('user', 'assistant')
    """  # noqa: S608

    # Views can't bind prepared parameters, so Codex rows are first
    # materialized into a real table (parsed once, here), and the lazy view
    # unions the always-fresh Claude select with that snapshot. Empty when
    # ``codex_glob`` is ``None`` or matches nothing, so the union is a no-op.
    conn.execute("DROP TABLE IF EXISTS codex_raw_messages")
    _create_codex_raw_messages_table(conn, codex_glob)
    conn.execute(_create_raw_messages_union(conn, claude_select, relation="VIEW"))

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
            ANY_VALUE(rm.provider) AS provider,
            ANY_VALUE(rm.harness) AS harness,
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
        f"""
        SELECT *, {TTL_OBSERVED_SQL} AS ttl_observed FROM (
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
        )
        """,  # noqa: S608
    )

    # ``cache_requests``: one row per API request, chain-ordered, carrying the
    # gap that decides cache warmth plus the 5m/1h counterfactual costs. The
    # single home of the cache-TTL detection rule — the session-detail
    # divider, the tokenscape event track and the cost-overview panel all read
    # it, so no two surfaces can disagree about what a cache break is.
    _make("cache_requests", CACHE_REQUESTS_BODY)

    # Per-session counterfactual rollup (diagnostics; the setting is per
    # user/project, so the actionable rollups are project/global).
    _make("session_cache_ttl", SESSION_CACHE_TTL_BODY)

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

    # Session titles: the original request for Codex approval sidecars when
    # available, otherwise the first meaningful main-conversation prompt.
    _make(
        "session_titles",
        """
        WITH first_prompts AS (
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
              AND NOT COALESCE(is_sidechain, FALSE)
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
              AND COALESCE(
                  json_extract_string(message, '$.content[0].text'),
                  json_extract_string(message, '$.content'),
                  ''
              ) NOT LIKE '<environment_context>%'
              AND COALESCE(
                  json_extract_string(message, '$.content[0].text'),
                  json_extract_string(message, '$.content'),
                  ''
              ) NOT LIKE 'The following is the Codex agent history%'
        )
        SELECT
            ls.session_id,
            COALESCE(
                NULLIF(csm.title, ''),
                fp.first_prompt,
                NULLIF(csm.agent_path, ''),
                NULLIF(csm.agent_nickname, '')
            ) AS first_prompt
        FROM logical_sessions ls
        LEFT JOIN first_prompts fp ON fp.session_id = ls.session_id AND fp.rn = 1
        LEFT JOIN codex_session_metadata csm ON csm.session_id = ls.session_id
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
    "CREATE INDEX idx_creq_session ON cache_requests(session_id)",
    "CREATE INDEX idx_creq_sidechain ON cache_requests(is_sidechain)",
    "CREATE INDEX idx_sttl_session ON session_cache_ttl(session_id)",
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
        ls.provider,
        ls.harness,
        COALESCE(tc.tool_count, 0) AS tool_count,
        COALESCE(fr_agg.files_read, 0) AS files_read,
        COALESCE(fw_agg.files_edited, 0) AS files_edited,
        COALESCE(fr_agg.files_read_only, 0) AS files_read_only,
        COALESCE(fr_agg.files_outside, 0) AS files_outside,
        fp.first_prompt,
        cmd.commands,
        sc.cost_usd,
        COALESCE(lc.has_long_context, FALSE) AS has_long_context
    FROM logical_sessions ls
    LEFT JOIN session_titles fp ON ls.session_id = fp.session_id
    LEFT JOIN {TOOL_COUNTS_SUBQUERY} ON ls.session_id = tc.session_id
    LEFT JOIN {FILE_READS_SUBQUERY} ON ls.session_id = fr_agg.session_id
    LEFT JOIN {FILE_WRITES_SUBQUERY} ON ls.session_id = fw_agg.session_id
    LEFT JOIN {COMMAND_LIST_SUBQUERY} ON ls.session_id = cmd.session_id
    LEFT JOIN {SESSION_COST_SUBQUERY} ON ls.session_id = sc.session_id
    LEFT JOIN (
        SELECT session_id, BOOL_OR({LONG_CONTEXT_REQUEST_SQL}) AS has_long_context
        FROM assistant_message_costs
        GROUP BY session_id
    ) lc ON ls.session_id = lc.session_id
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
