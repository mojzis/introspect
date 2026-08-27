"""Shared read-only SQL guard used by the MCP `run_sql` tool and the HTTP API.

Three layers guard the ad-hoc SQL surface, and it matters which one is load
bearing:

1. **The engine configuration** (:func:`introspect.db.connect_read_hardened`)
   is the primary boundary. ``enable_external_access=false`` plus a locked
   configuration is what actually stops ``read_csv('/etc/passwd')``,
   ``glob('/home/**')``, ``COPY ... TO``, ``ATTACH`` and extension loading —
   on a ``read_only=True`` connection all of those otherwise succeed.
2. **This module's validator** is defense in depth. It parses the statement
   with DuckDB's own parser and insists on exactly one ``SELECT``. Its job is
   to turn "the engine would have refused this" into a readable error, and to
   keep catching the obvious cases if the engine config is ever loosened.
3. **Bounded execution** (:func:`execute_bounded`) stops a *legal* query from
   eating the machine: wall clock, row count, byte count, cell width.

See ``docs/security.md`` for the threat model.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

# Hostnames that resolve to loopback but aren't parseable as IPs.
_LOOPBACK_HOSTNAMES = {"localhost"}


def is_loopback_host(host: str) -> bool:
    """True if binding to `host` only accepts local (loopback) connections.

    ``127.0.0.1`` / ``::1`` (and the ``127.0.0.0/8`` range) and ``localhost``
    are loopback; ``0.0.0.0`` / ``::`` / any routable address are not. This is
    the gate for exposing the HTTP SQL API, and also backs the per-request
    ``Origin`` check — a loopback *bind* does not stop a web page from
    pointing at ``http://127.0.0.1:8347``.

    Bracketed IPv6 (``[::1]``, as it appears in a URL authority) is accepted.
    """
    candidate = host.strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if candidate in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # Unknown hostname — fail closed rather than resolving DNS here.
        return False


# Hard cap on rows returned by the MCP `run_sql` tool — kept small because an
# LLM context is the consumer.
MCP_SQL_ROW_CAP = 500
# Hard cap for the local HTTP SQL API — a notebook doing analysis legitimately
# wants more rows than fit in an LLM context, but it's still bounded so a
# runaway query can't stream unbounded results.
API_SQL_ROW_CAP = 10_000
# Per-cell width caps, applied before the byte cap so one pathological column
# can't consume the whole budget. Named separately because the MCP formatter
# has to size its own column clip against this one.
MCP_SQL_CELL_CAP = 200
API_SQL_CELL_CAP = 4_000

# Suffix appended to a value clipped by the per-cell cap.
CELL_TRUNCATION_MARKER = "…[truncated]"


@dataclass(frozen=True)
class SqlBudget:
    """Everything one ad-hoc SQL caller is allowed to spend.

    The five limits always travel together — validation reads
    ``max_sql_bytes``, execution reads the rest — so they are one value
    rather than five parameters threaded through every call site.
    """

    #: Wall clock, seconds. See :func:`resolve_timeout` for the env override.
    timeout_s: float
    #: Rows returned, pushed into the planner as an outer ``LIMIT``.
    row_cap: int
    #: Estimated serialized size of the whole result.
    byte_cap: int
    #: Characters per string cell before :data:`CELL_TRUNCATION_MARKER`.
    cell_cap: int
    #: Length of the SQL text itself, in UTF-8 bytes.
    max_sql_bytes: int
    #: Environment variable that overrides ``timeout_s``.
    timeout_env_var: str


# An LLM context is the consumer: small everything, and a shorter clock than
# the API because a stalled tool call blocks a conversation.
MCP_BUDGET = SqlBudget(
    timeout_s=20.0,
    row_cap=MCP_SQL_ROW_CAP,
    byte_cap=64 * 1024,
    cell_cap=MCP_SQL_CELL_CAP,
    max_sql_bytes=8 * 1024,
    timeout_env_var="INTROSPECT_MCP_SQL_TIMEOUT_SECONDS",
)

# A notebook doing real analysis: room for a genuine result set, still bounded
# so one query can't stream forever or eat the process.
API_BUDGET = SqlBudget(
    timeout_s=30.0,
    row_cap=API_SQL_ROW_CAP,
    byte_cap=8 * 1024 * 1024,
    cell_cap=API_SQL_CELL_CAP,
    max_sql_bytes=32 * 1024,
    timeout_env_var="INTROSPECT_SQL_TIMEOUT_SECONDS",
)


def resolve_timeout(budget: SqlBudget) -> float:
    """``budget.timeout_s``, or its environment override when one is valid.

    A non-numeric or non-positive override falls back to the default rather
    than disabling the timeout — "0 means unlimited" is exactly the footgun
    the timeout exists to prevent, and the timeout is the only thing that
    stops a base-case-less recursive CTE.
    """
    raw = os.environ.get(budget.timeout_env_var, "")
    try:
        value = float(raw)
    except ValueError:
        return budget.timeout_s
    return value if value > 0 else budget.timeout_s


# Rows pulled per ``fetchmany`` while enforcing the byte cap.
_FETCH_BATCH = 1_000
# Flat byte estimate for a scalar cell (numbers, booleans, NULL) once
# :func:`normalize_cell` has run. Every non-scalar is a string by then and is
# measured exactly; this only keeps the aggregate estimate honest for the rest.
_SCALAR_CELL_BYTES = 16

_SQL_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_COMMENT_LINE = re.compile(r"--[^\n]*")
# Only single-quoted strings are SQL literals; double-quoted tokens are
# identifiers and must not be blanked.
_SQL_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

# Belt-and-braces identifier denylist. The hardened engine already refuses
# every one of these (``PermissionException``/``CatalogException``); the
# denylist exists to (a) return a message that says *why* rather than leaking
# a DuckDB internal error, and (b) keep failing loudly if the engine config is
# ever loosened. Matched only where the name is used as a function call, and
# only outside string literals, so a query *searching the logs* for the text
# ``read_csv`` still runs.
_DENIED_FUNCTIONS = (
    "read_csv",
    "read_csv_auto",
    "read_text",
    "read_blob",
    "read_json",
    "read_json_auto",
    "read_json_objects",
    "read_ndjson",
    "read_ndjson_auto",
    "read_ndjson_objects",
    "read_parquet",
    "sniff_csv",
    "glob",
    "parquet_metadata",
    "parquet_schema",
    "parquet_file_metadata",
    "parquet_kv_metadata",
    "parquet_bloom_probe",
    "sqlite_scan",
    "sqlite_attach",
    "postgres_scan",
    "postgres_scan_pushdown",
    "postgres_query",
    "mysql_scan",
    "mysql_query",
    "iceberg_scan",
    "delta_scan",
    "duckdb_secrets",
    "duckdb_extensions",
)
# ``read_json*`` / ``read_ndjson*`` / ``parquet_*`` are enumerated above rather
# than globbed so the error can name the exact function; the trailing wildcard
# below catches variants a future DuckDB adds.
_DENYLIST_RE = re.compile(
    r"\b(" + "|".join(_DENIED_FUNCTIONS) + r"|read_json\w*|read_ndjson\w*|parquet_\w*"
    r")\s*\(",
    re.IGNORECASE,
)


class SqlTimeoutError(Exception):
    """Raised when :func:`execute_bounded` interrupts a query on the clock."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        super().__init__(f"Query timed out after {timeout_s:g}s.")


@dataclass(frozen=True)
class BoundedResult:
    """What :func:`execute_bounded` managed to read within its budgets."""

    columns: list[str]
    #: Cells are already normalized (:func:`normalize_cell`) and clipped, so
    #: every value is ``None``/``bool``/``int``/``float``/``str``.
    rows: list[tuple[Any, ...]]
    truncated: bool
    #: Human-readable cap(s) that shortened the result, ``"; "``-joined, or
    #: None when none did. The row and byte caps stop the read; the cell cap
    #: only shortens values. All of them are reported — a result can hit the
    #: row cap *and* have had wide cells clipped, and the caller wants both.
    truncation_reason: str | None = None


def _strip_noise(sql: str) -> str:
    """Blank comments and string-literal contents for identifier scanning."""
    stripped = _SQL_COMMENT_BLOCK.sub(" ", sql)
    stripped = _SQL_COMMENT_LINE.sub(" ", stripped)
    return _SQL_STRING_LITERAL.sub("''", stripped)


def validate_read_only_sql(
    sql: str, *, max_bytes: int = API_BUDGET.max_sql_bytes
) -> str | None:
    """Return an error message if `sql` isn't a safe read-only query, else None.

    Uses ``duckdb.extract_statements`` — DuckDB's own parser — rather than a
    keyword regex, which gets four things right that hand-rolled scanning got
    wrong:

    * comments and string literals are the parser's problem, not ours, so
      ``SELECT ';' AS x`` and ``SELECT 1 /* ; DROP TABLE t */`` are single
      statements and ``select 'a;b'; drop table t`` is correctly two;
    * DuckDB's FROM-first syntax (``FROM t SELECT a``) parses as a ``SELECT``
      instead of being rejected for starting with the wrong keyword;
    * ``PRAGMA``/``SET``/``INSTALL``/``LOAD``/``ATTACH``/``COPY``/``CALL`` get
      their own statement types, so none of them can hide behind a leading
      comment;
    * a syntax error is reported as a syntax error, before execution.

    ``DESCRIBE``/``SHOW`` also parse as ``SELECT`` and are read-only, so they
    are permitted.
    """
    if len(sql.encode("utf-8")) > max_bytes:
        return f"SQL is too long (limit {max_bytes} bytes)."
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        # ParserException is the expected case; catching the base class keeps
        # any other engine-level parse failure a 400 rather than a 500.
        return f"SQL parse error: {exc}"
    if not statements:
        return "SQL is empty."
    if len(statements) > 1:
        return "Multiple statements are not allowed."
    statement_type = statements[0].type
    if statement_type != duckdb.StatementType.SELECT:
        return f"Only read-only queries are allowed (got: {statement_type.name})."
    return _denied_function(statements[0].query)


def _denied_function(sql: str) -> str | None:
    """Error message naming the first denylisted function call in `sql`."""
    match = _DENYLIST_RE.search(_strip_noise(sql))
    if match is None:
        return None
    return (
        f"Function {match.group(1)!r} is not allowed — it reads outside the "
        "conversation database."
    )


def clamp_row_limit(limit: int, cap: int) -> int:
    """Clamp a caller-supplied row limit to [1, cap]."""
    return max(1, min(limit, cap))


def wrap_with_row_cap(sql: str, capped_limit: int) -> str:
    """Wrap `sql` so the row cap is applied by the planner, not just by fetch.

    The newline before the closing paren is load bearing: without it a query
    ending in a ``--`` comment (``SELECT 1 -- note``) swallowed the paren and
    produced a syntax error.

    Safe to string-interpolate `capped_limit` because callers pass it through
    :func:`clamp_row_limit` (a clamped int) and `sql` has already passed
    :func:`validate_read_only_sql`.
    """
    inner = sql.strip().rstrip(";").strip()
    return f"SELECT * FROM (\n{inner}\n) AS _introspect_q LIMIT {capped_limit}"  # noqa: S608


def normalize_cell(value: Any) -> Any:
    """Coerce a DuckDB cell to the JSON-serializable form callers send on.

    This runs *before* the per-cell and byte caps, and that ordering is the
    whole point. DuckDB hands back ``list`` for LIST, ``dict`` for STRUCT and
    MAP, ``bytes`` for BLOB — none of which have a width the caps could see,
    so ``SELECT list(repeat('x', 1000000)) FROM range(300)`` used to be
    counted as one 16-byte cell and shipped in full. Normalizing first means
    every cell reaching the cell cap is a scalar or a ``str``, so the caps
    apply to what actually goes on the wire.

    ints/floats/bools/None/str pass through unchanged; ``Decimal`` (e.g. cost
    columns) becomes ``float`` so notebooks get a number to compute on;
    everything else DuckDB may hand back (UUID, datetime, date, bytes,
    interval, list, dict) is stringified — lossless enough for analysis and
    always serializable.

    ``nan`` / ``inf`` / ``-inf`` are the exception among floats: JSON has no
    spelling for them, and Starlette renders with ``allow_nan=False``, so
    passing one through raises inside the response constructor — past the
    handler's error handling, i.e. a 500 with no error envelope. They are
    stringified like any other non-JSON value. DuckDB produces them from
    ordinary arithmetic (``1.0 / 0.0``), not just from literals.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _clip_row(row: Sequence[Any], cell_cap: int) -> tuple[tuple[Any, ...], bool]:
    """Normalize and clip one row; the flag says whether a cell was clipped.

    The flag is what lets the result report the cell cap as a truncation
    reason. Unlike the row and byte caps it does not stop the read — the rest
    of the result is still worth returning — but silently handing back a
    shortened value would be the same bug the byte cap exists to prevent.
    """
    clipped = False
    values: list[Any] = []
    for value in row:
        normalized = normalize_cell(value)
        if isinstance(normalized, str) and len(normalized) > cell_cap:
            normalized = normalized[:cell_cap] + CELL_TRUNCATION_MARKER
            clipped = True
        values.append(normalized)
    return tuple(values), clipped


def _row_bytes(row: Sequence[Any]) -> int:
    """Serialized-size estimate, in UTF-8 bytes, for one clipped row.

    Encoding rather than taking ``len`` matters on this corpus: cell caps are
    in characters, and a 4 000-character cell of CJK or emoji is four times
    that on the wire. Counting characters would let an 8 MB budget ship a
    32 MB response. The encode is bounded — cells are clipped to ``cell_cap``
    before they get here.
    """
    return sum(
        len(value.encode("utf-8")) if isinstance(value, str) else _SCALAR_CELL_BYTES
        for value in row
    )


def execute_bounded(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    budget: SqlBudget,
    *,
    timeout_s: float | None = None,
) -> BoundedResult:
    """Run a validated read-only query under wall-clock and size budgets.

    Blocking — run it in a worker thread from async code
    (``await asyncio.to_thread(execute_bounded, ...)``), otherwise a slow
    query stalls the whole event loop: the web UI, the MCP endpoint and the
    background refresh all share it.

    ``conn`` must be the object that will execute the query: ``interrupt()``
    is per connection/cursor, so timing out a parent while a cursor runs does
    nothing. Pass a cursor if that's what you're executing on.

    Budgets, in the order they bite:

    * ``timeout_s`` — a ``threading.Timer`` calls ``conn.interrupt()``, which
      DuckDB surfaces as ``InterruptException``; re-raised as
      :class:`SqlTimeoutError`. This is the only thing that stops
      ``WITH RECURSIVE`` with no base case. Defaults to
      ``resolve_timeout(budget)``; pass it explicitly only to override the
      environment (tests do).
    * ``budget.row_cap`` — pushed into the planner as an outer ``LIMIT`` so
      DuckDB doesn't materialize more than the cap, then re-checked while
      fetching. Callers narrowing it for one request (an explicit ``limit``
      argument) should hand in a ``replace()``d budget.
    * ``budget.cell_cap`` — per-cell clip, applied to the *normalized* cell
      (see :func:`normalize_cell`) and before the byte accounting, so a LIST,
      STRUCT, MAP or BLOB is measured at the width it will be serialized to
      rather than waved through as an opaque object. Clipping shortens values
      without stopping the read, but still marks the result truncated.
    * ``budget.byte_cap`` — aggregate estimate across rows. ``LIMIT`` cannot
      bound this (a single ``string_agg`` row is unbounded), and ``fetchall``
      would build the Python objects outside DuckDB's memory accounting.

    Raises:
        SqlTimeoutError: the wall-clock budget expired.
        duckdb.Error: any other engine-level failure, including
            ``OutOfMemoryException`` when the query exceeds ``memory_limit``.
    """
    row_cap, byte_cap, cell_cap = budget.row_cap, budget.byte_cap, budget.cell_cap
    if timeout_s is None:
        timeout_s = resolve_timeout(budget)
    wrapped = wrap_with_row_cap(sql, row_cap + 1)
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cursor = conn.execute(wrapped)
        columns = [d[0] for d in (cursor.description or [])]
        rows: list[tuple[Any, ...]] = []
        total_bytes = 0
        stopped_by: str | None = None
        cells_clipped = False
        while stopped_by is None:
            batch = cursor.fetchmany(_FETCH_BATCH)
            if not batch:
                break
            for row in batch:
                if len(rows) >= row_cap:
                    stopped_by = f"row cap ({row_cap})"
                    break
                clipped_row, had_clip = _clip_row(row, cell_cap)
                cells_clipped = cells_clipped or had_clip
                total_bytes += _row_bytes(clipped_row)
                rows.append(clipped_row)
                if total_bytes > byte_cap:
                    stopped_by = f"byte cap ({byte_cap} bytes)"
                    break
        caps = [stopped_by] if stopped_by else []
        if cells_clipped:
            caps.append(f"cell cap ({cell_cap} chars)")
        reason = "; ".join(caps) or None
    except duckdb.InterruptException as exc:
        raise SqlTimeoutError(timeout_s) from exc
    finally:
        timer.cancel()
    return BoundedResult(
        columns=columns,
        rows=rows,
        truncated=reason is not None,
        truncation_reason=reason,
    )
