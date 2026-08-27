"""Local-only HTTP SQL API.

Lets a notebook or script run ad-hoc read-only SQL against the materialized
DuckDB — the HTTP counterpart of the CLI ``query`` command and the MCP
``run_sql`` tool. Gated on the server being bound to a loopback address
(``request.app.state.sql_api_enabled``, set in the app lifespan); when
disabled the routes 404 so the endpoint isn't advertised.

Every guard is shared with the MCP tool via :mod:`introspect.sql_query`: the
statement validator, the row/byte/cell caps and the wall-clock timeout. The
connection itself is hardened in :func:`introspect.db.connect_read_hardened`
— that, not this module, is what stops a query reaching the filesystem.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

import duckdb
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from introspect.api.handlers._helpers import conn
from introspect.db import DEFAULT_DB_PATH, configured_memory_limit
from introspect.sql_query import (
    API_BUDGET,
    API_SQL_ROW_CAP,
    SqlTimeoutError,
    clamp_row_limit,
    execute_bounded,
    validate_read_only_sql,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from introspect.sql_query import BoundedResult

# Default row limit when the caller doesn't specify one. Kept modest so a
# careless ``SELECT * FROM raw_data`` doesn't stream thousands of rows by
# accident; raise it explicitly (up to API_SQL_ROW_CAP) when you mean to.
DEFAULT_LIMIT = 100


class QueryRequest(BaseModel):
    """Body for ``POST /api/query``."""

    sql: str
    limit: int = Field(default=DEFAULT_LIMIT, ge=1)


def _jsonable(value: object) -> object:
    """Coerce a DuckDB cell to a JSON-serializable value.

    ints/floats/bools/None/str pass through unchanged; ``Decimal`` (e.g. cost
    columns) becomes ``float`` so notebooks get a number to compute on;
    everything else DuckDB may hand back (UUID, datetime, date, bytes,
    interval) is stringified — lossless enough for analysis and always
    serializable.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _require_sql_api(request: Request) -> None:
    """Fail closed with 404 unless the local-only SQL API is enabled.

    404 (rather than 403) so a disabled server doesn't advertise that the
    endpoint exists. Enablement is decided once at startup in the app
    lifespan (loopback bind only) and read from ``app.state``.
    """
    if not getattr(request.app.state, "sql_api_enabled", False):
        raise HTTPException(status_code=404, detail="Not found.")


async def run_query(request: Request, body: QueryRequest) -> JSONResponse:
    """Execute a validated read-only SELECT query, returning JSON.

    Response shape::

        {"columns": [...], "rows": [[...], ...],
         "row_count": N, "truncated": bool, "truncation_reason": str | None}

    ``truncated`` is true when a cap stopped the read; ``truncation_reason``
    names which one. Caps come from ``API_BUDGET``: 10 000 rows, 8 MB of
    serialized results, 4 000 characters per cell, 32 KB of SQL text, and a
    30 s wall clock (``INTROSPECT_SQL_TIMEOUT_SECONDS``). Validation, timeout
    and SQL errors all return HTTP 400 with ``{"error": ...}``.

    The query runs in a worker thread: DuckDB's ``execute`` is blocking, and
    on the event loop a slow query would freeze the web UI, the MCP endpoint
    and the background refresh along with it.
    """
    _require_sql_api(request)

    error = validate_read_only_sql(body.sql, max_bytes=API_BUDGET.max_sql_bytes)
    if error:
        return JSONResponse({"error": error}, status_code=400)

    db = conn(request)
    budget = replace(API_BUDGET, row_cap=clamp_row_limit(body.limit, API_SQL_ROW_CAP))
    try:
        result: BoundedResult = await asyncio.to_thread(
            execute_bounded, db, body.sql, budget
        )
    except SqlTimeoutError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except duckdb.OutOfMemoryException as exc:
        db_path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
        limit = configured_memory_limit(db_path)
        return JSONResponse(
            {"error": f"Query exceeded the {limit} memory limit: {exc}"},
            status_code=400,
        )
    except duckdb.Error as exc:
        return JSONResponse(
            {"error": f"SQL error ({type(exc).__name__}): {exc}"},
            status_code=400,
        )

    return JSONResponse(
        {
            "columns": result.columns,
            "rows": [[_jsonable(v) for v in row] for row in result.rows],
            "row_count": len(result.rows),
            "truncated": result.truncated,
            "truncation_reason": result.truncation_reason,
        }
    )


async def schema(request: Request) -> JSONResponse:
    """Return the views/tables and their columns available to ``/api/query``.

    JSON counterpart of the MCP ``describe_schema`` tool — lets a notebook
    discover column names and types before writing a query.
    """
    _require_sql_api(request)

    db = conn(request)
    rows = db.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()

    tables: dict[str, list[dict[str, str]]] = {}
    for table_name, column_name, data_type in rows:
        tables.setdefault(table_name, []).append(
            {"column": column_name, "type": data_type}
        )

    return JSONResponse({"tables": tables})
