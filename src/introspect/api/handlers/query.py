"""Local-only HTTP SQL API.

Lets a notebook or script run ad-hoc read-only SQL against the materialized
DuckDB — the HTTP counterpart of the CLI ``query`` command and the MCP
``run_sql`` tool. Gated on the server being bound to a loopback address
(``request.app.state.sql_api_enabled``, set in the app lifespan); when
disabled the routes 404 so the endpoint isn't advertised.

The SELECT-only guard, row-cap wrapping, and loopback check are shared with
the MCP tool via :mod:`introspect.sql_query`.
"""

from __future__ import annotations

from decimal import Decimal

import duckdb
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from introspect.api.handlers._helpers import conn
from introspect.sql_query import (
    API_SQL_ROW_CAP,
    clamp_row_limit,
    validate_read_only_sql,
    wrap_with_row_cap,
)

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
    """Execute a validated read-only SELECT / WITH query, returning JSON.

    Response shape::

        {"columns": [...], "rows": [[...], ...],
         "row_count": N, "truncated": bool}

    ``truncated`` is true when the result hit the row cap and more rows may
    exist. Validation and SQL errors return HTTP 400 with ``{"error": ...}``.
    """
    _require_sql_api(request)

    error = validate_read_only_sql(body.sql)
    if error:
        return JSONResponse({"error": error}, status_code=400)

    capped_limit = clamp_row_limit(body.limit, API_SQL_ROW_CAP)
    # Fetch one extra row to detect whether the cap actually truncated results.
    wrapped = wrap_with_row_cap(body.sql, capped_limit + 1)

    db = conn(request)
    try:
        cursor = db.execute(wrapped)
    except duckdb.Error as exc:
        return JSONResponse(
            {"error": f"SQL error ({type(exc).__name__}): {exc}"},
            status_code=400,
        )

    columns = [d[0] for d in (cursor.description or [])]
    rows = cursor.fetchall()
    truncated = len(rows) > capped_limit
    if truncated:
        rows = rows[:capped_limit]

    return JSONResponse(
        {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
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
