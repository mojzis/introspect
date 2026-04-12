"""SQL query page handler."""

import time
from collections import defaultdict

import duckdb
from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.sql_validation import validate_read_only_sql

from ._helpers import conn, parent, templates

# Row cap and cell truncation limits
SQL_ROW_CAP = 500
SQL_CELL_MAX = 200


def _get_schema(db: duckdb.DuckDBPyConnection) -> dict[str, list[dict[str, str]]]:
    """Return schema as {table_name: [{name, type}, ...]}."""
    rows = db.execute(
        "SELECT table_name, column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema = 'main' "
        "ORDER BY table_name, ordinal_position"
    ).fetchall()
    schema: dict[str, list[dict[str, str]]] = defaultdict(list)
    for table_name, column_name, data_type in rows:
        schema[table_name].append({"name": column_name, "type": data_type})
    return dict(schema)


async def sql_page(request: Request) -> HTMLResponse:
    """GET /sql — render the SQL query page with editor and schema sidebar."""
    db = conn(request)
    schema = _get_schema(db)
    return templates.TemplateResponse(
        request,
        "sql.html",
        {
            "parent": parent(request),
            "schema": schema,
        },
    )


async def sql_execute(request: Request) -> HTMLResponse:
    """POST /sql — execute a SQL query and return results fragment."""
    db = conn(request)
    form = await request.form()
    sql = str(form.get("sql", "")).strip()

    if not sql:
        return templates.TemplateResponse(
            request,
            "sql_results.html",
            {
                "error": "No SQL provided.",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "exec_time": 0,
            },
        )

    error = validate_read_only_sql(sql)
    if error:
        return templates.TemplateResponse(
            request,
            "sql_results.html",
            {
                "error": error,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "exec_time": 0,
            },
        )

    inner = sql.rstrip(";").strip()
    wrapped = f"SELECT * FROM ({inner}) AS _q LIMIT {SQL_ROW_CAP + 1}"  # noqa: S608

    start = time.monotonic()
    try:
        cursor = db.execute(wrapped)
    except duckdb.Error as exc:
        elapsed = time.monotonic() - start
        return templates.TemplateResponse(
            request,
            "sql_results.html",
            {
                "error": f"SQL error ({type(exc).__name__}): {exc}",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "exec_time": round(elapsed * 1000, 1),
            },
        )

    columns = [d[0] for d in (cursor.description or [])]
    raw_rows = cursor.fetchall()
    elapsed = time.monotonic() - start

    truncated = len(raw_rows) > SQL_ROW_CAP
    if truncated:
        raw_rows = raw_rows[:SQL_ROW_CAP]

    # Truncate cell values
    def fmt_cell(val: object) -> str | None:
        if val is None:
            return None
        text = str(val)
        if len(text) > SQL_CELL_MAX:
            return text[:SQL_CELL_MAX] + "..."
        return text

    rows = [[fmt_cell(v) for v in row] for row in raw_rows]

    return templates.TemplateResponse(
        request,
        "sql_results.html",
        {
            "error": None,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "exec_time": round(elapsed * 1000, 1),
        },
    )
