"""Raw data route handler."""

import json
import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import RAW_PER_PAGE, conn, parent, templates


async def raw_data(
    request: Request,
    page: int,
    session: str,
    record_type: str,
) -> HTMLResponse:
    """Raw unfiltered JSONL records with all fields."""
    db = conn(request)

    where_clauses: list[str] = []
    params: list[str | int] = []
    if session.strip():
        where_clauses.append("sessionId LIKE ?")
        params.append(f"{session.strip()}%")
    if record_type.strip():
        where_clauses.append("type = ?")
        params.append(record_type.strip())

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    total = db.execute(
        f"SELECT COUNT(*) FROM raw_data {where}",  # nosec B608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / RAW_PER_PAGE))
    offset = (page - 1) * RAW_PER_PAGE

    result = db.execute(
        f"SELECT * FROM raw_data {where} LIMIT ? OFFSET ?",  # nosec B608
        [*params, RAW_PER_PAGE, offset],
    )
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()

    # Get distinct types for filter dropdown
    record_types = db.execute(
        "SELECT DISTINCT type FROM raw_data WHERE type IS NOT NULL ORDER BY type"
    ).fetchall()

    # Build records as list of {column, value, is_json, preview} dicts
    preview_len = 100
    records = []
    for row in rows:
        fields = []
        for col, val in zip(columns, row, strict=True):
            if val is None:
                continue
            val_str = str(val).strip()
            # Try to pretty-print JSON objects/arrays
            is_json = False
            if isinstance(val, (dict, list)):
                try:
                    val_str = json.dumps(val, indent=2, ensure_ascii=False)
                    is_json = True
                except (TypeError, ValueError):
                    pass
            elif isinstance(val, str) and val.strip()[:1] in ("{", "["):
                try:
                    parsed = json.loads(val)
                    val_str = json.dumps(parsed, indent=2, ensure_ascii=False)
                    is_json = True
                except (json.JSONDecodeError, ValueError):
                    pass
            long = len(val_str) > preview_len
            preview = val_str[:preview_len] + "..." if long else val_str
            fields.append(
                {
                    "column": col,
                    "value": val_str,
                    "preview": preview,
                    "long": long,
                    "is_json": is_json,
                }
            )
        records.append(fields)

    return templates.TemplateResponse(
        request,
        "raw.html",
        {
            "parent": parent(request),
            "records": records,
            "columns": columns,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "filter_session": session,
            "filter_type": record_type,
            "record_types": [r[0] for r in record_types],
        },
    )
