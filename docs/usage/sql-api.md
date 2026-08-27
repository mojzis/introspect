# SQL API

When the web server is bound to a loopback address (the default `127.0.0.1`),
it also exposes a small read-only HTTP SQL API so a notebook or script can query
the same materialized DuckDB the CLI and MCP use.

!!! danger "Local-only by design"
    The SQL API is **disabled** whenever the server is bound to a non-loopback
    host (e.g. `--host 0.0.0.0`), and can be force-disabled even on loopback with
    `INTROSPECT_SQL_API=off`.

## Endpoints

### `POST /api/query`

Body: `{"sql": "...", "limit": 100}`

Response:

```json
{
  "columns": ["..."],
  "rows": [["..."]],
  "row_count": 42,
  "truncated": false
}
```

Only single `SELECT` / `WITH` queries are allowed; writes, `ATTACH`, `INSTALL`,
`LOAD`, `PRAGMA`, `COPY`, and multi-statement scripts are rejected. The default
limit is 100; rows are capped at 10 000. The same validator backs the MCP
`run_sql` tool — see
[the SQL guard](../architecture.md#read-only-sql-guard-sql_querypy).

### `GET /api/schema`

Returns `{"tables": {name: [{"column", "type"}]}}` for discovering views and
columns.

## Example

```python
import httpx
import pandas as pd

BASE = "http://127.0.0.1:8347"
resp = httpx.post(
    f"{BASE}/api/query",
    json={
        "sql": "SELECT project, cost_usd FROM session_stats ORDER BY cost_usd DESC",
        "limit": 20,
    },
)
data = resp.json()
df = pd.DataFrame(data["rows"], columns=data["columns"])
```

## What it does not do

- It is not exposed off-machine. Both routes 404 unless the server is bound to
  loopback, and 404 (rather than 403) so a publicly bound server doesn't
  advertise that the API exists at all.
- It has no authentication. Anything that can reach loopback on that port can
  read your conversation logs.
- It cannot write. One `SELECT` / `WITH` statement per request, capped rows, no
  DDL and no `COPY`.
