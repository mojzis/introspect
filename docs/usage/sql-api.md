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

**Required header:** `X-Introspect-Client: 1`. Without it the request is
refused with 403. It exists to force a CORS preflight that a cross-origin
`fetch` cannot satisfy — see [Security](../security.md#http-boundary). Any
non-browser client (curl, httpx, a notebook) just sets it.

Response:

```json
{
  "columns": ["..."],
  "rows": [["..."]],
  "row_count": 42,
  "truncated": false,
  "truncation_reason": null
}
```

Exactly one `SELECT` statement is allowed — `WITH` and DuckDB's FROM-first
form (`FROM t SELECT …`) count as `SELECT`. Writes, `ATTACH`, `INSTALL`,
`LOAD`, `PRAGMA`, `SET`, `COPY`, multi-statement scripts, and functions that
read outside the database (`read_csv`, `read_text`, `glob`, `sqlite_scan`, …)
are all rejected. The connection itself has filesystem, network and extension
access disabled, so those are refused by the engine too.

**Limits.** The default row limit is 100. A query is bounded at 10 000 rows,
8 MB of total output, 4 000 characters per cell, 32 KB of SQL text, and a 30 s
wall clock. When a cap stops the read, `truncated` is `true` and
`truncation_reason` names which one. A timeout, an out-of-memory, and a SQL
error all return 400 with `{"error": ...}`.

The same validator and the same bounded executor back the MCP `run_sql` tool —
see [the SQL guard](../architecture.md#read-only-sql-guard-sql_querypy) and
[Security](../security.md).

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
    headers={"X-Introspect-Client": "1"},
)
data = resp.json()
df = pd.DataFrame(data["rows"], columns=data["columns"])
```

## What it does not do

- It is not exposed off-machine. Both routes 404 unless the server is bound to
  loopback, and 404 (rather than 403) so a publicly bound server doesn't
  advertise that the API exists at all.
- It is not reachable from a web page. `Host` must be loopback, a non-loopback
  `Origin` is refused, and `POST` needs a header a drive-by `fetch` cannot set.
- It cannot read your filesystem. `enable_external_access` is off and the
  DuckDB configuration is locked, so a `SELECT` cannot reach a file, a URL, or
  an extension.
- It cannot eat the machine. Every query runs under a wall clock and row, byte
  and memory caps, in a worker thread so the rest of the server keeps serving.
- It has no authentication. Any non-browser process that can reach loopback on
  that port can read your conversation logs.
- It cannot write. One `SELECT` statement per request, capped rows, no DDL and
  no `COPY`.

The reasoning behind each of these, and what is left uncovered, is in
[Security](../security.md).
