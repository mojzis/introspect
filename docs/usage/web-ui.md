# Web UI

The web UI is a FastAPI application with an HTMX + Alpine.js frontend. It serves
the session dashboard, tool/command stats, cost breakdowns, and search.

```bash
introspy serve
# Runs on http://127.0.0.1:8347 by default
```

Bind a different port or host:

```bash
introspy serve --port 3000 --host 0.0.0.0
```

!!! warning "Binding to a non-loopback host"
    When the server is bound to a non-loopback host (e.g. `--host 0.0.0.0`), the
    local [SQL API](sql-api.md) is automatically disabled. Only bind to a public
    interface on a trusted network.

If the target port is busy, the server falls forward to the next free port.

## What's inside

- **Dashboard** — landing page with headline stats.
- **Sessions** — list, detail view, and a per-session cost-bloat panel.
- **Tools / Bash / MCP** — tool-call statistics, including failures.
- **Insights** — cost overviews (Pareto / portfolio / binary splits) and daily
  and hourly cost charts rendered server-side with Plotly.
- **Search** — full-text search across all message types.
- **Raw** — inspect the underlying JSONL records.

## Refreshing data

A background loop polls the JSONL files and rebuilds the database
automatically (every 10 minutes by default). You can also trigger a rebuild
from the "Refresh now" button in the UI, and scope how much history is loaded
with the window picker (`1` / `7` / `30` days, or the current calendar month).

See [Configuration](../configuration.md) for the environment variables that
control refresh behaviour.
