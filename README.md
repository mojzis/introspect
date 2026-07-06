# Introspect

Explore and search your Claude Code conversation logs using SQL, full-text search, a web UI, or an MCP server.

Published on PyPI as [`introspy`](https://pypi.org/project/introspy/).

## Installation

You need [uv](https://docs.astral.sh/uv/). If you don't have it (it also takes care of Python for you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Just trying it out?** Run it directly, nothing to install:

```bash
uvx introspy@latest serve
```

Use `introspy@latest`, not a bare `uvx introspy` — `uvx` caches its environment
indefinitely, so the bare form quietly keeps running the first version it
downloaded. `@latest` always fetches the current release.

**Using it regularly?** Install it as a tool, then call it by name:

```bash
uv tool install introspy
introspy serve

# Installed tools stay pinned — pull new releases with:
uv tool upgrade introspy
```

Introspect checks PyPI once a day in the background and prints a one-line hint
to stderr when a newer release is out. It never self-updates, sends nothing but
a plain request for the `introspy` release metadata, and honours
`INTROSPECT_VERSION_CHECK=off`.

## Usage

### Web UI

```bash
introspy serve
# Runs on http://127.0.0.1:8347 by default
introspy serve --port 3000 --host 0.0.0.0
```

### CLI

```bash
# List recent sessions
introspy sessions

# Show summary statistics
introspy stats

# Search conversation logs
introspy search "some query"

# Show tool call history
introspy tools
introspy tools --failed
introspy tools --name Bash

# Run an ad-hoc SQL query
introspy query "SELECT * FROM logical_sessions LIMIT 5"

# Rebuild the search index
introspy refresh
```

### MCP Server

```bash
introspy mcp
```

This starts an MCP server over stdio for integration with Claude Code.

Alternatively, the web server exposes the same MCP tools over HTTP at
`http://127.0.0.1:8347/mcp`. To launch a Claude Code session wired up to it:

```bash
introspy claude
```

`introspy claude` starts `introspy serve` automatically in the background when
nothing is listening on the target port (log at `~/.introspect/serve.log`).
The MCP config is passed inline to `claude`, so the server is only registered
for that session — no changes to your global Claude Code config.

Any arguments after `--` are forwarded verbatim to the `claude` CLI, so you can
pass normal Claude Code options alongside:

```bash
introspy claude -- --model opus --resume
introspy claude -- -p "what are the most expensive sessions"
```

Once connected, try asking Claude:

> what are the most expensive sessions

### SQL API (local notebooks)

When the web server is bound to a loopback address (the default
`127.0.0.1`), it also exposes a small read-only HTTP SQL API so a notebook or
script can query the same materialized DuckDB the CLI and MCP use. It is
**disabled** whenever the server is bound to a non-loopback host (e.g.
`--host 0.0.0.0`), and can be force-disabled even on loopback with
`INTROSPECT_SQL_API=off`.

- `POST /api/query` — body `{"sql": "...", "limit": 100}` → `{"columns": [...],
  "rows": [[...]], "row_count": N, "truncated": bool}`. Only single
  `SELECT` / `WITH` queries are allowed; writes, `ATTACH`, `PRAGMA`, `COPY`,
  and multi-statement scripts are rejected. Rows are capped at 10 000.
- `GET /api/schema` — `{"tables": {name: [{"column", "type"}]}}` for
  discovering views and columns.

```python
import httpx
import pandas as pd

BASE = "http://127.0.0.1:8347"
resp = httpx.post(
    f"{BASE}/api/query",
    json={"sql": "SELECT project, cost_usd FROM session_stats ORDER BY cost_usd DESC", "limit": 20},
)
data = resp.json()
df = pd.DataFrame(data["rows"], columns=data["columns"])
```

## Development

```bash
# Install dependencies (including dev tools)
uv sync

# Auto-format and fix lint issues
uv run poe fix

# Run lint, typecheck, security scan, and tests
uv run poe check

# Run tests only
uv run poe test

# Run all checks including dead-code and unused-deps
uv run poe check-all
```

Tests run in parallel via `pytest-xdist`.

### Code exploration tools

```bash
# Type-aware code search (LSP-quality, by symbol name)
uv run tyf show <name>      # definition + signature + usages
uv run tyf refs <name>      # find all usages
uv run tyf members <Class>  # view class API

# Structural clone detection — spot extraction opportunities
uv run biston scan --suggest .
uv run biston overview .
```

### Worktrees

```bash
# Create ~/worktrees/introspect-<branch> from a fresh origin/main
# (fetches, branches, copies settings, runs uv sync)
uv run poe worktree <branch>
```
