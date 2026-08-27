# Introspect

Explore and search your **Claude Code conversation logs** using SQL, full-text
search, a web UI, or an MCP server.

Published on PyPI as [`introspy`](https://pypi.org/project/introspy/).

---

Every Claude Code session is written to a JSONL file under
`~/.claude/projects/**/*.jsonl`. Introspect reads those files into a
[DuckDB](https://duckdb.org/) database and gives you four ways to explore them:

<div class="grid cards" markdown>

-   :material-monitor-dashboard: __Web UI__

    Sessions, tool calls, cost analytics — Pareto, tokenscape, trajectory,
    cache loss — and search.

    [:octicons-arrow-right-24: Web UI](usage/web-ui.md)

-   :material-console: __CLI__

    Typer commands for sessions, stats, search, tool history, and ad-hoc SQL.

    [:octicons-arrow-right-24: CLI](usage/cli.md)

-   :material-robot: __MCP server__

    Expose your logs to Claude Code or Codex — search, inspect, query, and
    seeded investigation prompts, over MCP.

    [:octicons-arrow-right-24: MCP server](usage/mcp.md)

-   :material-database-search: __SQL API__

    A local-only read-only HTTP endpoint so notebooks can query the same DuckDB.

    [:octicons-arrow-right-24: SQL API](usage/sql-api.md)

</div>

## Quick start

Nothing to install — run it directly with [uv](https://docs.astral.sh/uv/):

```bash
uvx introspy serve
```

Then open <http://127.0.0.1:8347>. See [Installation](installation.md) for the
persistent-install path.

## What you can ask

Once the MCP server is connected to Claude Code, you can ask questions like:

> what are the most expensive sessions

> which tools fail most often, by rate rather than count

> show me every session that touched `pricing.py`

> where did the cost run away in session `abc123` — find the tail

> which of my opening prompts started blind: no files named, no skill loaded

> how much did I pay to rebuild caches I let go cold last week

> list the query templates, then adapt the cost one to group by project

## Using this project with an LLM

There is a machine-readable [`llms.txt`](llms.md) index and a full
[`llms-full.txt`](llms.md) dump of this documentation, ready to paste into an
LLM when you want help using or extending Introspect.
