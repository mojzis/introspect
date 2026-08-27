# CLI

The CLI is built with [Typer](https://typer.tiangolo.com/) and renders output
as [Rich](https://rich.readthedocs.io/) tables. Read commands materialize the
same on-disk DuckDB the web server builds, so the CLI and UI stay in sync.

Both `introspy` and `introspect` are installed as entry points; they are
interchangeable.

For every option of every command, see the generated
[CLI reference](cli-reference.md).

## Commands

| Command | What it answers / does |
|---|---|
| `sessions` | Recent sessions with timestamps, message counts, model, cwd. |
| `tools` | Tool-call history; `--failed` for errors only, `--name` to filter. |
| `stats` | Summary statistics across your logs. |
| `search` | Full-text search across conversations (BM25, ILIKE fallback). |
| `query` | Ad-hoc read-only SQL against the views. |
| `raw` | Raw unfiltered JSONL records — all fields, nothing dropped. |
| `tables` | List the views and tables `query` can read. |
| `materialize` | Build the on-disk DuckDB; scope with `-d <days>`. |
| `refresh` | Rebuild the search corpus table and FTS index. |
| `serve` | Launch the [web UI](web-ui.md) (and the HTTP MCP endpoint). |
| `devserve` | Web UI with auto-reload and a per-branch dev database. |
| `mcp` | Run the [MCP server](mcp.md) over stdio. |
| `claude` | Launch Claude Code wired to the HTTP MCP endpoint. |
| `codex` | Launch Codex wired to the HTTP MCP endpoint. |

## Typical use

```bash
# List recent sessions
introspy sessions

# Summary statistics
introspy stats

# Full-text search
introspy search "some query"

# Tool-call history
introspy tools --failed
introspy tools --name Bash

# Ad-hoc SQL
introspy query "SELECT * FROM logical_sessions LIMIT 5"

# Rebuild the materialized tables and the search index
introspy refresh
```

## Studying real data

`introspy query` is the quickest way to explore the schema against your own
logs:

```bash
introspy query "SELECT project, cost_usd FROM session_stats ORDER BY cost_usd DESC LIMIT 10"
```

`introspy tables` lists every available relation. See
[Architecture](../architecture.md) for what each one contains and
[JSONL schema](../schema.md) for the raw records underneath.

## What the CLI does not do

- It never writes to your conversation logs. It only ever writes the derived
  DuckDB — read commands query it read-only, but will build it on first use
  if no server has materialized it yet.
- `query` accepts a single `SELECT` / `WITH` statement only; it is not a
  DuckDB shell.
- Only one process can hold the write lock. If a server is already running,
  write commands report that instead of failing with a traceback.
