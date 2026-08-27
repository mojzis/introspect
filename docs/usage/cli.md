# CLI

The CLI is built with [Typer](https://typer.tiangolo.com/) and renders output
as [Rich](https://rich.readthedocs.io/) tables. Read commands materialize the
same on-disk DuckDB the web server builds, so the CLI and UI stay in sync.

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

# Rebuild the search index / materialized tables
introspy refresh
```

## Command reference

| Command | What it does |
|---|---|
| `sessions` | List recent sessions with metadata. |
| `tools` | Tool-call history; filter with `--failed` and `--name`. |
| `stats` | Summary statistics across your logs, ending with a prompt-cache TTL recommendation per project. |
| `cache-ttl` | Would a 1h or 5m prompt-cache TTL have been cheaper? `--verify` shows the simulation's parity residuals; `--subagents` scores sidechain traffic. |
| `search` | Full-text search across conversations. |
| `query` | Run a read-only ad-hoc SQL query against the views. |
| `raw` | Inspect raw JSONL records. |
| `tables` | List the available views/tables. |
| `materialize` | Build the on-disk DuckDB (scope with `-d <days>`). |
| `serve` | Launch the [web UI](web-ui.md). |
| `devserve` | Launch the web UI with uvicorn auto-reload. |
| `mcp` | Start the [MCP server](mcp.md) over stdio. |
| `claude` | Launch a Claude Code session wired to the HTTP MCP endpoint. |
| `codex` | Launch a Codex session wired to the HTTP MCP endpoint. |
| `refresh` | Rebuild the search index / materialized tables. |

Run any command with `--help` for its full set of options.

## Choosing a prompt-cache TTL

`promptCacheTtl` trades two costs against each other: a 5m cache expires
during pauses and pays to rebuild the whole prefix, while 1h keeps it warm
but charges 2x input (against 1.25x) on **every** incremental write, not just
the rebuilds it avoids. A waste figure alone cannot settle it, so
`cache-ttl` replays every request under both policies:

```bash
introspy cache-ttl
```

Read the margin, not just the sign — under about 2% is inside the model's
error and is not a decision. Gaps longer than an hour are reported separately
as breaks: no setting recovers them, so they are not evidence for switching.
Subagents are scored apart (`--subagents`), since they have their own
`subagentPromptCacheTtl`.

Before trusting the numbers, check that simulating the TTL your sessions were
*actually* billed at reproduces their bills:

```bash
introspy cache-ttl --verify
```

A worst residual near 0% means the gap definition is sound. Anything larger
means it is misclassifying warmth, and the recommendation should be ignored
until that is explained.

## Studying real data

`introspy query` is the quickest way to explore the schema against your own
logs:

```bash
introspy query "SELECT project, cost_usd FROM session_stats ORDER BY cost_usd DESC LIMIT 10"
```

`introspy tables` lists every available view. See the
[Architecture](../architecture.md) and [JSONL schema](../schema.md) pages for
what each one contains.
