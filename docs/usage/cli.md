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
| `stats` | Summary statistics across your logs, ending with a prompt-cache TTL recommendation per project. |
| `cache-ttl` | Would a 1h or 5m prompt-cache TTL have been cheaper? `--verify` runs the simulation's parity gate; `--subagents` scores sidechain traffic separately. |
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

# Would a 1h or 5m prompt-cache TTL have been cheaper?
introspy cache-ttl

# Rebuild the materialized tables and the search index
introspy refresh
```

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
