# Configuration

Introspect is configured through environment variables.

| Variable | Default | Description |
|---|---|---|
| `INTROSPECT_DB_PATH` | `~/.introspect/introspect.duckdb` | Database file location. |
| `INTROSPECT_JSONL_GLOB` | `~/.claude/projects/**/*.jsonl` | Glob pattern for conversation logs. |
| `INTROSPECT_DAYS` | resolved from `INTROSPECT_REFRESH_WINDOW` | Days of history to load (`0` = no limit). Set explicitly by `serve` / `materialize` (`-d`); takes precedence over the window picker on lifespan startup. |
| `INTROSPECT_REFRESH_WINDOW` | `30` | Window picker token: `1`, `7`, `30`, or `month` (calendar-month-to-date). |
| `INTROSPECT_REFRESH_INTERVAL_SECONDS` | `600` | Background refresh poll interval; `0` disables auto-refresh. |
| `INTROSPECT_RESOLVE_PROJECTS` | `1` | When `0`, skip git worktree resolution for project names. |
| `INTROSPECT_MAX_OBJECT_SIZE_MB` | largest JSONL file + 1MB, clamped to 32–512MB | Override DuckDB's `maximum_object_size` (per-line JSON buffer) during materialization. Lower it on memory-constrained machines (e.g. WSL). |
| `INTROSPECT_THREADS` | derived from available RAM ÷ per-thread buffer, capped at CPU count | Override the DuckDB thread count used during materialization. Fewer threads means less peak memory. |
| `INTROSPECT_SQL_API` | on (loopback only) | Set to `off` to force-disable the local [SQL API](usage/sql-api.md) even on loopback. |
| `INTROSPECT_VERSION_CHECK` | `on` | Set to `off` to disable the [update check](#update-check) — the network call and the nag. |

## Update check

Introspect distributed via `uvx introspy` and `uv tool install introspy` does
not auto-update, so it can quietly tell you when a newer release is out. On
interactive commands (never on `mcp`, never when output is piped or `CI` is
set) it may print a single line to **stderr**:

```
introspy 0.3.0 is available (you have 0.2.3) — run: uvx introspy@latest  |  uv tool upgrade introspy
```

**What leaves your machine.** To learn the latest version, a short-lived
background thread makes one plain `GET` to the public PyPI JSON endpoint
(`https://pypi.org/pypi/introspy/json`). That request carries **no query
parameters, no identifiers, and no telemetry** — the only thing PyPI/Fastly can
infer is that some machine asked for the `introspy` release metadata. Your
conversation logs never leave your machine; this is the sole network call
Introspect ever makes, and it is informational only — Introspect never
self-updates. The result is cached under `~/.introspect/version_check.json` and
re-checked at most once a day. Set `INTROSPECT_VERSION_CHECK=off` to disable the
check, the network call, and the nag entirely.

## Refresh behaviour

On startup the web server materializes the JSONL logs into DuckDB tables. When
`INTROSPECT_REFRESH_INTERVAL_SECONDS > 0`, a background task polls the JSONL
file mtimes and rebuilds into a sidecar database, then atomically swaps it over
the live one. The manual "Refresh now" button and the `refresh_data` MCP tool
can wake the loop early.

The **window picker** scopes materialization to `1`, `7`, or `30` days, or the
current calendar month. Changing it forces a rebuild on the next tick.
