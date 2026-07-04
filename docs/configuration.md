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
| `INTROSPECT_SQL_API` | on (loopback only) | Set to `off` to force-disable the local [SQL API](usage/sql-api.md) even on loopback. |

## Refresh behaviour

On startup the web server materializes the JSONL logs into DuckDB tables. When
`INTROSPECT_REFRESH_INTERVAL_SECONDS > 0`, a background task polls the JSONL
file mtimes and rebuilds into a sidecar database, then atomically swaps it over
the live one. The manual "Refresh now" button and the `refresh_data` MCP tool
can wake the loop early.

The **window picker** scopes materialization to `1`, `7`, or `30` days, or the
current calendar month. Changing it forces a rebuild on the next tick.
