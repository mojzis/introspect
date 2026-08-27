# Configuration

Introspect is configured through environment variables. Everything the code
reads is listed below; `tests/test_docs_drift.py` fails if `src/` grows an
`INTROSPECT_*` variable that this page doesn't mention.

## Environment variables

| Variable | Default | Read by | Description |
|---|---|---|---|
| `INTROSPECT_DB_PATH` | `~/.introspect/introspect.duckdb` | CLI, web app, update check | Database file location. |
| `INTROSPECT_JSONL_GLOB` | `~/.claude/projects/**/*.jsonl` | web app | Glob for Claude Code conversation logs. |
| `INTROSPECT_CODEX_GLOB` | `~/.codex/sessions/**/*.jsonl` | web app | Glob for Codex CLI rollout logs. A missing directory or non-matching glob is a silent no-op. |
| `INTROSPECT_DAYS` | resolved from `INTROSPECT_REFRESH_WINDOW` | web app | Days of history to load (`0` = no limit). `serve` / `devserve` set it from `-d`; it takes precedence over the window picker on startup. |
| `INTROSPECT_REFRESH_WINDOW` | `30` | web app | Window-picker token: `1`, `7`, `30`, or `month` (calendar-month-to-date). An unrecognized value logs a warning and falls back to the default. |
| `INTROSPECT_REFRESH_INTERVAL_SECONDS` | `600` | web app | Background refresh poll interval; `0` disables auto-refresh. |
| `INTROSPECT_RESOLVE_PROJECTS` | `1` | web app | When `0`, skip git worktree resolution for project names. Set by `serve --no-resolve-projects`. |
| `INTROSPECT_HOST` | unset | set by `serve`, read by the web app | The address the server bound to. Gates the [SQL API](usage/sql-api.md) — see below. |
| `INTROSPECT_SQL_API` | unset (on, loopback only) | web app | Set to `off` to force-disable the SQL API even on loopback. |
| `INTROSPECT_VERSION_CHECK` | unset (on) | update check | Set to `off` (or `0` / `false` / `no`) to disable the [update check](#update-check) — the network call and the nag. |
| `INTROSPECT_VERSION_CHECK_INTERVAL` | `86400` | update check | Seconds between PyPI checks. Escape hatch for testing; a non-numeric or non-positive value falls back to the default. |

`serve` writes `INTROSPECT_DAYS`, `INTROSPECT_HOST`, and (with
`--no-resolve-projects`) `INTROSPECT_RESOLVE_PROJECTS` into the environment
before handing off to uvicorn, so the app's lifespan sees the flags you passed
on the command line.

## SQL API gating

`INTROSPECT_HOST` is how the CLI tells the app which address it bound to. The
app exposes the local [SQL API](usage/sql-api.md) only when that host is
loopback (`127.0.0.0/8`, `::1`, `localhost`) **and** `INTROSPECT_SQL_API` is
not `off`.

The check fails closed: if `INTROSPECT_HOST` is unset — for example when the
app is launched under bare `uvicorn` instead of `introspy serve` — the host is
not *known* to be loopback, so the API stays disabled and `/api/query` and
`/api/schema` return 404.

## Update check

Introspect distributed via `uvx introspy` and `uv tool install introspy` does
not auto-update, so it can quietly tell you when a newer release is out. On
interactive commands (never on `mcp`, never when output is piped or `CI` — or
another CI marker such as `GITHUB_ACTIONS` — is set) it may print a single line
to **stderr**:

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
