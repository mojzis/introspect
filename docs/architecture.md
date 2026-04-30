# Architecture

Introspect is a tool for exploring Claude Code conversation logs. It provides three interfaces — a CLI, a web UI, and an MCP server — all built on a shared DuckDB database layer that reads `~/.claude/projects/**/*.jsonl` files.

## Project Structure

```
src/introspect/
├── cli.py                  # Typer CLI commands
├── db.py                   # DuckDB schema, materialization, lazy views
├── refresh.py              # Background refresh loop + window picker
├── projects.py             # Git worktree-aware cwd → canonical project
├── pricing.py              # Hardcoded model pricing (Python + SQL CASE)
├── sql_fragments.py        # Shared SQL building blocks (cost / tool / file rollups)
├── search.py               # Full-text search (BM25 / ILIKE fallback)
├── api/
│   ├── main.py             # FastAPI app, lifespan, middleware
│   ├── routes.py           # Route definitions
│   └── handlers/
│       ├── _helpers.py     # Shared utilities (pagination, SQL fragments, templates)
│       ├── dashboard.py    # Landing page
│       ├── sessions.py     # Session list, detail, cost-bloat panel
│       ├── search.py       # Search results
│       ├── tools.py        # Tool call stats
│       ├── bash.py         # Bash command stats
│       ├── mcps.py         # MCP tool stats
│       ├── stats.py        # Insights & analytics
│       ├── cost_overview.py    # Pareto / portfolio / binary splits
│       ├── cost_breakdown.py   # Daily & hourly cost charts (nolegend Plotly)
│       ├── refresh.py      # Manual refresh trigger + status fragment
│       └── raw.py          # Raw JSONL records
├── mcp/
│   ├── server.py           # FastMCP server factory
│   ├── _register.py        # Tool registration
│   ├── tools.py            # MCP tool implementations
│   └── refresh_bridge.py   # Module-level handle so MCP tools can reach app.state
└── templates/
    ├── base.html           # Full page layout (HTMX + Alpine.js)
    ├── partial.html        # Fragment-only wrapper for HTMX requests
    ├── dashboard.html
    ├── sessions.html
    ├── session_detail.html
    ├── search.html
    ├── tools.html
    ├── bash.html
    ├── mcps.html
    ├── stats.html
    ├── cost_overview.html
    ├── raw.html
    └── _*.html             # HTMX partials (_refresh_indicator, _daily_cost_panel,
                            #   _hourly_cost_panel, _cost_portfolio_panel,
                            #   _session_cost, _session_cost_bloat,
                            #   _session_messages)
```

## Entry Points

### CLI (`cli.py`)

Built with Typer. Commands: `sessions`, `tools`, `stats`, `search`, `query`, `raw`, `tables`, `materialize`, `serve`, `devserve`, `mcp`, `refresh`. Output is formatted with Rich tables.

`serve` and `devserve` share an internal `_run_web_ui` launcher; `devserve` adds uvicorn auto-reload. Both probe the configured port and fall forward to the next free one if it's busy. Read commands (`sessions`, `tools`, …) call `ensure_materialized()` before opening the read connection so the CLI shares the same on-disk DB the server builds.

### Web UI (`api/main.py`)

A FastAPI application launched via `introspect serve`. Key aspects:

- **Lifespan startup**: opens a writable connection, calls `materialize_views()` to load JSONL into DuckDB tables (with indexes), builds the search corpus, then closes the writer. Per-request connections are opened read-only against the on-disk DB. Materialization always runs on startup — the lazy-view path (see [Database Layer](#database-layer-dbpy)) is reserved for `get_connection()` callers that haven't materialized.
- **Background refresh** (`refresh.py`): when `INTROSPECT_REFRESH_INTERVAL_SECONDS > 0` (default `600`), a task polls JSONL mtime and rebuilds into a sidecar file, then atomically `os.replace`s it over the live DB. An `asyncio.Event` (`app.state.refresh_trigger`) lets the manual "Refresh now" button and the MCP `refresh_data` tool wake the loop early.
- **Window picker**: the user can scope materialization to `1`, `7`, `30` days, or `month` (calendar-month-to-date). The choice lives on `app.state.refresh_window` and forces a rebuild on the next tick when it changes.
- **Middleware**: `db_middleware` opens a fresh per-request DuckDB read-only connection on `request.state.conn` so in-flight queries are decoupled from the background swap.
- **Routes** (`routes.py`): map URL paths to handler functions in `handlers/`.
- **Handler pattern**: each handler queries via `conn(request)`, builds dynamic SQL with parameterized filters, paginates results (1-based, fetch `size+1` to detect next page), and renders a Jinja2 template. Cost-bearing handlers reuse `SESSION_COST_SUBQUERY` from `sql_fragments.py`.
- **HTMX integration**: `parent(request)` returns `"base.html"` for full page loads or `"partial.html"` for HTMX fragment requests, enabling SPA-like navigation. The refresh indicator and cost panels are HTMX-swapped fragments.
- **Charts**: cost-breakdown views build server-side `plotly.graph_objects.Figure` objects styled with the [`nolegend`](https://github.com/mojzis/nolegend) Tufte template and embed them as JSON for `Plotly.newPlot` to render client-side.
- **Frontend**: Jinja2 templates with HTMX for navigation and Alpine.js for client-side interactivity.

### MCP Server (`mcp/server.py`)

Built with FastMCP. Tools (registered in `_register.py`):

- `search_conversations(query, limit, offset, cwd_prefix, role, since, session_id, require_all)` — full-text search across sessions, with filters and pagination.
- `get_session(session_id)` — fetch full session content.
- `recent_sessions(n)` — list recent sessions with metadata.
- `tool_failures(command_prefix, limit)` — find failed tool calls.
- `run_sql(sql, limit)` — execute a read-only `SELECT` / `WITH` query (multi-statement, `ATTACH`, `PRAGMA`, `COPY`, etc. are rejected by a validator).
- `describe_schema()` — list views/tables and their columns from `information_schema`.
- `refresh_data()` — wake the server's refresh loop and wait for the rebuild (only available when running embedded in `introspect serve`; the stdio MCP returns "unavailable").

Runs over stdio transport (`introspect mcp`) or mounted as an HTTP endpoint within the web app at `/mcp`. The HTTP mount is built dynamically inside the lifespan and replaces a placeholder `FastAPI()` so the MCP session manager runs concurrently with request handling.

`mcp/refresh_bridge.py` is a tiny module-level holder that lets stateless MCP tool functions reach the live `app.state` for `refresh_data`. It enforces single-app registration to surface accidental multi-app setups.

## Database Layer (`db.py`)

DuckDB reads JSONL files and exposes them through a fixed schema. Two creation paths share the same SELECT bodies (`_create_relation` dispatches between TABLE and VIEW):

- **Materialized tables** (`materialize_views()`): the web UI startup and `introspect materialize` build base tables with indexes. The on-disk DB is reused by all CLI commands and MCP tools through `ensure_materialized()` / `get_read_connection()`.
- **Lazy views** (`_create_views()`): used only when callers reach for a connection without materializing first. Created over the JSONL glob with `read_json_auto`. The `project_map` table is created empty in this mode so joins still resolve.

`materialize_views` records the build timestamp in a `materialize_meta` table; `read_last_materialized()` exposes it for the CLI's "Last materialized" banner. A `DatabaseLockedError` (subclass of `duckdb.IOException`) is raised when another process holds the write lock so the CLI can show a friendly "another Introspect is running" message instead of a traceback.

JSONL loading falls back to a per-file probe (`_filter_parseable_files`) if the bulk read fails, so a single corrupt file can't take down the whole load. An empty Claude home is handled by `_create_empty_raw_tables` — schema-shaped stubs keep downstream queries valid.

### Core tables / views

| Name | Description |
|---|---|
| `raw_data` | Direct JSONL records with added `filename` column |
| `raw_messages` | Filtered user/assistant messages with extracted `role` and `model` |
| `project_map` | `cwd` → canonical project path / name (worktree-aware via `projects.py`) |
| `logical_sessions` | Session summaries: timestamps, duration, message counts, model, cwd, project, branch |
| `assistant_message_costs` | Per-assistant-message token usage, deduplicated by API `message.id` (raw_messages can contain duplicate copies of the same response) |
| `tool_calls` | Tool invocations joined with results, including execution time and error status |
| `session_messages_enriched` | One row per content block, classified into a `kind` (`agent_text`, `agent_thinking`, `agent_tool_call`, `tool_result`, `slash_command`, `human_prompt`, `subagent_prompt`) — used by the session detail page |
| `conversation_turns` | Ordered user/assistant text turns per session |
| `session_titles` | First meaningful user prompt per session (filters out `/clear` and command tags) |
| `message_commands` | Extracted `<command-name>` tags from user messages |
| `file_reads` | One row per `Read` tool call with extracted `file_path` |
| `file_writes` | One row per `Edit` / `Write` / `MultiEdit` / `NotebookEdit` call |
| `session_stats` | Listing-page rollup: `logical_sessions` + tool / file / command / cost subqueries |
| `search_corpus` | Searchable text extracted from all message types |
| `materialize_meta` | Single-row stamp recording the latest `materialize_views` time |

### Shared SQL fragments (`sql_fragments.py`)

Pure SQL building blocks consumed by both `db.py` (materializing `session_stats`) and the FastAPI handlers (live queries):

- `TOOL_COUNTS_SUBQUERY`, `TOOL_COUNTS_WITH_ERRORS_SUBQUERY`
- `FILE_READS_SUBQUERY`, `FILE_WRITES_SUBQUERY`
- `COMMAND_LIST_SUBQUERY` (with `OBVIOUS_COMMANDS` filter for built-in slash commands)
- `SESSION_COST_SUBQUERY` and `session_cost_subquery_filtered(timestamp_where)` (built from the per-row pricing CASE expressions in `pricing.py`)

Keeping these in a leaf module avoids inverting the layering — `db.py` would otherwise have to import from `api.handlers._helpers`. `_helpers.py` re-exports the names for backward compatibility with handler call sites.

## Pricing (`pricing.py`)

Hardcoded snapshot of Anthropic API pricing (USD per 1M tokens), keyed by model-name **prefix** so dated suffixes (`claude-haiku-4-5-20251001`) match. Exposes both:

- `rates_for(model)` and `compute_cost_usd(...)` — Python helpers for per-row cost in handlers.
- `PRICING_INPUT_RATE_SQL`, `PRICING_OUTPUT_RATE_SQL`, etc. — DuckDB `CASE` expressions used by the cost subquery so mixed-model sessions sort correctly without materializing every assistant message in Python.

Unknown models log once at WARNING (LRU-bounded) and bill at $0.

## Search (`search.py`)

Two ranking strategies:

1. **BM25** (preferred): DuckDB's FTS extension. Availability is detected and cached at startup.
2. **ILIKE fallback**: When FTS is unavailable, scores by count of matching terms.

The `search_corpus` table is rebuilt by `build_search_corpus(conn)` from user messages, assistant text blocks, tool inputs, and tool results.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `INTROSPECT_DB_PATH` | `~/.introspect/introspect.duckdb` | Database file location |
| `INTROSPECT_JSONL_GLOB` | `~/.claude/projects/**/*.jsonl` | Glob pattern for conversation logs |
| `INTROSPECT_DAYS` | resolved from `INTROSPECT_REFRESH_WINDOW` | Days of history to load (`0` = no limit). Set explicitly by `serve` / `materialize` (`-d`); takes precedence over the window picker on lifespan startup. |
| `INTROSPECT_REFRESH_WINDOW` | `30` | Window picker token: `1`, `7`, `30`, or `month` (calendar-month-to-date) |
| `INTROSPECT_REFRESH_INTERVAL_SECONDS` | `600` | Background refresh poll interval; `0` disables auto-refresh |
| `INTROSPECT_RESOLVE_PROJECTS` | `1` | When `0`, skip git worktree resolution for project names |

## Testing

Tests live in `tests/` and use pytest:

| File | Scope |
|---|---|
| `test_db.py` | Database views, materialization, indexes |
| `test_search.py` | FTS availability, corpus building, BM25 & ILIKE search |
| `test_mcp_tools.py` | MCP tool implementations |
| `test_routes.py` | All web handlers (filters, pagination, sorting, HTMX) |
| `test_cli.py` | Typer commands and banners |
| `test_pricing.py` | Python ↔ SQL pricing parity |
| `test_projects.py` | Git worktree → canonical project resolution |
| `test_refresh.py` | Background refresh loop, window changes, manual triggers |
| `e2e/test_crawl.py`, `e2e/test_flows.py` | End-to-end browse over real fixture JSONL |

### Fixtures (`conftest.py`)

- `make_user_message()` / `make_assistant_message()` — build realistic JSONL records
- `write_jsonl()` — write test data to temp files
- `glob_pattern()` — return glob for temp directory
- `_patched_client()` — context manager providing a test client with patched DB

## Dev Tooling

| Tool | Purpose |
|---|---|
| `uv` | Package manager and virtual environment |
| `ruff` | Linting and formatting |
| `ty` | Type checking (beta) |
| `pytest` | Test runner with coverage |
| `bandit` | Security scanning |
| `poethepoet` | Task runner (`poe check`, `poe fix`, `poe test`, `poe check-all`) |
