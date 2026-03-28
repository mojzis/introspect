# Architecture

Introspect is a tool for exploring Claude Code conversation logs. It provides three interfaces — a CLI, a web UI, and an MCP server — all built on a shared DuckDB database layer that reads `~/.claude/projects/**/*.jsonl` files.

## Project Structure

```
src/introspect/
├── cli.py                  # Typer CLI commands
├── db.py                   # DuckDB views & materialization
├── search.py               # Full-text search (BM25 / ILIKE fallback)
├── api/
│   ├── main.py             # FastAPI app, lifespan, middleware
│   ├── routes.py           # Route definitions
│   └── handlers/
│       ├── _helpers.py     # Shared utilities (pagination, SQL fragments, templates)
│       ├── dashboard.py    # Dashboard handler
│       ├── sessions.py     # Session list & detail
│       ├── search.py       # Search results
│       ├── tools.py        # Tool call stats
│       ├── mcps.py         # MCP tool stats
│       ├── stats.py        # Insights & analytics
│       └── raw.py          # Raw JSONL records
├── mcp/
│   ├── server.py           # FastMCP server factory
│   ├── _register.py        # Tool registration
│   └── tools.py            # MCP tool implementations
└── templates/
    ├── base.html           # Full page layout (HTMX + Alpine.js)
    ├── partial.html        # Fragment-only wrapper for HTMX requests
    ├── dashboard.html
    ├── sessions.html
    ├── session_detail.html
    ├── search.html
    ├── tools.html
    ├── mcps.html
    ├── stats.html
    └── raw.html
```

## Entry Points

### CLI (`cli.py`)

Built with Typer. Commands include `sessions`, `tools`, `stats`, `search`, `query`, `raw`, `tables`, `materialize`, `serve`, `mcp`, and `refresh`. Output is formatted with Rich tables.

### Web UI (`api/main.py`)

A FastAPI application launched via `introspect serve`. Key aspects:

- **Lifespan**: On startup, materializes JSONL data into DuckDB, builds the search corpus, creates a shared read-only connection, and mounts the MCP server at `/mcp`.
- **Middleware**: `db_middleware` attaches a per-request DuckDB cursor to `request.state.conn`.
- **Routes** (`routes.py`): Maps URL paths to handler functions in `handlers/`.
- **Handler pattern**: Each handler queries the database via `conn(request)`, builds dynamic SQL with parameterized filters, paginates results (1-based, fetch `size+1` to detect next page), and renders a Jinja2 template.
- **HTMX integration**: `parent(request)` returns `"base.html"` for full page loads or `"partial.html"` for HTMX fragment requests, enabling SPA-like navigation.
- **Frontend**: Jinja2 templates with HTMX for navigation and Alpine.js for client-side interactivity.

### MCP Server (`mcp/server.py`)

Built with FastMCP. Exposes tools for Claude Code integration:

- `search_conversations(query, limit)` — full-text search across sessions
- `get_session(session_id)` — fetch full session content
- `recent_sessions(n)` — list recent sessions with metadata
- `tool_failures(command_prefix, limit)` — find failed tool calls

Runs over stdio transport (`introspect mcp`) or mounted as an HTTP endpoint within the web app at `/mcp`.

## Database Layer (`db.py`)

DuckDB reads JSONL files and exposes them through SQL views. Two modes:

- **Lazy views**: Created on-the-fly over raw JSONL files (no persistence, used for read-only access).
- **Materialized tables**: `materialize_views()` loads JSONL into DuckDB tables with indexes for faster queries. Controlled by `INTROSPECT_DAYS` (default: 10 days, 0 = all).

### Core Views

| View | Description |
|---|---|
| `raw_data` | Direct JSONL records with added `filename` column |
| `raw_messages` | Filtered user/assistant messages with extracted `role` and `model` |
| `logical_sessions` | Session summaries: timestamps, duration, message counts, model, cwd, branch |
| `tool_calls` | Tool invocations joined with results, including execution time and error status |
| `conversation_turns` | Ordered user/assistant message pairs per session |
| `session_titles` | First meaningful user prompt per session (filters out `/clear` and system messages) |
| `message_commands` | Extracted `<command-name>` tags from user messages |
| `search_corpus` | Searchable text extracted from all message types |

## Search (`search.py`)

Two ranking strategies:

1. **BM25** (preferred): Uses DuckDB's FTS extension for probabilistic relevance ranking. Availability is detected and cached at startup.
2. **ILIKE fallback**: When FTS is unavailable, searches with `ILIKE` and scores by count of matching terms.

The search corpus is materialized from `search_corpus` view, extracting text from user messages, assistant text blocks, tool use inputs, and tool results.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `INTROSPECT_DB_PATH` | `~/.introspect/introspect.duckdb` | Database file location |
| `INTROSPECT_JSONL_GLOB` | `~/.claude/projects/**/*.jsonl` | Glob pattern for conversation logs |
| `INTROSPECT_DAYS` | `10` | Days of history to load (0 = all) |

## Testing

Tests live in `tests/` and use pytest:

| File | Scope |
|---|---|
| `test_db.py` | Database views, materialization, indexes |
| `test_search.py` | FTS availability, corpus building, BM25 & ILIKE search |
| `test_mcp_tools.py` | MCP tool implementations |
| `test_routes.py` | All web handlers (filters, pagination, sorting, HTMX) |

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
