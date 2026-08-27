# Architecture

Introspect reads coding-agent conversation logs (`~/.claude/projects/**/*.jsonl`,
plus Codex rollout logs when present) into DuckDB and exposes them through four
interfaces — a CLI, a web UI, an MCP server, and a local HTTP SQL API — all
built on the same relations.

This page is the map: what each module owns, what each relation contains, and
which invariants the code depends on. For settings see
[Configuration](configuration.md); for the raw records see
[JSONL schema](schema.md).

## Project structure

```
src/introspect/
├── cli.py                  # Typer CLI commands
├── db.py                   # DuckDB schema, materialization, lazy views
├── codex.py                # Codex rollout-log transcoder (Python, not SQL)
├── refresh.py              # Background refresh loop + window picker
├── projects.py             # Git worktree-aware cwd → canonical project
├── pricing.py              # Model pricing (Python rates + SQL CASE)
├── sql_fragments.py        # Shared SQL building blocks (cost / tool / file rollups)
├── query_templates.py      # Curated SQL investigation registry (leaf module)
├── sql_query.py            # Read-only SQL guard shared by run_sql and /api/query
├── search.py               # Full-text search (BM25 / ILIKE fallback)
├── version_check.py        # Once-a-day PyPI update check + stderr nag
├── api/
│   ├── main.py             # FastAPI app, lifespan, middleware, SQL-API gating
│   ├── routes.py           # Route definitions
│   └── handlers/
│       ├── _helpers.py     # Shared utilities (pagination, SQL fragments, templates)
│       ├── dashboard.py    # Landing page
│       ├── sessions.py     # Session list, detail tabs, cost bloat, cache loss
│       ├── tokenscape.py   # Per-turn cost stripes for one session
│       ├── trajectory.py   # Tool-call sequence as a glyph strip
│       ├── subagents.py    # Per-agent cost / file breakdown for one session
│       ├── triggers.py     # First-prompt triggers page
│       ├── search.py       # Search results
│       ├── tools.py        # Tool call stats
│       ├── bash.py         # Bash command stats
│       ├── mcps.py         # MCP tool stats
│       ├── stats.py        # Insights & analytics
│       ├── cost_overview.py    # Pareto / portfolio / binary splits / spend shapes
│       ├── cost_breakdown.py   # Daily & hourly cost charts (nolegend Plotly)
│       ├── query.py        # Local-only JSON SQL API
│       ├── refresh.py      # Manual refresh trigger + status fragment
│       └── raw.py          # Raw JSONL records
├── mcp/
│   ├── server.py           # FastMCP server factory + client instructions
│   ├── _register.py        # Tool / prompt registration and registry wiring
│   ├── tools.py            # MCP tool implementations
│   ├── prompts.py          # MCP prompt seeds for exploratory templates
│   └── refresh_bridge.py   # Module-level handle so MCP tools can reach app.state
└── templates/
    ├── base.html           # Full page layout (HTMX + Alpine.js)
    ├── partial.html        # Fragment-only wrapper for HTMX requests
    ├── dashboard.html, sessions.html, session_detail.html, search.html,
    ├── tools.html, bash.html, mcps.html, stats.html, triggers.html,
    ├── cost_overview.html, raw.html
    └── _*.html             # HTMX partials and macros:
                            #   _refresh_indicator, _daily_cost_panel,
                            #   _hourly_cost_panel, _cost_portfolio_panel,
                            #   _spend_shapes (split/spark macros),
                            #   _session_cost, _session_cost_bloat,
                            #   _session_messages, _session_tokenscape,
                            #   _session_trajectory, _session_subagents
```

## Entry points

### CLI (`cli.py`)

Built with Typer, output formatted with Rich. See the
[CLI reference](usage/cli-reference.md) for every command and option.

`serve` and `devserve` share an internal `_run_web_ui` launcher; `devserve`
adds uvicorn auto-reload and a per-branch dev database. Both probe the
configured port and fall forward to the next free one if it's busy, and export
`INTROSPECT_HOST` so the app knows whether it is loopback-bound. Read commands
call `ensure_materialized()` before opening a read connection, so the CLI
shares the on-disk DB the server builds.

`claude` and `codex` launch a coding agent wired to the HTTP MCP endpoint —
see [MCP server](usage/mcp.md#dedicated-agent-sessions).

### Web UI (`api/main.py`)

A FastAPI application launched via `introspy serve`.

- **Lifespan startup**: opens a writable connection, calls `materialize_views()`
  to load JSONL into DuckDB tables (with indexes), builds the search corpus,
  then closes the writer. Per-request connections are opened read-only against
  the on-disk DB. Materialization always runs on startup — the lazy-view path
  (see [Database layer](#database-layer-dbpy)) is reserved for
  `get_connection()` callers that haven't materialized.
- **SQL API gating**: `_configure_sql_api()` records `app.state.sql_api_enabled`
  from `INTROSPECT_HOST` and `INTROSPECT_SQL_API`. It fails closed — an unset
  host counts as "not known to be loopback".
- **Background refresh** (`refresh.py`): when
  `INTROSPECT_REFRESH_INTERVAL_SECONDS > 0` (default `600`), a task polls JSONL
  mtime and rebuilds into a sidecar file, then atomically `os.replace`s it over
  the live DB. An `asyncio.Event` (`app.state.refresh_trigger`) lets the manual
  "Refresh now" button and the MCP `refresh_data` tool wake the loop early.
- **Window picker**: materialization can be scoped to `1`, `7`, `30` days, or
  `month` (calendar-month-to-date). The choice lives on
  `app.state.refresh_window` and forces a rebuild on the next tick when it
  changes.
- **Middleware**: `db_middleware` opens a fresh per-request read-only connection
  on `request.state.conn`, so in-flight queries are decoupled from the
  background swap.
- **Handler pattern**: each handler queries via `conn(request)`, builds dynamic
  SQL with parameterized filters, paginates (1-based, fetch `size+1` to detect
  the next page), and renders a Jinja2 template. Cost-bearing handlers reuse
  `SESSION_COST_SUBQUERY` from `sql_fragments.py` rather than hand-rolling cost
  math.
- **HTMX integration**: `parent(request)` returns `"base.html"` for full page
  loads or `"partial.html"` for HTMX fragment requests. The refresh indicator,
  cost panels, and drill-downs are HTMX-swapped fragments.
- **Charts**: built server-side as `plotly.graph_objects.Figure` objects styled
  with the [`nolegend`](https://github.com/mojzis/nolegend) Tufte template and
  embedded as JSON for `Plotly.newPlot` to render client-side.

### MCP server (`mcp/server.py`)

Built with FastMCP. `create_mcp_server()` registers tools (`register_tools`) and
prompts (`register_prompts`) and attaches the client-facing `INSTRUCTIONS`
blob — schema orientation for clients that have no other context about the
data. Runs over stdio (`introspy mcp`) or mounted at `/mcp` in the web app; the
HTTP mount is built inside the lifespan and replaces a placeholder `FastAPI()`
so the MCP session manager runs concurrently with request handling.

`mcp/refresh_bridge.py` is a module-level holder that lets stateless MCP tool
functions reach the live `app.state` for `refresh_data`. It enforces
single-app registration to surface accidental multi-app setups.

See [MCP server](usage/mcp.md) for the tool and prompt catalogue.

### SQL API (`api/handlers/query.py`)

`POST /api/query` and `GET /api/schema`, registered unconditionally but 404-ing
when `app.state.sql_api_enabled` is false, so the endpoint isn't advertised on a
publicly bound server. See [SQL API](usage/sql-api.md).

## Database layer (`db.py`)

DuckDB reads JSONL files and exposes them through a fixed schema. Two creation
paths share the same SELECT bodies (`_create_relation` dispatches between TABLE
and VIEW):

- **Materialized tables** (`materialize_views()`): the web UI startup and
  `introspy materialize` build base tables with indexes. The on-disk DB is
  reused by all CLI commands and MCP tools through `ensure_materialized()` /
  `get_read_connection()`.
- **Lazy views** (`_create_views()`): used only when callers reach for a
  connection without materializing first. Created over the JSONL glob with
  `read_json_auto`. The `project_map` table is created empty in this mode so
  joins still resolve.

`materialize_views` records the build timestamp in `materialize_meta`;
`read_last_materialized()` exposes it for the CLI's "Last materialized" banner.
A `DatabaseLockedError` (subclass of `duckdb.IOException`) is raised when
another process holds the write lock, so the CLI can show "another Introspect
is running" instead of a traceback.

JSONL loading falls back to a per-file probe (`_filter_parseable_files`) if the
bulk read fails, so one corrupt file can't take down the whole load. An empty
Claude home is handled by `_create_empty_raw_tables` — schema-shaped stubs keep
downstream queries valid.

### Codex ingestion

When `codex_glob` is given (non-`None`), Codex CLI rollout logs are transcoded
in Python (`codex.py`), inserted one file at a time into `codex_raw_messages`
via `unnest($1::STRUCT(...)[], recursive := true)`, and `UNION ALL BY NAME`-ed
into the Claude `raw_messages` SELECT — one row set spanning both sources,
tagged by `provider` (`anthropic` / `openai`) and `harness` (`claude-code` /
`codex`). A `codex_glob` matching nothing (no `~/.codex/sessions`) is a silent
no-op, mirroring the empty-Claude-home guard. `provider` / `harness` propagate
through `logical_sessions` and `session_stats` via `ANY_VALUE`.

### Relations

The drop-list at the top of `materialize_views()` is the canonical list of
relations; every entry appears below. `search_corpus` is built separately by
`search.build_search_corpus()`.

| Name | Key columns | What it is |
|---|---|---|
| `raw_data` | `filename`, `type`, `timestamp`, `sessionId`, `uuid`, `parentUuid`, `message`, `toolUseResult`, `attachment` | Direct JSONL records with an added `filename`, camelCase field names preserved. Claude only — Codex rows never appear here. The only relation that keeps `type='attachment'` records. |
| `codex_raw_messages` | same shape as `raw_messages` | Transcoded Codex rollout messages. Always created; zero rows when Codex isn't requested or its glob matches nothing. |
| `raw_messages` | `file_path`, `type`, `timestamp`, `session_id`, `uuid`, `parent_uuid`, `is_sidechain`, `cwd`, `role`, `model`, `message`, `tool_use_result`, `provider`, `harness` | User/assistant messages only (`type IN ('user','assistant')`), snake_cased, with `role`/`model` extracted. Claude `UNION ALL` Codex. Every derived relation except `session_context_loads` builds on this. |
| `project_map` | `cwd`, `canonical_path`, `project_name` | `cwd` → canonical project, worktree-aware via `projects.py`. Empty in lazy-view mode. |
| `logical_sessions` | `session_id`, `started_at`, `ended_at`, `duration`, `user_messages`, `assistant_messages`, `model`, `cwd`, `project`, `git_branch`, `entrypoint`, `provider`, `harness` | One row per session: timestamps, duration, message counts, provenance. |
| `assistant_message_costs` | `session_id`, `uuid`, `timestamp`, `is_sidechain`, `model`, `message_id`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `cache_creation_5m`, `cache_creation_1h` | Per-assistant-message token usage, deduplicated by API `message.id` (`raw_messages` can hold duplicate copies of one response). The base for every cost figure. |
| `tool_calls` | `session_id`, `called_at`, `tool_name`, `tool_use_id`, `tool_input`, `is_error`, `tool_use_result`, `result_at`, `execution_time` | Tool invocations joined with their results. `is_error` is the string `'true'`, not a SQL boolean. |
| `session_messages_enriched` | `session_id`, `uuid`, `parent_uuid`, `timestamp`, `block_idx`, `is_sidechain`, `model`, `kind`, `text`, `thinking_text`, `tool_name`, `tool_use_id`, `tool_input` | One row per content block, classified into a `kind` (`agent_text`, `agent_thinking`, `agent_tool_call`, `tool_result`, `slash_command`, `human_prompt`, `subagent_prompt`). Backs the session detail page. |
| `conversation_turns` | `session_id`, `timestamp`, `type`, `role`, `uuid`, `turn_order`, `content_text` | Ordered user/assistant text turns per session. |
| `session_titles` | `session_id`, `first_prompt` | First meaningful user prompt per session (drops `/clear` and command tags). |
| `message_commands` | `session_id`, `uuid`, `timestamp`, `command` | `<command-name>` tags extracted from user messages. |
| `session_context_loads` | `session_id`, `timestamp`, `load_kind`, `name`, `char_len` | Harness-injected context, one row per load: `load_kind` is `claude_md` \| `file_ref` \| `skill_listing` \| `mcp` \| `hook`. Reads `raw_data` attachments directly, since `raw_messages` drops them. Listing/reminder chatter is filtered out. The *root* and global `CLAUDE.md` arrive inline as a first-message `<system-reminder>`, not as attachments, so they are not counted here. |
| `file_reads` | `session_id`, `tool_use_id`, `called_at`, `file_path` | One row per `Read` tool call. |
| `file_writes` | `session_id`, `tool_use_id`, `called_at`, `tool_name`, `file_path` | One row per `Edit` / `Write` / `MultiEdit` / `NotebookEdit` call. |
| `session_stats` | `logical_sessions` columns plus `tool_count`, `files_read`, `files_edited`, `files_read_only`, `files_outside`, `first_prompt`, `commands`, `cost_usd` | The listing-page rollup, and the relation to reach for first. `cost_usd` here is the canonical per-session cost. |
| `search_corpus` | `rowid`, `session_id`, `timestamp`, `role`, `content_text` | Searchable text from user messages, assistant text blocks, tool inputs, and tool results. Built by `build_search_corpus()`, indexed by DuckDB FTS when available. |
| `materialize_meta` | `materialized_at` | Single-row stamp recording the latest `materialize_views` time. |

### Shared SQL fragments (`sql_fragments.py`)

Pure SQL building blocks consumed by `db.py` (materializing `session_stats`),
the FastAPI handlers, and `query_templates.py`:

| Fragment | Purpose |
|---|---|
| `TOOL_COUNTS_SUBQUERY`, `TOOL_COUNTS_WITH_ERRORS_SUBQUERY` | Per-session tool counts. |
| `FILE_READS_SUBQUERY`, `FILE_WRITES_SUBQUERY` | Per-session file activity, including read-only and outside-project counts. |
| `COMMAND_LIST_SUBQUERY` | Slash commands per session, with the `OBVIOUS_COMMANDS` filter for built-ins. |
| `SKILLS_INVOKED_ROLLUP_SQL` | `Skill` tool calls per session — covers both typed `/name` and model-triggered skills. |
| `CONTEXT_LOADS_ROLLUP_SQL` | `session_context_loads` rolled up per session (`auto_loaded_claude_md`, `n_auto_loaded_files`, `skill_menu_loaded`). |
| `CACHE_READ_COST_SQL`, `CACHE_WRITE_COST_SQL`, `OUTPUT_COST_SQL`, `COST_EXPR_SQL` | Per-row cost expressions built from `pricing.py`'s CASE expressions. |
| `SESSION_COST_SUBQUERY`, `session_cost_subquery_filtered(timestamp_where)` | Per-session cost, optionally windowed by timestamp. |

Keeping these in a leaf module avoids inverting the layering — `db.py` would
otherwise have to import from `api.handlers._helpers`. `_helpers.py` re-exports
the names for handler call sites.

## Read-only SQL guard (`sql_query.py`)

`sql_query.py` is the **primary safety boundary** for both the MCP `run_sql`
tool and `POST /api/query` — not the read-only DuckDB connection.

A `read_only=True` connection still permits some side-effecting statements: for
example `COPY ... TO '/file'` can write outside the database. So the real guard
is a validator that strips comments and string literals, rejects anything
containing a `;` (no multi-statement scripts), and requires the first keyword to
be `SELECT` or `WITH`. That single check is what blocks `ATTACH`, `INSTALL`,
`LOAD`, `PRAGMA`, `COPY`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, and
`CALL`. Do not weaken it on the assumption that the connection already
protects you.

Two more pieces live here:

- **Row caps.** `clamp_row_limit()` bounds the caller's limit, and
  `wrap_with_row_cap()` re-wraps the query as `SELECT * FROM (...) LIMIT n` so
  the cap is applied by the planner rather than at fetch time. `run_sql` caps at
  `MCP_SQL_ROW_CAP = 500` (an LLM context is the consumer); the HTTP API caps at
  `API_SQL_ROW_CAP = 10_000` (a notebook legitimately wants more).
- **`is_loopback_host()`.** The gate for exposing the HTTP SQL API. Bound to
  loopback, the OS itself refuses non-local TCP, so no per-request client check
  is needed. Unknown hostnames fail closed rather than triggering DNS
  resolution.

## Query-template registry (`query_templates.py`)

One registry of curated SQL investigations, fanned out to three adapters. The
registry is a leaf module — it imports only `sql_fragments` and stdlib — so
every adapter can depend on it without a cycle.

Each `QueryTemplate` carries a `name`, the `question` it answers, `sql` using
DuckDB `$named` placeholders (so the same string both binds for execution and
reads self-documenting in the cookbook), a tuple of `Param`s, a `note` about
pitfalls, and a `kind`:

| `kind` | Adapter | Behaviour |
|---|---|---|
| `deterministic` | MCP tool | `_register_deterministic_template_tools()` registers a tool per entry. The tool binds params and executes the registry SQL through `mcp.tools.run_query_template()`, so tool and cookbook can't diverge. |
| `exploratory` | MCP prompt | `register_prompts()` registers a prompt per entry. `mcp/prompts.py` renders a seed message from the entry's question, SQL, and note, plus question-specific follow-ups. |
| both | Cookbook | `list_query_templates()` renders every entry, whatever its kind, as a reference for the model to adapt and run through `run_sql`. |

**The shadowing rule.** A hand-registered tool wins over a generated one.
`expensive_sessions` is a `deterministic` template, but its hand-built tool is
richer than a literal-SQL passthrough (Pareto analysis, spend shape, cost
split), so generation skips any name already in `registered_names`. The entry
stays in the registry as cookbook material and as a parity-test fixture.

**Adding a template.** Append a `QueryTemplate` to `QUERY_TEMPLATES`. Then:

- `kind="deterministic"` → add an adapter function to `deterministic_tool_fns`
  in `_register.py` (or hand-register a tool of the same name).
- `kind="exploratory"` → add a prompt function to `mcp/prompts.py` and to
  `exploratory_prompt_fns` in `_register.py`.

`_wire_template_adapters()` raises at registration time in both directions: an
adapter naming a template that doesn't exist, or a template of that kind with
no adapter. So a half-added template fails loudly at server startup rather than
silently registering nothing.

**What the tests enforce.** `tests/test_query_templates.py` checks that every
entry's SQL binds and executes against a fixture DB, that every entry has
sample params (so a new template can't escape coverage), that the cookbook
renders every entry's name and note, that each exploratory prompt's seed
message contains its entry's SQL and note, and that `tool_failure_rate` (the
tool) returns exactly what the registry SQL returns
(`test_tool_failure_rate_parity_with_registry_sql`). Registration itself is the
other half of the guard — see `_wire_template_adapters()` above.
`tests/test_docs_drift.py` checks every template name is mentioned in the docs.

## Pricing (`pricing.py`)

A hardcoded snapshot of published API list prices in USD per 1M tokens, keyed by
model-name **prefix** so dated suffixes (`claude-haiku-4-5-20251001`) match.
Prefixes are sorted longest-first at import so the most specific one wins in
both the Python and the SQL lookup path. Anthropic and OpenAI (Codex
`gpt-5.6-*`) models are both covered. There is no live fetch.

`Rates` is a five-field named tuple:

| Field | Meaning |
|---|---|
| `input` | Uncached input tokens. |
| `cache_write_5m` | Writing the 5-minute ephemeral cache (input × 1.25). |
| `cache_write_1h` | Writing the 1-hour ephemeral cache (input × 2.0). |
| `cache_read` | Reading from cache — the provider's published cached rate, not derived. |
| `output` | Generated tokens. |

**The 5m/1h split.** `assistant_message_costs` extracts
`usage.cache_creation.ephemeral_5m_input_tokens` and
`ephemeral_1h_input_tokens` into `cache_creation_5m` / `cache_creation_1h`.
Older records only carry the flat `cache_creation_input_tokens`; when both split
fields are zero and the total is non-zero, the **fallback** bills the whole
total at the 5-minute rate (`CACHE_WRITE_FALLBACK_SQL` in SQL,
`cache_miss_premium_usd` / `compute_cost_usd` in Python). That's the cheaper of
the two, so a legacy record is never over-billed.

Other exports:

- `rates_for(model)` and `compute_cost_usd(...)` — Python per-row cost.
- `PRICING_INPUT_RATE_SQL`, `PRICING_OUTPUT_RATE_SQL`,
  `PRICING_CACHE_READ_RATE_SQL`, `PRICING_CACHE_WRITE_5M_RATE_SQL`,
  `PRICING_CACHE_WRITE_1H_RATE_SQL` — DuckDB `CASE` expressions, so mixed-model
  sessions price correctly without materializing every message in Python.
  `tests/test_pricing.py` checks the SQL and Python totals agree.
- `CACHE_TTL_SECONDS = 300` — Anthropic's default ephemeral cache TTL, used by
  [cache-loss detection](usage/web-ui.md#cache-loss).
- `cache_miss_premium_usd(...)` — the premium paid for cache-write tokens that
  would have been cache reads on a warm cache.

**Unknown models bill at $0.** `rates_for()` returns zero rates for `None`, the
empty string, `<synthetic>`, and any unrecognized model, logging once per name
at WARNING (LRU-bounded). The SQL `CASE` expressions end in `ELSE 0`. So an
unpriced model silently contributes nothing to cost totals rather than
guessing.

**Known understatements.** Rates are list prices with no request-level
modifiers. Anthropic records `usage.speed == 'fast'` (2× on Opus 5 / Opus 4.8)
and `usage.inference_geo == 'us'` (1.1×); neither is modelled, so a session that
used either is understated by that multiplier. OpenAI's long-context tier (2×
input) isn't recorded in Codex logs at all, so every `gpt-5.6-*` call is billed
at the standard rate.

## Search (`search.py`)

Two ranking strategies:

1. **BM25** (preferred): DuckDB's FTS extension. Availability is detected and
   cached at startup.
2. **ILIKE fallback**: when FTS is unavailable, scores by count of matching
   terms.

`build_search_corpus(conn)` rebuilds `search_corpus` from user messages,
assistant text blocks, tool inputs, and tool results.

## Update check (`version_check.py`)

A once-a-day PyPI lookup that prints a single stderr line when a newer
`introspy` is out. It never blocks the command, never runs under `mcp`, a
non-TTY, or CI, and never self-updates. Behaviour and the exact network call are
documented in [Configuration](configuration.md#update-check).

## Configuration

Every environment variable, its default, and what reads it lives in
[Configuration](configuration.md).

## Testing

Tests live in `tests/` and run with pytest (parallel via `pytest-xdist`,
randomized order via `pytest-randomly`).

| Area | Files | Scope |
|---|---|---|
| Database | `test_db.py` | Relations, materialization, indexes, Codex union, empty-home stubs |
| Search | `test_search.py` | FTS availability, corpus building, BM25 & ILIKE |
| Pricing | `test_pricing.py` | Python ↔ SQL parity, cache split, fallback |
| Codex | `test_codex.py` | Rollout-log transcoding |
| Query templates | `test_query_templates.py` | Registry integrity, param/SQL agreement, tool ↔ registry parity |
| MCP | `test_mcp_tools.py` | Tool implementations and registration |
| CLI | `test_cli.py` | Typer commands, banners, launcher config |
| Update check | `test_version_check.py` | Gating, caching, nag formatting |
| Projects | `test_projects.py` | Git worktree → canonical project resolution |
| Refresh | `test_refresh.py` | Background loop, window changes, manual triggers |
| Web routes | `tests/routes/` (~25 files) | One file per page or feature — filters, pagination, sorting, HTMX fragments, charts, cost math, tokenscape, trajectory, subagents, triggers, SQL API |
| Docs | `test_docs_drift.py` | Commands, env vars, relations, MCP tools/prompts, templates, routes, and nav are all mentioned in the docs |
| End-to-end | `e2e/test_crawl.py`, `e2e/test_flows.py` | Browse every route over real fixture JSONL |

### Fixtures (`conftest.py`)

- `make_user_message()` / `make_assistant_message()` — build realistic JSONL
  records
- `write_jsonl()` — write test data to temp files
- `glob_pattern()` — return the glob for a temp directory
- `_patched_client()` — context manager providing a test client with a patched
  DB (in `tests/routes/conftest.py`)

## Dev tooling

| Tool | Purpose |
|---|---|
| `uv` | Package manager and virtual environment |
| `ruff` | Linting and formatting |
| `ty` | Type checking (beta) |
| `pytest` | Test runner with coverage |
| `vulture` / `deptry` | Dead code and unused dependencies |
| `biston` | Structural clone detection |
| `poethepoet` | Task runner (`poe check`, `poe fix`, `poe test`, `poe check-all`, `poe docs-cli`) |
