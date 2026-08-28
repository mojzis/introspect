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
├── db.py                   # DuckDB schema, materialization, lazy views, hardened reads
├── codex.py                # Codex rollout-log transcoder (Python, not SQL)
├── refresh.py              # Background refresh loop + window picker
├── projects.py             # Git worktree-aware cwd → canonical project
├── pricing.py              # Model pricing (Python rates + SQL CASE)
├── sql_fragments.py        # Shared SQL building blocks (cost / tool / file rollups)
├── query_templates.py      # Curated SQL investigation registry (leaf module)
├── sql_query.py            # SQL validator + bounded executor for run_sql and /api/query
├── search.py               # Full-text search (BM25 / ILIKE fallback)
├── version_check.py        # Once-a-day PyPI update check + stderr nag
├── api/
│   ├── main.py             # FastAPI app, lifespan, middleware, SQL-API gating
│   ├── routes.py           # Route definitions
│   ├── errors.py           # Exception handlers: the one error-response policy
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
    ├── error.html          # Full-page error (non-HTMX requests)
    └── _*.html             # HTMX partials and macros:
                            #   _error (error fragment appended to #errors),
                            #   _refresh_indicator, _daily_cost_panel,
                            #   _hourly_cost_panel, _cost_portfolio_panel,
                            #   _spend_shapes (split/spark macros),
                            #   _pagination (shared Prev/Next + size picker),
                            #   _session_cost, _session_cost_bloat,
                            #   _session_messages, _session_tokenscape,
                            #   _session_trajectory, _session_subagents
```

## Entry points

### CLI (`cli.py`)

Built with Typer, output formatted with Rich. See the
[CLI reference](usage/cli-reference.md) for every command and option.

`serve` and `devserve` share an internal `_run_web_ui` launcher; `devserve`
adds uvicorn auto-reload, a per-branch dev database, and `INTROSPECT_DEBUG=1`
so failures render their traceback in the browser. Both probe the
configured port and fall forward to the next free one if it's busy, and export
`INTROSPECT_HOST` so the app knows whether it is loopback-bound. Read commands
call `ensure_materialized()` before opening a read connection, so the CLI
shares the on-disk DB the server builds.

`claude` and `codex` launch a coding agent wired to the HTTP MCP endpoint —
see [MCP server](usage/mcp.md#dedicated-agent-sessions).

`mcp` installs its own SIGINT handler *before* starting the stdio server, for
two reasons. asyncio's runner claims SIGINT only while the current handler is
still `signal.default_int_handler`, so registering first is what keeps ours in
force; and the transport parks its stdin read on a worker thread, so a
`KeyboardInterrupt` raised in the event loop cannot cancel it — the process
would hang until the client sends another byte, then unwind as an anyio
exception group. The handler therefore writes its line with a raw `os.write`
(a signal can land mid-write on stderr's buffer, and re-entering that buffer
raises) and ends the process with `os._exit`. The `except` guard around
`server.run` is a backstop for interrupts arriving outside that window.

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
- **Middleware**, three of them: `db_middleware` opens a fresh per-request
  *hardened* read connection (`connect_read_hardened`) on
  `request.state.conn`, so in-flight queries are decoupled from the background
  swap; `host_guard` rejects a `Host` outside the loopback allowlist, but only
  when the bind is itself loopback (`host_allowlist_applies`); `local_api_guard`
  rejects a non-loopback `Origin` on `/api/query`, `/api/schema` and `/mcp`, and
  requires an `X-Introspect-Client` header on `POST /api/query`. `CORSMiddleware`
  is deliberately absent — see [Security](security.md#http-boundary).
- **Handler pattern**: each handler queries via `conn(request)`, builds dynamic
  SQL with parameterized filters, paginates (1-based, fetch `size+1` to detect
  the next page), and renders a Jinja2 template. Cost-bearing handlers reuse
  `SESSION_COST_SUBQUERY` (per-session) or `COST_EXPR_SQL` (per-row) from
  `sql_fragments.py` rather than hand-rolling cost math.
- **HTMX integration**: `parent(request)` returns `"base.html"` for full page
  loads or `"partial.html"` for HTMX fragment requests — both off the single
  `is_htmx(request)` header test. The refresh indicator, cost panels, and
  drill-downs are HTMX-swapped fragments.
- **Error responses** (`api/errors.py`): three exception handlers give every
  failure a visible, honest representation. See
  [Error handling](usage/web-ui.md#error-handling).
- **Charts**: built server-side as `plotly.graph_objects.Figure` objects styled
  with the [`nolegend`](https://github.com/mojzis/nolegend) Tufte template and
  embedded as JSON for `Plotly.newPlot` to render client-side.

### MCP server (`mcp/server.py`)

Built with FastMCP. `create_mcp_server(bind_host)` registers tools
(`register_tools`) and prompts (`register_prompts`) and attaches the
client-facing `INSTRUCTIONS` blob — schema orientation for clients that have no
other context about the data. `bind_host` selects the transport's
`TransportSecuritySettings`: a loopback bind (and stdio, which passes the
default) gets the SDK's DNS-rebinding protection with loopback host and origin
allowlists; a deliberate non-loopback bind turns it off, mirroring
`api.main.host_allowlist_applies`. Runs over stdio (`introspy mcp`) or mounted at `/mcp` in the web app; the
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

`session_meta` also carries Codex subagent paths and generated nicknames. For
approval-review sidecars, the recorded user message is an agent-history
envelope; `codex_session_metadata` extracts its original first user request for
the display title rather than using the envelope boilerplate.

### Relations

The drop-list at the top of `materialize_views()` is the canonical list of
relations; every entry appears below. `search_corpus` is built separately by
`search.build_search_corpus()`.

| Name | Key columns | What it is |
|---|---|---|
| `raw_data` | `filename`, `type`, `timestamp`, `sessionId`, `uuid`, `parentUuid`, `message`, `toolUseResult`, `attachment` | Direct JSONL records with an added `filename`, camelCase field names preserved. Claude only — Codex rows never appear here. The only relation that keeps `type='attachment'` records. |
| `codex_raw_messages` | same shape as `raw_messages` | Transcoded Codex rollout messages. Native response-item IDs are deduplicated per logical session so parent-transcript replays do not repeat; rows without a native ID (including synthesized enrichment) are preserved. Always created; zero rows when Codex isn't requested or its glob matches nothing. |
| `codex_session_metadata` | `session_id`, `title`, `agent_path`, `agent_nickname`, `parent_thread_id` | One row per Codex logical session carrying session-meta naming/context. `title` is the original human request extracted from an approval-review envelope when available. |
| `raw_messages` | `file_path`, `type`, `timestamp`, `session_id`, `uuid`, `parent_uuid`, `is_sidechain`, `cwd`, `role`, `model`, `message`, `tool_use_result`, `provider`, `harness` | User/assistant messages only (`type IN ('user','assistant')`), snake_cased, with `role`/`model` extracted. Claude `UNION ALL` Codex. Every derived relation except `session_context_loads` builds on this. |
| `project_map` | `cwd`, `canonical_path`, `project_name` | `cwd` → canonical project, worktree-aware via `projects.py`. Empty in lazy-view mode. |
| `logical_sessions` | `session_id`, `started_at`, `ended_at`, `duration`, `user_messages`, `assistant_messages`, `model`, `cwd`, `project`, `git_branch`, `entrypoint`, `provider`, `harness` | One row per session: timestamps, duration, message counts, provenance. |
| `assistant_message_costs` | `session_id`, `uuid`, `timestamp`, `is_sidechain`, `model`, `message_id`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `cache_creation_5m`, `cache_creation_1h`, `ttl_observed` | Per-assistant-message token usage, deduplicated by API `message.id` (`raw_messages` can hold duplicate copies of one response). The base for every cost figure. `ttl_observed` (`5m` / `1h` / `mixed` / `unknown`) records which prompt-cache TTL the row was actually billed at. |
| `session_stats` | session metadata, activity counts, `cost_usd`, `has_long_context` | Materialized summary for the session list and quick lookups. `has_long_context` marks a session containing a `gpt-5.6-*` request above 272K input tokens. |
| `cache_requests` | `session_id`, `uuid`, `message_id`, `timestamp`, `trigger_ts`, `response_end_ts`, `is_sidechain`, `model`, `seq`, `prefix_total`, `prev_prefix_total`, `common_prefix`, `gap_seconds`, `gap_bucket`, `ttl_observed`, `structural_invalidation`, `prefix_shrank`, `cache_miss`, `gap_recoverable`, `gap_unrecoverable`, `miss_premium_usd`, `warm_5m`, `warm_1h`, `cost_5m_usd`, `cost_1h_usd`, `cost_observed_usd` | One row per API request, chain-ordered by timestamp within `(session_id, is_sidechain)`. The **only** definition of a prompt-cache break — the session divider, the tokenscape event track and the cost-overview panel all read it — plus both TTL policies replayed over the same requests. See [Cache TTL](#cache-ttl-cache_ttlpy). |
| `session_cache_ttl` | `session_id`, `is_sidechain`, `cost_5m`, `cost_1h`, `delta`, `cost_observed`, `n_requests`, `n_gaps_recoverable`, `n_gaps_unrecoverable`, `n_structural`, `ttl_observed_dominant`, `recoverable_waste_usd`, `unrecoverable_break_usd`, `recoverable_prefix_tokens` | Per-session rollup of `cache_requests`, for ad-hoc SQL. Diagnostics only: `promptCacheTtl` is set per user/project, so act on the project/global rollups, not on this. |
| `tool_calls` | `session_id`, `called_at`, `tool_name`, `tool_use_id`, `tool_input`, `is_error`, `tool_use_result`, `result_at`, `execution_time` | Tool invocations joined with their results. `is_error` is the string `'true'`, not a SQL boolean. |
| `session_messages_enriched` | `session_id`, `uuid`, `parent_uuid`, `timestamp`, `block_idx`, `is_sidechain`, `model`, `kind`, `text`, `thinking_text`, `tool_name`, `tool_use_id`, `tool_input` | One row per content block, classified into a `kind` (`agent_text`, `agent_thinking`, `agent_tool_call`, `tool_result`, `slash_command`, `human_prompt`, `subagent_prompt`). Backs the session detail page. |
| `conversation_turns` | `session_id`, `timestamp`, `type`, `role`, `uuid`, `turn_order`, `content_text` | Ordered user/assistant text turns per session. |
| `session_titles` | `session_id`, `first_prompt` | Display title per session: the extracted original request for a Codex approval sidecar when available; otherwise the first meaningful main-conversation user prompt (dropping `/clear`, command tags, context wrappers, and subagent boilerplate). Codex agent path/nickname is a final fallback. |
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
| `INPUT_COST_SQL`, `OUTPUT_COST_SQL`, `CACHE_READ_COST_SQL`, `CACHE_WRITE_COST_SQL`, `COST_EXPR_SQL` | Per-row cost expressions built from `pricing.py`'s CASE expressions — one per component, plus their sum. `CACHE_WRITE_FALLBACK_SQL` supplies the legacy-record fallback the last two apply. |
| `SESSION_COST_SUBQUERY`, `session_cost_subquery_filtered(timestamp_where)` | Per-session cost, optionally windowed by timestamp. |

Keeping these in a leaf module avoids inverting the layering — `db.py` would
otherwise have to import from `api.handlers._helpers`. `_helpers.py` re-exports
the names for handler call sites.

## Read-only SQL guard (`sql_query.py`)

Both ad-hoc SQL surfaces — the MCP `run_sql` tool and `POST /api/query` — go
through the same three layers, and it matters which one is load bearing.
[Security](security.md) has the threat model; this is the code map.

**The engine configuration is the boundary.**
`db.connect_read_hardened()` is the only place the codebase opens a read-only
connection to the main DB. It opens with resource caps, loads FTS, then sets
`enable_external_access = false` (plus the extension and secret toggles) and
`lock_configuration = true`. A `read_only=True` connection on its own permits
plenty: `read_csv('/etc/passwd')`, `glob('/home/**')`, `COPY ... TO '/file'`,
`ATTACH`. The locked configuration is what stops them.

Two ordering constraints, both verified on DuckDB 1.5.3 and documented in the
factory's docstring: `LOAD fts` must precede the external-access disable, and
the factory must skip its SETs on an already-locked instance. Settings are
instance-global, and DuckDB refuses a second connection to a file whose
instance was opened with a different config — which is why one factory serves
the API middleware, `run_sql` and `describe_schema` alike.

**`sql_query.validate_read_only_sql()` is defense in depth.** It parses with
`duckdb.extract_statements()` and requires exactly one `SELECT` statement,
which correctly handles comments, string literals, `WITH`, `WITH RECURSIVE`
and DuckDB's FROM-first syntax without hand-rolled scanning. A denylist of
file- and network-reading functions rides along so callers get a readable
error instead of a `PermissionException`, and so a loosened engine config
fails loudly.

**`sql_query.execute_bounded()` bounds resources.** One shared executor for
both callers, taking a frozen `SqlBudget` (`MCP_BUDGET` / `API_BUDGET`) rather
than five loose limits: an outer `LIMIT` for rows, `fetchmany` batching for a
byte cap, per-cell clipping, and a `threading.Timer` calling
`conn.interrupt()` for wall clock. Cells go through `normalize_cell()` first —
LIST, STRUCT, MAP and BLOB become strings there — which is what gives the cell
and byte caps a width to measure instead of an opaque Python object. Narrow one limit for a single request with
`dataclasses.replace()`. `interrupt()` is per connection or cursor, so the object handed to
`execute_bounded` must be the one executing. The API handler calls it through
`asyncio.to_thread` — `db.execute` is blocking, and on the event loop one slow
query froze the UI, the MCP endpoint and the refresh loop together.

Also here: **`is_loopback_host()`**, which gates exposure of the HTTP SQL API
and backs the per-request `Origin` check in `api/main.py`. Unknown hostnames
fail closed rather than triggering DNS resolution.

The HTTP boundary itself lives in `api/main.py`: `host_guard` (a loopback
`Host` allowlist, applied only when the bind is itself loopback — see
`host_allowlist_applies`), an `Origin` check and the `X-Introspect-Client`
requirement in `local_api_guard`, and deliberately **no** `CORSMiddleware`.

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

## Cache TTL (`cache_ttl.py`)

Two questions, deliberately kept apart:

1. **What did idling cost?** A pause longer than the cache TTL means the
   next request rebuilds the prefix at the write rate instead of reading it.
2. **Would a different TTL have been cheaper?** 1h charges 2x input on every
   incremental write, against 1.25x for 5m — so it buys back the rebuilds in
   the 5-60 minute band and pays a surcharge on everything else.

The second cannot be read off the first, which is why the module simulates
both policies rather than reporting a waste figure.

**One detection rule.** `cache_requests` is the only place a cache break is
defined. The session-detail divider, the tokenscape event track and the
cost-overview panel all read it; before this there were two rules with
different thresholds (300s vs 270s, different secondary conditions) that
could disagree about the same session.

**Gap semantics.** Measured from the end of the previous response (the last
logged block of that `message.id`) to whatever triggered the next request —
a human prompt *or* a tool result. A tool that runs for six minutes expires
the cache exactly like a coffee break does. Anthropic's TTL refreshes on
every hit, so the gap since the previous *request* is the right clock, not
the gap since the cache was first written.

**Ordering** is by wall-clock timestamp within `(session_id, is_sidechain)`.
`parent_uuid` would be the more principled chain, but real transcripts break
it — parallel tool calls and harness rewrites leave dangling parents.

**The counterfactual.** For `T ∈ {300, 3600}`, independent of what was
observed:

```
warm(T) = gap_seconds <= T AND NOT structural_invalidation AND seq > 1
warm : read = common_prefix;  write = prefix_total - read
cold : read = 0;              write = prefix_total
cost(T) = read*rate_read + write*rate_write(T) + input*rate_in + output*rate_out
```

Assumptions, stated so they can be argued with:

- **Prefix invariance.** `prefix_total = cache_read + cache_creation` is a
  property of the conversation, not the TTL — the same messages get re-sent
  either way. Only the read/write split moves. This is what makes the
  comparison honest.
- **Common prefix.** A request that was observed *warm* reports its reusable
  overlap directly as `cache_read_tokens` (cache-breakpoint granularity
  included). A request that *missed* reports only whatever residue survived,
  so it falls back to `min(prev_prefix_total, prefix_total)` — exact for the
  ordinary append-only case.
- **Structural invalidations are excluded.** Reading back ~nothing after a
  sub-5-minute gap means the prefix changed (model or effort switch,
  `/compact`, a tool-set change), not that time ran out. Identical cost under
  both policies, so attributing it to pausing would be wrong.
- **Gaps over an hour are breaks, not waste.** No TTL Claude Code offers
  recovers them; counting them inflates the apparent upside of switching, so
  they are reported separately.
- **Subscription dollars are API-equivalent.** Costs are list API prices; no
  plan coefficient is recorded in the transcripts, so treat the margin as a
  ratio as much as a dollar figure.
- **Sidechains never merge into the main verdict.** Subagents carry their own
  `subagentPromptCacheTtl`, and concurrent subagents interleave in wall-clock
  order, so their gaps are noisier. Scored separately, always.
- **One TTL bucket per request.** Mixed-TTL billing positions inside a single
  request are not modelled.

**The gate.** `introspy cache-ttl --verify` replays the TTL each uniform-TTL
session was *actually* billed at and compares against the observed bill. Any
non-zero residual means the gap definition misclassified a request's warmth,
and nothing built on the simulation is trustworthy until it is explained. The
same command reports 5m/1h split coverage by month.

Surfaces: the cost-overview portfolio panel, `introspy cache-ttl`, the
`Prompt-cache TTL` line in `introspy stats`, and the `cache_ttl_choice`
query template / MCP tool.

## Pricing (`pricing.py`)

A hardcoded snapshot of published API list prices in USD per 1M tokens, keyed by
model-name **prefix** so dated suffixes (`claude-haiku-4-5-20251001`) match.
Prefixes are sorted longest-first at import so the most specific one wins in
both the Python and the SQL lookup path. Anthropic and OpenAI (Codex
`gpt-5.6-*`) models are both covered. There is no live fetch.

Codex's internal `codex-auto-review` model has no published price or underlying
public-model identifier in its local logs. Introspect therefore estimates it
at Luna rates provisionally; this is tracked in
[openai/codex#20981](https://github.com/openai/codex/issues/20981) and will be
revisited when OpenAI provides authoritative pricing data.

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
total at the 5-minute rate (`CACHE_WRITE_FALLBACK_SQL` and
`cache_ttl.CACHE_CREATION_EFFECTIVE_SQL` in SQL, `compute_cost_usd` in Python).
That's the cheaper of the two, so a legacy record is never over-billed.

Other exports:

- `rates_for(model, input_tokens=...)` and `compute_cost_usd(...)` — Python
  per-row cost. `gpt-5.6-*` rows above 272K input tokens use OpenAI's
  long-context table; the threshold is applied before every token class is
  priced, including output.
- `is_priced(model)` — whether the model is in the table at all, so a surface
  showing dollars can distinguish "no rates for this model" from "cheap". The
  MCP `get_session` cost block uses it to append `[unpriced]`.
- `PRICING_INPUT_RATE_SQL`, `PRICING_OUTPUT_RATE_SQL`,
  `PRICING_CACHE_READ_RATE_SQL`, `PRICING_CACHE_WRITE_5M_RATE_SQL`,
  `PRICING_CACHE_WRITE_1H_RATE_SQL` — DuckDB `CASE` expressions, so mixed-model
  sessions price correctly without materializing every message in Python.
  `tests/test_pricing.py` checks the SQL and Python totals agree.
- `CACHE_TTL_SECONDS = 300` — Anthropic's default ephemeral cache TTL. Just the
  constant; the detection rule built on it (and the 1h alternative) lives in
  [`cache_ttl.py`](#cache-ttl-cache_ttlpy), which is also where the cache-miss
  premium is now computed — per request, in the `cache_requests` view.

**Unknown models bill at $0.** `rates_for()` returns zero rates for `None`, the
empty string, `<synthetic>`, and any unrecognized model, logging once per name
at WARNING (LRU-bounded). The SQL `CASE` expressions end in `ELSE 0`. So an
unpriced model contributes nothing to cost totals rather than guessing — silently
everywhere except the MCP `get_session` cost block, which names it via
`is_priced()`.

**Known understatements.** Rates are list prices with no request-level
modifiers. Anthropic records `usage.speed == 'fast'` (2× on Opus 5 / Opus 4.8)
and `usage.inference_geo == 'us'` (1.1×); neither is modelled, so a session that
used either is understated by that multiplier. OpenAI's `gpt-5.6-*` long-context
tier is selected from the recorded per-request input count: above 272K input
tokens, input, cached input, and cache writes are 2× their short-context
prices, while output is 1.5×.

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
