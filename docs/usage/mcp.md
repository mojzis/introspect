# MCP server

Introspect exposes its data to Claude Code (or any MCP client) as tools and
prompts, so you can ask questions about your own conversation history in natural
language.

At connect time the server also sends an instructions blob orienting the client
on the schema — which relation to reach for, which tool beats hand-rolled SQL,
and the cost-attribution rules. A client with no other context about your logs
still knows where to start.

## Over stdio

```bash
introspy mcp
```

Starts an MCP server over stdio, for registering in a client's MCP config.

## Over HTTP

The web server exposes the same tools and prompts at
`http://127.0.0.1:8347/mcp`.

## Tools

Nine hand-built tools, plus the ones generated per deterministic
[query template](#query-templates).

| Tool | Parameters | What it does |
|---|---|---|
| `search_conversations` | `query`, `limit`, `offset`, `cwd_prefix`, `role`, `since`, `session_id`, `require_all` | Full-text search across sessions (BM25 when available), with filters and pagination. |
| `get_session` | `session_id` | Fetch the full content of one session. |
| `recent_sessions` | `n` | List the most recent N sessions with metadata. |
| `tool_failures` | `command_prefix`, `limit` | List failed tool calls, optionally filtered by tool-name prefix. |
| `tool_failure_rate` | `limit`, `since`, `min_calls` | Rank tools by failure *rate* and count. `min_calls` suppresses low-N noise. Generated from the query-template registry. |
| `cache_ttl_choice` | `limit`, `since`, `sidechain` | Would a 1h or 5m prompt-cache TTL have been cheaper, per project? Replays every request under both policies and reports the margin — a thin margin comes back as "either" rather than a recommendation. Ask this here rather than inferring it from cache-miss waste, which cannot answer it. Generated from the query-template registry. |
| `expensive_sessions` | `limit`, `since` | Sessions ranked by cost with a `[pareto]` marker on the rows that together make 80% of spend, plus cost split, spend shape, subagent flag, and commands used. Mirrors the web Cost Overview Pareto table. |
| `run_sql` | `sql`, `limit` | Execute one read-only `SELECT` query. Capped at 500 rows / 64 KB / 20 s. |
| `describe_schema` | — | List relations available to `run_sql` with their columns. Call it before writing SQL. |
| `list_query_templates` | `kind` | Render the curated SQL cookbook — see [below](#query-templates). |
| `refresh_data` | — | Wake the refresh loop and wait for the rebuild. Only available when running embedded in `introspy serve`; the stdio server returns "unavailable". |

`run_sql` accepts exactly one `SELECT` statement (`WITH` and DuckDB's
FROM-first form count as one). Writes, `ATTACH`, `INSTALL`, `LOAD`, `PRAGMA`,
`SET`, `COPY`, multi-statement scripts, and functions that read outside the
database (`read_csv`, `read_text`, `glob`, `sqlite_scan`, …) are rejected. Its
connection has filesystem, network and extension access disabled and its
configuration locked, so the engine refuses those independently of the
validator — which matters here, because the logs `run_sql` reads are full of
untrusted text that could ask a model to try exactly that.

Every call is bounded at 500 rows, 64 KB of output, 200 characters per cell,
8 KB of SQL text, and a 20 s wall clock. The caps are measured against the
*serialized* value, so a LIST, STRUCT, MAP or BLOB cell counts for what it will
weigh in the rendered table rather than as one opaque object. A truncated
result names every cap that fired on its last line. The cell cap only shortens
values — seeing it there means every row is present but some were too wide, so
re-running with a smaller `limit` will not help; select narrower columns or
`substr()` instead. See [the SQL guard](../architecture.md#read-only-sql-guard-sql_querypy)
and [Security](../security.md).

## Query templates

`list_query_templates` renders a cookbook of curated SQL investigations. Each
entry shows the question it answers, its `$named` parameters, the SQL itself,
and a note carrying the non-obvious schema knowledge needed to adapt it (dedup
rules, what a flag column actually means). They are **starting points to adapt**,
not canned answers.

| Template | Kind | Question |
|---|---|---|
| `expensive_sessions` | deterministic | Which sessions cost the most? |
| `tool_failure_rate` | deterministic | Which tools fail most, by rate and count? |
| `cache_ttl_choice` | deterministic | Which prompt-cache TTL should I set — 5m or 1h? |
| `session_cost_tail` | exploratory | For one session, where does cumulative cost decouple from progress? |
| `first_prompt_triggers` | exploratory | What did each opening prompt reference and trigger? |
| `topic_to_cost` | exploratory | Sessions about a topic, ranked by cost. |

A `deterministic` template ("one fixed query answers this") is also registered
as an MCP tool that binds params and executes the registry SQL directly — check
the tool list first. The exception is `expensive_sessions`: the hand-built tool
of that name is richer than the template, so it takes precedence and no tool is
generated. An `exploratory` template ("the value is in adapting and following
threads") becomes a prompt instead.

Filter with `kind="deterministic"` or `kind="exploratory"`. See
[the registry](../architecture.md#query-template-registry-query_templatespy)
for how to add one.

## Prompts

Three MCP prompts, one per exploratory template. Each expands into a single seed
message built from that registry entry — its question, canonical SQL, and note —
plus follow-ups specific to the investigation. The seed is a starting point to
adapt; nothing in it is a finished answer.

| Prompt | Arguments | Use it when |
|---|---|---|
| `session_cost_tail` | `session_id` | One session cost far more than it should have and you want to find *where* it went wrong. Seeds a cumulative-cost walk over the session's turns and asks for the inflection point, then for `get_session` on what happened around it. |
| `first_prompt_triggers` | `project`, `limit` | You want to improve how you *open* sessions. Seeds the per-session trigger rollup and asks you to correlate prompt wording with what actually loaded — the sessions with zero referenced paths and zero skills are the ones that started blind. |
| `topic_to_cost` | `query`, `limit` | "How much did the auth refactor cost me?" Seeds a substring match over the search corpus joined to cost, and points at `search_conversations` (BM25) first for better relevance. |

Prompts seed an investigation; tools return an answer. Reach for a prompt when
the interesting part is the judgment call, not the query.

## Dedicated agent sessions

`introspy claude` and `introspy codex` launch a coding agent wired to the HTTP
MCP endpoint for one session only.

```bash
introspy claude
introspy claude -- --model opus --resume
introspy claude -- -p "what are the most expensive sessions"

introspy codex
introspy codex -- --model gpt-5.4
introspy codex -- "what are the most expensive sessions"
```

Both:

- **Start the server if needed.** If nothing is listening on the target port,
  `introspy serve` is launched in the background (log at
  `~/.introspect/serve.log`) and stopped again when the agent exits. A server
  that was already running is never touched. Pass `--keep-server` to leave an
  auto-started server up.
- **Change nothing on disk.** The MCP config is passed inline to `claude` and as
  command-line configuration overrides to `codex`. Neither your Claude Code
  settings nor your Codex configuration is modified; the server is registered
  for that session only.
- **Inject session instructions.** The same text goes in as Claude Code's
  appended system prompt and as Codex's developer instructions: this session is
  for analyzing conversation logs, so prefer the `mcp__introspect__*` tools over
  reading `~/.claude/projects` JSONL directly — the relations already handle
  session stitching, cost attribution, and project resolution. It names
  `expensive_sessions` for ranked cost analysis and tells the agent to call
  `describe_schema` before writing SQL and `list_query_templates` for starting
  points. `tests/test_cli.py` fails if a registered tool goes unmentioned there.
- **Pre-grant one permission.** For Claude Code, the rule `mcp__introspect`
  covers every tool on the server, so a log-analysis session doesn't prompt on
  each query. Nothing else is pre-approved.

Anything after `--` is forwarded verbatim to the underlying CLI. Use `--` to
separate Introspect's own options (`--port`, `--host`, `--keep-server`) from the
ones meant for the agent.

## What the MCP server does not do

- It cannot modify your logs. Every tool is read-only, and `refresh_data` only
  rebuilds the derived DuckDB.
- `run_sql` is not a DuckDB shell — one `SELECT` statement, 500 rows maximum,
  no filesystem or network access, 20 s of wall clock. For larger result sets
  from a notebook, use the [SQL API](sql-api.md).
- It serves no data over the network beyond the loopback interface it is bound
  to, and makes no outbound calls.

For the relations these tools query, see [Architecture](../architecture.md) and
[JSONL schema](../schema.md).
