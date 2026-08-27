# Introspect

Explore Claude Code (and Codex) conversation logs via CLI, web UI, MCP server.

## Architecture

- `db.py` — DuckDB schema over `~/.claude/projects/**/*.jsonl`; materialized at server startup, lazy views as fallback. The drop-list at the top of `materialize_views()` is the canonical relation list
- `codex.py` — Codex rollout-log transcoder; rows land in `codex_raw_messages` and `UNION ALL BY NAME` into `raw_messages`, tagged `provider` / `harness`
- `refresh.py` — background rebuild loop + window picker (`1`/`7`/`30`/`month`)
- `pricing.py` — model pricing as Python `Rates` + DuckDB `CASE` SQL; five rates per model (input, cache_write_5m, cache_write_1h, cache_read, output). Unknown models bill $0
- `cache_ttl.py` — the single prompt-cache-break detection rule (`cache_requests` view) plus the 5m-vs-1h counterfactual and its rollups; verify with `introspy cache-ttl --verify`
- `sql_fragments.py` — shared SQL building blocks (cost / tool / file / command / skills / context-loads rollups)
- `query_templates.py` — one registry of curated SQL investigations, leaf module; `kind` fans out to three adapters (cookbook tool, deterministic MCP tools, MCP prompts)
- `sql_query.py` — shared SQL guard for MCP `run_sql` and the HTTP SQL API: `validate_read_only_sql` (one `SELECT`, via `duckdb.extract_statements`, plus a function denylist), `execute_bounded` (wall clock / rows / bytes / cell width), `is_loopback_host`. The **engine config**, not this module, is the boundary — see `docs/security.md`
- `db.connect_read_hardened()` — the single read-only connection factory. `enable_external_access=false` + `lock_configuration=true` + memory/thread/temp caps. `LOAD fts` must precede the disable; the SETs must be skipped on an already-locked instance. Never open a read-only connection to the main DB anywhere else — `tests/e2e/test_sql_hardening.py` fails the build if you do (DuckDB rejects a second connection with a different config)
- `version_check.py` — once-a-day PyPI update check + stderr nag
- `projects.py` — git worktree-aware `cwd` → canonical project
- `search.py` — FTS via BM25, ILIKE fallback
- `api/handlers/query.py` — local-only `POST /api/query` + `GET /api/schema` JSON SQL API; gated on `app.state.sql_api_enabled` (loopback bind only, set in `main.py` lifespan from `INTROSPECT_HOST`, fails closed)
- `api/main.py` — three middlewares: `db_middleware` (per-request hardened read connection), `host_guard` (loopback `Host` allowlist `ALLOWED_HOSTS`, enforced only when `host_allowlist_applies(bind_host)` — hand-rolled, *not* Starlette's `TrustedHostMiddleware`, because the allowlist depends on the bind host the lifespan reads), `local_api_guard` (`Origin` + `X-Introspect-Client` on `/api/query` / `/api/schema` / `/mcp`). **Never** add `CORSMiddleware`
- `api/routes.py` → `api/handlers/<name>.py` → `templates/<name>.html`
- `api/handlers/_helpers.py` — shared: `parent(request)`, `conn(request)`, pagination, sort allowlists; re-exports SQL fragments
- `mcp/` — FastMCP tools + prompts mounted on FastAPI; `_register.py` wires the query-template registry; `refresh_bridge.py` plumbs `app.state` to stateless tool fns
- `cli.py` — Typer commands

## Key Patterns

- **Adding a page**: handler in `handlers/`, route in `routes.py`, template, tests in `tests/routes/`
- **DB access**: `request.state.conn` (read-only, per-request), `json_extract()` for JSON fields, `# noqa: S608` for dynamic SQL
- **Pagination**: 1-based, fetch `size+1` to detect next page
- **HTMX**: `parent(request)` selects `base.html` (full) vs `partial.html` (fragment)
- **Charts**: build `plotly.graph_objects.Figure` server-side, style with `nolegend.activate()`, embed JSON for `Plotly.newPlot` (see `/python-review` skill `nolegend`)
- **Cache breaks**: never re-derive a TTL threshold — read `cache_requests` (`cache_miss`, `gap_recoverable`, `gap_unrecoverable`). Waste is capped at 1h gaps; anything longer is a break no setting recovers
- **Ad-hoc SQL**: never hand-roll validation or execution for `run_sql` / `/api/query` — both go through `validate_read_only_sql` then `execute_bounded`, and the API wraps the latter in `asyncio.to_thread` (blocking `db.execute` on the event loop freezes the UI, MCP and refresh together)
- **Cost SQL**: reuse `SESSION_COST_SUBQUERY` / `session_cost_subquery_filtered()` from `sql_fragments.py` — never hand-roll cost math in handlers
- **Adding a query template**: append to `QUERY_TEMPLATES`, then add the matching adapter — `deterministic` → `deterministic_tool_fns` in `_register.py` (or a hand-registered tool of the same name, which shadows the generated one, as `expensive_sessions` does); `exploratory` → a fn in `mcp/prompts.py` plus `exploratory_prompt_fns`. `_wire_template_adapters()` raises on either half missing
- **Materialization**: `materialize_views()` runs on web startup and rebuilds derived tables (incl. `session_stats`, `assistant_message_costs`, `session_messages_enriched`, `session_context_loads`); CLI commands call `ensure_materialized()` so they share the on-disk DB
- **Relations** (`db.py`): `raw_data`, `codex_raw_messages`, `raw_messages`, `project_map`, `logical_sessions`, `assistant_message_costs`, `tool_calls`, `session_messages_enriched`, `conversation_turns`, `session_titles`, `message_commands`, `session_context_loads`, `file_reads`, `file_writes`, `session_stats`, `cache_requests`, `session_cache_ttl`, `search_corpus`, `materialize_meta`
- **Docs**: user-facing docs live in `docs/` (published); planning notes go in `docs/plans/` (excluded from the build). `tests/test_docs_drift.py` fails when a command, env var, relation, MCP tool/prompt, template, or route isn't mentioned in the docs; security-relevant behaviour belongs in `docs/security.md` — fix the docs, don't weaken the test. Regenerate the CLI reference with `uv run poe docs-cli`

## Test Fixtures (`conftest.py`)

`make_user_message()`, `make_assistant_message()`, `write_jsonl()`, `glob_pattern()`. Route tests use `_patched_client()` context manager (defined in `tests/routes/conftest.py`).

## Commands

- `uv run introspect query "SELECT ..."` — ad-hoc SQL against the relations (use this to study real data; `introspect tables` lists them)
- `uv run poe check` — run lint, typecheck, vulns, then tests
- `uv run poe fix` — auto-format and fix lint issues
- `uv run poe test` — run tests only
- `uv run poe check-all` — run all checks including dead-code and unused-deps
- `uv run poe docs-cli` — regenerate `docs/usage/cli-reference.md` from `--help`
- `uv run mkdocs build --strict` — build the docs site (needs `uv sync --group docs`)
- `uv run poe worktree <branch>` — create `~/worktrees/introspect-<branch>` from a fresh `origin/main` (fetches, branches, copies `.claude/settings.local.json`, runs `uv sync`). See `scripts/worktree.sh`.

## Worktrees

User keeps worktrees under `~/worktrees/introspect-<branch>`. To set one up, ask: "set up a worktree for `<branch>`" and Claude will run `poe worktree <branch>`. The DuckDB at `~/.introspect/introspect.duckdb` is shared across worktrees (reads fine; avoid concurrent writes/refreshes).

## Code Search (`tyf`)

This project has `tyf` (ty-find) — type-aware code search that gives LSP-quality results by symbol name. Prefer it over grep for Python symbols. Reserve grep for string literals, config values, TODOs, non-Python files.

- `uv run tyf show <name>` — definition + signature + usages (flags: `-d` docs, `-r` refs, `-t` test refs, `--all`)
- `uv run tyf find <Symbol>` — locate definition
- `uv run tyf refs <name>` — find all usages
- `uv run tyf members <Class>` — view class API
- `uv run tyf list <file.py>` — file outline

## Clone Detection (`biston`)

Structural clone detector for Python — finds groups of functions that are structurally similar even when names/literals/argument order differ. Run after producing multiple similar functions, or when refactoring, to spot extraction opportunities.

- `uv run biston scan --suggest .` — find clones with anti-unified template proposals
- `uv run biston scan --threshold 0.8 .` — stricter matching (default 0.7)
- `uv run biston scan --min-lines 10 .` — ignore tiny functions (default 5)
- `uv run biston scan --tests-only .` — test-duplication scan
- `uv run biston overview .` — condensed file-centric summary
- `uv run biston stats .` — aggregate counts

## Stack

uv, ruff (lint/format), ty (type check), tyf (code search), biston (clone detection), pytest, poethepoet (task runner), mkdocs-material (docs)

## Notes

- ty is in beta — may produce false positives. Prefer `# ty: ignore[rule]` over blanket suppression.
- Pre-commit hook auto-fixes and restages files. Only blocks on unfixable issues.
- All user-facing features must have tests. When adding new routes, template variables, query parameters, or UI functionality, add corresponding tests in `tests/routes/`.
- **IMPORTANT**: After completing any task, you MUST run the `/python-review` skill to review all changes. Apply all 🔴 Must Fix and 🟡 Should Fix findings before marking work as complete.
- **IMPORTANT**: Then run the `/docs-review` skill to check the diff against `docs/`, `README.md`, and `CLAUDE.md`. Apply all 🔴 Must Fix findings before marking work as complete.
