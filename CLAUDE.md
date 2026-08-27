# Introspect

Explore Claude Code conversation logs via CLI, web UI, MCP server.

## Architecture

- `db.py` — DuckDB schema over `~/.claude/projects/**/*.jsonl`; materialized at server startup, lazy views as fallback
- `refresh.py` — background rebuild loop + window picker (`1`/`7`/`30`/`month`)
- `pricing.py` — model pricing as Python rates + DuckDB `CASE` SQL
- `sql_fragments.py` — shared SQL building blocks (cost / tool / file / command rollups)
- `projects.py` — git worktree-aware `cwd` → canonical project
- `search.py` — FTS via BM25, ILIKE fallback
- `sql_query.py` — shared read-only SQL guard (SELECT/WITH-only validator, row-cap wrapping, loopback host check) used by both MCP `run_sql` and the HTTP SQL API
- `api/handlers/query.py` — local-only `POST /api/query` + `GET /api/schema` JSON SQL API; gated on `app.state.sql_api_enabled` (loopback bind only, set in `main.py` lifespan from `INTROSPECT_HOST`)
- `api/routes.py` → `api/handlers/<name>.py` → `templates/<name>.html`
- `api/handlers/_helpers.py` — shared: `parent(request)`, `conn(request)`, pagination, sort allowlists; re-exports SQL fragments
- `mcp/` — FastMCP tools mounted on FastAPI; `refresh_bridge.py` plumbs `app.state` to stateless tool fns
- `cli.py` — Typer commands

## Key Patterns

- **Adding a page**: handler in `handlers/`, route in `routes.py`, template, tests in `tests/routes/`
- **DB access**: `request.state.conn` (read-only, per-request), `json_extract()` for JSON fields, `# noqa: S608` for dynamic SQL
- **Pagination**: 1-based, fetch `size+1` to detect next page
- **HTMX**: `parent(request)` selects `base.html` (full) vs `partial.html` (fragment)
- **Charts**: build `plotly.graph_objects.Figure` server-side, style with `nolegend.activate()`, embed JSON for `Plotly.newPlot` (see `/python-review` skill `nolegend`)
- **Cost SQL**: reuse `SESSION_COST_SUBQUERY` / `session_cost_subquery_filtered()` from `sql_fragments.py` — never hand-roll cost math in handlers
- **Materialization**: `materialize_views()` runs on web startup and rebuilds derived tables (incl. `session_stats`, `assistant_message_costs`, `session_messages_enriched`); CLI commands call `ensure_materialized()` so they share the on-disk DB
- **Views/tables** (`db.py`): `raw_data`, `raw_messages`, `project_map`, `logical_sessions`, `assistant_message_costs`, `tool_calls`, `session_messages_enriched`, `conversation_turns`, `session_titles`, `message_commands`, `file_reads`, `file_writes`, `session_stats`, `search_corpus`, `materialize_meta`

## Test Fixtures (`conftest.py`)

`make_user_message()`, `make_assistant_message()`, `write_jsonl()`, `glob_pattern()`. Route tests use `_patched_client()` context manager (defined in `tests/routes/conftest.py`).

## Commands

- `uv run introspect query "SELECT ..."` — ad-hoc SQL against the views (use this to study real data; `introspect views` lists them)
- `uv run poe check` — run lint, typecheck, vulns, then tests
- `uv run poe fix` — auto-format and fix lint issues
- `uv run poe test` — run tests only
- `uv run poe check-all` — run all checks including dead-code and unused-deps
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

uv, ruff (lint/format), ty (type check), tyf (code search), biston (clone detection), pytest, poethepoet (task runner)

## Notes

- ty is in beta — may produce false positives. Prefer `# ty: ignore[rule]` over blanket suppression.
- Pre-commit hook auto-fixes and restages files. Only blocks on unfixable issues.
- All user-facing features must have tests. When adding new routes, template variables, query parameters, or UI functionality, add corresponding tests in `tests/routes/`.
- **IMPORTANT**: After completing any task, you MUST run the `/python-review` skill to review all changes. Apply all 🔴 Must Fix and 🟡 Should Fix findings before marking work as complete.
- **IMPORTANT**: Then run the `/docs-review` skill to check the diff against `docs/`, `README.md`, and `CLAUDE.md`. Apply all 🔴 Must Fix findings before marking work as complete.
