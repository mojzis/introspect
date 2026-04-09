# Introspect

Explore Claude Code conversation logs via CLI, web UI, MCP server.

## Architecture

- `db.py` — DuckDB views over `~/.claude/projects/**/*.jsonl`
- `api/routes.py` → `api/handlers/<name>.py` → `templates/<name>.html`
- `_helpers.py` — shared: `parent(request)`, `conn(request)`, pagination, sort allowlists
- `search.py` — FTS via BM25, ILIKE fallback
- `mcp/` — FastMCP tools mounted on FastAPI
- `cli.py` — Typer commands

## Key Patterns

- **Adding a page**: handler in `handlers/`, route in `routes.py`, template, tests in `test_routes.py`
- **DB access**: `request.state.conn`, `json_extract()` for JSON fields, `# noqa: S608` for dynamic SQL
- **Pagination**: 1-based, fetch `size+1` to detect next page
- **HTMX**: `parent(request)` selects `base.html` (full) vs `partial.html` (fragment)
- **Views** (`db.py`): `raw_data`, `raw_messages`, `logical_sessions`, `tool_calls`, `conversation_turns`, `session_titles`, `search_corpus`

## Test Fixtures (`conftest.py`)

`make_user_message()`, `make_assistant_message()`, `write_jsonl()`, `glob_pattern()`. Route tests use `_patched_client()` context manager.

## Commands

- `uv run poe check` — run lint, typecheck, vulns, then tests
- `uv run poe fix` — auto-format and fix lint issues
- `uv run poe test` — run tests only
- `uv run poe check-all` — run all checks including dead-code and unused-deps

## Stack

uv, ruff (lint/format), ty (type check), pytest, poethepoet (task runner)

## Notes

- ty is in beta — may produce false positives. Prefer `# ty: ignore[rule]` over blanket suppression.
- Pre-commit hook auto-fixes and restages files. Only blocks on unfixable issues.
- All user-facing features must have tests. When adding new routes, template variables, query parameters, or UI functionality, add corresponding tests in `tests/test_routes.py`.
- **IMPORTANT**: After completing any task, you MUST run the `/python-review` skill to review all changes. Apply all 🔴 Must Fix and 🟡 Should Fix findings before marking work as complete.
