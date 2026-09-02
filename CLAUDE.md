# Introspect

Explore Claude Code (and Codex) conversation logs via CLI, web UI, MCP server.

## Architecture

- `db.py` — DuckDB schema over `~/.claude/projects/**/*.jsonl`; materialized at server startup, lazy views as fallback. The drop-list at the top of `materialize_views()` is the canonical relation list
- `codex.py` — Codex rollout-log transcoder; rows land in `codex_raw_messages` and `UNION ALL BY NAME` into `raw_messages`, tagged `provider` / `harness`; session-meta naming/context lands in `codex_session_metadata`
- `refresh.py` — progressive preview/warm-snapshot startup + background sidecar rebuild loop; `RefreshTarget` / `LoadingState` are the shared lifecycle contract for picker targets (`1`/`7`/`30`/`month` or numeric days, with `0` for all data)
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
- `api/errors.py` — the single error-response policy: `HTTPException`, validation errors and unhandled exceptions become an HTMX fragment, a full page, or JSON, always with the real status code
- `api/main.py` — three middlewares: `db_middleware` (per-request hardened read connection), `host_guard` (loopback `Host` allowlist `ALLOWED_HOSTS`, enforced only when `host_allowlist_applies(bind_host)` — hand-rolled, *not* Starlette's `TrustedHostMiddleware`, because the allowlist depends on the bind host the lifespan reads), `local_api_guard` (`Origin` + `X-Introspect-Client` on `/api/query` / `/api/schema` / `/mcp`). **Never** add `CORSMiddleware`
- `api/routes.py` → `api/handlers/<name>.py` → `templates/<name>.html`
- `api/handlers/_helpers.py` — shared: `parent(request)`, `conn(request)`, pagination, sort allowlists; re-exports SQL fragments
- `mcp/` — FastMCP tools + prompts mounted on FastAPI; `_register.py` wires the query-template registry; `refresh_bridge.py` plumbs `app.state` to stateless tool fns
- `cli.py` — Typer commands

## Key Patterns

- **Adding a page**: handler in `handlers/`, route in `routes.py`, template, tests in `tests/routes/`
- **DB access**: `request.state.conn` (read-only, per-request), `json_extract()` for JSON fields, `# noqa: S608` for dynamic SQL
- **Pagination**: 1-based, fetch `size+1` to detect next page. The session-detail Messages tab is the exception: it windows in SQL by block ordinal via `MessageWindow` / `_resolve_message_window()` (`handlers/sessions.py`), and deep links pass `?focus=<uuid|tool_use_id>` so the page holding an anchor renders
- **HTMX**: `parent(request)` selects `base.html` (full) vs `partial.html` (fragment); both go through `is_htmx(request)` — the one place `HX-Request` is read
- **Errors**: never return 200 for a failure. `api/errors.py` holds the whole policy — an HTMX request gets `_error.html` retargeted to `#errors` (`HX-Reswap: beforeend`, so the original target survives), a plain request gets `error.html`, `/api/*` and `/mcp` keep their JSON. Tracebacks are always logged, and only rendered under `INTROSPECT_DEBUG` on a loopback bind. `base.html` refuses to swap any error response lacking `X-Introspect-Error-Rendered`. A panel that degrades to an inline notice (`chart_error`, the tokenscape/trajectory/subagents tab fallbacks) is a successful request, not a failure — leave those at 200
- **Charts**: build `plotly.graph_objects.Figure` server-side, style with `nolegend.activate()`, embed JSON for `Plotly.newPlot` (see `/python-review` skill `nolegend`)
- **Cache breaks**: never re-derive a TTL threshold — read `cache_requests` (`cache_miss`, `gap_recoverable`, `gap_unrecoverable`). Waste is capped at 1h gaps; anything longer is a break no setting recovers
- **Ad-hoc SQL**: never hand-roll validation or execution for `run_sql` / `/api/query` — both go through `validate_read_only_sql` then `execute_bounded`, and the API wraps the latter in `asyncio.to_thread` (blocking `db.execute` on the event loop freezes the UI, MCP and refresh together)
- **Cost SQL**: reuse `SESSION_COST_SUBQUERY` / `session_cost_subquery_filtered()` from `sql_fragments.py` for per-session rollups, and `COST_EXPR_SQL` (or one of its `*_COST_SQL` components) when you need per-row cost — never hand-roll cost math in handlers. The legacy cache_creation fallback lives inside those expressions and must be applied per row: folding it into an aggregate drops legacy tokens whenever the same model also logged a modern record
- **Adding a query template**: append to `QUERY_TEMPLATES`, then add the matching adapter — `deterministic` → `deterministic_tool_fns` in `_register.py` (or a hand-registered tool of the same name, which shadows the generated one, as `expensive_sessions` does); `exploratory` → a fn in `mcp/prompts.py` plus `exploratory_prompt_fns`. `_wire_template_adapters()` raises on either half missing
- **Materialization**: `materialize_views()` runs on web startup and rebuilds derived tables (incl. `session_stats`, `assistant_message_costs`, `session_messages_enriched`, `session_context_loads`); CLI commands call `ensure_materialized()` so they share the on-disk DB
- **Relations** (`db.py`): `raw_data`, `codex_raw_messages`, `codex_session_metadata`, `raw_messages`, `project_map`, `logical_sessions`, `assistant_message_costs`, `tool_calls`, `session_messages_enriched`, `conversation_turns`, `session_titles`, `message_commands`, `session_context_loads`, `file_reads`, `file_writes`, `session_stats`, `cache_requests`, `session_cache_ttl`, `search_corpus`, `materialize_meta`
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
- `level=minor uv run poe release` — bump the version (default `patch`), commit, tag `v<version>`, push; the tag triggers the PyPI publish. `level` is an env var, so it must precede the command. See [docs/development.md](docs/development.md#releasing-a-new-version).

## Worktrees

User keeps worktrees under `~/worktrees/introspect-<branch>`. To set one up, ask: "set up a worktree for `<branch>`" and Claude will run `poe worktree <branch>`. The DuckDB at `~/.introspect/introspect.duckdb` is shared across worktrees (reads fine; avoid concurrent writes/refreshes).

## Toolbox

Five CLI tools ship as dev dependencies. Every one of them documents itself —
run `uv run <tool> guide` first (`--help` on the tools that have no `guide`
subcommand) and follow its own conventions rather than guessing at flags.

**On every commit** (`scripts/pre-commit.sh`, installed by `uv run poe setup`):
ruff format + autofix, ruff check, `ty check`, then `biston` on the staged
files, then only the tests the diff impacts. Bypass with `git commit
--no-verify`.

**On demand**: `zorilla` (test-quality) and the wider `biston` / `gerenuk`
reports below.

Refresh all five to their latest versions with:

```
uv sync --upgrade-package madoqua --upgrade-package gerenuk \
  --upgrade-package biston --upgrade-package zorilla --upgrade-package ty-find
```

(`madoqua` is not published on PyPI as of this writing, so it is not a
dependency here and the flag above is a no-op for it — see "Commit hook".)

### Code Search (`tyf`, from the `ty-find` package)

Type-aware code search that gives LSP-quality results by symbol name.
**Agents must use `uv run tyf` instead of grep for symbol definitions and
references in this repo.** Reserve grep for string literals, config values,
TODOs, and non-Python files. Note the binary is `tyf`, not `ty-find`; it runs
a background daemon that auto-starts on first use (`uv run tyf daemon status`).

- `uv run tyf show <name>` — definition + signature + usages (flags: `-d` docs, `-r` refs, `-t` test refs, `--all`)
- `uv run tyf find <Symbol>` — locate definition
- `uv run tyf refs <name>` — find all usages (`-t` splits out test references)
- `uv run tyf members <Class>` — view class API
- `uv run tyf list <file.py>` — file outline

### Impacted Tests (`gerenuk` + `tyf`)

`gerenuk changed-symbols` reports which symbols the working tree changed;
`scripts/impacted_tests.py` feeds those to `tyf refs --tests` and prints the
test files that reach them. The commit hook runs just those instead of the
full suite. Selection is conservative: a change to `conftest.py`,
`pyproject.toml` or `uv.lock`, an unmappable symbol, or any tool error falls
back to the whole suite rather than risking a silent coverage gap.

- `uv run poe impacted-tests` — list the test files the current diff impacts
- `uv run gerenuk audit <file.py>...` — symbols nothing references, and symbols only tests reach
- `uv run gerenuk doctor` — check that `tyf` and the workspace resolve

### Clone Detection (`biston`)

Structural clone detector for Python — finds groups of functions that are
structurally similar even when names/literals/argument order differ. Runs in
the commit hook scoped to staged files; run it wider after producing multiple
similar functions or when refactoring. Configured under `[tool.biston.scan]`
in `pyproject.toml` (first-party code only, threshold 0.75).

- `uv run biston scan --suggest .` — find clones with anti-unified template proposals
- `uv run biston scan --tests-only .` — test-duplication scan (tests are outside the configured scan set)
- `uv run biston overview .` — condensed file-centric summary
- `uv run biston stats .` — aggregate counts

### Test Quality (`zorilla`)

A pytest test-smell linter: sleeps in tests, conditional logic, assertion
roulette, hardcoded external resources. **Deliberately not in the hook or
CI** — it is an on-demand check to run when the test suite has grown.

- `uv run poe test-smells` — lint `tests/` (equivalently `uv run zorilla check tests`)
- `uv run zorilla stats tests` — findings per rule, always exits 0
- `uv run zorilla overview tests` — findings grouped by file
- `uv run zorilla explain ZR004` — long-form docs for a rule

### Commit hook

`scripts/pre-commit.sh` is the single hook; `uv run poe setup` copies it into
`.git/hooks/pre-commit`. The task asked for `madoqua` to own this, but
`madoqua` is not published on PyPI (404 on both the JSON and simple indexes),
so the existing script stays the carrier and every check was consolidated into
it. Swapping in `madoqua` later means porting the four stages in that file to
its config — nothing else in this repo depends on the hook's shape.

## Stack

uv, ruff (lint/format), ty (type check), tyf (code search), gerenuk (changed-symbol / test impact), biston (clone detection), zorilla (test-smell lint), pytest, poethepoet (task runner), mkdocs-material (docs)

## Notes

- ty is in beta — may produce false positives. Prefer `# ty: ignore[rule]` over blanket suppression.
- Pre-commit hook auto-fixes and restages files, then gates on ruff, `ty`, `biston`, and the tests the diff impacts. See "Toolbox" above.
- All user-facing features must have tests. When adding new routes, template variables, query parameters, or UI functionality, add corresponding tests in `tests/routes/`.
- **IMPORTANT**: After completing any task, you MUST run the `/python-review` skill to review all changes. Apply all 🔴 Must Fix and 🟡 Should Fix findings before marking work as complete.
- **IMPORTANT**: Then run the `/docs-review` skill to check the diff against `docs/`, `README.md`, and `CLAUDE.md`. Apply all 🔴 Must Fix findings before marking work as complete.
