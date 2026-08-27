# Development

Introspect uses [uv](https://docs.astral.sh/uv/) for dependency management and
[poethepoet](https://poethepoet.natn.io/) as the task runner.

## Common tasks

```bash
# Install dependencies (including dev tools)
uv sync

# Auto-format and fix lint issues
uv run poe fix

# Run lint, typecheck, security scan, and tests
uv run poe check

# Run tests only
uv run poe test

# Run all checks including dead-code and unused-deps
uv run poe check-all
```

Tests run in parallel via `pytest-xdist`.

## Code exploration tools

```bash
# Type-aware code search (LSP-quality, by symbol name)
uv run tyf show <name>      # definition + signature + usages
uv run tyf refs <name>      # find all usages
uv run tyf members <Class>  # view class API

# Structural clone detection — spot extraction opportunities
uv run biston scan --suggest .
uv run biston overview .
```

## Claude Code skills

Skills under `.claude/skills/` extend Claude Code when working in this repo:

| Skill | Purpose |
|---|---|
| `/python-review` | Deep code-quality review — design, naming, performance, test quality. |
| `/docs-review` | Checks the diff against `docs/`, `README.md`, and `CLAUDE.md`, and fixes reference docs that are missing or stale. |
| `nolegend` | Tufte-style Plotly conventions for the server-side `go.Figure` charts. |

After finishing a change, run `/python-review` and then `/docs-review`, and
apply every 🔴 Must Fix finding before marking the work complete.

## Worktrees

```bash
# Create ~/worktrees/introspect-<branch> from a fresh origin/main
# (fetches, branches, copies settings, runs uv sync)
uv run poe worktree <branch>
```

The DuckDB at `~/.introspect/introspect.duckdb` is shared across worktrees
(reads fine; avoid concurrent writes/refreshes).

## Adding a page to the app

The web app follows a consistent handler → route → template pattern:

1. Add a handler in `src/introspect/api/handlers/`.
2. Register it in `api/routes.py`.
3. Add a Jinja2 template under `src/introspect/templates/`.
4. Add tests in `tests/routes/`.

All user-facing features must have tests. See the
[Architecture](architecture.md) reference for the full layout.

## Building these docs

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme.

```bash
# Install the docs dependency group
uv sync --group docs

# Live-preview at http://127.0.0.1:8000
uv run mkdocs serve

# Build the static site into ./site (and generate llms.txt)
uv run mkdocs build --strict
```

Pushing to `main` deploys the site to GitHub Pages via the
`.github/workflows/docs.yml` workflow.
