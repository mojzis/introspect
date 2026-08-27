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

## Agentic skills

Skills under `.claude/skills/` extend the coding agent when working in this
repo:

| Skill | Purpose |
|---|---|
| `/python-review` | Deep code-quality review — design, naming, performance, test quality. |
| `/docs-review` | Checks the diff against `docs/`, `README.md`, and `CLAUDE.md`, and fixes reference docs that are missing or stale. |
| `nolegend` | Tufte-style Plotly conventions for the server-side `go.Figure` charts. |

After finishing a change, run `/python-review` and then `/docs-review`, and
apply every 🔴 Must Fix finding before marking the work complete.

The same skills are available in Codex: `.agents/skills` is a symlink to
`.claude/skills`, and Codex scans `.agents/skills` from the working directory
up to the repo root, following symlinks. Add a skill once under
`.claude/skills/` and both agents see it — `codex /skills` lists what Codex
discovered, and `$` mentions one. `.codex/config.toml` sets
`project_doc_fallback_filenames = ["CLAUDE.md"]`, so Codex reads the same
project instructions as Claude Code rather than needing its own `AGENTS.md`.

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

## Database connections

There is exactly one way to open a read-only connection to the main database:
`db.connect_read_hardened()`. It applies the resource caps, loads FTS, disables
external access, and locks the configuration — see
[Security](security.md#engine-configuration-the-real-boundary). Calling
`duckdb.connect(..., read_only=True)` anywhere else is a build failure:
`tests/e2e/test_sql_hardening.py` parses `src/` and fails on any such call
outside `db.py`. It is also a runtime problem, not just a policy one — DuckDB
refuses a second connection to a file whose instance was opened with a
different configuration, so an unhardened caller breaks every hardened one in
the same process.

Run that test after any DuckDB upgrade; it is what says whether the boundary
still holds.

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

## Releasing a new version

Releases are cut from `main` with one task:

```bash
# Patch release (0.3.0 -> 0.3.1)
uv run poe release

# Minor or major — `level` is read from the environment, so it goes
# *before* the command; `poe release level=minor` cuts a patch release.
level=minor uv run poe release
level=major uv run poe release
```

It runs `uv version --bump <level>` (updating `pyproject.toml` and `uv.lock`),
commits everything tracked as `Release v<version>`, tags `v<version>`, and
pushes the branch and the tag. Because the commit is a `git commit -am`, it
sweeps up every modified tracked file — start from an up-to-date `main` with a
clean tree and `uv run poe check` green.

Pushing the tag triggers `.github/workflows/release.yml`, which builds the sdist
and wheel with `uv build` and publishes them to PyPI as
[`introspy`](https://pypi.org/project/introspy/) through the `pypi` deployment
environment. Publishing uses PyPI trusted publishing (OIDC), so there is no API
token to rotate — but the environment does need to approve the run if it is
configured with reviewers.

After the workflow finishes, confirm the new version is the one PyPI serves:

```bash
curl -s https://pypi.org/pypi/introspy/json | jq -r .info.version
```

Installed copies find out on their own: `version_check.py` asks PyPI for the
latest version at most once a day and nags on stderr from the commands where
that is harmless — `introspy mcp` stays silent so its stdio channel stays
clean. See [update check](configuration.md#update-check) for the opt-out and
the interval override.
