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

## Toolbox

Four CLI tools ship as dev dependencies. Each documents itself — run
`uv run <tool> guide` first (or `--help` on the tools without a `guide`
subcommand) rather than guessing at flags.

| Tool | Runs | Purpose |
|---|---|---|
| `tyf` (`ty-find`) | on demand + in the hook | Type-aware code search by symbol name |
| `gerenuk` | in the hook | Which symbols the diff changed; feeds test selection |
| `biston` | in the hook | Structural clone detection |
| `zorilla` | on demand only | pytest test-smell lint |

Refresh them all to their latest versions:

```bash
uv sync --upgrade-package madoqua --upgrade-package gerenuk \
  --upgrade-package biston --upgrade-package zorilla --upgrade-package ty-find
```

```bash
# Type-aware code search (LSP-quality, by symbol name).
# Prefer this over grep for Python symbols. Runs a daemon that auto-starts.
uv run tyf show <name>      # definition + signature + usages
uv run tyf refs <name>      # find all usages (-t splits out test references)
uv run tyf members <Class>  # view class API
uv run tyf daemon status    # inspect the background LSP server

# Structural clone detection — spot extraction opportunities.
# Configured under [tool.biston.scan] in pyproject.toml.
uv run biston scan --suggest .
uv run biston overview .
uv run biston scan --tests-only .   # tests sit outside the configured scan set

# Which tests does the current diff impact?
uv run poe impacted-tests
uv run gerenuk audit <file.py>...   # unreferenced and test-only symbols

# Test quality — on demand, when the suite has grown. Not in the hook or CI.
uv run poe test-smells              # = zorilla check tests
uv run zorilla stats tests
uv run zorilla explain ZR004
```

## Commit hook

`uv run poe setup` installs `scripts/pre-commit.sh` as `.git/hooks/pre-commit`.
It is the only hook, and it runs four stages, each reporting its own duration:

1. `ruff format` + `ruff check --fix` on staged Python, re-staged.
2. `ruff check --no-fix` and `ty check`.
3. `biston scan --files-from -` — clone pairs involving a staged file. The
   whole tree stays in the comparison, so a staged file cloning an untouched
   one is still caught.
4. `gerenuk changed-symbols` → `tyf refs --tests` → `pytest` on just the
   impacted test files, via `scripts/impacted_tests.py`.

Stage 4 is conservative by construction: a change to `conftest.py`,
`pyproject.toml` or `uv.lock`, a symbol that maps to no test, or any tool
error makes the selector exit non-zero, and the hook then runs the full
suite. It never turns an inconclusive answer into a skipped test.

Bypass the hook with `git commit --no-verify`.

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

Signal failures by raising `HTTPException` — never by returning a 200 whose
body says something went wrong. `api/errors.py` turns anything raised into an
error fragment, a full page, or JSON, picked from the request, so a handler
needs no error markup and no per-element `hx-` attributes of its own. See
[Error handling](usage/web-ui.md#error-handling) for what the user sees. A
panel that degrades to a partial result — a chart that could not be built, a
tab with no data — is a successful request and stays a 200.

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
