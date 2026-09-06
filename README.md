# Introspect

Explore and search your Claude Code (and Codex) conversation logs using SQL,
full-text search, a web UI, or an MCP server.

Every session your coding agent runs is written to a JSONL file under
`~/.claude/projects/`. Introspect reads those files into DuckDB and lets you ask
what they cost, which tools failed, where a session went off the rails, and what
your opening prompt actually loaded.

Published on PyPI as [`introspy`](https://pypi.org/project/introspy/).
Full documentation: **<https://mojzis.github.io/introspect/>**

## Quick start

You need [uv](https://docs.astral.sh/uv/). Then:

```bash
uvx introspy@latest serve
```

Open <http://127.0.0.1:8347>. Nothing is installed and nothing leaves your
machine.

Using it regularly? `uv tool install introspy`, then `introspy serve`.
See [Installation](https://mojzis.github.io/introspect/installation/).

## Documentation

| Page | What's there |
|---|---|
| [Installation](https://mojzis.github.io/introspect/installation/) | `uvx` vs `uv tool install`, where your data comes from, the update check |
| [Web UI](https://mojzis.github.io/introspect/usage/web-ui/) | Every page, the cost analytics, tokenscape, trajectory, cache loss, triggers |
| [CLI](https://mojzis.github.io/introspect/usage/cli/) | What each command answers — plus a [generated reference](https://mojzis.github.io/introspect/usage/cli-reference/) for every option |
| [MCP server](https://mojzis.github.io/introspect/usage/mcp/) | Tools, query templates, prompts, and the `introspy claude` / `introspy codex` launchers |
| [SQL API](https://mojzis.github.io/introspect/usage/sql-api/) | Local-only JSON SQL endpoint for notebooks |
| [Architecture](https://mojzis.github.io/introspect/architecture/) | Module map, every relation and its columns, pricing, the read-only SQL guard |
| [JSONL schema](https://mojzis.github.io/introspect/schema/) | The raw records on disk, including the `usage` block |
| [Configuration](https://mojzis.github.io/introspect/configuration/) | Every `INTROSPECT_*` environment variable |
| [Development](https://mojzis.github.io/introspect/development/) | `poe` tasks, the toolbox (`tyf`, `gerenuk`, `biston`, `zorilla`), the commit hook, worktrees, building the docs |
| [For LLMs](https://mojzis.github.io/introspect/llms/) | `llms.txt` and `llms-full.txt` |

## Development

```bash
uv sync
uv run poe check   # lint, typecheck, dead code, deps, vulns, tests
uv run poe fix     # auto-format and fix lint
```

See [Development](https://mojzis.github.io/introspect/development/) for the rest.
