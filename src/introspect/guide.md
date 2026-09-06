# introspect guide

introspect is not part of the toolbox setup on purpose: it reads your own
Claude Code and Codex conversation logs, so a human opts in, not a setup
prompt. This page is the pitch and the first three commands. Nothing on it
writes to those logs.

## What it records

Nothing new. Claude Code already writes every session to
`~/.claude/projects/**/*.jsonl`, and Codex to `~/.codex/sessions/**/*.jsonl`.
introspect reads those files into a DuckDB database at
`~/.introspect/introspect.duckdb` and turns them into relations you can
query: sessions with duration, model and working directory; every tool call
with its input, result and error flag; tokens and dollars per assistant
message, session and project; file reads and writes; slash commands and
skills; prompt-cache hits, misses and the gaps that broke them; and full-text
search over all of it. `introspy tables` lists the relations.

## What it costs

- Disk: one database file, a few times the size of the logs it reads.
- Time: the first command builds the database; later ones reuse it and
  refresh what changed. `introspy serve` starts on the last 10 days and fills
  the rest in behind you.
- Network: none for your logs, ever. The one request it makes is a daily
  PyPI check for a newer release; `INTROSPECT_VERSION_CHECK=off` stops it.
- Exposure: the web UI binds to 127.0.0.1, and every SQL path, including
  `introspy query`, runs on a connection with filesystem and network access
  disabled. Your logs are the whole attack surface, and they stay put.

## Try it in three commands

```bash
introspy stats
introspy sessions
introspy query "SELECT project, cost_usd FROM session_stats ORDER BY cost_usd DESC LIMIT 10"
```

`stats` is the overview: sessions, tool calls, failures, and which
prompt-cache TTL each project should be on. `sessions` is the recent list.
`query` is read-only SQL over every relation `introspy tables` names, with
`session_stats` as the one to reach for first.

## When the terminal is not enough

- `introspy serve` opens the web UI at http://127.0.0.1:8347: sessions, tool
  calls, where the money went, search.
- `introspy mcp` serves the same data to Claude Code or Codex as MCP tools, so
  you ask "what did my agent do all day?" in prose. `introspy claude` launches
  Claude Code already wired to it.

Install with `uv tool install introspy`, or keep using `uvx introspy@latest`.
Both `introspy` and `introspect` are installed as entry points.

next: run `introspy stats`
