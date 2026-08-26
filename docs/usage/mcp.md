# MCP server

Introspect exposes its data to Claude Code (or any MCP client) as a set of
tools, so you can ask questions about your own conversation history in natural
language.

## Over stdio

```bash
introspy mcp
```

This starts an MCP server over stdio for integration with Claude Code.

## Over HTTP

The web server exposes the same MCP tools over HTTP at
`http://127.0.0.1:8347/mcp`. To launch a Claude Code session wired up to it:

```bash
introspy claude
```

`introspy claude` starts `introspy serve` automatically in the background when
nothing is listening on the target port (log at `~/.introspect/serve.log`). The
MCP config is passed inline to `claude`, so the server is only registered for
that session — no changes to your global Claude Code config.

Any arguments after `--` are forwarded verbatim to the `claude` CLI, so you can
pass normal Claude Code options alongside:

```bash
introspy claude -- --model opus --resume
introspy claude -- -p "what are the most expensive sessions"
```

Once connected, try asking Claude:

> what are the most expensive sessions

### Codex

To launch a Codex session with the same temporary HTTP MCP connection:

```bash
introspy codex
introspy codex -- --model gpt-5.4
introspy codex -- "what are the most expensive sessions"
```

`introspy codex` starts `introspy serve` when needed and passes the MCP URL
and log-analysis developer instructions as Codex command-line configuration
overrides. It does not modify your Codex configuration. Pass `--keep-server`
to leave an auto-started server running after Codex exits.

## Available tools

| Tool | Description |
|---|---|
| `search_conversations` | Full-text search across sessions, with filters (`cwd_prefix`, `role`, `since`, `session_id`, `require_all`) and pagination. |
| `get_session` | Fetch the full content of a session. |
| `recent_sessions` | List recent sessions with metadata. |
| `tool_failures` | Find failed tool calls, optionally filtered by command prefix. |
| `run_sql` | Execute a read-only `SELECT` / `WITH` query. Writes, `ATTACH`, `PRAGMA`, `COPY`, and multi-statement scripts are rejected by a validator. |
| `describe_schema` | List views/tables and their columns. |
| `refresh_data` | Wake the refresh loop and wait for the rebuild (only when running embedded in `introspy serve`). |

For the underlying schema these tools query, see the
[Architecture](../architecture.md) and [JSONL schema](../schema.md) pages.
