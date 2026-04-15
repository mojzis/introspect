# Introspect

Explore and search your Claude Code conversation logs using SQL, full-text search, a web UI, or an MCP server.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync
```

Activate the project's virtual environment before running the `introspect`
commands below (or prefix each one with `uv run`):

```bash
source .venv/bin/activate
```

## Usage

### CLI

```bash
# List recent sessions
introspect sessions

# Show summary statistics
introspect stats

# Search conversation logs
introspect search "some query"

# Show tool call history
introspect tools
introspect tools --failed
introspect tools --name Bash

# Run an ad-hoc SQL query
introspect query "SELECT * FROM logical_sessions LIMIT 5"

# Rebuild the search index
introspect refresh
```

### Web UI

```bash
introspect serve
# Runs on http://127.0.0.1:8000 by default
introspect serve --port 3000 --host 0.0.0.0
```

### MCP Server

```bash
introspect mcp
```

This starts an MCP server over stdio for integration with Claude Code.

Alternatively, the web server exposes the same MCP tools over HTTP at
`http://127.0.0.1:8000/mcp`. To launch a Claude Code session wired up to it:

```bash
# In one terminal
introspect serve

# In another
uv run poe claude
```

The `claude` poe task runs `claude --mcp-config .claude/mcp.json`, so the MCP
server is only registered for that session — no changes to your global Claude
Code config.

## Development

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
