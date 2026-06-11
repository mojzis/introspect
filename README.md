# Introspect

Explore and search your Claude Code conversation logs using SQL, full-text search, a web UI, or an MCP server.

Published on PyPI as [`introspy`](https://pypi.org/project/introspy/).

## Installation

You need [uv](https://docs.astral.sh/uv/). If you don't have it (it also takes care of Python for you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Just trying it out?** Run it directly, nothing to install:

```bash
uvx introspy serve
```

**Using it regularly?** Install it as a tool, then call it by name:

```bash
uv tool install introspy
introspy serve
```

## Usage

### Web UI

```bash
introspy serve
# Runs on http://127.0.0.1:8347 by default
introspy serve --port 3000 --host 0.0.0.0
```

### CLI

```bash
# List recent sessions
introspy sessions

# Show summary statistics
introspy stats

# Search conversation logs
introspy search "some query"

# Show tool call history
introspy tools
introspy tools --failed
introspy tools --name Bash

# Run an ad-hoc SQL query
introspy query "SELECT * FROM logical_sessions LIMIT 5"

# Rebuild the search index
introspy refresh
```

### MCP Server

```bash
introspy mcp
```

This starts an MCP server over stdio for integration with Claude Code.

Alternatively, the web server exposes the same MCP tools over HTTP at
`http://127.0.0.1:8347/mcp`. To launch a Claude Code session wired up to it:

```bash
# In one terminal
introspy serve

# In another
introspy claude
```

The MCP config is passed inline to `claude`, so the server is only registered
for that session — no changes to your global Claude Code config.

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
