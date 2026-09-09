# Functional QA: standalone MCP startup

Run the isolated synthetic consumer from the repository root:

```bash
uv run python scripts/qa_mcp_startup.py
```

The script creates two synthetic Claude sessions in a temporary directory,
spawns `introspy mcp` twice over the real stdio transport, and removes the
directory afterward. It performs `initialize`, `tools/list`, and
`recent_sessions` calls, then prints one JSON object with measured cold and
warm initialize/tool-discovery timings and the synthetic results.

The cold run demonstrates the loading response, the one-day preview, and the
later authoritative result containing both sessions. The warm run starts from
the database produced by the cold run, exercises the immediately available
warm snapshot, and checks the subsequent authoritative result. No pytest,
conversation logs, shared `~/.introspect` database, or real home-directory
glob is used as evidence.
