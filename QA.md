# Functional QA: standalone MCP startup

Run the isolated synthetic consumer from the repository root:

```bash
uv run python scripts/qa_mcp_startup.py
```

The script spawns real `introspy mcp` stdio processes for cold startup, warm
startup, terminal startup and directory-preparation failures, unlimited history (`INTROSPECT_DAYS=0`),
and a larger synthetic transcript containing 10,000 messages. It performs
`initialize`, `tools/list`, `recent_sessions`, and `refresh_data` calls, then
prints measured timings and observed results as JSON. Handshake and discovery
must complete within five seconds. The route has a two-minute overall timeout.

Cold and warm runs must expose partial data before publishing complete,
unmarked results. Unlimited cold startup must publish complete, ready results
directly. Every successful run disables periodic refresh with interval `0`;
a later request to change the window must report manual refresh unavailable
and retain the startup target. The failure case must preserve its cause in
both data and refresh responses.

All inputs and databases are disposable synthetic files, removed afterward.
No pytest, conversation logs, shared `~/.introspect` database, or real
home-directory glob is used as evidence.
