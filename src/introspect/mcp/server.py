"""MCP server for introspect."""

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from introspect.mcp._register import register_prompts, register_tools
from introspect.sql_query import is_loopback_host

# The streamable-HTTP endpoint is mounted at /mcp on the same loopback-bound
# app as the web UI, so it is reachable from any page the user has open in a
# browser. The SDK's DNS-rebinding protection checks Host and Origin inside
# the transport, which also covers the sub-app's own paths.
#
# ``[::1]`` is spelled bracketed here because that is how it appears in a Host
# header. Ports are enumerated because the SDK matches the header verbatim,
# unlike Starlette's TrustedHostMiddleware, which strips the port; the
# wildcard entries cover a server started on a non-default port.
_LOOPBACK_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
_LOOPBACK_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[*_LOOPBACK_HOSTS, *(f"{h}:*" for h in _LOOPBACK_HOSTS)],
    allowed_origins=[
        *(f"http://{h}" for h in _LOOPBACK_HOSTS),
        *(f"http://{h}:*" for h in _LOOPBACK_HOSTS),
    ],
)
# A deliberate non-loopback bind (``serve --host 0.0.0.0``) is the user asking
# to reach this from another machine, and we cannot know which hostnames they
# will use. Mirrors ``api.main.host_allowlist_applies``.
_OPEN_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

# Sent to MCP clients at initialize. Most users connect from outside this
# repo and have no other context about the data, so this carries the schema
# orientation that CLAUDE.md provides locally.
INSTRUCTIONS = """\
Introspect explores Claude Code conversation logs (~/.claude/projects/**/*.jsonl)
materialized into a read-only DuckDB.

Workflow: call `describe_schema` first, then `run_sql` (single SELECT/WITH
statement, capped at 500 rows / 64 KB / 20 s) for anything the canned tools
don't cover.
Prefer these tools over reading the JSONL files directly — the views already
handle session stitching, cost attribution, and project resolution.

Key views:
- session_stats — one row per session: project, started_at, duration, model,
  cost_usd, tool_count, files_read, files_edited, first_prompt
- logical_sessions — session metadata (continuation-aware)
- tool_calls — every tool invocation: tool_name, tool_input, is_error,
  execution_time
- conversation_turns — ordered user/assistant text per session
- assistant_message_costs — per-message token counts and cost
- file_reads / file_writes, message_commands — file and slash-command activity
- session_context_loads — harness-injected context per session (one row each):
  load_kind (claude_md | file_ref | skill_listing | mcp | hook), name, char_len

Tips:
- Costs: use session_stats.cost_usd or assistant_message_costs; don't
  recompute from raw token counts.
- For ranked expensive sessions with cost split, spend shape, and Pareto
  analysis, call `expensive_sessions` instead of hand-rolling SQL — it
  mirrors the web Cost Overview page and accepts an optional `since` filter.
- Raw JSONL fields live in raw_data / raw_messages; use json_extract() for
  nested values.
- Data refreshes every ~10 minutes; call `refresh_data` to pick up a session
  that just ended.

Example — top sessions by cost:
  SELECT project, started_at::date AS day, round(cost_usd, 2) AS usd,
         left(first_prompt, 60) AS prompt
  FROM session_stats ORDER BY cost_usd DESC LIMIT 10
"""


def create_mcp_server(bind_host: str = "") -> FastMCP:
    """Create a fresh MCP server instance with all tools registered.

    ``bind_host`` is the address the HTTP server bound to, used only to
    decide whether the transport enforces its loopback host/origin
    allowlists. The default — an empty string, and what the stdio entry point
    passes — enforces them; stdio has no HTTP transport for them to apply to.
    """
    security = (
        _OPEN_TRANSPORT_SECURITY
        if bind_host and not is_loopback_host(bind_host)
        else _LOOPBACK_TRANSPORT_SECURITY
    )
    server = FastMCP(
        "introspect",
        instructions=INSTRUCTIONS,
        transport_security=security,
    )
    # Serve the streamable HTTP endpoint at the sub-app root so that mounting
    # it at `/mcp` in FastAPI yields a final path of `/mcp`, not `/mcp/mcp`.
    server.settings.streamable_http_path = "/"
    register_tools(server)
    register_prompts(server)
    return server
