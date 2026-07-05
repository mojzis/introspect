"""MCP server for introspect."""

from mcp.server.fastmcp import FastMCP

from introspect.mcp._register import register_prompts, register_tools

# Sent to MCP clients at initialize. Most users connect from outside this
# repo and have no other context about the data, so this carries the schema
# orientation that CLAUDE.md provides locally.
INSTRUCTIONS = """\
Introspect explores Claude Code conversation logs (~/.claude/projects/**/*.jsonl)
materialized into a read-only DuckDB.

Workflow: call `describe_schema` first, then `run_sql` (single SELECT/WITH
statement, capped at 500 rows) for anything the canned tools don't cover.
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


def create_mcp_server() -> FastMCP:
    """Create a fresh MCP server instance with all tools registered."""
    server = FastMCP("introspect", instructions=INSTRUCTIONS)
    # Serve the streamable HTTP endpoint at the sub-app root so that mounting
    # it at `/mcp` in FastAPI yields a final path of `/mcp`, not `/mcp/mcp`.
    server.settings.streamable_http_path = "/"
    register_tools(server)
    register_prompts(server)
    return server
