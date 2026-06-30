"""Register MCP tools on a FastMCP instance."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_tools(mcp: FastMCP) -> None:
    """Register all introspect MCP tools on the given server instance."""
    from introspect.mcp.tools import (  # noqa: PLC0415
        describe_schema,
        expensive_sessions,
        get_session,
        list_query_templates,
        recent_sessions,
        refresh_data,
        run_sql,
        search_conversations,
        tool_failures,
    )

    mcp.tool()(search_conversations)
    mcp.tool()(get_session)
    mcp.tool()(recent_sessions)
    mcp.tool()(tool_failures)
    mcp.tool()(expensive_sessions)
    mcp.tool()(run_sql)
    mcp.tool()(describe_schema)
    mcp.tool()(refresh_data)
    mcp.tool()(list_query_templates)
