"""MCP server for introspect."""

from mcp.server.fastmcp import FastMCP

from introspect.mcp._register import register_tools


def create_mcp_server() -> FastMCP:
    """Create a fresh MCP server instance with all tools registered."""
    server = FastMCP("introspect")
    register_tools(server)
    return server
