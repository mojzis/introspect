"""MCP server for introspect."""

from mcp.server.fastmcp import FastMCP

from introspect.mcp._register import register_tools


def create_mcp_server() -> FastMCP:
    """Create a fresh MCP server instance with all tools registered."""
    server = FastMCP("introspect")
    # Serve the streamable HTTP endpoint at the sub-app root so that mounting
    # it at `/mcp` in FastAPI yields a final path of `/mcp`, not `/mcp/mcp`.
    server.settings.streamable_http_path = "/"
    register_tools(server)
    return server
