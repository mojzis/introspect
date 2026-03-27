"""MCP server for introspect."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("introspect")

# Register tools by importing the module (side effect: decorators run)
import introspect.mcp.tools as _tools  # noqa: F401, E402
