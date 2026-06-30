"""Register MCP tools on a FastMCP instance."""

from __future__ import annotations

from types import FunctionType
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
        tool_failure_rate,
        tool_failures,
    )
    from introspect.query_templates import templates_by_kind  # noqa: PLC0415

    registered_names: set[str] = set()

    def _register(fn: FunctionType) -> None:
        mcp.tool()(fn)
        registered_names.add(fn.__name__)

    _register(search_conversations)
    _register(get_session)
    _register(recent_sessions)
    _register(tool_failures)
    _register(expensive_sessions)
    _register(run_sql)
    _register(describe_schema)
    _register(refresh_data)
    _register(list_query_templates)

    # Deterministic-template adapters: every kind="deterministic" entry in
    # the query-template registry gets a dedicated MCP tool that binds and
    # executes the registry SQL directly — *unless* a tool of that name was
    # already registered above. `expensive_sessions` IS a deterministic
    # template (see query_templates.py), but its hand-built tool above is
    # richer (Pareto + spend shape + split) than a literal-SQL passthrough
    # would be, so it's deliberately skipped here rather than re-registered
    # or clobbered — the `registered_names` check below is what keeps the
    # richer tool from being shadowed by a generated one.
    _deterministic_tool_fns = {"tool_failure_rate": tool_failure_rate}
    _deterministic_templates = templates_by_kind("deterministic")
    _template_names = {t.name for t in _deterministic_templates}
    _orphaned_fns = set(_deterministic_tool_fns) - _template_names
    if _orphaned_fns:
        verb = "is" if len(_orphaned_fns) == 1 else "are"
        raise RuntimeError(  # noqa: TRY003
            f"_deterministic_tool_fns in register_tools references "
            f"{sorted(_orphaned_fns)}, which {verb} not a kind='deterministic' "
            "template in query_templates.py. Remove the stale entry or fix "
            "the template's kind/name."
        )
    for template in _deterministic_templates:
        if template.name in registered_names:
            continue
        fn = _deterministic_tool_fns.get(template.name)
        if fn is not None:
            _register(fn)
