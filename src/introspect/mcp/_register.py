"""Register MCP tools on a FastMCP instance."""

from __future__ import annotations

from collections.abc import Callable
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
        tool_failures,
    )

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

    _register_deterministic_template_tools(mcp, registered_names, _register)


def _register_deterministic_template_tools(
    mcp: FastMCP,
    registered_names: set[str],
    register: Callable[[FunctionType], None],
) -> None:
    """Register the MCP tools backing every kind="deterministic" template.

    Every `kind="deterministic"` entry in the query-template registry gets a
    dedicated MCP tool that binds and executes the registry SQL directly —
    *unless* a tool of that name was already registered (via `register`)
    above. `expensive_sessions` IS a deterministic template (see
    `query_templates.py`), but its hand-built tool is richer (Pareto + spend
    shape + split) than a literal-SQL passthrough would be, so it's
    deliberately skipped here rather than re-registered or clobbered — the
    `registered_names` check is what keeps the richer tool from being
    shadowed by a generated one.

    Mirrors `register_prompts`' loud-failure shape: both directions of
    registry/adapter drift raise rather than silently registering nothing —
    a stale fn with no matching template, or a template with no fn and no
    pre-existing hand-built tool.
    """
    from introspect.mcp.tools import tool_failure_rate  # noqa: PLC0415
    from introspect.query_templates import templates_by_kind  # noqa: PLC0415

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
        if fn is None:
            raise RuntimeError(  # noqa: TRY003
                f"No MCP tool registered for kind='deterministic' template "
                f"{template.name!r}. Add it to _deterministic_tool_fns in "
                "_register_deterministic_template_tools, or hand-register a "
                "tool of the same name in register_tools."
            )
        register(fn)


def register_prompts(mcp: FastMCP) -> None:
    """Register all introspect MCP prompts on the given server instance.

    Exploratory-template adapters: every kind="exploratory" entry in the
    query-template registry gets a dedicated MCP prompt that seeds an
    investigation from the registry's SQL/note — the registry stays the
    source of truth by looping over `templates_by_kind("exploratory")` and
    pairing each entry with its named prompt function.
    """
    from introspect.mcp.prompts import (  # noqa: PLC0415
        session_cost_tail,
        topic_to_cost,
    )
    from introspect.query_templates import templates_by_kind  # noqa: PLC0415

    _exploratory_prompt_fns = {
        "session_cost_tail": session_cost_tail,
        "topic_to_cost": topic_to_cost,
    }
    _exploratory_templates = templates_by_kind("exploratory")
    _template_names = {t.name for t in _exploratory_templates}
    _orphaned_fns = set(_exploratory_prompt_fns) - _template_names
    if _orphaned_fns:
        verb = "is" if len(_orphaned_fns) == 1 else "are"
        raise RuntimeError(  # noqa: TRY003
            f"_exploratory_prompt_fns in register_prompts references "
            f"{sorted(_orphaned_fns)}, which {verb} not a kind='exploratory' "
            "template in query_templates.py. Remove the stale entry or fix "
            "the template's kind/name."
        )
    for template in _exploratory_templates:
        fn = _exploratory_prompt_fns.get(template.name)
        if fn is None:
            raise RuntimeError(  # noqa: TRY003
                f"No prompt function registered for kind='exploratory' "
                f"template {template.name!r}. Add it to "
                "_exploratory_prompt_fns in register_prompts."
            )
        mcp.prompt()(fn)
