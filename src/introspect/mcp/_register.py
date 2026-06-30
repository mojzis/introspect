"""Register MCP tools on a FastMCP instance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from introspect.query_templates import TemplateKind


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

    _register_deterministic_template_tools(registered_names, _register)


@dataclass(frozen=True)
class _AdapterSpec:
    """Static, per-call-site description used to phrase drift errors."""

    kind: TemplateKind
    fns_var_name: str
    caller_name: str
    adapter_label: str
    extra_missing_hint: str = ""


def _wire_template_adapters(
    spec: _AdapterSpec,
    fns: dict[str, FunctionType],
    register: Callable[[FunctionType], None],
    skip_names: frozenset[str] = frozenset(),
) -> None:
    """Wire a {template name: adapter fn} mapping against the registry.

    Shared by `_register_deterministic_template_tools` (MCP tools) and
    `register_prompts` (MCP prompts): both raise rather than silently
    registering nothing when the registry and the adapter dict drift apart —
    either `fns` names a template that doesn't exist (orphaned fn), or a
    `kind=...` template (not in `skip_names`) has no matching fn (missing
    adapter).
    """
    from introspect.query_templates import templates_by_kind  # noqa: PLC0415

    templates = templates_by_kind(spec.kind)
    template_names = {t.name for t in templates}
    orphaned_fns = set(fns) - template_names
    if orphaned_fns:
        verb = "is" if len(orphaned_fns) == 1 else "are"
        raise RuntimeError(  # noqa: TRY003
            f"{spec.fns_var_name} in {spec.caller_name} references "
            f"{sorted(orphaned_fns)}, which {verb} not a kind={spec.kind!r} "
            "template in query_templates.py. Remove the stale entry or fix "
            "the template's kind/name."
        )
    for template in templates:
        if template.name in skip_names:
            continue
        fn = fns.get(template.name)
        if fn is None:
            raise RuntimeError(  # noqa: TRY003
                f"No {spec.adapter_label} registered for kind={spec.kind!r} "
                f"template {template.name!r}. Add it to {spec.fns_var_name} "
                f"in {spec.caller_name}.{spec.extra_missing_hint}"
            )
        register(fn)


def _register_deterministic_template_tools(
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
    """
    from introspect.mcp.tools import tool_failure_rate  # noqa: PLC0415

    deterministic_tool_fns = {"tool_failure_rate": tool_failure_rate}
    spec = _AdapterSpec(
        kind="deterministic",
        fns_var_name="deterministic_tool_fns",
        caller_name="_register_deterministic_template_tools",
        adapter_label="MCP tool",
        extra_missing_hint=(
            " Or hand-register a tool of the same name in register_tools."
        ),
    )
    _wire_template_adapters(
        spec, deterministic_tool_fns, register, frozenset(registered_names)
    )


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

    exploratory_prompt_fns = {
        "session_cost_tail": session_cost_tail,
        "topic_to_cost": topic_to_cost,
    }

    def _register_prompt(fn: FunctionType) -> None:
        mcp.prompt()(fn)

    spec = _AdapterSpec(
        kind="exploratory",
        fns_var_name="exploratory_prompt_fns",
        caller_name="register_prompts",
        adapter_label="prompt function",
    )
    _wire_template_adapters(spec, exploratory_prompt_fns, _register_prompt)
