"""Curated SQL investigation templates — the registry adapters fan out from.

Leaf module: imports only from ``introspect.sql_fragments`` (which itself
wraps ``introspect.pricing``) and stdlib. No imports from ``introspect.api``,
``introspect.mcp``, ``introspect.search``, or ``introspect.db`` — mirrors why
``sql_fragments.py`` is a leaf.

Each :class:`QueryTemplate` carries everything the three adapters need:

- the **cookbook** (``mcp.tools.list_query_templates``) renders every entry
  as a reference the model adapts and runs through ``run_sql``;
- **deterministic** entries get registered as MCP tools that bind params and
  execute the registry SQL directly;
- **exploratory** entries get registered as MCP prompts that seed an
  investigation.

SQL uses DuckDB ``$named`` placeholders so the same string both binds for
execution and reads self-documenting in the cookbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from introspect.sql_fragments import COST_EXPR_SQL

TemplateKind = Literal["deterministic", "exploratory"]


@dataclass(frozen=True)
class Param:
    """One bindable parameter of a :class:`QueryTemplate`.

    ``default`` is intentionally typed loosely (``object | None``) rather than
    tied to ``type`` — this is metadata for rendering the cookbook, not a
    runtime coercion/validation schema.
    """

    name: str
    type: str
    required: bool
    default: object | None
    description: str


@dataclass(frozen=True)
class QueryTemplate:
    """A named, parameterized SQL investigation."""

    name: str
    question: str
    sql: str
    params: tuple[Param, ...]
    note: str
    kind: TemplateKind


# Shared across every template with an optional row cap — avoids three
# near-identical `Param("limit", "int", False, 20, "Max rows to return.")`
# literals drifting out of sync.
_LIMIT_PARAM = Param(
    name="limit",
    type="int",
    required=False,
    default=20,
    description="Max rows to return.",
)

QUERY_TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate(
        name="expensive_sessions",
        question="Which sessions cost the most?",
        sql=(
            "SELECT session_id, project, started_at, model, "
            "round(cost_usd, 2) AS cost_usd\n"
            "FROM session_stats\n"
            "WHERE cost_usd IS NOT NULL\n"
            "  AND ($since IS NULL OR started_at >= $since::TIMESTAMP)\n"
            "ORDER BY cost_usd DESC\n"
            "LIMIT $limit"
        ),
        params=(
            _LIMIT_PARAM,
            Param(
                name="since",
                type="str",
                required=False,
                default=None,
                description=(
                    "ISO date/timestamp; only sessions started at or after "
                    "this point. None means all time."
                ),
            ),
        ),
        note=(
            "Cost comes from session_stats.cost_usd (already deduped via "
            "assistant_message_costs); never sum raw_messages — duplicate "
            "copies of the same response over-count cost. Not registered as "
            "an MCP tool: the existing richer `expensive_sessions` tool "
            "(Pareto + spend shape + split) already covers this question — "
            "this entry is cookbook + integrity-test only."
        ),
        kind="deterministic",
    ),
    QueryTemplate(
        name="tool_failure_rate",
        question="Which tools fail most, by rate and count?",
        sql=(
            "SELECT tool_name,\n"
            "       COUNT(*) AS calls,\n"
            "       COUNT(*) FILTER (WHERE is_error = 'true') AS failures,\n"
            "       COUNT(*) FILTER (WHERE is_error = 'true') * 1.0 "
            "/ COUNT(*) AS failure_rate\n"
            "FROM tool_calls\n"
            "WHERE ($since IS NULL OR called_at >= $since::TIMESTAMP)\n"
            "GROUP BY tool_name\n"
            "HAVING COUNT(*) >= $min_calls\n"
            "ORDER BY failure_rate DESC, calls DESC\n"
            "LIMIT $limit"
        ),
        params=(
            _LIMIT_PARAM,
            Param(
                name="since",
                type="str",
                required=False,
                default=None,
                description=(
                    "ISO date/timestamp; only calls at or after this point. "
                    "None means all time."
                ),
            ),
            Param(
                name="min_calls",
                type="int",
                required=False,
                default=5,
                description=(
                    "Suppress low-N noise: only tools with at least this "
                    "many calls are included."
                ),
            ),
        ),
        note=(
            "is_error is compared as the string 'true', not a SQL boolean — "
            "tool_calls stores it as JSON/bool serialized to text. "
            "min_calls filters out tools called once or twice whose failure "
            "rate is meaningless. Not yet a dedicated tool — run this SQL "
            "via run_sql for now."
        ),
        kind="deterministic",
    ),
    QueryTemplate(
        name="session_cost_tail",
        question=(
            "For one session, where does cumulative cost decouple from "
            "progress — is there a derailment tail?"
        ),
        sql=(
            "SELECT\n"
            "    timestamp,\n"
            "    ROW_NUMBER() OVER (ORDER BY timestamp) AS turn_index,\n"
            f"    ({COST_EXPR_SQL}) / 1e6 AS msg_cost,\n"
            f"    SUM(({COST_EXPR_SQL}) / 1e6) OVER (\n"
            "        ORDER BY timestamp ROWS UNBOUNDED PRECEDING\n"
            "    ) AS cum_cost\n"
            "FROM assistant_message_costs\n"
            "WHERE session_id = $session_id\n"
            "ORDER BY timestamp"
        ),
        params=(
            Param(
                name="session_id",
                type="str",
                required=True,
                default=None,
                description=(
                    "Session to inspect (pass to get_session for full context)."
                ),
            ),
        ),
        note=(
            "Rows are already deduped by message.id in "
            "assistant_message_costs. Cost is computed per-row from the "
            "pricing CASE expression (COST_EXPR_SQL), so mixed-model "
            "sessions price correctly. Exploratory: locate the inflection "
            "point where cum_cost accelerates relative to turn_index, and "
            "report the tail's share of total cost — that judgment call is "
            "why this is a prompt, not a fixed tool."
        ),
        kind="exploratory",
    ),
    QueryTemplate(
        name="topic_to_cost",
        question="Sessions about <topic>, ranked by cost.",
        sql=(
            "SELECT ss.session_id, ss.project, round(ss.cost_usd, 2) "
            "AS cost_usd\n"
            "FROM session_stats ss\n"
            "JOIN (\n"
            "    SELECT DISTINCT session_id FROM search_corpus\n"
            "    WHERE content_text ILIKE '%' || $query || '%'\n"
            ") m USING (session_id)\n"
            "WHERE ss.cost_usd IS NOT NULL\n"
            "ORDER BY ss.cost_usd DESC\n"
            "LIMIT $limit"
        ),
        params=(
            Param(
                name="query",
                type="str",
                required=True,
                default=None,
                description="Substring to search for.",
            ),
            _LIMIT_PARAM,
        ),
        note=(
            "ILIKE is a coarse substring match over search_corpus — it has "
            "no ranking and no AND/OR semantics across words. For better "
            "relevance, prefer the search_conversations tool (BM25) to find "
            "the topic, then drill into the priciest hit with get_session. "
            "This entry stays self-contained over search_corpus because "
            "query_templates.py cannot import search.py (leaf constraint)."
        ),
        kind="exploratory",
    ),
)


def get_template(name: str) -> QueryTemplate | None:
    """Look up a template by name, or None if no such template exists.

    Not yet called internally — scaffolding for the Phase 2 deterministic
    tool registration, which will look up a template by name to bind and
    execute its SQL.
    """
    for template in QUERY_TEMPLATES:
        if template.name == name:
            return template
    return None


def templates_by_kind(kind: TemplateKind) -> tuple[QueryTemplate, ...]:
    """Return all templates of the given kind."""
    return tuple(t for t in QUERY_TEMPLATES if t.kind == kind)
