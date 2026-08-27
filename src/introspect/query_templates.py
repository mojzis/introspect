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

from introspect.sql_fragments import (
    CONTEXT_LOADS_ROLLUP_SQL,
    COST_EXPR_SQL,
    SKILLS_INVOKED_ROLLUP_SQL,
)

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

# Heuristic regex for file/dir references in a first prompt. Catches @mentions,
# rooted paths (/a/b, ./a/b, ~/a/b), and any token with a slash + file
# extension; deliberately misses bare filenames without a slash to avoid
# matching prose. Shared so ``first_prompt_triggers`` and the ``/triggers`` web
# page can't drift.
FIRST_PROMPT_PATH_REGEX = (
    r"(?:@[\w./~-]+)|(?:[~.]?/[\w.~-]+/[\w./~-]+)|(?:[\w-]+/[\w./~-]*\.[\w]+)"
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
            "rate is meaningless. Also registered as the `tool_failure_rate` "
            "MCP tool, which executes this exact SQL and formats the rows — "
            "prefer that tool over run_sql for this question."
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
        name="first_prompt_triggers",
        question=(
            "For each session, what did the opening prompt reference and "
            "trigger — file/dir paths, skills, slash commands?"
        ),
        sql=(
            "WITH first_prompts AS (\n"  # noqa: S608 — path regex is a constant
            "    SELECT session_id, project, started_at, first_prompt, commands\n"
            "    FROM session_stats\n"
            "    WHERE first_prompt IS NOT NULL\n"
            "      AND ($project IS NULL OR project = $project)\n"
            "),\n"
            "refs AS (\n"
            "    SELECT session_id, regexp_extract_all(\n"
            "        first_prompt,\n"
            "        '" + FIRST_PROMPT_PATH_REGEX + "'"
            "\n    ) AS referenced_paths\n"
            "    FROM first_prompts\n"
            "),\n"
            "skills AS (" + SKILLS_INVOKED_ROLLUP_SQL + "),\n"
            "loads AS (" + CONTEXT_LOADS_ROLLUP_SQL + ")\n"
            "SELECT\n"
            "    fp.session_id,\n"
            "    fp.project,\n"
            "    fp.started_at,\n"
            "    len(r.referenced_paths) AS n_referenced_paths,\n"
            "    r.referenced_paths,\n"
            "    COALESCE(s.skills_invoked, 0) AS skills_invoked,\n"
            "    COALESCE(l.auto_loaded_claude_md, FALSE)\n"
            "        AS auto_loaded_claude_md,\n"
            "    COALESCE(l.n_auto_loaded_files, 0) AS n_auto_loaded_files,\n"
            "    COALESCE(l.skill_menu_loaded, FALSE) AS skill_menu_loaded,\n"
            "    fp.commands,\n"
            "    left(fp.first_prompt, 200) AS first_prompt\n"
            "FROM first_prompts fp\n"
            "LEFT JOIN refs r USING (session_id)\n"
            "LEFT JOIN skills s USING (session_id)\n"
            "LEFT JOIN loads l USING (session_id)\n"
            "ORDER BY fp.started_at DESC\n"
            "LIMIT $limit"
        ),
        params=(
            _LIMIT_PARAM,
            Param(
                name="project",
                type="str",
                required=False,
                default=None,
                description=(
                    "Restrict to one project (session_stats.project). "
                    "None means all projects."
                ),
            ),
        ),
        note=(
            "Answers 'did my opening prompt set the session up right?'. "
            "referenced_paths is a HEURISTIC regex over first_prompt: it "
            "catches @mentions, rooted paths (/a/b, ./a/b, ~/a/b), and any "
            "token with a slash + file extension. It deliberately misses "
            "bare filenames without a slash (e.g. 'fix db.py') to avoid "
            "matching prose like 'e.g.'; widen the pattern if you want those. "
            "skills_invoked counts Skill tool calls (kind='agent_tool_call', "
            "tool_name='Skill') — this covers BOTH skills the user typed as "
            "/name AND skills the model auto-triggered, since both emit a "
            "Skill tool_use. commands reuses session_stats.commands (the "
            "canonical COMMAND_LIST_SUBQUERY rollup — noise commands like "
            "/clear and /compact already filtered out), so it can't drift "
            "from the sessions listing. The auto_loaded_* columns roll up "
            "session_context_loads (harness-injected context): "
            "auto_loaded_claude_md is TRUE when a nested CLAUDE.md / "
            ".claude/rules file loaded on directory entry; n_auto_loaded_files "
            "counts @-file expansions; skill_menu_loaded is TRUE when the "
            "skill menu was shown. (The ROOT/global CLAUDE.md arrives inline "
            "as a first-message <system-reminder>, not an attachment, so it is "
            "NOT counted — near-universal anyway.) Exploratory because the "
            "payoff is comparing prompt WORDING against what loaded: e.g. "
            "sessions where n_referenced_paths=0 and skills_invoked=0 are "
            "candidates for a vaguer opening prompt."
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
    QueryTemplate(
        name="cache_ttl_choice",
        question="Which prompt-cache TTL should I set — 5m or 1h?",
        sql=(
            "SELECT\n"
            "    COALESCE(ls.project, '?') AS project,\n"
            "    COUNT(*) AS n_requests,\n"
            "    COUNT(*) FILTER (WHERE cr.gap_recoverable) "
            "AS n_gaps_recoverable,\n"
            "    COUNT(*) FILTER (WHERE cr.gap_unrecoverable) "
            "AS n_breaks_over_1h,\n"
            "    COUNT(*) FILTER (WHERE cr.structural_invalidation) "
            "AS n_structural,\n"
            "    mode(cr.ttl_observed) FILTER (WHERE cr.ttl_observed <> "
            "'unknown') AS ttl_observed,\n"
            "    round(SUM(cr.cost_5m_usd), 2) AS cost_5m,\n"
            "    round(SUM(cr.cost_1h_usd), 2) AS cost_1h,\n"
            "    round(SUM(cr.cost_1h_usd) - SUM(cr.cost_5m_usd), 2) "
            "AS delta_1h_minus_5m,\n"
            "    round(100 * (SUM(cr.cost_1h_usd) - SUM(cr.cost_5m_usd))\n"
            "          / NULLIF(LEAST(SUM(cr.cost_5m_usd), "
            "SUM(cr.cost_1h_usd)), 0), 1) AS margin_pct\n"
            "FROM cache_requests cr\n"
            "LEFT JOIN logical_sessions ls ON ls.session_id = cr.session_id\n"
            "WHERE cr.is_sidechain = $sidechain\n"
            "  AND ($since IS NULL OR cr.timestamp >= $since::TIMESTAMP)\n"
            "GROUP BY 1\n"
            "ORDER BY GREATEST(SUM(cr.cost_5m_usd), SUM(cr.cost_1h_usd)) DESC\n"
            "LIMIT $limit"
        ),
        params=(
            _LIMIT_PARAM,
            Param(
                name="sidechain",
                type="bool",
                required=False,
                default=False,
                description=(
                    "False scores the main conversation (promptCacheTtl). "
                    "True scores subagents, which have their own "
                    "subagentPromptCacheTtl — never merge the two."
                ),
            ),
            Param(
                name="since",
                type="str",
                required=False,
                default=None,
                description=(
                    "ISO date/timestamp; only requests at or after this "
                    "point. None means all time."
                ),
            ),
        ),
        note=(
            "Negative delta_1h_minus_5m means 1h is cheaper. Read margin_pct "
            "before acting: under ~2% is inside the simulation's modelling "
            "error and is not a decision. The comparison works because "
            "prefix_total (cache_read + cache_creation) is a property of the "
            "conversation, not of the TTL — only the read/write split moves, "
            "and cost_5m_usd / cost_1h_usd in cache_requests replay each "
            "request under both policies. n_gaps_recoverable counts gaps "
            "between the TTL a row was actually billed at and 60 min — the "
            "only ones a longer TTL rescues; n_breaks_over_1h is reported "
            "separately because no setting recovers those, and folding them "
            "in overstates the case for switching. Group by project, not "
            "session: the setting is per user/project, so per-session rows "
            "(session_cache_ttl) are diagnostics only."
        ),
        kind="deterministic",
    ),
)


def get_template(name: str) -> QueryTemplate | None:
    """Look up a template by name, or None if no such template exists.

    Used by deterministic-tool adapters (e.g. ``mcp.tools.tool_failure_rate``)
    to fetch the registry entry whose SQL they bind and execute.
    """
    for template in QUERY_TEMPLATES:
        if template.name == name:
            return template
    return None


def templates_by_kind(kind: TemplateKind) -> tuple[QueryTemplate, ...]:
    """Return all templates of the given kind."""
    return tuple(t for t in QUERY_TEMPLATES if t.kind == kind)
