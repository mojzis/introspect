"""MCP prompt definitions for introspect's exploratory query templates.

Each `kind="exploratory"` entry in the query-template registry
(``query_templates.py``) gets a prompt function here, registered via
``mcp.prompt()`` in ``_register.py``. A prompt expands into a single seed
message — built FROM the registry entry (its canonical ``$named`` SQL plus
its ``note``), with 1-2 suggested follow-ups specific to the question. The
seed text is a starting point to adapt, not a canned answer; nothing here
hand-duplicates SQL or notes that already live in the registry.
"""

from __future__ import annotations

from introspect.query_templates import QueryTemplate, get_template

_STARTING_POINT = (
    "This is a starting point to adapt, not a canned answer — adjust the "
    "SQL for what you're actually investigating, then run it through "
    "`run_sql`."
)


def _seed_text(template: QueryTemplate, follow_ups: list[str]) -> str:
    """Render a prompt seed message from a registry entry.

    Combines the entry's question, canonical SQL, note, and the caller's
    suggested follow-ups into the single user message FastMCP sends back for
    ``get_prompt``. The single place that assembles seed text from the
    registry, so the two exploratory prompts can't drift in shape.
    """
    follow_up_lines = "\n".join(f"- {f}" for f in follow_ups)
    return (
        f"Q: {template.question}\n\n"
        f"SQL:\n{template.sql}\n\n"
        f"Note: {template.note}\n\n"
        f"Suggested follow-ups:\n{follow_up_lines}\n\n"
        f"{_STARTING_POINT}"
    )


def session_cost_tail(session_id: str) -> str:
    """Seed an investigation into one session's cumulative cost tail.

    Expands the `session_cost_tail` registry entry — per-message cost and a
    running cumulative-cost total over `assistant_message_costs`, ordered by
    turn — into a starting prompt that asks the model to locate the
    inflection point where cumulative cost decouples from progress and
    report what share of total cost the tail accounts for.
    """
    template = get_template("session_cost_tail")
    if template is None:  # pragma: no cover - defensive, registry can't drop this
        return "Error: 'session_cost_tail' template not found in registry."
    return _seed_text(
        template,
        [
            "Locate the turn where cum_cost starts accelerating relative "
            f"to turn_index for session_id={session_id!r}, and report what "
            "fraction of the session's total cost falls after that turn "
            "(the 'tail').",
            "Once you've found the inflection point, call `get_session` "
            "to read what actually happened around that turn — a "
            "derailment, a large file dump, or legitimate heavy work?",
        ],
    )


def topic_to_cost(query: str, limit: int = 20) -> str:
    """Seed an investigation into the cost of sessions about a topic.

    Expands the `topic_to_cost` registry entry — a coarse ILIKE substring
    match over `search_corpus` joined to `session_stats` for cost, capped at
    `limit` — into a starting prompt that suggests the BM25-ranked
    `search_conversations` tool for better relevance before drilling into
    the priciest hit.
    """
    template = get_template("topic_to_cost")
    if template is None:  # pragma: no cover - defensive, registry can't drop this
        return "Error: 'topic_to_cost' template not found in registry."
    return _seed_text(
        template,
        [
            f"Try `search_conversations(query={query!r})` first — BM25 "
            "ranks relevance far better than this ILIKE substring match.",
            "Once you've identified the relevant sessions, drill into the "
            f"priciest one (up to {limit} candidates here) with "
            "`get_session` to see what made it expensive.",
        ],
    )
