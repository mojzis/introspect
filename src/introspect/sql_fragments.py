"""Shared SQL fragments used by both ``db.py`` (when materializing the
``session_stats`` rollup) and the FastAPI handlers (when querying).

These are pure SQL strings — no FastAPI / web layer imports — so keeping
them in a leaf module lets ``db.py`` import them directly without inverting
the layering (``db`` -> ``api.handlers._helpers`` -> ``db``).

``introspect.api.handlers._helpers`` re-exports the names below for
backwards compatibility with handler call sites.
"""

from introspect.pricing import (
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
)

# Reusable SQL fragment for per-session tool counts.
TOOL_COUNTS_SUBQUERY = """(
    SELECT session_id, COUNT(*) AS tool_count
    FROM tool_calls GROUP BY session_id
) tc"""

# Built-in / meta commands that don't reflect real work — hidden from the UI.
OBVIOUS_COMMANDS: frozenset[str] = frozenset(
    {
        "/clear",
        "/compact",
        "/config",
        "/cost",
        "/doctor",
        "/exit",
        "/fast",
        "/help",
        "/init",
        "/listen",
        "/login",
        "/logout",
        "/model",
        "/quit",
        "/status",
        "/terminal-setup",
        "/vim",
    }
)

OBVIOUS_COMMANDS_SQL = "(" + ", ".join(f"'{c}'" for c in sorted(OBVIOUS_COMMANDS)) + ")"

COMMAND_LIST_SUBQUERY = (
    "(SELECT session_id,"  # noqa: S608
    " string_agg(DISTINCT command, ', ' ORDER BY command) AS commands"
    " FROM message_commands"
    f" WHERE command NOT IN {OBVIOUS_COMMANDS_SQL}"
    " GROUP BY session_id) cmd"
)

TOOL_COUNTS_WITH_ERRORS_SUBQUERY = """(
    SELECT session_id,
           COUNT(*) AS tool_count,
           COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
    FROM tool_calls GROUP BY session_id
) tc"""

# Reusable SQL fragments for per-session file metrics
# (backed by file_reads / file_writes views).
FILE_READS_SUBQUERY = """(
    SELECT
        fr.session_id,
        COUNT(DISTINCT fr.file_path) AS files_read,
        COUNT(DISTINCT fr.file_path) FILTER (
            WHERE fr.file_path NOT IN (
                SELECT DISTINCT fw.file_path FROM file_writes fw
                WHERE fw.session_id = fr.session_id
            )
        ) AS files_read_only,
        COUNT(DISTINCT fr.file_path) FILTER (
            WHERE NOT starts_with(fr.file_path, COALESCE(ls.cwd, ''))
        ) AS files_outside
    FROM file_reads fr
    JOIN logical_sessions ls ON fr.session_id = ls.session_id
    GROUP BY fr.session_id
) fr_agg"""

FILE_WRITES_SUBQUERY = """(
    SELECT session_id, COUNT(DISTINCT file_path) AS files_edited
    FROM file_writes GROUP BY session_id
) fw_agg"""

# Per-session count of Skill tool calls — covers both user-typed /name and
# model-auto-triggered skills (both emit a Skill tool_use).
SKILLS_INVOKED_ROLLUP_SQL = (
    "SELECT session_id, count(*) AS skills_invoked"
    " FROM session_messages_enriched"
    " WHERE kind = 'agent_tool_call' AND tool_name = 'Skill'"
    " GROUP BY session_id"
)

# Per-session rollup of harness-injected context loads (session_context_loads).
# Shared by the ``first_prompt_triggers`` query template and the ``/triggers``
# web page so the classification can't drift between them.
CONTEXT_LOADS_ROLLUP_SQL = (
    "SELECT session_id,"
    " bool_or(load_kind = 'claude_md') AS auto_loaded_claude_md,"
    " count(*) FILTER (WHERE load_kind = 'file_ref') AS n_auto_loaded_files,"
    " bool_or(load_kind = 'skill_listing') AS skill_menu_loaded"
    " FROM session_context_loads GROUP BY session_id"
)


# ---------------------------------------------------------------------------
# Hoisted cost-expression fragments — shared by the session-cost subquery
# and the spend-shape (R/W bar + sparkline) query so the math cannot drift.
# ---------------------------------------------------------------------------

# Legacy schema fallback: when the newer per-tier cache_creation fields are
# both zero, treat the raw cache_creation_tokens as 5m writes (Anthropic's
# historical default).
CACHE_WRITE_FALLBACK_SQL: str = (
    "(CASE WHEN cache_creation_5m = 0 AND cache_creation_1h = 0 "
    "THEN cache_creation_tokens ELSE 0 END)"
)

# Cache-read dollar cost for one assistant message row (pre /1e6 division).
CACHE_READ_COST_SQL: str = f"cache_read_tokens * ({PRICING_CACHE_READ_RATE_SQL})"

# Cache-write dollar cost for one assistant message row (pre /1e6 division).
# Includes the legacy fallback billed at the 5m rate.
CACHE_WRITE_COST_SQL: str = (
    f"cache_creation_5m * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
    f" + cache_creation_1h * ({PRICING_CACHE_WRITE_1H_RATE_SQL})"
    f" + {CACHE_WRITE_FALLBACK_SQL} * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
)

# Output-token dollar cost for one assistant message row (pre /1e6 division).
OUTPUT_COST_SQL: str = f"output_tokens * ({PRICING_OUTPUT_RATE_SQL})"

# Total per-row dollar cost (pre /1e6 division).
COST_EXPR_SQL: str = (
    f"input_tokens * ({PRICING_INPUT_RATE_SQL})"
    f" + {OUTPUT_COST_SQL}"
    f" + {CACHE_READ_COST_SQL}"
    f" + {CACHE_WRITE_COST_SQL}"
)


def _build_session_cost_subquery(timestamp_where: str = "") -> str:
    """Assemble the per-session $ cost subquery, plumbing in the rate CASE strings.

    Reads from the deduped ``assistant_message_costs`` view and computes cost
    per row (so mixed-model sessions roll up correctly) — done in DuckDB
    rather than Python so the sessions list can ``ORDER BY cost_usd`` without
    materializing every assistant message.

    The ``cc_fallback`` term covers the legacy schema where
    ``usage.cache_creation_input_tokens`` is set but the
    ``cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens`` sub-fields
    are zero — bill those tokens at the 5m write rate (Anthropic's older
    default).  Mirrors the Python fallback in ``fetch_token_usage``.

    ``timestamp_where`` is spliced into the inner SELECT as a WHERE clause
    when non-empty. Trust contract: callers must pass only validated SQL
    (no user input) — used by the cost-overview portfolio panel to scope
    the per-session rollup to a chosen day or hour.
    """
    cost_expr = COST_EXPR_SQL
    where_clause = f" WHERE {timestamp_where}" if timestamp_where else ""
    sql = (
        f"(SELECT session_id, SUM(({cost_expr}) / 1000000.0) AS cost_usd "  # noqa: S608
        f"FROM assistant_message_costs{where_clause} GROUP BY session_id) sc"
    )
    return sql


def session_cost_subquery_filtered(timestamp_where: str) -> str:
    """Per-session cost subquery scoped to a timestamp predicate.

    Trust contract: ``timestamp_where`` MUST be built from validated inputs
    (e.g. parsed YYYY-MM-DD strings), never raw user input.
    """
    return _build_session_cost_subquery(timestamp_where)


SESSION_COST_SUBQUERY = _build_session_cost_subquery()
