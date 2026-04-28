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
    cc_fallback = (
        "(CASE WHEN cache_creation_5m = 0 AND cache_creation_1h = 0 "
        "THEN cache_creation_tokens ELSE 0 END)"
    )
    cost_expr = (
        f"input_tokens * ({PRICING_INPUT_RATE_SQL})"
        f" + output_tokens * ({PRICING_OUTPUT_RATE_SQL})"
        f" + cache_read_tokens * ({PRICING_CACHE_READ_RATE_SQL})"
        f" + cache_creation_5m * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
        f" + cache_creation_1h * ({PRICING_CACHE_WRITE_1H_RATE_SQL})"
        f" + {cc_fallback} * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
    )
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
