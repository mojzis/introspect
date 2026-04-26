"""Shared helpers, constants, and template setup for route handlers."""

import logging
import re
from pathlib import Path

import duckdb
from fastapi import Request
from fastapi.templating import Jinja2Templates

from introspect.pricing import (
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
    compute_cost_usd,
)

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE_DEFAULT = 50
SESSIONS_PAGE_SIZES = [25, 50, 100, 200]

# Allowed sort columns for sessions page (resolved against session_stats ss).
SESSIONS_SORT_COLS = {
    "started_at": "ss.started_at",
    "duration": "ss.duration",
    "user_msgs": "ss.user_messages",
    "asst_msgs": "ss.assistant_messages",
    "tool_calls": "ss.tool_count",
    "model": "ss.model",
    "project": "ss.project",
    "branch": "ss.git_branch",
    "title": "ss.first_prompt",
    "files_read": "ss.files_read",
    "files_edited": "ss.files_edited",
    "files_read_only": "ss.files_read_only",
    "files_outside": "ss.files_outside",
    "cost": "ss.cost_usd",
}
SESSIONS_SORT_DEFAULT = "started_at"

RAW_PER_PAGE = 20

_XML_TAG_RE = re.compile(r"<[^>]+>")
# <command-message> duplicates the <command-name> for slash-command / skill
# invocations (e.g. "<command-name>marimo-pair</command-name>"
# "<command-message>/marimo-pair</command-message>"), so drop it entirely
# instead of leaving the repeated name in the title.
_COMMAND_MESSAGE_RE = re.compile(r"<command-message>.*?</command-message>", re.DOTALL)


def clean_title(raw: str) -> str:
    """Strip all XML-style tags from session titles."""
    without_msg = _COMMAND_MESSAGE_RE.sub("", raw)
    # Replace tags with a space so adjacent block contents don't run together
    # (e.g. "<command-name>commit</command-name><command-args>fix</command-args>"
    # becomes "commit fix", not "commitfix"); then collapse whitespace runs.
    detagged = _XML_TAG_RE.sub(" ", without_msg)
    return " ".join(detagged.split())


def parent(request: Request) -> str:
    """Return the base template: full page for normal requests, partial for HTMX."""
    if request.headers.get("HX-Request"):
        return "partial.html"
    return "base.html"


def conn(request: Request):
    """Get the DuckDB connection from request state."""
    return request.state.conn


DEFAULT_PAGE_SIZE = 50

# Reusable SQL fragment for per-session tool counts.
TOOL_COUNTS_SUBQUERY = """(
    SELECT session_id, COUNT(*) AS tool_count
    FROM tool_calls GROUP BY session_id
) tc"""


def _build_session_cost_subquery() -> str:
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
    sql = (
        f"(SELECT session_id, SUM(({cost_expr}) / 1000000.0) AS cost_usd "  # noqa: S608
        "FROM assistant_message_costs GROUP BY session_id) sc"
    )
    return sql


SESSION_COST_SUBQUERY = _build_session_cost_subquery()


def build_cost_attribution_sql(amc_filter: str = "") -> str:
    """Build the full cost-attribution query used by both cost views.

    The result query yields one row per deduped assistant message with the
    bloat classifier inputs attached (``user_block_type``, ``tool_name``,
    ``tool_input``). Two consumers share this query:

    * ``_build_cost_context`` (session-detail Cost tab) — scopes to one
      session via ``WHERE session_id = ?``.
    * ``_build_huge_reads_split`` (Cost Overview) — runs unscoped across
      every session.

    Keeping the classifier CTE in one place means the two paths can't drift
    — "huge reads" on the overview must agree with what the session-detail
    Cost tab calls Read-category cache creation.

    Args:
        amc_filter: a SQL ``WHERE`` clause (including the keyword) to scope
            the inner ``assistant_message_costs`` read, e.g.
            ``"WHERE session_id = ?"``. Empty string means "all sessions".

    The returned SQL yields columns:
        session_id, uuid, timestamp, is_sidechain, model,
        input_tokens, output_tokens, cache_read_tokens, cc_total,
        cache_creation_5m, cache_creation_1h,
        user_block_type, tool_name, tool_input
    """
    # amc_filter is a trusted module-local literal ("" or "WHERE session_id = ?"
    # — never user input), so splicing it into the query is safe.
    amc_cte = f"WITH amc AS (SELECT * FROM assistant_message_costs {amc_filter})"  # noqa: S608
    return amc_cte + _COST_ATTRIBUTION_STATIC_TAIL


# Static tail of the cost-attribution SQL — the only variable part is the
# ``amc`` CTE's WHERE clause (handled by :func:`build_cost_attribution_sql`),
# so the rest lives in one place as a module constant.
_COST_ATTRIBUTION_STATIC_TAIL = """,
        parent_blocks AS (
            SELECT
                amc.uuid AS amc_uuid,
                json_extract_string(
                    u.message, '$.content[' || i.idx || '].type'
                ) AS block_type,
                json_extract_string(
                    u.message, '$.content[' || i.idx || '].tool_use_id'
                ) AS block_tool_use_id,
                i.idx AS block_idx
            FROM amc
            JOIN raw_messages u
              ON u.uuid = amc.parent_uuid AND u.type = 'user'
              AND json_array_length(json_extract(u.message, '$.content')) > 0,
              generate_series(
                  0,
                  CAST(json_array_length(
                      json_extract(u.message, '$.content')
                  ) - 1 AS BIGINT)
              ) AS i(idx)
        ),
        chosen_block AS (
            -- Prefer a tool_result block (any index) over a text block.
            SELECT amc_uuid,
                   FIRST(block_type ORDER BY
                         CASE WHEN block_type = 'tool_result' THEN 0 ELSE 1 END,
                         block_idx) AS user_block_type,
                   FIRST(block_tool_use_id ORDER BY
                         CASE WHEN block_type = 'tool_result' THEN 0 ELSE 1 END,
                         block_idx) AS tool_use_id
            FROM parent_blocks
            GROUP BY amc_uuid
        )
        SELECT
            amc.session_id,
            amc.uuid,
            amc.timestamp,
            amc.is_sidechain,
            amc.model,
            amc.input_tokens,
            amc.output_tokens,
            amc.cache_read_tokens,
            amc.cache_creation_tokens AS cc_total,
            amc.cache_creation_5m,
            amc.cache_creation_1h,
            cb.user_block_type,
            tc.tool_name,
            tc.tool_input
        FROM amc
        LEFT JOIN chosen_block cb ON cb.amc_uuid = amc.uuid
        LEFT JOIN tool_calls tc ON tc.tool_use_id = cb.tool_use_id
        ORDER BY amc.timestamp
"""


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


def format_duration(total_seconds: float) -> str:
    """Format seconds as M:SS string."""
    secs = int(total_seconds)
    return f"{secs // 60}:{secs % 60:02d}"


# Columns selected by SESSION_INFO_SELECT (positional, resolved against
# the ``session_stats ss`` rollup table/view).
SESSION_INFO_SELECT = """
    ss.session_id,
    ss.started_at,
    ss.ended_at,
    ss.duration,
    ss.user_messages,
    ss.assistant_messages,
    ss.model,
    ss.project,
    ss.git_branch,
    ss.first_prompt,
    ss.tool_count,
    ss.files_read,
    ss.files_edited,
    ss.files_read_only,
    ss.files_outside,
    ss.commands,
    ss.cost_usd
"""

# All per-session aggregates already live in ``session_stats``; no further
# joins are needed for the listing query.
SESSION_INFO_JOINS = ""

_EMPTY_SESSION_INFO: dict[str, object] = {
    "date": "",
    "start_time": "",
    "end_time": "",
    "duration": "",
    "user_msgs": 0,
    "asst_msgs": 0,
    "model": "",
    "project": "",
    "branch": "",
    "title": "",
    "tool_count": 0,
    "files_read": 0,
    "files_edited": 0,
    "files_read_only": 0,
    "files_outside": 0,
    "commands": "",
    "cost_usd": None,
    "cost": "—",
}


_COST_DISPLAY_THRESHOLD = 0.01


def format_cost(value: float | None) -> str:
    """Render a USD cost the way the sessions list / cost tab want it.

    Returns ``"—"`` for unknown / null / negative, ``"<$0.01"`` for sub-cent
    costs, ``"$0.00"`` for exact zero, and ``"$0.42"`` style otherwise.
    """
    if value is None or value < 0:
        return "—"
    if value == 0:
        return "$0.00"
    if value < _COST_DISPLAY_THRESHOLD:
        return "<$0.01"
    return f"${value:.2f}"


def session_row_to_dict(row: tuple) -> dict:
    """Convert a SESSION_INFO_SELECT row to a template-friendly dict."""
    (
        session_id,
        started_at,
        ended_at,
        duration,
        user_msgs,
        asst_msgs,
        model,
        project,
        git_branch,
        first_prompt,
        tool_count,
        files_read,
        files_edited,
        files_read_only,
        files_outside,
        commands,
        cost_usd,
    ) = row
    dur_str = format_duration(duration.total_seconds()) if duration else ""
    cost_value = float(cost_usd) if cost_usd is not None else None
    return {
        "id": session_id,
        "date": str(started_at)[5:10] if started_at else "",
        "start_time": str(started_at)[11:16] if started_at else "",
        "end_time": str(ended_at)[11:16] if ended_at else "",
        "duration": dur_str,
        "user_msgs": user_msgs or 0,
        "asst_msgs": asst_msgs or 0,
        "model": model or "",
        "project": project or "",
        "branch": git_branch or "",
        "title": clean_title(first_prompt or "")[:120],
        "tool_count": tool_count or 0,
        "files_read": files_read or 0,
        "files_edited": files_edited or 0,
        "files_read_only": files_read_only or 0,
        "files_outside": files_outside or 0,
        "commands": commands or "",
        "cost_usd": cost_value,
        "cost": format_cost(cost_value),
    }


def escape_ilike(s: str) -> str:
    """Escape ILIKE special characters so they match literally."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def fetch_distinct_projects(
    db: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Return sorted list of distinct project names."""
    rows = db.execute("""
        SELECT DISTINCT project
        FROM logical_sessions
        WHERE project IS NOT NULL
        ORDER BY project
    """).fetchall()
    return [r[0] for r in rows]


_EMPTY_TOKEN_USAGE: dict = {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "cache_creation": 0,
    "cache_creation_5m": 0,
    "cache_creation_1h": 0,
    "cost_usd": 0.0,
    "cost": "—",
}


def fetch_token_usage(
    db: duckdb.DuckDBPyConnection,
    *,
    session_id: str | None = None,
) -> dict:
    """Fetch deduped token usage + estimated $ cost.

    Reads from ``assistant_message_costs`` (deduped by ``message.id``) so
    callers don't need to know about the raw_messages duplication bug.

    Always returns a dict (empty totals + ``"—"`` cost when the query fails
    or there is no data) so templates don't need ``or {}`` guards.

    Cost is computed in Python so we can apply the per-row cache_creation
    fallback (``cc_total > 0`` with both 5m/1h zero implies legacy 5m).
    """
    session_filter = ""
    params: list[str] = []
    if session_id is not None:
        session_filter = "WHERE session_id = ?"
        params.append(session_id)

    try:
        rows = db.execute(
            f"""
            SELECT
                model,
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(cache_read_tokens), 0),
                COALESCE(SUM(cache_creation_tokens), 0),
                COALESCE(SUM(cache_creation_5m), 0),
                COALESCE(SUM(cache_creation_1h), 0)
            FROM assistant_message_costs
            {session_filter}
            GROUP BY model
        """,  # noqa: S608
            params,
        ).fetchall()
    except duckdb.CatalogException:
        # Lazy-mode read connections may not yet have the derived view; the
        # caller can still render with empty totals.  Any other failure is a
        # bug we want to surface, not a zeroed-out template.
        log.warning("assistant_message_costs view missing", exc_info=True)
        return dict(_EMPTY_TOKEN_USAGE)

    totals = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "cache_creation_5m": 0,
        "cache_creation_1h": 0,
    }
    cost_usd = 0.0
    for row in rows:
        model = row[0]
        in_tok, out_tok, cr_tok, cc_tok, cc_5m, cc_1h = (int(v or 0) for v in row[1:])
        totals["input"] += in_tok
        totals["output"] += out_tok
        totals["cache_read"] += cr_tok
        totals["cache_creation"] += cc_tok
        totals["cache_creation_5m"] += cc_5m
        totals["cache_creation_1h"] += cc_1h
        # Legacy schema: usage.cache_creation_input_tokens with no 5m/1h
        # breakdown — bill at the 5m rate (Anthropic's older default).
        eff_5m, eff_1h = cc_5m, cc_1h
        if cc_5m == 0 and cc_1h == 0 and cc_tok > 0:
            eff_5m = cc_tok
        cost_usd += compute_cost_usd(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cr_tok,
            cache_creation_5m=eff_5m,
            cache_creation_1h=eff_1h,
        )

    return {
        **totals,
        "cost_usd": cost_usd,
        "cost": format_cost(cost_usd),
    }
