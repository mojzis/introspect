"""Cost Overview route handler.

Builds a portfolio-level view of Claude Code costs:

* **Pareto table** — sessions ranked by cost desc, cut at the first row that
  crosses 80% cumulative share. That crossing row is emphasised visually.
* **Hero stats** — grand total, Pareto-session count, the "N sessions =
  80% of $X" headline.
* **Binary splits** — three two-row mini-tables showing the cost impact of
  subagent presence, huge-reads, and skill/slash-command usage.

The session-level ``cost_usd`` values are sourced from
:data:`_helpers.SESSION_COST_SUBQUERY` so the totals here always match the
sessions-list page exactly.

Time-window filter (``/cost-overview/portfolio?day=...&hour=...``)
------------------------------------------------------------------
When the user clicks a bar in the daily or hourly cost chart, the
portfolio panel reruns under a timestamp window. The semantic is:

* The **cost aggregation** (`SESSION_COST_SUBQUERY`) and therefore the
  Pareto rows narrow to assistant_message_costs rows in the window.
  Sessions absent from the window simply don't contribute.
* **Subagent / skill classifiers stay all-time** session properties.
  Filtering them to the window would require, e.g. a Task tool call
  *inside hour 14* — a noisy and unintuitive definition. Keeping them
  all-time means "of the cost incurred at hour 14, this much came from
  sessions that ever used a subagent".
* **Huge-reads stays per-session** (its read-cost-ratio is intrinsic);
  only the cost denominator narrows with the window, so the threshold
  test runs against the same $ values shown in the chart.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import duckdb
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from introspect.pricing import cache_miss_premium_usd
from introspect.sql_fragments import session_cost_subquery_filtered

from ._helpers import (
    OBVIOUS_COMMANDS_SQL,
    SESSION_COST_SUBQUERY,
    build_cost_attribution_sql,
    clean_title,
    conn,
    format_cost,
    parent,
    parse_day,
    parse_hour,
    templates,
)
from .cost_breakdown import (
    DEFAULT_BREAKDOWN,
    build_daily_panel_context,
)
from .sessions import cache_loss_event_rows

# Type alias: half-open ISO-formatted timestamp window (inclusive start,
# exclusive end). ``None`` means no filter.
TimeWindow = tuple[str, str]

# Pareto cumulative-cost cutoff — sessions are kept until their *previous*
# row's cumulative share crosses this fraction (so the row that tips the
# 80% line is the last one included).
PARETO_CUTOFF = 0.80

# Huge-reads thresholds — both guards must fire for a session to classify.
# 10% of session cost keeps the shape honest even for tiny sessions; the
# 100k-token floor filters out trivial ones where 10% is meaningless.
HUGE_READS_COST_FRACTION = 0.10
HUGE_READS_MIN_TOKENS = 100_000

# Suppress the "X% of total" suffix on the cache-loss stat card when the
# share rounds below this threshold — at that point the percentage is
# noisier than informative.
CACHE_LOSS_PCT_DISPLAY_THRESHOLD = 0.1


def _window_for(day: str | None, hour: str | None) -> TimeWindow | None:
    """Build a half-open ``[start, end)`` ISO-timestamp window.

    Returns ``None`` for the unfiltered case. ``day`` alone yields one
    24-hour window; ``day`` + ``hour`` yields a one-hour window with
    rollover handled by ``timedelta``.
    """
    if not day:
        return None
    start = datetime.fromisoformat(f"{day}T{hour or '00'}:00:00")
    delta = timedelta(hours=1) if hour else timedelta(days=1)
    end = start + delta
    return (start.isoformat(sep=" "), end.isoformat(sep=" "))


def _filter_label(day: str | None, hour: str | None) -> str:
    """Human-readable chip label for the active filter."""
    if not day:
        return ""
    if hour:
        return f"{day} {hour}:00"
    return day


def _cost_subquery(window: TimeWindow | None) -> str:
    """Pick the unfiltered or filtered SESSION_COST_SUBQUERY for ``window``.

    The window endpoints are produced by :func:`_window_for` from validated
    day/hour inputs, so splicing them into the SQL is safe.
    """
    if window is None:
        return SESSION_COST_SUBQUERY
    start, end = window
    return session_cost_subquery_filtered(
        f"timestamp >= '{start}' AND timestamp < '{end}'"
    )


def _build_panel_context(
    db: duckdb.DuckDBPyConnection,
    window: TimeWindow | None,
) -> dict[str, Any]:
    """Build the Pareto + binary-splits context for the portfolio panel.

    Shared by the full page render (window=None) and the HTMX fragment
    swap. ``_build_pareto`` already runs the per-session cost rollup, so
    the splits derive ``cost_rows`` from its rows — the rollup runs once
    per request, not twice.
    """
    pareto = _build_pareto(db, window)
    cost_rows = [(r["session_id"], r["cost_usd"]) for r in pareto["rows"]]
    return {
        "pareto": pareto,
        "subagent_split": _build_subagent_split(db, cost_rows),
        "huge_reads_split": _build_huge_reads_split(db, cost_rows, window),
        "skill_split": _build_skill_split(db, cost_rows),
        "cache_loss": _aggregate_cache_loss(
            db, window, total_cost_usd=pareto["total_cost_usd"]
        ),
    }


def _aggregate_cache_loss(
    db: duckdb.DuckDBPyConnection,
    window: TimeWindow | None,
    *,
    total_cost_usd: float,
) -> dict[str, Any]:
    """Sum the cache-miss premium across all cache-loss events in ``window``.

    Reuses ``cache_loss_event_rows`` so the detection rule stays in lockstep
    with the per-session marker. Window filters on the *rebuild* assistant
    timestamp, so totals line up with the cost chart's bucketing.
    """
    rows = cache_loss_event_rows(db, timestamp_window=window)
    # Row layout: (user_uuid, user_ts, prev_asst_ts, next_asst_uuid,
    #              model, cc_total, cc_5m, cc_1h)
    cost_usd = sum(
        cache_miss_premium_usd(
            model=row[4],
            cc_total=int(row[5] or 0),
            cc_5m=int(row[6] or 0),
            cc_1h=int(row[7] or 0),
        )
        for row in rows
    )
    pct_of_total = 100.0 * cost_usd / total_cost_usd if total_cost_usd > 0 else 0.0
    return {
        "count": len(rows),
        "cost_usd": cost_usd,
        "cost": format_cost(cost_usd),
        "pct_of_total": pct_of_total,
        "show_pct": pct_of_total >= CACHE_LOSS_PCT_DISPLAY_THRESHOLD,
    }


async def cost_overview(request: Request) -> HTMLResponse:
    """Render the /cost-overview page."""
    db = conn(request)
    panel = _build_panel_context(db, window=None)
    daily_panel = build_daily_panel_context(db, DEFAULT_BREAKDOWN)

    return templates.TemplateResponse(
        request,
        "cost_overview.html",
        {
            "parent": parent(request),
            "is_filtered": False,
            "filter_label": "",
            **panel,
            **daily_panel,
        },
    )


async def cost_portfolio_panel(
    request: Request,
    day: str | None,
    hour: str | None,
) -> HTMLResponse:
    """Render the portfolio fragment, optionally scoped to a time window."""
    if hour and not day:
        raise HTTPException(status_code=400, detail="hour requires day")
    try:
        day_str = parse_day(day) if day else None
        hour_str = parse_hour(hour) if hour else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    window = _window_for(day_str, hour_str)

    db = conn(request)
    panel = _build_panel_context(db, window)
    return templates.TemplateResponse(
        request,
        "_cost_portfolio_panel.html",
        {
            "parent": parent(request),
            "is_filtered": window is not None,
            "filter_label": _filter_label(day_str, hour_str),
            **panel,
        },
    )


def _fetch_cost_rows(
    db: duckdb.DuckDBPyConnection,
    window: TimeWindow | None = None,
) -> list[tuple]:
    """Return ``[(session_id, cost_usd), ...]`` for every positive-cost session.

    Shared denominator for all three binary splits — guarantees they agree
    on totals and the :data:`SESSION_COST_SUBQUERY` aggregation only runs
    once per page load rather than once per split.

    When ``window`` is set, the per-session rollup only sums assistant
    messages whose ``timestamp`` falls inside the half-open interval —
    sessions outside drop out entirely.
    """
    subquery = _cost_subquery(window)
    return db.execute(
        f"SELECT session_id, cost_usd FROM {subquery} "  # noqa: S608
        "WHERE cost_usd IS NOT NULL AND cost_usd > 0"
    ).fetchall()


def _build_pareto(
    db: duckdb.DuckDBPyConnection,
    window: TimeWindow | None = None,
) -> dict[str, Any]:
    """Rank sessions by cost desc and cut at 80% cumulative share.

    Returns::

        {
            "rows": [ {session_id, title, project, started_at, cost_usd,
                       cost, cumulative_usd, cumulative, cum_frac,
                       is_cutoff}, ... ],
            "total_cost_usd": float,
            "total_cost": str,
            "pareto_session_count": int,
            "total_session_count": int,
            "pareto_cost_usd": float,
            "pareto_cost": str,
        }
    """
    # SESSION_COST_SUBQUERY carries its own ``sc`` alias; the outer CTE below
    # uses a distinct ``session_costs`` name to avoid shadowing it.
    subquery = _cost_subquery(window)
    rows = db.execute(
        f"""
        WITH session_costs AS (
            SELECT session_id, cost_usd FROM {subquery}
        ),
        ranked AS (
            SELECT
                session_id,
                cost_usd,
                SUM(cost_usd) OVER () AS grand_total,
                SUM(cost_usd) OVER (
                    ORDER BY cost_usd DESC, session_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative
            FROM session_costs
            WHERE cost_usd IS NOT NULL AND cost_usd > 0
        )
        SELECT
            r.session_id,
            r.cost_usd,
            r.cumulative,
            r.grand_total,
            r.cumulative / NULLIF(r.grand_total, 0) AS cum_frac,
            ls.started_at,
            ls.project,
            st.first_prompt
        FROM ranked r
        LEFT JOIN logical_sessions ls ON ls.session_id = r.session_id
        LEFT JOIN session_titles st ON st.session_id = r.session_id
        ORDER BY r.cost_usd DESC, r.session_id
        """  # noqa: S608
    ).fetchall()

    total_cost_usd = float(rows[0][3]) if rows else 0.0

    out_rows: list[dict[str, Any]] = []
    pareto_cost_usd = 0.0
    cutoff_seen = False
    prev_cum_frac = 0.0
    for row in rows:
        session_id = row[0]
        cost_usd = float(row[1] or 0.0)
        cumulative = float(row[2] or 0.0)
        cum_frac = float(row[4] or 0.0)
        started_at = row[5]
        project = row[6] or ""
        raw_title = row[7] or ""

        # Keep rows whose *previous* cum_frac is below the 80% threshold;
        # that way the row that tips the cutoff is included but no more.
        if cutoff_seen:
            in_pareto = False
        elif prev_cum_frac >= PARETO_CUTOFF:
            in_pareto = False
            cutoff_seen = True
        else:
            in_pareto = True

        is_cutoff_row = in_pareto and cum_frac >= PARETO_CUTOFF
        if in_pareto:
            pareto_cost_usd = cumulative

        out_rows.append(
            {
                "session_id": session_id,
                "title": clean_title(raw_title)[:120],
                "project": project,
                "started_at": str(started_at)[:16] if started_at else "",
                "cost_usd": cost_usd,
                "cost": format_cost(cost_usd),
                "cumulative_usd": cumulative,
                "cumulative": format_cost(cumulative),
                "cum_frac": cum_frac,
                "cum_pct": 100.0 * cum_frac,
                "in_pareto": in_pareto,
                "is_cutoff": is_cutoff_row,
            }
        )
        prev_cum_frac = cum_frac

    pareto_session_count = sum(1 for r in out_rows if r["in_pareto"])

    return {
        "rows": out_rows,
        "total_cost_usd": total_cost_usd,
        "total_cost": format_cost(total_cost_usd),
        "pareto_session_count": pareto_session_count,
        "total_session_count": len(out_rows),
        "pareto_cost_usd": pareto_cost_usd,
        "pareto_cost": format_cost(pareto_cost_usd),
    }


def _split_from_flagged_rows(
    flag_rows: list[tuple],
    cost_rows: list[tuple],
) -> dict[str, Any]:
    """Aggregate (flag, session_id, cost) triples into with/without totals.

    Args:
        flag_rows: list of ``(session_id, is_flagged_bool)`` for sessions we
            know about. Sessions missing from ``flag_rows`` are assumed
            non-flagged (False).
        cost_rows: list of ``(session_id, cost_usd)`` for every session
            with a positive cost.

    Returns a dict matching the contract in ``context.md``::

        {"with": {"sessions", "cost_usd", "mean_cost_usd", "cost",
                  "mean_cost", "pct_cost"},
         "without": {...},
         "pct_cost_with": float}
    """
    flag_by_session = {s: bool(f) for s, f in flag_rows}
    totals = {
        True: {"sessions": 0, "cost_usd": 0.0},
        False: {"sessions": 0, "cost_usd": 0.0},
    }
    for session_id, cost_usd in cost_rows:
        bucket = flag_by_session.get(session_id, False)
        entry = totals[bucket]
        entry["sessions"] += 1
        entry["cost_usd"] += float(cost_usd or 0.0)

    grand_total = totals[True]["cost_usd"] + totals[False]["cost_usd"]

    def _format(bucket: dict[str, float]) -> dict[str, Any]:
        sessions = int(bucket["sessions"])
        cost_usd = float(bucket["cost_usd"])
        mean = cost_usd / sessions if sessions else 0.0
        pct = 100.0 * cost_usd / grand_total if grand_total else 0.0
        return {
            "sessions": sessions,
            "cost_usd": cost_usd,
            "cost": format_cost(cost_usd),
            "mean_cost_usd": mean,
            "mean_cost": format_cost(mean),
            "pct_cost": pct,
        }

    with_row = _format(totals[True])
    without_row = _format(totals[False])
    pct_cost_with = with_row["pct_cost"]
    return {
        "with": with_row,
        "without": without_row,
        "pct_cost_with": pct_cost_with,
    }


def _build_subagent_split(
    db: duckdb.DuckDBPyConnection,
    cost_rows: list[tuple],
) -> dict[str, Any]:
    """Subagent-presence split: sidechain messages OR Task/Agent tool calls."""
    flag_rows = db.execute(
        """
        SELECT DISTINCT session_id, TRUE AS has_subagent
        FROM (
            SELECT session_id FROM assistant_message_costs
            WHERE is_sidechain = TRUE
            UNION ALL
            SELECT session_id FROM tool_calls
            WHERE tool_name IN ('Task', 'Agent')
        ) sub
        """
    ).fetchall()
    return _split_from_flagged_rows(flag_rows, cost_rows)


def _build_huge_reads_split(
    db: duckdb.DuckDBPyConnection,
    cost_rows: list[tuple],
    window: TimeWindow | None = None,
) -> dict[str, Any]:
    """Huge-reads split.

    A session counts as "with huge reads" when its Read-category cache
    creation is *both* ≥ 10% of session cost *and* ≥ 100k tokens. Both
    guards matter — percentage alone over-flags tiny sessions, and the
    token count alone under-flags short bursts of small reads.

    The Read classifier is the shared ``chosen_block`` CTE in
    ``build_cost_attribution_sql`` — the same classifier the session-detail
    Cost tab calls Read, so the two views can't drift.

    ``cost_rows`` is the shared ``[(session_id, cost_usd), ...]`` pull from
    :data:`SESSION_COST_SUBQUERY`. We use its totals as the denominator for
    the ≥10% threshold so the fraction and the reported "Total $" in the
    with/without table are measured against the same source — keeping the
    headline claim ("totals match the sessions-list page exactly") true for
    the threshold itself, not just the report.

    When ``window`` is set, both the cost denominator (already filtered in
    ``cost_rows``) and the read-token tally narrow to the same timestamp
    interval — keeping the ratio meaningful inside the window.
    """
    # Import inside the function to avoid a circular import with sessions.py
    # (sessions imports from _helpers; _helpers must not import from
    # sessions; cost_overview is leaf).
    from introspect.api.handlers.sessions import _classify_bucket  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    # Use SESSION_COST_SUBQUERY totals as the denominator — single source of
    # truth with the sessions-list cost column and the Pareto table.
    session_totals = {s: float(c or 0.0) for s, c in cost_rows}

    # One big scan for every session. We only need session_id, model,
    # cc_total, cc_5m, cc_1h and the classifier inputs; the rest of the
    # attribution row is ignored here.
    if window is None:
        amc_filter = ""
        params: tuple[Any, ...] = ()
    else:
        amc_filter = "WHERE timestamp >= ? AND timestamp < ?"
        params = window
    attrib_rows = db.execute(build_cost_attribution_sql(amc_filter), params).fetchall()

    # Per-session Read-category cache-creation tokens + cost.
    per_session: dict[str, dict[str, float]] = {}
    for row in attrib_rows:
        (
            session_id,
            _uuid,
            _timestamp,
            _is_side,
            model,
            _in_tok,
            _out_tok,
            _cr_tok,
            cc_total,
            cc_5m,
            cc_1h,
            user_block_type,
            tool_name,
            tool_input_raw,
        ) = row
        cc_total_i = int(cc_total or 0)
        cc_5m_i = int(cc_5m or 0)
        cc_1h_i = int(cc_1h or 0)
        if cc_total_i == 0:
            continue
        _bucket, category = _classify_bucket(
            tool_name=tool_name,
            tool_input_raw=tool_input_raw,
            user_block_type=user_block_type,
        )
        if category != "Read":
            continue
        eff_5m = (
            cc_total_i
            if (cc_5m_i == 0 and cc_1h_i == 0 and cc_total_i > 0)
            else cc_5m_i
        )
        read_cc_cost = compute_cost_usd(
            model=model,
            cache_creation_5m=eff_5m,
            cache_creation_1h=cc_1h_i,
        )
        entry = per_session.setdefault(
            session_id,
            {"read_cc_cost": 0.0, "read_cc_tokens": 0.0},
        )
        entry["read_cc_cost"] += read_cc_cost
        entry["read_cc_tokens"] += cc_total_i

    flag_rows: list[tuple] = []
    for session_id, agg in per_session.items():
        tokens = agg["read_cc_tokens"]
        total_cost = session_totals.get(session_id, 0.0)
        read_cost = agg["read_cc_cost"]
        frac = read_cost / total_cost if total_cost else 0.0
        flagged = tokens >= HUGE_READS_MIN_TOKENS and frac >= HUGE_READS_COST_FRACTION
        flag_rows.append((session_id, flagged))

    return _split_from_flagged_rows(flag_rows, cost_rows)


def _build_skill_split(
    db: duckdb.DuckDBPyConnection,
    cost_rows: list[tuple],
) -> dict[str, Any]:
    """Skill / slash-command split.

    A session counts as "with skills" when at least one ``<command-name>``
    tag appears that isn't in the :data:`OBVIOUS_COMMANDS` built-in
    allowlist (``/clear``, ``/compact``, ...). Built-in commands don't
    reflect real work, so they mustn't flip a session into the "with
    skills" bucket.
    """
    flag_rows = db.execute(
        f"""
        SELECT DISTINCT session_id, TRUE AS has_skill
        FROM message_commands
        WHERE command NOT IN {OBVIOUS_COMMANDS_SQL}
        """  # noqa: S608
    ).fetchall()
    return _split_from_flagged_rows(flag_rows, cost_rows)
