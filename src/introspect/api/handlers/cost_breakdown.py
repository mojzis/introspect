"""Daily / hourly cost breakdown views with Plotly bar charts.

Renders a stacked bar chart of cost per day (optionally split by model or
project) and an HTMX-loaded hourly drill-down for a clicked day.

The bar chart is built with `nolegend
<https://github.com/mojzis/nolegend>`_ — a Tufte-style Plotly template —
and embedded as a JSON figure that ``Plotly.newPlot`` instantiates client
side. The template is registered lazily on first chart render so importing
this module is side-effect-free.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import nolegend
import plotly.graph_objects as go
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from introspect.pricing import (
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
)

from ._helpers import conn, format_cost, parent, templates

ALLOWED_BREAKDOWNS: tuple[str, ...] = ("total", "model", "project")
DEFAULT_BREAKDOWN = "total"

# --- Chart visual contract ---------------------------------------------
# Hard ceiling on traces per chart. Stacked bars stop conveying segment
# identity past ~8 colours (identity is delivered through the
# hovertemplate, since the chart has no legend), and the tufte palette
# runs out of perceptually distinct hues around the same point. The long
# tail is folded into "Other" so any bucket has at most ``MAX_GROUPS``
# distinct stacks.
MAX_GROUPS = 8

# Direct-label at most this many groups so the labels themselves stay
# uncluttered — the rest are still distinguishable by colour and hover.
LABEL_TOP_N = 4

# Skip annotating segments smaller than this fraction of the chart's
# grand total. Tiny labels with arrows sticking into bigger neighbours
# look noisy and don't help identification.
LABEL_MIN_SHARE = 0.04

# Single-element list lets us flip the flag without a ``global`` statement
# (and without exposing a settable module attribute).
_template_activated: list[bool] = [False]


def _ensure_template() -> None:
    """Register nolegend's "tufte" template the first time a chart is built.

    Doing it here rather than at import keeps module import side-effect-free
    so importing the handler from a CLI or test that doesn't render charts
    doesn't mutate Plotly's default-template state.
    """
    if not _template_activated[0]:
        nolegend.activate()
        _template_activated[0] = True


def _normalise_breakdown(value: str) -> str:
    return value if value in ALLOWED_BREAKDOWNS else DEFAULT_BREAKDOWN


def _per_message_cost_expr() -> str:
    """Per-row USD cost expression for use inside a CTE over assistant_message_costs.

    Unqualified — the inner CTE selects from ``assistant_message_costs`` only
    so the bare ``model`` reference in :data:`PRICING_*_RATE_SQL` is
    unambiguous. Mirrors :func:`_helpers._build_session_cost_subquery` (same
    rate columns, same legacy cache-creation fallback) so daily/hourly
    totals match the sessions-list cost column to the cent.
    """
    cc_fallback = (
        "(CASE WHEN cache_creation_5m = 0 AND cache_creation_1h = 0 "
        "THEN cache_creation_tokens ELSE 0 END)"
    )
    return (
        f"input_tokens * ({PRICING_INPUT_RATE_SQL})"
        f" + output_tokens * ({PRICING_OUTPUT_RATE_SQL})"
        f" + cache_read_tokens * ({PRICING_CACHE_READ_RATE_SQL})"
        f" + cache_creation_5m * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
        f" + cache_creation_1h * ({PRICING_CACHE_WRITE_1H_RATE_SQL})"
        f" + {cc_fallback} * ({PRICING_CACHE_WRITE_5M_RATE_SQL})"
    )


def _fetch_aggregated(
    db: duckdb.DuckDBPyConnection,
    *,
    bucket_expr: str,
    where_sql: str = "",
    params: tuple[Any, ...] = (),
) -> list[tuple[str, str, str, float]]:
    """Aggregate cost by ``(bucket, model, project)`` and return tuples.

    ``bucket_expr`` is a trusted module-local literal evaluated inside the
    inner CTE against ``assistant_message_costs`` columns
    (``CAST(timestamp AS DATE)`` for the daily view,
    ``strftime(date_trunc('hour', timestamp::TIMESTAMP), '%H:00')`` for the
    hourly view). User input never reaches this string.
    """
    cost_expr = _per_message_cost_expr()
    sql = f"""
        WITH per_msg AS (
            SELECT
                session_id,
                model,
                CAST({bucket_expr} AS VARCHAR) AS bucket,
                ({cost_expr}) / 1000000.0 AS cost_usd
            FROM assistant_message_costs
            {where_sql}
        )
        SELECT
            per_msg.bucket,
            COALESCE(per_msg.model, 'unknown') AS model_name,
            COALESCE(ls.project, 'unknown') AS project_name,
            SUM(per_msg.cost_usd) AS cost_usd
        FROM per_msg
        LEFT JOIN logical_sessions ls ON ls.session_id = per_msg.session_id
        GROUP BY per_msg.bucket, model_name, project_name
        HAVING SUM(per_msg.cost_usd) > 0
        ORDER BY per_msg.bucket
    """  # noqa: S608
    return db.execute(sql, list(params)).fetchall()


def _collapse_to_breakdown(
    rows: list[tuple[str, str, str, float]],
    breakdown: str,
) -> dict[str, dict[str, float]]:
    """Reduce ``(bucket, model, project, cost)`` tuples to ``{bucket: {group: cost}}``.

    ``breakdown == "total"`` collapses every row into a single ``"Total"``
    group so the chart renders a single trace.
    """
    out: dict[str, dict[str, float]] = {}
    for bucket, model, project, cost_usd in rows:
        if breakdown == "model":
            group = model
        elif breakdown == "project":
            group = project
        else:
            group = "Total"
        bucket_entry = out.setdefault(bucket, {})
        bucket_entry[group] = bucket_entry.get(group, 0.0) + float(cost_usd or 0.0)
    return out


def _group_totals(bucketed: dict[str, dict[str, float]]) -> dict[str, float]:
    """Sum ``bucketed`` across buckets to ``{group: total_cost}``."""
    totals: dict[str, float] = {}
    for groups in bucketed.values():
        for group, cost in groups.items():
            totals[group] = totals.get(group, 0.0) + cost
    return totals


def _cap_groups(
    bucketed: dict[str, dict[str, float]],
    max_groups: int = MAX_GROUPS,
) -> dict[str, dict[str, float]]:
    """Keep the top ``max_groups - 1`` groups; fold the rest into "Other".

    A returned mapping has at most ``max_groups`` keys per bucket — the top
    real groups by total cost plus an ``"Other"`` aggregate when the tail
    needs collapsing.
    """
    totals = _group_totals(bucketed)
    if len(totals) <= max_groups:
        return bucketed
    keep = {
        g for g, _ in sorted(totals.items(), key=lambda kv: -kv[1])[: max_groups - 1]
    }
    out: dict[str, dict[str, float]] = {}
    for bucket, groups in bucketed.items():
        new_groups: dict[str, float] = {}
        for group, cost in groups.items():
            target = group if group in keep else "Other"
            new_groups[target] = new_groups.get(target, 0.0) + cost
        out[bucket] = new_groups
    return out


def _build_figure(
    bucketed: dict[str, dict[str, float]],
    *,
    breakdown: str,
    x_title: str,
) -> go.Figure:
    """Build a stacked bar chart from ``{bucket: {group: cost}}`` data."""
    _ensure_template()
    buckets = sorted(bucketed.keys())
    # Order groups by total descending so the largest sits at the bottom of
    # the stack, mirroring the conventional Pareto reading order.
    ordered_groups = [
        g for g, _ in sorted(_group_totals(bucketed).items(), key=lambda kv: -kv[1])
    ]

    multi_series = breakdown != "total"
    fig = go.Figure()
    for group in ordered_groups:
        ys = [bucketed.get(b, {}).get(group, 0.0) for b in buckets]
        fig.add_trace(
            go.Bar(
                name=group,
                x=buckets,
                y=ys,
                hovertemplate=f"<b>%{{x}}</b><br>{group}: $%{{y:.2f}}<extra></extra>",
            )
        )

    # No legend — segment identity comes from direct annotations on top
    # groups (see ``_compute_top_group_annotations``) plus colour.
    # ``hovermode="closest"`` is the only mode whose tooltip is anchored
    # to the segment under the cursor; ``"x"`` and ``"x unified"`` both
    # position relative to the plot area (not the cursor) and rendered
    # detached from short stacked-bar charts in practice.
    # No internal title — the surrounding card already has a ``<h2>``,
    # and the redundant in-chart title pushed the plot area down enough
    # to throw off hover hit-testing on the bottom row of bars.
    fig.update_layout(
        template="tufte",
        barmode="stack" if multi_series else "group",
        showlegend=False,
        hovermode="closest",
        xaxis_title=x_title,
        yaxis_title="USD",
        height=360,
        margin={"l": 60, "r": 30, "t": 20, "b": 60},
    )
    if multi_series:
        annotations = _compute_top_group_annotations(
            bucketed, ordered_groups=ordered_groups
        )
        for ann in annotations:
            fig.add_annotation(**ann)
    return fig


def _compute_top_group_annotations(
    bucketed: dict[str, dict[str, float]],
    *,
    ordered_groups: list[str],
) -> list[dict[str, Any]]:
    """Build annotation kwargs for the top groups' peak segments.

    Pure helper — returns the list of ``add_annotation`` kwargs so the
    placement arithmetic (peak bucket, cumulative-below offset, segment
    centre) is unit-testable without parsing a Plotly figure.

    ``ordered_groups`` must match the trace stacking order in
    :func:`_build_figure` (largest total first → bottom of stack), so
    the cumulative offsets here line up with the rendered stacks.
    """
    totals = _group_totals(bucketed)
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return []
    label_threshold = grand_total * LABEL_MIN_SHARE
    top_groups = ordered_groups[:LABEL_TOP_N]

    out: list[dict[str, Any]] = []
    for group in top_groups:
        peak_bucket, peak_value, bucket_groups = max(
            ((b, g.get(group, 0.0), g) for b, g in bucketed.items()),
            key=lambda bvg: bvg[1],
        )
        if peak_value < label_threshold:
            continue
        # Cumulative height of stacked segments below this one — matches
        # the trace add-order in _build_figure (bottom-to-top by total).
        below = 0.0
        for og in ordered_groups:
            if og == group:
                break
            below += bucket_groups.get(og, 0.0)
        center_y = below + peak_value / 2
        # Plain in-segment label — no arrow. The arrowed variant rendered
        # detached from the chart in some layouts (likely a tufte-template
        # / stacked-bar / showarrow quirk we couldn't pin down). Pinning
        # the label to the segment centre with ``showarrow=False`` removes
        # the offset / axref / ayref machinery entirely.
        out.append(
            {
                "x": peak_bucket,
                "y": center_y,
                "xref": "x",
                "yref": "y",
                "xanchor": "center",
                "yanchor": "middle",
                "text": group,
                "showarrow": False,
                "font": {"size": 10, "color": "#ffffff"},
                "bgcolor": "rgba(0,0,0,0.45)",
                "borderpad": 3,
            }
        )
    return out


def _build_panel_context(  # noqa: PLR0913
    db: duckdb.DuckDBPyConnection,
    *,
    breakdown: str,
    bucket_expr: str,
    x_title: str,
    where_sql: str = "",
    params: tuple[Any, ...] = (),
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Shared aggregate → figure pipeline for both daily and hourly panels.

    Returns ``(base_context, bucketed)`` so callers can attach view-specific
    extras (``day_count``, ``day``) without re-aggregating.
    """
    bd = _normalise_breakdown(breakdown)
    rows = _fetch_aggregated(
        db,
        bucket_expr=bucket_expr,
        where_sql=where_sql,
        params=params,
    )
    bucketed = _cap_groups(_collapse_to_breakdown(rows, bd))
    fig = _build_figure(bucketed, breakdown=bd, x_title=x_title)
    total_cost = sum(_group_totals(bucketed).values())
    base_context = {
        "chart_json": fig.to_json(),
        "breakdown": bd,
        "has_data": bool(bucketed),
        "total_cost": format_cost(total_cost),
    }
    return base_context, bucketed


def build_daily_panel_context(
    db: duckdb.DuckDBPyConnection,
    breakdown: str,
) -> dict[str, Any]:
    """Build the template context for the daily-cost panel.

    Public seam used by both the route handler (HTMX swaps) and
    ``cost_overview`` (initial render of the embedded panel).
    """
    base, bucketed = _build_panel_context(
        db,
        breakdown=breakdown,
        bucket_expr="CAST(timestamp AS DATE)",
        x_title="Day",
    )
    base["breakdown_options"] = list(ALLOWED_BREAKDOWNS)
    base["day_count"] = len(bucketed)
    return base


def _build_hourly_panel_context(
    db: duckdb.DuckDBPyConnection,
    day_str: str,
    breakdown: str,
) -> dict[str, Any]:
    """Build the template context for the hourly-cost panel for ``day_str``."""
    base, _ = _build_panel_context(
        db,
        breakdown=breakdown,
        bucket_expr="strftime(date_trunc('hour', timestamp::TIMESTAMP), '%H:00')",
        x_title="Hour (UTC)",
        where_sql="WHERE CAST(timestamp AS DATE) = ?",
        params=(day_str,),
    )
    base["day"] = day_str
    return base


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_day(day_str: str) -> str:
    """Validate a YYYY-MM-DD day string.

    A pattern check is sufficient — the validated string is bound as a
    DuckDB query parameter (not interpolated), and DuckDB rejects a
    non-date string with a clear error if one slips through. We don't
    parse to a Python date because :class:`datetime.date` carries no
    timezone, and the column is compared as a SQL ``DATE`` regardless.
    """
    if not _DAY_RE.match(day_str):
        raise HTTPException(status_code=400, detail="Invalid day format")
    return day_str


async def daily_panel(request: Request, breakdown: str) -> HTMLResponse:
    """Return the daily-cost panel fragment (chart + controls)."""
    context = build_daily_panel_context(conn(request), breakdown)
    context["parent"] = parent(request)
    return templates.TemplateResponse(request, "_daily_cost_panel.html", context)


async def hourly_panel(
    request: Request,
    day: str,
    breakdown: str,
) -> HTMLResponse:
    """Return the hourly-cost panel fragment for ``day``."""
    day_str = _parse_day(day)
    context = _build_hourly_panel_context(conn(request), day_str, breakdown)
    context["parent"] = parent(request)
    return templates.TemplateResponse(request, "_hourly_cost_panel.html", context)
