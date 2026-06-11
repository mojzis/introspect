"""Tests for the Cost Breakdown panel (daily/hourly charts)."""

import json
import tempfile
from pathlib import Path

import pytest

from .cost_helpers import (
    _cost_overview_setup,
    _dup_jsonl,
    _materialize_and_run,
    _multi_day_specs,
    _multi_model_specs,
    _run_with_client,
)


def test_cost_overview_daily_panel_embedded():
    """The /cost-overview page must embed the daily-chart container."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            text = response.text
            assert "Daily breakdown" in text
            assert 'id="daily-cost-panel"' in text
            assert 'id="daily-cost-chart"' in text
            # Data attributes the base.html bootstrap reads to wire the
            # chart up — drilldown click handler + breakdown for URL.
            assert 'class="cost-chart"' in text
            assert 'data-figure-id="daily-cost-chart-data"' in text
            assert 'data-on-click="hourly-drilldown"' in text
            assert '<script type="application/json"' in text

        _run_with_client(tmp, _check)


def test_cost_overview_breakdown_fragment_renders_total():
    """Breakdown fragment endpoint returns the daily-cost panel (default total)."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            response = client.get("/cost-overview/breakdown?breakdown=total")
            assert response.status_code == 200
            assert 'id="daily-cost-panel"' in response.text
            # Days appear on the x axis
            assert "2026-04-21" in response.text
            assert "2026-04-23" in response.text

        _run_with_client(tmp, _check)


def test_cost_overview_breakdown_total_collapses_to_single_trace():
    """Total breakdown collapses every row into one trace named "Total".

    Asserted on the structured chart JSON rather than substring-matching
    the HTML because Plotly's JSON output is order-stable but the embedded
    HTML around it is not.
    """
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())
        ctx = _materialize_and_run(tmp, lambda c: build_daily_panel_context(c, "total"))
        fig = json.loads(ctx["chart_json"])
        assert len(fig["data"]) == 1
        assert fig["data"][0]["name"] == "Total"


def test_cost_overview_breakdown_by_model_traces():
    """Model-mode panel should produce one trace per distinct model.

    Two models in the fixture → two stacked-bar traces, named after each
    model. Both totals must add up to the day's grand total via the
    chart's y-values.
    """
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_model_specs())
        ctx = _materialize_and_run(tmp, lambda c: build_daily_panel_context(c, "model"))
        assert ctx["has_data"]
        assert ctx["day_count"] == 1
        fig = json.loads(ctx["chart_json"])
        names = sorted(trace["name"] for trace in fig["data"])
        # claude-opus-4-7 input is $5/M, claude-sonnet-4-6 input rate
        # differs — but presence and naming is the contract under test.
        assert names == ["claude-opus-4-7", "claude-sonnet-4-6"]


@pytest.mark.parametrize("breakdown", ["total", "model", "project"])
def test_cost_overview_breakdown_hides_legend(breakdown):
    """Chart never renders a legend — segment identity is in hover."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_model_specs())
        ctx = _materialize_and_run(
            tmp, lambda c: build_daily_panel_context(c, breakdown)
        )
        fig = json.loads(ctx["chart_json"])
        assert fig["layout"]["showlegend"] is False
        if breakdown == "model":
            # Confirms the multi-series path still produces ≥2 traces — the
            # original regression was a legend appearing on multi-series
            # charts, so a single-trace pass would not exercise it.
            assert len(fig["data"]) >= 2


@pytest.mark.parametrize("breakdown", ["total", "model", "project"])
def test_cost_overview_breakdown_uses_closest_hovermode(breakdown):
    """Hover label must anchor to the segment under the cursor (closest mode).

    ``"x"`` and ``"x unified"`` both position relative to the plot area
    and rendered detached from the chart in our short stacked-bar layout
    — ``"closest"`` is the only mode that keeps the popup on the bar.
    """
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_model_specs())
        ctx = _materialize_and_run(
            tmp, lambda c: build_daily_panel_context(c, breakdown)
        )
        fig = json.loads(ctx["chart_json"])
        assert fig["layout"]["hovermode"] == "closest"


def test_compute_top_group_annotations_skips_below_threshold():
    """Groups under LABEL_MIN_SHARE are not annotated."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        LABEL_MIN_SHARE,
        _compute_top_group_annotations,
    )

    # "tiny" sits at ~1% of the grand total — well under the 4% threshold.
    bucketed = {
        "2026-04-09": {"big": 100.0, "tiny": 1.0},
        "2026-04-10": {"big": 80.0, "tiny": 0.5},
    }
    ordered = ["big", "tiny"]
    grand_total = sum(sum(g.values()) for g in bucketed.values())
    assert 1.0 / grand_total < LABEL_MIN_SHARE  # sanity: fixture below threshold

    anns = _compute_top_group_annotations(bucketed, ordered_groups=ordered)
    labels = {a["text"] for a in anns}
    assert labels == {"big"}


def test_compute_top_group_annotations_caps_at_top_n():
    """At most LABEL_TOP_N groups get direct labels."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        LABEL_TOP_N,
        _compute_top_group_annotations,
    )

    # Six equal-sized groups, all above the share threshold (≈16.7% each).
    groups = [f"g{i}" for i in range(6)]
    bucketed = {"2026-04-09": dict.fromkeys(groups, 10.0)}
    anns = _compute_top_group_annotations(bucketed, ordered_groups=groups)
    assert len(anns) == LABEL_TOP_N
    assert {a["text"] for a in anns} == set(groups[:LABEL_TOP_N])


def test_compute_top_group_annotations_centre_y_matches_stack_offset():
    """Centre-y of a labelled segment equals cumulative-below + value/2."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        _compute_top_group_annotations,
    )

    # ordered_groups = ["a", "b", "c"]: stack is a (bottom) → b → c (top).
    # On 2026-04-10 the values are a=10, b=20, c=5; "b" peaks here, so its
    # centre is at 10 + 20/2 = 20.
    bucketed = {
        "2026-04-09": {"a": 50.0, "b": 5.0, "c": 1.0},
        "2026-04-10": {"a": 10.0, "b": 20.0, "c": 5.0},
    }
    anns = _compute_top_group_annotations(bucketed, ordered_groups=["a", "b", "c"])
    by_group = {a["text"]: a for a in anns}
    assert by_group["b"]["x"] == "2026-04-10"
    assert by_group["b"]["y"] == 20.0


def test_cost_overview_breakdown_annotates_top_groups():
    """Multi-series chart direct-labels the top groups at their peak bucket."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_model_specs())
        ctx = _materialize_and_run(tmp, lambda c: build_daily_panel_context(c, "model"))
        fig = json.loads(ctx["chart_json"])
        annotations = fig["layout"].get("annotations") or []
        # Filter to annotations pinned to data coordinates — these are the
        # group labels we add (the tufte template's annotation defaults
        # don't add any of their own).
        labels = {ann.get("text") for ann in annotations if ann.get("xref") == "x"}
        # Both fixture models exceed the 4% share threshold so both must
        # appear as direct labels.
        assert {"claude-opus-4-7", "claude-sonnet-4-6"} <= labels


def test_cost_overview_breakdown_total_has_no_annotations():
    """Single-series chart adds no group labels (nothing to direct-label)."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())
        ctx = _materialize_and_run(tmp, lambda c: build_daily_panel_context(c, "total"))
        fig = json.loads(ctx["chart_json"])
        annotations = fig["layout"].get("annotations") or []
        labelled = [ann for ann in annotations if ann.get("xref") == "x"]
        assert labelled == []


def test_cost_overview_breakdown_invalid_falls_back_to_total():
    """Unknown breakdown value collapses to the default."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())
        ctx = _materialize_and_run(
            tmp, lambda c: build_daily_panel_context(c, "garbage")
        )
        assert ctx["breakdown"] == "total"
        fig = json.loads(ctx["chart_json"])
        assert fig["data"][0]["name"] == "Total"


def test_cost_overview_hourly_drilldown_for_known_day():
    """Hourly endpoint returns the chart for a day with cost recorded."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            response = client.get("/cost-overview/breakdown/2026-04-21?breakdown=total")
            assert response.status_code == 200
            text = response.text
            assert "Hourly cost — 2026-04-21" in text
            # Hour bucket comes from the 10:00:01 timestamp in the fixture.
            assert "10:00" in text

        _run_with_client(tmp, _check)


def test_cost_overview_hourly_empty_day_has_no_chart():
    """Hourly endpoint for a day with no cost shows the empty-state message."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            response = client.get("/cost-overview/breakdown/1999-01-01?breakdown=total")
            assert response.status_code == 200
            assert "No cost recorded" in response.text

        _run_with_client(tmp, _check)


def test_cost_overview_hourly_invalid_day_returns_400():
    """Malformed day path must reject early — never hits DuckDB."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            response = client.get("/cost-overview/breakdown/not-a-date?breakdown=total")
            assert response.status_code == 400

        _run_with_client(tmp, _check)


def test_hourly_chart_colors_match_daily_chart():
    """Each group's bar color must match between the daily and hourly panels.

    Otherwise a project blue in the monthly bar can render as orange (or
    fold differently) when the user drills into its day, which is visually
    disorienting because the colour-to-identity mapping silently flips.
    """
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        _build_hourly_panel_context,
        build_daily_panel_context,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_model_specs())

        def _both(c):
            return (
                build_daily_panel_context(c, "model"),
                _build_hourly_panel_context(c, "2026-04-21", "model"),
            )

        daily_ctx, hourly_ctx = _materialize_and_run(tmp, _both)
        daily_fig = json.loads(daily_ctx["chart_json"])
        hourly_fig = json.loads(hourly_ctx["chart_json"])

        daily_colors = {
            trace["name"]: trace.get("marker", {}).get("color")
            for trace in daily_fig["data"]
        }
        hourly_colors = {
            trace["name"]: trace.get("marker", {}).get("color")
            for trace in hourly_fig["data"]
        }
        # Multi-series sanity: at least two groups, otherwise a same-colour
        # accident would still pass.
        assert len(daily_colors) >= 2
        assert len(hourly_colors) >= 2
        for group, color in hourly_colors.items():
            assert color is not None, f"{group} has no pinned colour"
            assert color == daily_colors[group], (
                f"{group}: daily={daily_colors[group]!r} hourly={color!r}"
            )


def test_canonical_color_map_returns_empty_for_total():
    """Single-series 'total' breakdown needs no map — Plotly's default suffices."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        _canonical_color_map,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())
        result = _materialize_and_run(tmp, lambda c: _canonical_color_map(c, "total"))
        assert result == {}


def test_cost_breakdown_collapses_groups_above_cap():
    """The MAX_GROUPS cap merges the long tail into an "Other" bucket."""
    from introspect.api.handlers.cost_breakdown import (  # noqa: PLC0415
        MAX_GROUPS,
        _cap_groups,
    )

    bucketed = {
        "2026-04-21": {f"g{i}": float(i + 1) for i in range(MAX_GROUPS + 3)},
    }
    capped = _cap_groups(bucketed)
    assert "Other" in capped["2026-04-21"]
    assert len(capped["2026-04-21"]) == MAX_GROUPS


def test_fetch_token_usage_dedup():
    """Direct unit test: deduped totals should equal a single message's usage."""
    from introspect.api.handlers._helpers import fetch_token_usage  # noqa: PLC0415
    from introspect.db import get_connection, materialize_views  # noqa: PLC0415

    from ..conftest import glob_pattern  # noqa: PLC0415

    sid = "ftu-dedup-session-0000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _dup_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        conn = get_connection(db_path, glob_pattern(tmp))
        materialize_views(conn, glob_pattern(tmp), 0, resolve_projects=False)
        usage = fetch_token_usage(conn, session_id=sid)
        conn.close()
        assert usage is not None
        # 1M input tokens for ONE message — the duplicate must not double it
        assert usage["input"] == 1_000_000
        assert usage["output"] == 1_000_000
        assert usage["cost_usd"] == pytest.approx(30.0)
