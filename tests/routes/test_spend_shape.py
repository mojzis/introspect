"""Tests for the spend-shape columns (R/W bar + sparkline) on cost overview."""

import tempfile
from pathlib import Path

import pytest

from ..conftest import make_assistant_message, make_user_message
from .cost_helpers import (
    _cost_overview_setup,
    _materialize_and_run,
    _run_with_client,
    _session_at_cost,
)


def _session_with_cache(
    session_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_5m: int = 0,
    cache_creation_1h: int = 0,
    timestamp_day: str = "2026-04-21",
    timestamp_hour: str = "10",
    timestamp_minute: str = "00",
) -> list[dict]:
    """Build a minimal two-message JSONL with explicit cache token counts.

    Usage keys follow the assistant message schema:
    - ``output_tokens`` -> ``output_tokens`` in the view
    - ``cache_read_input_tokens`` -> ``cache_read_tokens`` in the view
    - ``cache_creation.ephemeral_5m_input_tokens`` -> ``cache_creation_5m``
    - ``cache_creation.ephemeral_1h_input_tokens`` -> ``cache_creation_1h``
    """
    ts = f"{timestamp_day}T{timestamp_hour}:{timestamp_minute}:01.000Z"
    usage: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_creation_5m + cache_creation_1h,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_creation_5m,
            "ephemeral_1h_input_tokens": cache_creation_1h,
        },
    }
    return [
        make_user_message(
            session_id,
            "u1",
            None,
            f"{timestamp_day}T{timestamp_hour}:{timestamp_minute}:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            ts,
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a1",
            usage=usage,
        ),
    ]


def test_spend_shape_rw_bar_read_write_dollars():
    """Direct helper test: known cache R/W + output tokens yield correct split dollars.

    claude-opus-4-7: cache read=$0.50/M, cache write 5m=$6.25/M, output=$25/M.
    2M read => $1.00; 1M 5m-write => $6.25; 200k output => $5.00.
    Total = $12.25; px segments must sum to 64.
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _attach_spend_shapes,
        _build_pareto,
    )

    sid = "sess-shape-rw-01-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (
            sid,
            _session_with_cache(
                sid,
                cache_read_tokens=2_000_000,
                cache_creation_5m=1_000_000,
                output_tokens=200_000,
            ),
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _run(conn):
            pareto = _build_pareto(conn)
            _attach_spend_shapes(conn, pareto["rows"], None)
            return pareto["rows"]

        rows = _materialize_and_run(tmp, _run)

    assert len(rows) == 1
    split = rows[0]["split"]
    assert split is not None, "Expected cost-split shape for session with cache+output"
    assert split["read_usd"] == pytest.approx(1.00, rel=1e-3)
    assert split["write_usd"] == pytest.approx(6.25, rel=1e-3)
    assert split["output_usd"] == pytest.approx(5.00, rel=1e-3)
    assert split["read_px"] + split["write_px"] + split["output_px"] == 64


def test_spend_shape_sparkline_geometry():
    """Sparkline: points are time-ordered, y non-increasing, longer session wider.

    Session A has 3 messages spread over 60 min; session B has 2 messages
    spread over 5 min. Both are in the same pareto table so max_dur is
    driven by A. The sparkline for A should be wider than B.
    Cost is front-loaded in session A (all cost in first message).
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _attach_spend_shapes,
        _build_pareto,
    )

    sid_a = "sess-shape-spark-a-aaaa-aaaa-aaaaaaaaaaaa"
    sid_b = "sess-shape-spark-b-aaaa-aaaa-aaaaaaaaaaaa"

    # Session A: 3 messages at :00, :30, :60 (60-minute spread); cost concentrated early
    ts_a_base = "2026-04-21"
    lines_a = [
        make_user_message(
            sid_a,
            "u1",
            None,
            f"{ts_a_base}T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid_a,
            "a1",
            "u1",
            f"{ts_a_base}T10:00:01.000Z",
            [{"type": "text", "text": "step1"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_a}-a1",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},  # $5 early
        ),
        make_user_message(sid_a, "u2", "a1", f"{ts_a_base}T10:30:00.000Z", "more"),
        make_assistant_message(
            sid_a,
            "a2",
            "u2",
            f"{ts_a_base}T10:30:01.000Z",
            [{"type": "text", "text": "step2"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_a}-a2",
            usage={"input_tokens": 10, "output_tokens": 0},  # tiny
        ),
        make_user_message(sid_a, "u3", "a2", f"{ts_a_base}T11:00:00.000Z", "done"),
        make_assistant_message(
            sid_a,
            "a3",
            "u3",
            f"{ts_a_base}T11:00:01.000Z",
            [{"type": "text", "text": "step3"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_a}-a3",
            usage={"input_tokens": 10, "output_tokens": 0},  # tiny
        ),
    ]

    # Session B: 2 messages at :00, :05 (5-minute spread)
    lines_b = [
        make_user_message(
            sid_b,
            "u1",
            None,
            f"{ts_a_base}T12:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid_b,
            "a1",
            "u1",
            f"{ts_a_base}T12:00:01.000Z",
            [{"type": "text", "text": "ok1"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_b}-a1",
            usage={"input_tokens": 100, "output_tokens": 0},
        ),
        make_user_message(sid_b, "u2", "a1", f"{ts_a_base}T12:05:00.000Z", "done"),
        make_assistant_message(
            sid_b,
            "a2",
            "u2",
            f"{ts_a_base}T12:05:01.000Z",
            [{"type": "text", "text": "ok2"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_b}-a2",
            usage={"input_tokens": 100, "output_tokens": 0},
        ),
    ]

    specs = [(sid_a, lines_a), (sid_b, lines_b)]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _run(conn):
            pareto = _build_pareto(conn)
            _attach_spend_shapes(conn, pareto["rows"], None)
            return pareto["rows"]

        rows = _materialize_and_run(tmp, _run)

    by_sid = {r["session_id"]: r for r in rows}
    spark_a = by_sid[sid_a]["spark"]
    spark_b = by_sid[sid_b]["spark"]

    assert spark_a is not None, "Session A should have a sparkline"
    assert spark_b is not None, "Session B should have a sparkline"

    # Session A (60 min) should be wider than session B (5 min).
    assert spark_a["width"] > spark_b["width"]

    # Points in session A should be time-ordered (left to right: x increasing)
    # and y must be non-increasing (SVG y grows downward as cumulative grows).
    if spark_a["points"]:
        pairs = [
            (float(p.split(",")[0]), float(p.split(",")[1]))
            for p in spark_a["points"].split()
        ]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        # x must be non-decreasing (time order)
        assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
        # y must be non-increasing (cumulative cost grows downward in SVG coords)
        assert all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))


def test_spend_shape_degenerate_cases():
    """Degenerate: single-message session => spark dot; zero-cache => split is None.

    Also verifies that an output-only session (no cache) renders a full-width
    output segment (output_px == 64).
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _attach_spend_shapes,
        _build_pareto,
    )

    sid_single = "sess-shape-degen-single-aaaa-aaaaaaaaaa"
    sid_nocache = "sess-shape-degen-nocach-aaaa-aaaaaaaaaa"
    sid_output_only = "sess-shape-degen-outonly-aaa-aaaaaaaaaa"

    specs = [
        (sid_single, _session_at_cost(sid_single, 100_000)),
        (
            sid_nocache,
            _session_with_cache(
                sid_nocache,
                input_tokens=200_000,
            ),
        ),
        (
            sid_output_only,
            _session_with_cache(
                sid_output_only,
                output_tokens=200_000,
            ),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _run(conn):
            pareto = _build_pareto(conn)
            _attach_spend_shapes(conn, pareto["rows"], None)
            return pareto["rows"]

        rows = _materialize_and_run(tmp, _run)

    by_sid = {r["session_id"]: r for r in rows}

    # Single-message session: spark should not be None (returns dot form),
    # and no crash.
    spark_single = by_sid[sid_single]["spark"]
    assert spark_single is not None  # degrades to dot, not None

    # Input-only session: no cache, no output => split must be None (em-dash).
    split_nocache = by_sid[sid_nocache]["split"]
    assert split_nocache is None

    # Output-only session: no cache but output tokens => full-width output bar.
    split_output = by_sid[sid_output_only]["split"]
    assert split_output is not None, "Expected bar for output-only session"
    assert split_output["output_px"] == 64
    assert split_output["read_px"] == 0
    assert split_output["write_px"] == 0


def test_cost_overview_page_has_spend_shape_columns():
    """Route test: /cost-overview renders the new column headers and markup."""
    sid = "sess-shape-route-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (
            sid,
            _session_with_cache(
                sid,
                cache_read_tokens=500_000,
                cache_creation_5m=200_000,
            ),
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            text = response.text
            # New column headers
            assert "Cost split" in text
            assert "Spend shape" in text
            # SVG polyline or circle markup for the sparkline
            assert "<polyline" in text or "<circle" in text

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_window_scopes_shapes():
    """Window test: portfolio?day= only reflects in-window messages in shapes.

    Session has two assistant messages: one on day A with cache reads,
    one on day B with cache writes. When filtered to day A, only the
    read $ should show in the R/W bar (write_usd near zero).
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _attach_spend_shapes,
        _build_pareto,
    )

    sid = "sess-shape-window-aaaa-aaaa-aaaaaaaaaaaa"
    # Day A: cache read message
    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-20T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-20T10:00:01.000Z",
            [{"type": "text", "text": "read-heavy"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid}-a1",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 2_000_000,
                "cache_creation_input_tokens": 0,
            },
        ),
        # Day B: cache write message
        make_user_message(sid, "u2", "a1", "2026-04-21T10:00:00.000Z", "more"),
        make_assistant_message(
            sid,
            "a2",
            "u2",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "write-heavy"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid}-a2",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1_000_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 1_000_000,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
        ),
    ]
    specs = [(sid, lines)]

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        from introspect.api.handlers.cost_overview import _window_for  # noqa: PLC0415

        def _run(conn):
            window = _window_for("2026-04-20", None)
            pareto = _build_pareto(conn, window)
            _attach_spend_shapes(conn, pareto["rows"], window)
            return pareto["rows"]

        rows = _materialize_and_run(tmp, _run)

    # The day-A window captures 2M cache-read tokens (~$1.00) so the session
    # must appear.  Its split bar must be present (cache activity exists) and
    # write_usd and output_usd must be zero because only the read message is on day A.
    assert len(rows) == 1
    split = rows[0]["split"]
    assert split is not None
    assert split["write_usd"] == pytest.approx(0.0, abs=1e-6)
    assert split["output_usd"] == pytest.approx(0.0, abs=1e-6)
