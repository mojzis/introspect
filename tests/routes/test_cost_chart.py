"""Tests for the session cost chart (multi-series, inflection markers, slope)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from introspect.api.main import app

from ..conftest import (
    glob_pattern,
    local_client,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)
from .conftest import SID, _patched_client


def test_session_cost_tab_has_chart_view_map():
    """Cost tab embeds a Plotly figure with a four-view map for the toggle JS."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=cost")
        assert response.status_code == 200
        text = response.text
        # Chart container carries the data attrs the JS reads.
        assert 'id="session-cost-chart"' in text
        assert 'data-on-click="session-cost-marker"' in text
        # data-view-map JSON is HTML-escaped to &quot; — check both forms.
        assert "data-view-map=" in text
        for view_key in ("total", "agent", "category", "invocations"):
            assert view_key in text
        # The figure_json blob lives in a script tag of this id.
        assert 'id="session-cost-chart-data"' in text


def test_session_cost_chart_uses_plotly():
    """Chart renders as a Plotly figure JSON, not hand-rolled SVG polylines."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=cost")
        assert response.status_code == 200
        text = response.text
        # No more SVG polyline rendering for the cost chart.
        assert "<polyline" not in text
        # Figure JSON is embedded for the client-side bootstrap.
        assert 'application/json" id="session-cost-chart-data"' in text
        # Plotly-shaped JSON: at least the trace data and the layout.
        # Use a simple substring check instead of parsing.
        assert '"data"' in text
        assert '"type": "scatter"' in text or '"type":"scatter"' in text


def _spikey_cost_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL with one clear cost spike followed by many tiny messages.

    Need ≥ 6 messages for spike detection. The spike is 1M input tokens on
    message #3; every other message is tiny. With median ~0, threshold is
    clamped to $0.01, and the big message is well above that.
    """
    lines: list[dict] = [
        make_user_message(
            session_id,
            "u0",
            None,
            "2026-04-21T10:00:00.000Z",
            "start",
            tool_use_result={"content": "seed"},
        ),
    ]
    tiny_usage = {"input_tokens": 10, "output_tokens": 5}
    big_usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    # Pattern: tiny, tiny, BIG, tiny, tiny, tiny, tiny, tiny → 8 messages.
    spike_idx = 2
    for i in range(8):
        usage = big_usage if i == spike_idx else tiny_usage
        lines.append(
            make_assistant_message(
                session_id,
                f"a{i}",
                "u0",
                f"2026-04-21T10:00:{i + 1:02d}.000Z",
                [{"type": "text", "text": f"msg{i}"}],
                model="claude-opus-4-7",
                msg_id=f"msg-spiky-{i}",
                usage=usage,
            )
        )
    return write_jsonl(tmp_dir, session_id, lines)


def test_session_cost_chart_marker_in_customdata():
    """Spike detection emits a marker trace whose customdata carries kind+uuid.

    The client-side click handler reads ``customdata[0]`` (uuid) and fires
    the HTMX bloat-filter request — so missing customdata = broken
    interactivity.
    """
    sid = "spiky-session-000000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _spikey_cost_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            local_client(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            text = response.text
            # Marker overlay trace named "Markers" with at least one spike.
            assert '"name": "Markers"' in text or '"name":"Markers"' in text
            assert '"spike"' in text
            # Spike assistant uuid (a2 = the big-usage message) appears in
            # the figure JSON's customdata blob.
            assert '"a2"' in text


def _subagent_with_task_call_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """JSONL with a Task/Agent call + matching sidechain reply.

    Drives the per-agent-type chart view: the Task call supplies a
    subagent_type, and the sidechain assistant message incurs the cost that
    the per-type series should plot. Distinct from the earlier
    ``_subagent_jsonl`` fixture, which tests sidechain bloat *without* a
    preceding Task call.
    """
    tiny = {"input_tokens": 100, "output_tokens": 10}
    lines: list[dict] = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "kick off subagent",
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_task1",
                    "name": "Task",
                    "input": {
                        "description": "look around",
                        "prompt": "do research",
                        "subagent_type": "Explore",
                    },
                }
            ],
            model="claude-opus-4-7",
            msg_id="msg-main",
            usage=tiny,
        ),
        make_user_message(
            session_id,
            "s1",
            "a1",
            "2026-04-21T10:00:02.000Z",
            "do research",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "s2",
            "s1",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id="msg-sub",
            usage={"input_tokens": 10_000, "output_tokens": 500},
            is_sidechain=True,
        ),
        make_user_message(
            session_id,
            "u2",
            "s2",
            "2026-04-21T10:00:04.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_task1",
                    "content": "subagent done",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "subagent done"},
            source_tool_uuid="a1",
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def test_session_cost_tab_hides_invocations_view_without_task_calls():
    """No Task/Agent tool calls → has-subagents flag off, summary table absent."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=cost")
        assert response.status_code == 200
        text = response.text
        # data-has-subagents drives whether the JS toolbar renders the
        # "By invocation" button.
        assert 'data-has-subagents="0"' in text
        assert "Subagent invocations by cost" not in text


def test_session_cost_tab_shows_invocations_view_with_subagent_type():
    """Task call with subagent_type → has-subagents flag on, table + trace name."""
    sid = "subagent-session-00000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _subagent_with_task_call_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            local_client(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            text = response.text
            assert 'data-has-subagents="1"' in text
            # Legend trace name "#1 Explore" appears in the figure JSON, and
            # the same string drives the summary-table row.
            assert "#1 Explore" in text
            assert "Subagent invocations by cost" in text


def test_session_cost_tab_shows_chart_error_notice_on_failure():
    """If chart construction blows up, the cost tab still renders the
    per-model rollup and bloat tables with an inline chart-error notice."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(Path(tmp)) as client,
        patch(
            "introspect.api.handlers.sessions._build_chart_from_attrib",
            side_effect=ValueError("bad figure"),
        ),
    ):
        response = client.get(f"/sessions/{SID}?tab=cost")
        assert response.status_code == 200
        assert "Chart unavailable." in response.text
        assert "ValueError: bad figure" in response.text
        # The rest of the cost tab still renders.
        assert "Bloat attribution" in response.text
        assert "Top contributors" in response.text


def _short_session_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a 3-message session — below the inflection-detection minimum."""
    usage = {"input_tokens": 100, "output_tokens": 20}
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "hi",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id="msg-short-1",
            usage=usage,
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "bye"}],
            model="claude-opus-4-7",
            msg_id="msg-short-2",
            usage=usage,
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def test_inflection_detection_empty_on_short_session():
    """Sessions below _SPIKE_MIN_N produce no inflection markers."""
    from introspect.api.handlers.sessions import _SPIKE_MIN_N  # noqa: PLC0415

    # The fixture emits 2 assistant messages — must be below _SPIKE_MIN_N.
    assert _SPIKE_MIN_N > 2
    sid = "short-session-00000000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _short_session_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            local_client(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            # No marker anchors rendered — the Cost tab must not produce any
            # href to a message anchor when the session is below the minimum.
            marker_prefix = f'href="/sessions/{sid}?tab=messages#msg-'
            assert marker_prefix not in response.text


def test_slope_detector_fires_on_gradual_ramp():
    """Slope detector flags a sustained cost ramp even without a spike."""
    from introspect.api.handlers.sessions import (  # noqa: PLC0415
        _detect_inflection_points,
    )

    # 10 cheap messages, then 10 identical modest messages — each single
    # message is below the spike threshold (median*2), but the slope-window
    # delta jumps dramatically when the ramp starts.
    inc = [0.0005] * 10 + [0.004] * 10
    cum: list[float] = []
    running = 0.0
    for v in inc:
        running += v
        cum.append(running)
    uuids = [f"u{i}" for i in range(len(inc))]
    markers = _detect_inflection_points(uuids, inc, cum)
    # At least one slope marker should fire (the step-up in window delta).
    # No spikes — individual increments are all below $0.01.
    assert any(m["kind"] == "slope" for m in markers), markers
    assert all(m["kind"] != "spike" for m in markers), markers


def test_slope_detector_handles_zero_variance():
    """Constant-cost session must not over-fire slope markers (sigma=0)."""
    from introspect.api.handlers.sessions import (  # noqa: PLC0415
        _detect_inflection_points,
    )

    # All messages identical → every positive delta identical → sigma = 0.
    # Without the guard, the threshold collapses to 0 and the top-N filter
    # fires up to 5 arbitrary markers.
    inc = [0.001] * 15
    cum = [0.001 * (i + 1) for i in range(15)]
    uuids = [f"u{i}" for i in range(15)]
    markers = _detect_inflection_points(uuids, inc, cum)
    assert all(m["kind"] != "slope" for m in markers), markers


def test_slope_detector_handles_single_positive_delta():
    """Single positive slope delta must not fire (can't compute variance)."""
    from introspect.api.handlers.sessions import (  # noqa: PLC0415
        _detect_inflection_points,
    )

    # 10 messages: all zero except one $0.50 late in the session.  The
    # slope window captures exactly one positive delta — len < 2, so the
    # slope branch must short-circuit (the spike branch may still fire).
    inc = [0.0] * 14 + [0.5]
    cum: list[float] = []
    running = 0.0
    for v in inc:
        running += v
        cum.append(running)
    uuids = [f"u{i}" for i in range(len(inc))]
    markers = _detect_inflection_points(uuids, inc, cum)
    assert all(m["kind"] != "slope" for m in markers), markers
