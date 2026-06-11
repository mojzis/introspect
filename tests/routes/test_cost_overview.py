"""Tests for the Cost Overview page."""

import tempfile
from pathlib import Path

import pytest

from ..conftest import make_assistant_message, make_user_message, write_jsonl
from .cost_helpers import (
    _cache_loss_session_lines,
    _cost_overview_setup,
    _materialize_and_run,
    _multi_day_specs,
    _run_with_client,
    _session_at_cost,
)


def _subagent_overview_session(session_id: str, input_tokens: int) -> list[dict]:
    """Two-message session whose assistant record is a sidechain message."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_user_message(
            session_id,
            "su1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            "subagent",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sa1",
            "su1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-sa1",
            usage={"input_tokens": input_tokens, "output_tokens": 0},
            is_sidechain=True,
        ),
    ]
    return lines


def _huge_reads_session(session_id: str) -> list[dict]:
    """Build a session whose Read-category cache creation is ≥10% of cost
    and ≥100k tokens — i.e., classifies as "with huge reads".

    Strategy: a Read tool use followed by an assistant message whose
    parent is the tool_result, with 200k cache-creation tokens (well over
    the 100k floor) AND a dominant cache-creation cost share.
    """
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "please review",
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "tu-huge-read",
                    "name": "Read",
                    "input": {"file_path": "/repo/src/huge_file.py"},
                }
            ],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a1",
            usage={"input_tokens": 100, "output_tokens": 5},
        ),
        make_user_message(
            session_id,
            "u2",
            "a1",
            "2026-04-21T10:00:02.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-huge-read",
                    "content": "x" * 100,
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "x" * 100},
            source_tool_uuid="a1",
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a2",
            usage={
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 200_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 200_000,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
        ),
    ]
    return lines


def _skill_session_command_tag(session_id: str) -> list[dict]:
    """Session that uses a non-built-in skill via <command-name>."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "<command-name>marimo-pair</command-name>\nplease pair",
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a1",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
        ),
    ]
    return lines


def _skill_session_clear(session_id: str) -> list[dict]:
    """Session whose only command is /clear — a built-in that must NOT
    flip it into the "with skills" bucket.
    """
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "<command-name>/clear</command-name>",
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a1",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
        ),
    ]
    return lines


def _cross_day_subagent_session(
    session_id: str, day_a_tokens: int, day_b_tokens: int
) -> list[dict]:
    """Session with sidechain cost on Day A and regular cost on Day B.

    Used to pin the ``cost_overview`` design contract: subagent classifier
    is **all-time**, so when the portfolio panel is filtered to Day B,
    the session must still classify "with subagent" even though no
    sidechain message exists in the Day-B window.
    """
    day_a = "2026-04-20"
    day_b = "2026-04-21"
    return [
        # Day A: head user + sidechain user + sidechain assistant.
        make_user_message(
            session_id,
            "u1",
            None,
            f"{day_a}T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_user_message(
            session_id,
            "su1",
            "u1",
            f"{day_a}T10:00:01.000Z",
            "subagent",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sa1",
            "su1",
            f"{day_a}T10:00:02.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-sa1",
            usage={"input_tokens": day_a_tokens, "output_tokens": 0},
            is_sidechain=True,
        ),
        # Day B: regular (non-sidechain) assistant message — its cost is
        # what the Day-B filter window picks up.
        make_user_message(
            session_id,
            "u2",
            "sa1",
            f"{day_b}T10:00:00.000Z",
            "next day",
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            f"{day_b}T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a2",
            usage={"input_tokens": day_b_tokens, "output_tokens": 0},
        ),
    ]


def test_cost_overview_page_renders():
    """Cost Overview returns 200 and includes the hero total-cost string."""
    # Three sessions at known costs so the hero total is deterministic.
    # 4M+2M+1M = 7M input tokens * $5/M = $35.00.
    specs = [
        (
            "sess-render-01-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost("sess-render-01-aaaa-aaaa-aaaaaaaaaaaa", 4_000_000),
        ),
        (
            "sess-render-02-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost("sess-render-02-aaaa-aaaa-aaaaaaaaaaaa", 2_000_000),
        ),
        (
            "sess-render-03-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost("sess-render-03-aaaa-aaaa-aaaaaaaaaaaa", 1_000_000),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            text = response.text
            assert "Cost Overview" in text
            # Hero total: $35.00 (4M+2M+1M) * $5/M.
            assert "$35.00" in text
            # Cost-split legend under "Sessions ranked by cost". Anchor on
            # "Cost split:" (with colon) — unique to the legend, unlike
            # "cache read"/"cache write" which also occur in row tooltips.
            assert "Cost split:" in text
            assert "cache read" in text
            assert "cache write" in text
            assert text.index("Sessions ranked by cost") < text.index("Cost split:")

        _run_with_client(tmp, _check)


def test_cost_overview_pareto_cutoff_at_80pct():
    """Pareto cutoff must include rows up to and including the 80% crossing.

    Fixture: 10 sessions with input_tokens sized so cost_usd = tokens / 200k.
    Costs: 100, 50, 30, 20, 10, 5, 5, 3, 2, 1. Total $226; 80% = $180.80.
    Cumulative: 100, 150, 180, 200, 210, 215, 220, 223, 225, 226.
    Row 3 is $180 (79.6%, below cutoff). Row 4 is $200 (88.5%, first
    crossing) — row 4 is the cutoff and rows 1-4 are in Pareto; row 5 is
    NOT in Pareto.
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _build_pareto,
    )

    costs_usd = [100, 50, 30, 20, 10, 5, 5, 3, 2, 1]
    specs: list[tuple[str, list[dict]]] = []
    for rank, c in enumerate(costs_usd):
        # $1 == 200_000 input tokens at $5/M claude-opus-4-7 pricing.
        sid = f"sess-pareto-{rank:02d}-aaaa-aaaa-aaaaaaaaaaaa"
        specs.append((sid, _session_at_cost(sid, c * 200_000)))
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        pareto = _materialize_and_run(tmp, _build_pareto)

        # Rows are sorted by cost_usd DESC. Check the first 5 rows by
        # cost and their Pareto membership.
        rows = pareto["rows"]
        assert len(rows) == 10
        # Row 0: $100 (44.2% cum), in pareto, not cutoff.
        assert rows[0]["cost_usd"] == pytest.approx(100.0)
        assert rows[0]["in_pareto"]
        assert not rows[0]["is_cutoff"]
        # Row 1: $50 ($150, 66.4%), in pareto, not cutoff.
        assert rows[1]["in_pareto"]
        assert not rows[1]["is_cutoff"]
        # Row 2: $30 ($180, 79.6%), in pareto, not cutoff (still < 80%).
        assert rows[2]["in_pareto"]
        assert not rows[2]["is_cutoff"]
        # Row 3: $20 ($200, 88.5%), in pareto, IS cutoff (first ≥80%).
        assert rows[3]["in_pareto"]
        assert rows[3]["is_cutoff"]
        # Row 4: $10 ($210), NOT in pareto.
        assert not rows[4]["in_pareto"]
        assert not rows[4]["is_cutoff"]

        assert pareto["pareto_session_count"] == 4
        assert pareto["total_session_count"] == 10
        assert pareto["total_cost_usd"] == pytest.approx(226.0)


def test_cost_overview_subagent_split():
    """Subagent-presence split: with-row counts only sidechain sessions."""
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _build_subagent_split,
        _fetch_cost_rows,
    )

    sid_with = "sess-side-01-aaaa-aaaa-aaaaaaaaaaaa"
    sid_without = "sess-side-02-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (sid_with, _subagent_overview_session(sid_with, 2_000_000)),  # $10
        (sid_without, _session_at_cost(sid_without, 1_000_000)),  # $5
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        split = _materialize_and_run(
            tmp,
            lambda c: _build_subagent_split(c, _fetch_cost_rows(c)),
        )

        assert split["with"]["sessions"] == 1
        assert split["without"]["sessions"] == 1
        # Sidechain session cost = $10 (2M * $5/M).
        assert split["with"]["cost_usd"] == pytest.approx(10.0)
        # Non-sidechain cost = $5 (1M * $5/M).
        assert split["without"]["cost_usd"] == pytest.approx(5.0)


def test_cost_overview_huge_reads_split():
    """A session whose Read-category cache creation clears both guards
    (≥10% of session cost AND ≥100k tokens) classifies as "with huge reads".
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _build_huge_reads_split,
        _fetch_cost_rows,
    )

    sid_with = "sess-huge-01-aaaa-aaaa-aaaaaaaaaaaa"
    sid_without = "sess-huge-02-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (sid_with, _huge_reads_session(sid_with)),
        (sid_without, _session_at_cost(sid_without, 1_000_000)),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        split = _materialize_and_run(
            tmp,
            lambda c: _build_huge_reads_split(c, _fetch_cost_rows(c)),
        )

        # Only the huge-reads session classifies "with".
        assert split["with"]["sessions"] == 1
        assert split["without"]["sessions"] == 1
        # Pin the "without" side to the known baseline ($5.00 = 1M * $5/M).
        # A threshold inversion would route the huge-reads session into
        # "without" and push its cost above $5, so pinning this side
        # exactly catches a with/without swap.
        assert split["without"]["cost_usd"] == pytest.approx(5.0)
        # The "with" side gets the session whose ~$1.25 cost is dominated
        # by 200k cache-creation tokens — the ratio (not the absolute) is
        # what clears the 10% guard. Confirm the with-side cost matches
        # the huge-reads session, not the baseline.
        assert split["with"]["cost_usd"] < split["without"]["cost_usd"]


def test_cost_overview_skill_split():
    """/clear and other OBVIOUS_COMMANDS must not flip a session's classification.

    Fixture: session 1 uses <command-name>marimo-pair</command-name> (should
    classify as "with skills"); session 2 uses no commands; session 3 uses
    /clear only. Only session 1 classifies as "with"; sessions 2 and 3 are
    both "without".
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _build_skill_split,
        _fetch_cost_rows,
    )

    sid1 = "sess-skill-01-aaaa-aaaa-aaaaaaaaaaaa"
    sid2 = "sess-skill-02-aaaa-aaaa-aaaaaaaaaaaa"
    sid3 = "sess-skill-03-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (sid1, _skill_session_command_tag(sid1)),
        (sid2, _session_at_cost(sid2, 1_000_000)),
        (sid3, _skill_session_clear(sid3)),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        split = _materialize_and_run(
            tmp,
            lambda c: _build_skill_split(c, _fetch_cost_rows(c)),
        )

        # Only session 1 is "with skills"; sessions 2 and 3 are not.
        assert split["with"]["sessions"] == 1
        assert split["without"]["sessions"] == 2


# --- Portfolio panel time-window filter tests ---


def test_cost_overview_portfolio_filter_day_scopes_pareto():
    """``/cost-overview/portfolio?day=...`` narrows Pareto to that day."""
    sid_a = "sess-port-da-aaaa-aaaa-aaaaaaaaaaaa"
    sid_b = "sess-port-db-aaaa-aaaa-aaaaaaaaaaaa"
    sid_c = "sess-port-dc-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (sid_a, _session_at_cost(sid_a, 4_000_000, timestamp_day="2026-04-20")),
        (sid_b, _session_at_cost(sid_b, 2_000_000, timestamp_day="2026-04-21")),
        (sid_c, _session_at_cost(sid_c, 1_000_000, timestamp_day="2026-04-21")),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview/portfolio?day=2026-04-21")
            assert response.status_code == 200
            text = response.text
            # Filter chip rendered with the day label.
            assert "Filtered to" in text
            assert "2026-04-21" in text
            # Pareto totals: $10 + $5 = $15 (Day 21 sessions only).
            assert "$15.00" in text
            # Day-20 cost ($20.00) must NOT appear.
            assert "$20.00" not in text

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_filter_hour_scopes_pareto():
    """``?day=...&hour=14`` further narrows to a single hour."""
    sid_morning = "sess-port-am-aaaa-aaaa-aaaaaaaaaaaa"
    sid_afternoon = "sess-port-pm-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (
            sid_morning,
            _session_at_cost(
                sid_morning,
                2_000_000,
                timestamp_day="2026-04-21",
                timestamp_hour="10",
            ),
        ),
        (
            sid_afternoon,
            _session_at_cost(
                sid_afternoon,
                1_000_000,
                timestamp_day="2026-04-21",
                timestamp_hour="14",
            ),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview/portfolio?day=2026-04-21&hour=14")
            assert response.status_code == 200
            text = response.text
            assert "2026-04-21 14:00" in text
            # Only the 14:00 session ($5) appears.
            assert "$5.00" in text
            # The 10:00 session's $10 must not.
            assert "$10.00" not in text

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_invalid_day_returns_400():
    """Bad day format is rejected with 400, not silently treated as no-filter."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # No fixtures needed — the validator runs before any DB work.

        def _check(client):
            response = client.get("/cost-overview/portfolio?day=garbage")
            assert response.status_code == 400

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_hour_without_day_returns_400():
    """``?hour=14`` alone is meaningless and must be rejected."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        def _check(client):
            response = client.get("/cost-overview/portfolio?hour=14")
            assert response.status_code == 400

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_no_filter_matches_page_total():
    """Unfiltered portfolio call must agree with the page's hero total."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, _multi_day_specs())

        def _check(client):
            page = client.get("/cost-overview")
            assert page.status_code == 200
            fragment = client.get("/cost-overview/portfolio")
            assert fragment.status_code == 200
            # Multi-day specs sum to $35 (4M+2M+1M tokens * $5/M).
            assert "$35.00" in page.text
            assert "$35.00" in fragment.text
            # No filter ⇒ no chip on the fragment.
            assert "Filtered to" not in fragment.text

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_clear_link_targets_panel():
    """Filtered render must include a Clear link that re-fetches without filter."""
    sid = "sess-port-clr-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [(sid, _session_at_cost(sid, 1_000_000, timestamp_day="2026-04-21"))]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview/portfolio?day=2026-04-21")
            assert response.status_code == 200
            text = response.text
            assert 'hx-get="/cost-overview/portfolio"' in text
            assert 'hx-target="#cost-portfolio-panel"' in text
            assert "Clear" in text

        _run_with_client(tmp, _check)


def test_cost_overview_portfolio_subagent_classifier_stays_alltime():
    """Filtering to Day B must keep an all-time subagent flag on the session.

    A session with a sidechain message on Day A and regular cost on Day B,
    filtered to Day B, must still land in the "with" bucket — pins the
    documented semantic that subagent/skill classifiers are session-wide,
    only the cost aggregation narrows with the window.
    """
    from introspect.api.handlers.cost_overview import (  # noqa: PLC0415
        _build_panel_context,
        _window_for,
    )

    sid_with = "sess-port-sub-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [
        (sid_with, _cross_day_subagent_session(sid_with, 200_000, 1_000_000)),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        panel = _materialize_and_run(
            tmp,
            lambda c: _build_panel_context(c, _window_for("2026-04-21", None)),
        )

        # Pareto cost narrows to Day B only: 1M tokens * $5/M = $5.
        assert panel["pareto"]["total_cost_usd"] == pytest.approx(5.0)
        # Classifier stays all-time: the session's Day-A sidechain still
        # flips it into "with subagent" even though the window doesn't
        # include any sidechain message.
        subagent = panel["subagent_split"]
        assert subagent["with"]["sessions"] == 1
        assert subagent["without"]["sessions"] == 0
        assert subagent["with"]["cost_usd"] == pytest.approx(5.0)


def test_cost_overview_renders_with_titleless_session():
    """Sessions filtered out of ``session_titles`` must not crash the panel.

    Regression: the Pareto template slices ``session_id[:8]`` when ``title``
    is empty. DuckDB sometimes returns ``session_id`` as ``uuid.UUID`` (not
    subscriptable), so the dict builder must coerce to ``str``.
    """
    # UUID-shaped id so DuckDB infers the column as UUID (the production
    # type that breaks ``session_id[:8]`` slicing); a non-UUID-shaped id
    # would silently fall back to VARCHAR and not reproduce the bug.
    sid = "deadbeef-1234-5678-9abc-def012345678"
    # First user message is ``/clear`` — session_titles filters it out, so
    # the LEFT JOIN yields NULL/'' for first_prompt and the template falls
    # through to the session_id slice.
    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "/clear",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_jsonl(tmp, sid, lines)

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            # Title fallback rendered the first 8 chars of the session_id.
            assert sid[:8] in response.text

        _run_with_client(tmp, _check)


def test_cost_overview_cache_loss_stat_card():
    """Cost overview surfaces the cache-loss premium when events exist."""
    sid = "sess-loss-01-aaaa-aaaa-aaaaaaaaaaaa"
    specs = [(sid, _cache_loss_session_lines(sid, gap_minutes=6))]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            text = response.text
            assert "Wasted on cache misses" in text
            # opus-4-6 5m write premium = (6.25 - 0.50)/1M * 8500 ≈ $0.0489.
            # format_cost rounds to "$0.05".
            assert "$0.05" in text
            assert "1 event" in text

        _run_with_client(tmp, _check)


def test_cost_overview_cache_loss_card_hidden_without_events():
    """No cache-loss events → no stat card, no 'Wasted' label."""
    sid = "sess-loss-02-aaaa-aaaa-aaaaaaaaaaaa"
    # Same shape, but 4-min gap: under threshold.
    specs = [(sid, _cache_loss_session_lines(sid, gap_minutes=4))]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview")
            assert response.status_code == 200
            assert "Wasted on cache misses" not in response.text

        _run_with_client(tmp, _check)


def test_cost_overview_cache_loss_respects_window():
    """Windowed portfolio query only counts events whose rebuild lands inside."""
    sid_in = "sess-loss-in-aaaa-aaaa-aaaaaaaaaaaa"
    sid_out = "sess-loss-out-aaaa-aaaa-aaaaaaaaaaa"
    specs = [
        (
            sid_in,
            _cache_loss_session_lines(
                sid_in, gap_minutes=6, timestamp_day="2026-04-21"
            ),
        ),
        (
            sid_out,
            _cache_loss_session_lines(
                sid_out, gap_minutes=6, timestamp_day="2026-04-22"
            ),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _cost_overview_setup(tmp, specs)

        def _check(client):
            response = client.get("/cost-overview/portfolio?day=2026-04-21")
            assert response.status_code == 200
            text = response.text
            # In-window event surfaces.
            assert "Wasted on cache misses" in text
            assert "1 event" in text
            # Out-of-window event does not — count would say "2 event" if it did.
            assert "2 event" not in text

        _run_with_client(tmp, _check)
