"""Tests for tokenscape subagent drill-down charts and run grouping."""

import json
import tempfile
from pathlib import Path

import pytest

from ..conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)
from .conftest import SID, _patched_client, _write_sample_jsonl
from .tokenscape_helpers import (
    _tokenscape_session_jsonl,
    _tokenscape_subagent_session_jsonl,
)


def test_tokenscape_subagent_run_ties_out_to_run_bill():
    """One Task → one drill-down chart whose stripes sum to the run's bill."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape07"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_subagent_session_jsonl(tmp, sid)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    runs = ctx["subagent_runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["label"] == "agent: scan docs"
    assert run["turn_count"] == 2
    expected = compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=4,
        output_tokens=1_000,
        cache_creation_5m=50_000,
    ) + compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=4,
        output_tokens=500,
        cache_read_tokens=50_000,
        cache_creation_5m=21_000,
    )
    assert run["cost"] == pytest.approx(expected, rel=1e-6)
    fig = json.loads(run["figure_json"])
    stripes_total = sum(sum(t["y"]) for t in fig["data"])
    assert stripes_total == pytest.approx(run["cost"], abs=0.0051)
    # The agent's big Read surfaces as its own stripe — the drill-down's
    # whole point is showing what the agent read.
    assert any(t["name"] == "Read doc.md" for t in fig["data"])
    assert ctx["subagent_runs_hidden"] == {"count": 0, "cost": 0.0}
    # The drill-down carries the run's models and the same colour its
    # stripe got in the session-level stripes chart.
    assert run["models"] == ["claude-sonnet-4-6"]
    stripes_fig = json.loads(ctx["figure_json_stripes"])
    (stripe_color,) = [
        t["marker"]["color"]
        for t in stripes_fig["data"]
        if t["name"] == "agent: scan docs"
    ]
    assert run["fill"] == stripe_color


def test_tokenscape_subagent_label_survives_dangling_parent():
    """Conversation threads whose root parent_uuid exists nowhere in the
    log can't chain back to their prompt — the drill-down label falls
    back to the most recent preceding Task call, not 'agent: run'."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape10"
    model = "claude-sonnet-4-6"
    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "first prompt",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_task1",
                    "name": "Task",
                    "input": {
                        "description": "scan docs",
                        "prompt": "analyze the docs",
                        "subagent_type": "explore",
                    },
                }
            ],
            model=model,
            msg_id="msg-dp-1",
            usage={"input_tokens": 4, "output_tokens": 100},
        ),
        # Sidechain prompt is a parentless one-message thread; the
        # conversation roots separately at an assistant message whose
        # parent uuid is dangling (real Claude Code logs do this).
        make_user_message(
            sid,
            "sc-u1",
            None,
            "2026-04-21T10:00:02.000Z",
            "analyze the docs",
            is_sidechain=True,
        ),
        make_assistant_message(
            sid,
            "sc-a1",
            "nowhere-to-be-found",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "on it"}],
            model=model,
            msg_id="msg-dp-sc1",
            usage={
                "input_tokens": 4,
                "output_tokens": 500,
                "cache_creation_input_tokens": 50_000,
            },
            is_sidechain=True,
        ),
        make_assistant_message(
            sid,
            "sc-a2",
            "sc-a1",
            "2026-04-21T10:00:04.000Z",
            [{"type": "text", "text": "docs analyzed"}],
            model=model,
            msg_id="msg-dp-sc2",
            usage={
                "input_tokens": 4,
                "output_tokens": 500,
                "cache_read_input_tokens": 50_000,
                "cache_creation_input_tokens": 1_000,
            },
            is_sidechain=True,
        ),
        make_user_message(
            sid,
            "u2",
            "a1",
            "2026-04-21T10:00:06.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_task1",
                    "content": "agent finished",
                }
            ],
        ),
        make_assistant_message(
            sid,
            "a2",
            "u2",
            "2026-04-21T10:00:07.000Z",
            [{"type": "text", "text": "all done"}],
            model=model,
            msg_id="msg-dp-2",
            usage={"input_tokens": 4, "output_tokens": 50},
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_jsonl(tmp, sid, lines)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    runs = ctx["subagent_runs"]
    assert len(runs) == 1
    assert runs[0]["label"] == "agent: scan docs"
    # Both conversation messages merged into the labelled run — not just
    # the prompt-thread fragment that carried the label.
    assert runs[0]["turn_count"] == 2


def test_tokenscape_concurrent_dangling_threads_stay_separate_runs():
    """Two unlinkable sidechain threads overlapping in time (a forked
    skill, a nested agent) inherit the call's label but must NOT be
    concatenated — interleaving would fold the chart's time axis back
    on itself."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape12"
    model = "claude-sonnet-4-6"
    usage = {"input_tokens": 4, "output_tokens": 500}

    def sidechain_turn(uuid: str, parent: str, second: int, text: str) -> dict:
        return make_assistant_message(
            sid,
            uuid,
            parent,
            f"2026-04-21T10:00:{second:02d}.000Z",
            [{"type": "text", "text": text}],
            model=model,
            msg_id=f"msg-cd-{uuid}",
            usage={**usage, "cache_creation_input_tokens": 30_000},
            is_sidechain=True,
        )

    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "first prompt",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_task1",
                    "name": "Task",
                    "input": {
                        "description": "scan docs",
                        "prompt": "analyze the docs",
                        "subagent_type": "explore",
                    },
                }
            ],
            model=model,
            msg_id="msg-cd-1",
            usage=usage,
        ),
        # Thread A (dangling parent): 10:00:03 → 10:00:10.
        sidechain_turn("sc-a1", "gone-a", 3, "thread A start"),
        sidechain_turn("sc-a2", "sc-a1", 10, "thread A end"),
        # Thread B (dangling parent) overlaps A: 10:00:05 → 10:00:07.
        sidechain_turn("sc-b1", "gone-b", 5, "thread B start"),
        sidechain_turn("sc-b2", "sc-b1", 7, "thread B end"),
        make_user_message(
            sid,
            "u2",
            "a1",
            "2026-04-21T10:00:12.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_task1",
                    "content": "agent finished",
                }
            ],
        ),
        make_assistant_message(
            sid,
            "a2",
            "u2",
            "2026-04-21T10:00:13.000Z",
            [{"type": "text", "text": "all done"}],
            model=model,
            msg_id="msg-cd-2",
            usage=usage,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_jsonl(tmp, sid, lines)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    runs = ctx["subagent_runs"]
    assert len(runs) == 2
    assert {r["label"] for r in runs} == {"agent: scan docs"}
    assert sorted(r["turn_count"] for r in runs) == [2, 2]


def test_tokenscape_stripe_labels_use_dark_ink():
    """Stripe captions render in the high-contrast label font, not the
    greyish category border colours."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _STRIPE_LABEL_FONT,
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape11"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_session_jsonl(tmp, sid)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    fig = json.loads(ctx["figure_json_stripes"])
    annotations = fig["layout"]["annotations"]
    assert annotations
    for annotation in annotations:
        assert annotation["font"] == _STRIPE_LABEL_FONT


def test_tokenscape_parallel_subagents_split_by_parent_chain():
    """Interleaved sidechain threads split per Task via parent chains —
    a timestamp sweep would lump them into one run."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape08"
    model = "claude-sonnet-4-6"

    def sc_usage(cc: int, out: int) -> dict:
        return {
            "input_tokens": 4,
            "output_tokens": out,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": cc,
        }

    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "scan everything",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_taskA",
                    "name": "Task",
                    "input": {
                        "description": "scan docs",
                        "prompt": "analyze the docs",
                        "subagent_type": "explore",
                    },
                },
                {
                    "type": "tool_use",
                    "id": "toolu_taskB",
                    "name": "Task",
                    "input": {
                        "description": "scan code",
                        "prompt": "analyze the code",
                        "subagent_type": "explore",
                    },
                },
            ],
            model=model,
            msg_id="msg-par-1",
            usage={
                "input_tokens": 4,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 10_000,
            },
        ),
        # Two sidechain threads with interleaved timestamps but disjoint
        # parent chains and distinct root prompts.
        make_user_message(
            sid,
            "t1-u1",
            None,
            "2026-04-21T10:00:02.000Z",
            "analyze the docs",
            is_sidechain=True,
        ),
        make_user_message(
            sid,
            "t2-u1",
            None,
            "2026-04-21T10:00:02.500Z",
            "analyze the code",
            is_sidechain=True,
        ),
        make_assistant_message(
            sid,
            "t1-a1",
            "t1-u1",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "docs done"}],
            model=model,
            msg_id="msg-par-t1",
            usage=sc_usage(50_000, 1_000),
            is_sidechain=True,
        ),
        make_assistant_message(
            sid,
            "t2-a1",
            "t2-u1",
            "2026-04-21T10:00:03.500Z",
            [{"type": "text", "text": "code done"}],
            model=model,
            msg_id="msg-par-t2",
            usage=sc_usage(40_000, 800),
            is_sidechain=True,
        ),
        make_user_message(
            sid,
            "u2",
            "a1",
            "2026-04-21T10:00:06.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_taskA",
                    "content": "docs done",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_taskB",
                    "content": "code done",
                },
            ],
        ),
        make_assistant_message(
            sid,
            "a2",
            "u2",
            "2026-04-21T10:00:07.000Z",
            [{"type": "text", "text": "all done"}],
            model=model,
            msg_id="msg-par-2",
            usage={
                "input_tokens": 4,
                "output_tokens": 50,
                "cache_read_input_tokens": 10_000,
                "cache_creation_input_tokens": 500,
            },
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_jsonl(tmp, sid, lines)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    runs = ctx["subagent_runs"]
    assert len(runs) == 2
    by_label = {r["label"]: r for r in runs}
    expected_docs = compute_cost_usd(
        model=model, input_tokens=4, output_tokens=1_000, cache_creation_5m=50_000
    )
    expected_code = compute_cost_usd(
        model=model, input_tokens=4, output_tokens=800, cache_creation_5m=40_000
    )
    assert by_label["agent: scan docs"]["cost"] == pytest.approx(
        expected_docs, rel=1e-6
    )
    assert by_label["agent: scan code"]["cost"] == pytest.approx(
        expected_code, rel=1e-6
    )
    # Per-run traces carry matching colours in both charts. (Parallel
    # agents launched in one turn lump into one run in the timestamp
    # sweep, so a single agent trace is expected here.)
    stripes_fig = json.loads(ctx["figure_json_stripes"])
    stripe_colors = {
        t["name"]: t["marker"]["color"]
        for t in stripes_fig["data"]
        if t["name"].startswith("agent:")
    }
    bar_fig = json.loads(ctx["figure_json"])
    bar_colors = {
        t["name"]: t["marker"]["color"]
        for t in bar_fig["data"]
        if t["name"].startswith("agent:")
    }
    assert stripe_colors
    assert bar_colors == stripe_colors
    # The drill-down that owns the session stripe inherits its fill; the
    # one whose cost was lumped away falls back to the fold colour.
    # (Which of the two labels owns the stripe depends on the tie order
    # of the simultaneous Task calls, so match by label dynamically.)
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _OTHER_AGENTS_COLOR,
    )

    (stripe_label,) = stripe_colors
    (other_label,) = set(by_label) - {stripe_label}
    assert by_label[stripe_label]["fill"] == stripe_colors[stripe_label]
    assert by_label[other_label]["fill"] == _OTHER_AGENTS_COLOR[0]
    assert by_label[stripe_label]["models"] == [model]


def test_tokenscape_stripe_colors_distinguish_runs_and_rest():
    """Agent runs cycle distinct fills; 'everything else' is the hatched
    'rest' colour, not the system-prompt overhead colour."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _CATEGORY_FILL,
        _AgentRun,
        _build_stripes,
        _render_stripes_figure,
        _Turn,
    )

    turns = [
        _Turn(
            idx=i,
            ts=None,
            model="claude-sonnet-4-6",
            inp=4,
            cache_read=0,
            cache_creation=0,
            cc_5m=0,
            cc_1h=0,
            out=0,
            ctx=100,
            event=None,
        )
        for i in range(4)
    ]
    runs = [
        _AgentRun(label=f"agent: run {i}", cost=1.0, first_turn=0, last_turn=3)
        for i in range(3)
    ]
    # agent_costs exceeds the runs' sum by 0.5 → an "everything else" stripe.
    stripes = _build_stripes([], turns, [0.0] * 4, [3.5, 0.0, 0.0, 0.0], runs)

    agent_stripes = [s for s in stripes if s["category"] == "agent"]
    assert len(agent_stripes) == 3
    assert len({s["fill"] for s in agent_stripes}) == 3

    (rest,) = [s for s in stripes if s["label"] == "everything else"]
    assert rest["category"] == "rest"
    assert _CATEGORY_FILL["rest"] != _CATEGORY_FILL["overhead"]

    fig = json.loads(_render_stripes_figure(stripes, turns))
    by_name = {t["name"]: t for t in fig["data"]}
    assert by_name["everything else"]["marker"]["color"] == _CATEGORY_FILL["rest"]
    assert by_name["everything else"]["marker"]["pattern"]["shape"] == "/"
    assert (
        by_name["agent: run 0"]["marker"]["color"]
        != by_name["agent: run 1"]["marker"]["color"]
    )


def test_tokenscape_subagent_prompt_classified_as_prompt():
    """A sub-session's root prompt counts as a prompt, not residual."""
    from introspect.api.handlers.tokenscape import _classify_block  # noqa: PLC0415

    assert _classify_block(
        "subagent_prompt", None, None, "analyze the docs", set()
    ) == ("prompt", "prompt")


def test_tokenscape_subagent_selection_caps_charts():
    """At most 8 charts, cost-descending, sub-$0.10 runs folded away."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _select_subagent_runs,
    )

    costs = [5.0, 0.05, 3.0, 0.5, 0.2, 1.0, 0.01, 2.0, 0.8, 0.3, 0.15, 4.0]
    runs = [{"label": f"agent: r{i}", "cost": c} for i, c in enumerate(costs)]
    shown, hidden = _select_subagent_runs(runs)

    assert len(shown) == 8
    shown_costs = [r["cost"] for r in shown]
    assert shown_costs == sorted(shown_costs, reverse=True)
    assert shown_costs == [5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.5, 0.3]
    assert hidden["count"] == 4
    assert hidden["cost"] == pytest.approx(0.2 + 0.15 + 0.05 + 0.01)


def test_tokenscape_orphan_sidechain_yields_no_runs():
    """A lone sidechain prompt with no usage (parent on the main chain)
    produces no drill-down runs and the tab still renders."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _write_sample_jsonl(tmp)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, SID)
    assert ctx["subagent_runs"] == []
    assert ctx["subagent_runs_hidden"] == {"count": 0, "cost": 0.0}

    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=tokenscape")
        assert response.status_code == 200


def test_session_tokenscape_tab_renders_subagent_section():
    """Tokenscape tab embeds one drill-down chart per shown subagent run."""
    sub_sid = "deadbeef-0000-0000-0000-tokenscape09"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_subagent_session_jsonl(tmp, sub_sid)
        with _patched_client(tmp) as client:
            response = client.get(f"/sessions/{sub_sid}?tab=tokenscape")
            assert response.status_code == 200
            assert 'id="subagent-tokenscape-1-data"' in response.text
            assert "agent: scan docs" in response.text
            # Heading carries the run's stripe-matching colour swatch
            # and the model that ran the agent.
            from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
                _agent_run_color,
            )

            fill, border = _agent_run_color(0)
            assert f"background:{fill};border:1px solid {border}" in response.text
            assert "claude-sonnet-4-6" in response.text
