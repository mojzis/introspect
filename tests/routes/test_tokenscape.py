"""Tests for the tokenscape tab (basic rendering, unit tests, cost tie-out)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ..conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)
from .conftest import SID, _patched_client
from .tokenscape_helpers import _tokenscape_session_jsonl


def test_session_tokenscape_tab_renders():
    """Tokenscape tab loads and exposes the new tab in the strip."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=tokenscape")
        assert response.status_code == 200
        text = response.text
        assert "Tokenscape" in text
        assert "Where the money went" in text or "nothing to plot" in text


def test_session_tokenscape_tab_streamgraph_shape():
    """Sample fixture has tool_use + tool_result, so the tab embeds a figure."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=tokenscape")
        assert response.status_code == 200
        text = response.text
        assert 'id="session-tokenscape-data"' in text
        # Stacked-bar layout — assert on barmode so a regression to
        # area/streamgraph stands out.
        assert '"barmode": "stack"' in text or '"barmode":"stack"' in text


def test_tokenscape_classify_block():
    """Content blocks map to the right cost category."""
    from introspect.api.handlers.tokenscape import _classify_block  # noqa: PLC0415

    edited = {"/tmp/edited.py"}
    assert _classify_block("human_prompt", None, None, "fix the bug", edited) == (
        "prompt",
        "prompt",
    )
    assert _classify_block(
        "human_prompt",
        None,
        None,
        "Base directory for this skill: /home/u/.claude/skills/cf\n\n# cf",
        edited,
    ) == ("skill", "skill cf")
    assert _classify_block(
        "slash_command", None, None, "<command-message>cf</command-message>", edited
    ) == ("skill", "slash command")
    assert _classify_block(
        "human_prompt", None, None, "<system-reminder>x</system-reminder>", edited
    ) == ("overhead", "system reminders")
    assert _classify_block(
        "human_prompt", None, None, "<task-notification>x", edited
    ) == ("overhead", "system reminders")
    assert _classify_block(
        "tool_result", "Read", '{"file_path": "/tmp/edited.py"}', "", edited
    ) == ("read_edited", "Read edited.py")
    assert _classify_block(
        "tool_result", "Read", '{"file_path": "/tmp/other.md"}', "", edited
    ) == ("read", "Read other.md")
    assert _classify_block("tool_result", "Task", "{}", "", edited) == (
        "agent",
        "subagent result",
    )
    assert _classify_block("tool_result", "Bash", '{"command": "ls"}', "", edited) == (
        "tool",
        "Bash · ls",
    )
    # Assistant-side blocks are attributed via output_tokens, not chars.
    assert _classify_block("agent_text", None, None, "hi", edited) is None
    assert _classify_block("agent_tool_call", "Bash", None, "", edited) is None


def test_tokenscape_label_known_tools():
    """File-path labels surface the basename for Read, head for Bash, etc."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _tool_result_label,
    )

    assert (
        _tool_result_label("Read", '{"file_path": "/tmp/foo/bar.py"}') == "Read bar.py"
    )
    assert _tool_result_label("Bash", '{"command": "git log --oneline"}').startswith(
        "Bash · git log"
    )
    assert _tool_result_label("WebFetch", "{}") == "WebFetch"
    assert _tool_result_label("mcp__filesystem__read", "{}").startswith("mcp · ")
    assert _tool_result_label("", "") == "tool result"


def test_tokenscape_cost_ties_out_to_bill():
    """Sum of all allocated band costs equals the priced API bill."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape01"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_session_jsonl(tmp, sid)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    assert ctx["has_data"]
    assert ctx["turn_count"] == 3
    expected = sum(
        compute_cost_usd(
            model="claude-sonnet-4-6",
            input_tokens=4,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_creation_5m=cc,
        )
        for out, cr, cc in [(100, 0, 10_000), (50, 10_000, 2_100), (10, 12_100, 50)]
    )
    assert ctx["total_cost"] == pytest.approx(expected, rel=1e-6)
    # The 8k-char tool_result dominates turn 2's 2.1k-token delta and
    # should surface as a named band.
    assert any(b["label"] == "Read big.py" for b in ctx["top_bands"])
    # Category table covers the whole bill.
    assert sum(c["cost"] for c in ctx["category_totals"]) == pytest.approx(
        ctx["total_cost"], rel=1e-6
    )


def test_tokenscape_walk_attributes_context_deltas():
    """Assistant share = prev output_tokens; remainder goes to user blocks."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _fetch_chain_rows,
        _fetch_edited_files,
        _walk_rows,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape02"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_session_jsonl(tmp, sid)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        bands, turns = _walk_rows(
            _fetch_chain_rows(db, sid, sidechain=False), _fetch_edited_files(db, sid)
        )

    assert len(turns) == 3
    # Turn 2 delta = (4 + 10000 + 2100) - (4 + 0 + 10000) = 2100 tokens:
    # 100 to the previous assistant output, 2000 to the tool_result.
    by_label = {b.label: b for b in bands}
    assert by_label["Read big.py"].arrival == 1
    assert by_label["Read big.py"].tokens == pytest.approx(2000, abs=1)
    assistant_bands = [b for b in bands if b.category == "assistant"]
    assert assistant_bands[0].tokens == pytest.approx(100, abs=1)
    # First turn: prompt estimated by chars, residual = system prompt.
    overhead = [b for b in bands if b.category == "overhead"]
    assert overhead and overhead[0].arrival == 0
    assert overhead[0].tokens > 9000


def test_tokenscape_compact_closes_bands():
    """A huge context drop closes all bands and opens a summary baseline."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        _fetch_chain_rows,
        _fetch_edited_files,
        _walk_rows,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape03"

    def usage(cr: int, cc: int, out: int) -> dict:
        return {
            "input_tokens": 4,
            "output_tokens": out,
            "cache_read_input_tokens": cr,
            "cache_creation_input_tokens": cc,
        }

    lines = [
        make_user_message(
            sid,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "big prompt " * 100,
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            sid,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "working"}],
            model="claude-sonnet-4-6",
            msg_id="msg-c-1",
            usage=usage(0, 50_000, 200),
        ),
        make_assistant_message(
            sid,
            "a2",
            "u1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "more"}],
            model="claude-sonnet-4-6",
            msg_id="msg-c-2",
            usage=usage(50_000, 200, 100),
        ),
        # /compact: context collapses from ~50k to ~6k.
        make_user_message(sid, "u2", "a2", "2026-04-21T10:00:03.000Z", "continue"),
        make_assistant_message(
            sid,
            "a3",
            "u2",
            "2026-04-21T10:00:04.000Z",
            [{"type": "text", "text": "post-compact"}],
            model="claude-sonnet-4-6",
            msg_id="msg-c-3",
            usage=usage(0, 6_000, 50),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_jsonl(tmp, sid, lines)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        bands, turns = _walk_rows(
            _fetch_chain_rows(db, sid, sidechain=False), _fetch_edited_files(db, sid)
        )

    assert [t.event for t in turns] == [None, None, "compact"]
    pre_compact = [b for b in bands if b.arrival < 2]
    assert pre_compact and all(b.end == 1 for b in pre_compact)
    summary = [b for b in bands if b.arrival == 2 and b.label == "compact summary"]
    assert summary and summary[0].tokens > 4_000


def test_tokenscape_sidechain_costs_bucket_to_active_turn():
    """Subagent API calls land on the main turn running when they happened."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape04"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # Main-chain session plus one sidechain call between turns 2 and 3.
        _tokenscape_session_jsonl(tmp, sid)
        sidechain = make_assistant_message(
            sid,
            "sc1",
            "u2",
            "2026-04-21T10:00:03.500Z",
            [{"type": "text", "text": "subagent work"}],
            model="claude-sonnet-4-6",
            msg_id="msg-ts-sc1",
            usage={
                "input_tokens": 1_000,
                "output_tokens": 500,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            is_sidechain=True,
        )
        jsonl_path = tmp / "projects" / "test-project" / f"{sid}.jsonl"
        with jsonl_path.open("a") as f:
            f.write(json.dumps(sidechain) + "\n")
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    expected_sc = compute_cost_usd(
        model="claude-sonnet-4-6", input_tokens=1_000, output_tokens=500
    )
    subagents = next(
        (c for c in ctx["category_totals"] if c["label"] == "subagents"), None
    )
    assert subagents is not None
    assert subagents["cost"] == pytest.approx(expected_sc, rel=1e-6)
    # The run gets its own stripe in the stripes figure (fallback label —
    # the fixture has no Task/Agent tool call to name it after).
    fig = json.loads(ctx["figure_json_stripes"])
    assert any(t["name"] == "agent: run" for t in fig["data"])


def test_tokenscape_stripes_and_bill_split():
    """Variant B figure renders and the bill split reconciles to the total."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape05"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_session_jsonl(tmp, sid)
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    fig = json.loads(ctx["figure_json_stripes"])
    assert fig["data"], "stripes figure should have at least one trace"
    # Stripe areas (sum of all trace ys) reconcile to the bill, up to
    # the sub-half-cent residual the figure intentionally drops.
    stripes_total = sum(sum(t["y"]) for t in fig["data"])
    assert stripes_total == pytest.approx(ctx["total_cost"], abs=0.0051)
    assert sum(ctx["bill_split"].values()) == pytest.approx(ctx["total_cost"], rel=1e-6)


def test_tokenscape_unknown_model_bills_zero_without_crashing():
    """Unpriced (unknown-model) sessions produce a $0 bill and a 0% cache-read
    share instead of dividing by a zero total."""
    from introspect.api.handlers.tokenscape import (  # noqa: PLC0415
        build_tokenscape_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415

    sid = "deadbeef-0000-0000-0000-tokenscape06"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _tokenscape_session_jsonl(tmp, sid, model="experimental-model-x")
        db = get_connection(tmp / "t.duckdb", glob_pattern(tmp))
        ctx = build_tokenscape_context(db, sid)

    assert ctx["has_data"]
    assert ctx["turn_count"] == 3
    assert ctx["total_cost"] == 0
    assert ctx["cache_read_pct"] == 0


def test_session_tokenscape_tab_embeds_stripes_chart():
    """Tokenscape tab renders both the stripes and per-turn figures."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=tokenscape")
        assert response.status_code == 200
        assert 'id="session-tokenscape-stripes-data"' in response.text
        assert "Bill by token type" in response.text


def test_session_tokenscape_tab_shows_error_notice_on_failure():
    """If tokenscape building blows up, the tab degrades to an inline notice
    (with the exception text) instead of a 500."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(Path(tmp)) as client,
        patch(
            "introspect.api.handlers.sessions.build_tokenscape_context",
            side_effect=RuntimeError("boom"),
        ),
    ):
        response = client.get(f"/sessions/{SID}?tab=tokenscape")
        assert response.status_code == 200
        assert "Tokenscape unavailable." in response.text
        assert "RuntimeError: boom" in response.text
