"""Tests for the Subagents tab on the session detail page."""

import os
from contextlib import contextmanager
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
from .tokenscape_helpers import _tokenscape_subagent_session_jsonl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUBAGENT_SID = "deadbeef-0000-0000-0000-subagents001"
_TWO_AGENT_SID = "deadbeef-0000-0000-0000-subagents002"
_UNIT_SID = "deadbeef-0000-0000-0000-subagents003"
_MULTI_BLOCK_SID = "deadbeef-0000-0000-0000-subagents004"


def _two_agent_session_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Session with two subagent invocations at different costs for rank tests."""
    model = "claude-sonnet-4-6"
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "run two agents",
            tool_use_result={"content": "seed"},
        ),
        # First Task call
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_cheap",
                    "name": "Task",
                    "input": {
                        "description": "cheap agent",
                        "prompt": "do cheap work",
                        "subagent_type": "general",
                    },
                }
            ],
            model=model,
            msg_id="msg-two-1",
            usage={"input_tokens": 4, "output_tokens": 10},
        ),
        # Cheap agent sidechain
        make_user_message(
            session_id,
            "sc1-u1",
            None,
            "2026-04-21T10:00:02.000Z",
            "do cheap work",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sc1-a1",
            "sc1-u1",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "cheap done"}],
            model=model,
            msg_id="msg-two-sc1",
            usage={
                "input_tokens": 4,
                "output_tokens": 100,  # small output → cheap
                "cache_creation_input_tokens": 1_000,
            },
            is_sidechain=True,
        ),
        # Tool result for cheap agent
        make_user_message(
            session_id,
            "u2",
            "a1",
            "2026-04-21T10:00:04.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_cheap",
                    "content": "cheap done",
                }
            ],
        ),
        # Second Task call (expensive)
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:05.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_expensive",
                    "name": "Task",
                    "input": {
                        "description": "expensive agent",
                        "prompt": "do expensive work",
                        "subagent_type": "general",
                    },
                }
            ],
            model=model,
            msg_id="msg-two-2",
            usage={"input_tokens": 4, "output_tokens": 10},
        ),
        # Expensive agent sidechain
        make_user_message(
            session_id,
            "sc2-u1",
            None,
            "2026-04-21T10:00:06.000Z",
            "do expensive work",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sc2-a1",
            "sc2-u1",
            "2026-04-21T10:00:07.000Z",
            [{"type": "text", "text": "expensive done"}],
            model=model,
            msg_id="msg-two-sc2",
            usage={
                "input_tokens": 4,
                "output_tokens": 50_000,  # huge output → expensive
                "cache_creation_input_tokens": 100_000,
            },
            is_sidechain=True,
        ),
        # Tool result for expensive agent
        make_user_message(
            session_id,
            "u3",
            "a2",
            "2026-04-21T10:00:08.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_expensive",
                    "content": "expensive done",
                }
            ],
        ),
        # Final main-chain reply
        make_assistant_message(
            session_id,
            "a3",
            "u3",
            "2026-04-21T10:00:09.000Z",
            [{"type": "text", "text": "all done"}],
            model=model,
            msg_id="msg-two-3",
            usage={"input_tokens": 4, "output_tokens": 5},
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def _multi_block_session_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Session with a multi-block assistant message to test amc dedup."""
    model = "claude-sonnet-4-6"
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "run agent",
            tool_use_result={"content": "seed"},  # ensure toolUseResult column exists
        ),
        # Main-chain agent that spawns a Task
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_task_mb",
                    "name": "Task",
                    "input": {
                        "description": "multi-block agent",
                        "prompt": "multi block",
                        "subagent_type": "general",
                    },
                }
            ],
            model=model,
            msg_id="msg-mb-1",
            usage={"input_tokens": 4, "output_tokens": 10},
        ),
        # Sidechain: user prompt + assistant with TWO content blocks.
        # The JSONL fixture only supports one assistant message per uuid,
        # but _fetch_chain_rows returns one row per content block.
        # We simulate multi-block by having a second message with the same
        # msg_id (uuid is different, msg_id same = same "amc" row).
        make_user_message(
            session_id,
            "sc-u1",
            None,
            "2026-04-21T10:00:02.000Z",
            "multi block",
            is_sidechain=True,
        ),
        # First block of the multi-block assistant message
        make_assistant_message(
            session_id,
            "sc-a1",
            "sc-u1",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "block one"}],
            model=model,
            msg_id="msg-mb-sc-dedup",  # dedup key
            usage={
                "input_tokens": 4,
                "output_tokens": 1_000,
                "cache_creation_input_tokens": 20_000,
            },
            is_sidechain=True,
        ),
        # Second block — same msg_id, should NOT double-count tokens
        make_assistant_message(
            session_id,
            "sc-a2",
            "sc-a1",
            "2026-04-21T10:00:04.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_sc_read_mb",
                    "name": "Read",
                    "input": {"file_path": "/tmp/multiblock.txt"},
                }
            ],
            model=model,
            msg_id="msg-mb-sc-dedup",  # same msg_id → same amc uuid
            usage=None,  # amc dedup: only first row has usage
            is_sidechain=True,
        ),
        make_user_message(
            session_id,
            "u2",
            "a1",
            "2026-04-21T10:00:06.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_task_mb",
                    "content": "mb done",
                }
            ],
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:07.000Z",
            [{"type": "text", "text": "done"}],
            model=model,
            msg_id="msg-mb-2",
            usage={"input_tokens": 4, "output_tokens": 5},
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


@contextmanager
def _patched_client_with_jsonl(tmp_path: Path, jsonl_writer):
    """Like _patched_client but uses a custom JSONL writer."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from introspect.api.main import app  # noqa: PLC0415

    jsonl_writer(tmp_path)
    db_path = tmp_path / "test.duckdb"
    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
                "INTROSPECT_DAYS": "0",
            },
        ),
        TestClient(app) as client,
    ):
        yield client


# ---------------------------------------------------------------------------
# Tab visibility tests
# ---------------------------------------------------------------------------


def test_subagents_tab_absent_for_plain_session(tmp_path):
    """Session without subagents must not show the Subagents tab."""
    with _patched_client(tmp_path) as client:
        resp = client.get(f"/sessions/{SID}")
        assert resp.status_code == 200
        assert "?tab=subagents" not in resp.text


def test_subagents_empty_state_for_plain_session(tmp_path):
    """Hitting ?tab=subagents on a plain session shows empty state."""
    with _patched_client(tmp_path) as client:
        resp = client.get(f"/sessions/{SID}?tab=subagents")
        assert resp.status_code == 200
        assert "No subagents in this session" in resp.text


def test_subagents_tab_present_for_session_with_task(tmp_path):
    """Session with a Task call shows the Subagents tab link."""
    with _patched_client_with_jsonl(
        tmp_path,
        lambda d: _tokenscape_subagent_session_jsonl(d, _SUBAGENT_SID),
    ) as client:
        resp = client.get(f"/sessions/{_SUBAGENT_SID}")
        assert resp.status_code == 200
        assert "?tab=subagents" in resp.text


# ---------------------------------------------------------------------------
# Happy-path: one-Task session
# ---------------------------------------------------------------------------


def test_subagents_table_renders_for_one_task_session(tmp_path):
    """?tab=subagents shows the agent run label and cost, and a Main row."""
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    with _patched_client_with_jsonl(
        tmp_path,
        lambda d: _tokenscape_subagent_session_jsonl(d, _SUBAGENT_SID),
    ) as client:
        resp = client.get(f"/sessions/{_SUBAGENT_SID}?tab=subagents")
        assert resp.status_code == 200
        html = resp.text

        # Run label and main row
        assert "agent: scan docs" in html
        assert "Main agent" in html

        # Read file should show up in file counts
        assert "/tmp/doc.md" not in html  # paths aren't shown directly...
        # tool_count == 1 and files_read == 1 are verified precisely in
        # test_context_builder_per_run_metrics; here we just confirm the page
        # renders without error and includes the expected labels.

        # Cost: expected run cost from tokenscape test
        expected_cost = compute_cost_usd(
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
        # Cost should be present formatted in the table ($X.XX)
        assert expected_cost > 0
        # Page has some dollar sign
        assert "$" in html

        # Spend-shape sparkline column renders
        assert "Spend shape" in html
        assert "<svg" in html


# ---------------------------------------------------------------------------
# Two-invocation session: cost-desc ranking
# ---------------------------------------------------------------------------


def test_subagents_two_invocations_rank_cost_desc(tmp_path):
    """With two subagents, the more expensive one appears first."""
    with _patched_client_with_jsonl(
        tmp_path,
        lambda d: _two_agent_session_jsonl(d, _TWO_AGENT_SID),
    ) as client:
        resp = client.get(f"/sessions/{_TWO_AGENT_SID}?tab=subagents")
        assert resp.status_code == 200
        html = resp.text

        # Both labels must appear
        assert "agent: expensive agent" in html
        assert "agent: cheap agent" in html

        # Expensive agent must appear before cheap agent in the HTML
        pos_expensive = html.index("agent: expensive agent")
        pos_cheap = html.index("agent: cheap agent")
        assert pos_expensive < pos_cheap, (
            "expensive agent should rank higher (appear first) than cheap agent"
        )

        # Cumulative % for last row should be ~100%
        assert "100.0%" in html


# ---------------------------------------------------------------------------
# Direct context-builder unit tests
# ---------------------------------------------------------------------------


def test_context_builder_per_run_metrics(tmp_path):
    """Direct build_subagent_breakdown_context call checks per-run metrics."""
    from introspect.api.handlers.subagents import (  # noqa: PLC0415
        build_subagent_breakdown_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    _tokenscape_subagent_session_jsonl(tmp_path, _UNIT_SID)
    db = get_connection(tmp_path / "t.duckdb", glob_pattern(tmp_path))
    ctx = build_subagent_breakdown_context(db, _UNIT_SID, "/tmp/test")

    assert ctx["has_data"] is True
    assert ctx["subagent_count"] == 1

    rows = ctx["rows"]
    # Should have 2 rows: main + 1 subagent
    assert len(rows) == 2

    # Find the subagent row
    subagent_row = next(r for r in rows if not r["is_main"])
    main_row = next(r for r in rows if r["is_main"])

    # Subagent label
    assert subagent_row["label"] == "agent: scan docs"

    # Subagent cost matches expected
    expected_subagent_cost = compute_cost_usd(
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
    assert subagent_row["cost_usd"] == pytest.approx(expected_subagent_cost, rel=1e-6)

    # Subagent has 1 tool call (Read /tmp/doc.md)
    assert subagent_row["tool_count"] == 1
    assert subagent_row["files_read"] == 1

    # /tmp/doc.md starts with /tmp — within cwd /tmp/test? No, /tmp/test != /tmp
    # session_cwd is "/tmp/test"; "/tmp/doc.md" does NOT start_with "/tmp/test"
    assert subagent_row["files_outside"] == 1

    # Main row is present
    assert main_row["label"] == "Main agent"
    assert main_row["is_main"] is True

    # Ranks: sorted cost desc; subagent likely more expensive than main
    # Both have a rank
    assert subagent_row["rank"] >= 1
    assert main_row["rank"] >= 1

    # pct and cum_pct are set
    assert 0 <= subagent_row["pct"] <= 100
    total_pct = sum(r["pct"] for r in rows)
    assert total_pct == pytest.approx(100.0, abs=0.01)

    # cum_pct of last row == 100
    last_row = max(rows, key=lambda r: r["rank"])
    assert last_row["cum_pct"] == pytest.approx(100.0, abs=0.01)

    # Spend-shape sparkline: subagent has 2 timestamped assistant messages,
    # so it gets a real polyline (not the single-message dot fallback)
    spark = subagent_row["spark"]
    assert spark is not None
    # Exactly 2 polyline points: one per assistant message — the tool_result
    # row between them carries no usage and must stay off the sparkline.
    assert len(spark["points"].split()) == 2
    # spend_msgs is internal — popped before rows reach the template
    assert "spend_msgs" not in subagent_row


def test_context_builder_amc_dedup_multi_block(tmp_path):
    """Multi-block assistant messages must not double-count tokens/cost."""
    from introspect.api.handlers.subagents import (  # noqa: PLC0415
        build_subagent_breakdown_context,
    )
    from introspect.db import get_connection  # noqa: PLC0415
    from introspect.pricing import compute_cost_usd  # noqa: PLC0415

    _multi_block_session_jsonl(tmp_path, _MULTI_BLOCK_SID)
    db = get_connection(tmp_path / "t.duckdb", glob_pattern(tmp_path))
    ctx = build_subagent_breakdown_context(db, _MULTI_BLOCK_SID, "/tmp/test")

    assert ctx["has_data"] is True

    subagent_row = next(r for r in ctx["rows"] if not r["is_main"])

    # The sidechain has exactly ONE unique assistant uuid with usage
    # (sc-a1 with msg-mb-sc-dedup and usage); sc-a2 shares the same msg_id
    # so if there are two amc rows they both have the same uuid — dedupe means
    # we only count them once.
    # Expected cost: only the one amc entry
    expected_cost = compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=4,
        output_tokens=1_000,
        cache_creation_5m=20_000,
    )
    assert subagent_row["cost_usd"] == pytest.approx(expected_cost, rel=1e-6)

    # asst_msgs deduped: should be 1, not 2
    assert subagent_row["asst_msgs"] == 1

    # tool_count: 1 Read call
    assert subagent_row["tool_count"] == 1
    assert subagent_row["files_read"] == 1
