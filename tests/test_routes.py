"""Tests for introspect web UI routes."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from introspect.api.handlers._helpers import clean_title
from introspect.api.main import app

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "01234567-abcd-abcd-abcd-0123456789ab"


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing."""
    lines = [
        make_user_message(
            SID,
            "u1",
            None,
            "2026-03-27T10:00:00.000Z",
            "Hello, help me with tests",
        ),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-03-27T10:00:01.000Z",
            [{"type": "text", "text": "Sure, I can help!"}],
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
        make_assistant_message(
            SID,
            "a2",
            "a1",
            "2026-03-27T10:00:02.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_test1",
                    "name": "Bash",
                    "input": {"command": "echo hello", "description": "test"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u2",
            "a2",
            "2026-03-27T10:00:03.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test1",
                    "content": "hello\n",
                    "is_error": False,
                }
            ],
            tool_use_result={
                "stdout": "hello\n",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
            source_tool_uuid="a2",
        ),
        make_assistant_message(
            SID,
            "a3",
            "u2",
            "2026-03-27T10:00:04.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp1",
                    "name": "mcp__github__get_me",
                    "input": {},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3",
            "a3",
            "2026-03-27T10:00:05.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_mcp1",
                    "content": '{"login": "test"}',
                    "is_error": False,
                }
            ],
            tool_use_result={},
            source_tool_uuid="a3",
        ),
        # Read tool call (file inside project)
        make_assistant_message(
            SID,
            "a4",
            "u3",
            "2026-03-27T10:00:05.500Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/test/src/main.py"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3b",
            "a4",
            "2026-03-27T10:00:05.600Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read1",
                    "content": "file contents here",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "file contents here"},
            source_tool_uuid="a4",
        ),
        # Read tool call (file OUTSIDE project)
        make_assistant_message(
            SID,
            "a5",
            "u3b",
            "2026-03-27T10:00:05.700Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read2",
                    "name": "Read",
                    "input": {"file_path": "/home/user/other/config.yml"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3c",
            "a5",
            "2026-03-27T10:00:05.800Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read2",
                    "content": "config data",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "config data"},
            source_tool_uuid="a5",
        ),
        # Edit tool call
        make_assistant_message(
            SID,
            "a6",
            "u3c",
            "2026-03-27T10:00:05.900Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_edit1",
                    "name": "Edit",
                    "input": {
                        "file_path": "/tmp/test/src/main.py",
                        "old_string": "old",
                        "new_string": "new",
                    },
                }
            ],
        ),
        make_user_message(
            SID,
            "u3d",
            "a6",
            "2026-03-27T10:00:05.950Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_edit1",
                    "content": "ok",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "ok"},
            source_tool_uuid="a6",
        ),
        make_user_message(
            SID,
            "u4",
            "u3d",
            "2026-03-27T10:00:06.000Z",
            "<command-name>/commit</command-name>\nCommit my changes",
        ),
        # Sidechain user message — this is the prompt the main agent passed
        # to a subagent via the Task/Agent tool, NOT a human-typed prompt.
        make_user_message(
            SID,
            "s1",
            "a3",
            "2026-03-27T10:00:07.000Z",
            "Explore the database layer and report back findings.",
            is_sidechain=True,
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


@contextmanager
def _patched_client(tmp_path: Path):
    """Context manager that yields a TestClient with materialized test data."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"

    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_DAYS": "0",
            },
        ),
        TestClient(app) as client,
    ):
        yield client


def test_dashboard_returns_200():
    """Dashboard page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Dashboard" in response.text


def test_sessions_returns_200():
    """Sessions page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200


def test_session_detail_returns_200():
    """Session detail page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions/01234567-abcd-abcd-abcd-0123456789ab")
        assert response.status_code == 200


def test_search_returns_200():
    """Search page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search")
        assert response.status_code == 200


def test_tools_returns_200():
    """Tools page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/tools")
        assert response.status_code == 200


def test_mcps_returns_200():
    """MCPs page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps")
        assert response.status_code == 200
        assert "MCP Servers" in response.text


def test_mcps_filter_by_server():
    """MCPs page filters by server name."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps?server=github")
        assert response.status_code == 200
        assert "github" in response.text


def test_base_template_has_history_restore_spinner_fix():
    """Regression: htmx:historyRestore listener must clear the loading spinner."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "htmx:historyRestore" in response.text
        assert "loading-overlay" in response.text


def test_htmx_partial_response_excludes_spinner():
    """HTMX partial responses should not contain the loading overlay."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "loading-overlay" not in response.text


def test_stats_returns_200():
    """Stats page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200


def test_mcp_endpoint_mounted():
    """MCP streamable-HTTP endpoint is reachable at /mcp/, not /mcp/mcp."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # Trailing slash avoids the mount's 307 redirect; MCP transport
        # security rejects a bare GET but a non-5xx proves the sub-app
        # is wired up at /mcp/.
        response = client.get("/mcp/")
        assert 400 <= response.status_code < 500
        # Regression guard: the old double-prefixed path must not exist.
        assert client.get("/mcp/mcp").status_code == 404


# --- Raw page tests ---


def test_raw_returns_200():
    """Raw page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw")
        assert response.status_code == 200
        assert "Raw Data" in response.text


def test_raw_filter_by_type():
    """Raw page filters records by type and excludes other types."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw?type=user")
        assert response.status_code == 200
        assert "Raw Data" in response.text
        # Should show only user records (8 user messages in sample data:
        # initial prompt, 5 tool_results, slash command, subagent prompt)
        assert "8 records" in response.text


def test_raw_filter_by_session():
    """Raw page filters records by session ID prefix."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/raw?session={SID[:8]}")
        assert response.status_code == 200
        assert "Raw Data" in response.text


def test_raw_filter_by_type_and_session():
    """Raw page filters by both type and session simultaneously."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/raw?type=user&session={SID[:8]}")
        assert response.status_code == 200
        assert "Raw Data" in response.text


def test_raw_filter_no_results():
    """Raw page handles filter that matches nothing."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw?type=nonexistent")
        assert response.status_code == 200
        assert "No records found" in response.text


def test_raw_pagination():
    """Raw page supports pagination parameter."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw?page=1")
        assert response.status_code == 200


def test_raw_shows_record_count():
    """Raw page displays the total record count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw")
        assert response.status_code == 200
        assert "records" in response.text


def test_raw_type_dropdown_populated():
    """Raw page populates the type filter dropdown with available types."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw")
        assert response.status_code == 200
        assert "user" in response.text
        assert "assistant" in response.text


def test_raw_htmx_partial():
    """Raw page returns partial content for HTMX requests."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/raw", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "loading-overlay" not in response.text


# --- User-related action tests ---


def test_dashboard_shows_user_message_stats():
    """Dashboard includes user message data in session list."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        # Dashboard shows recent sessions which include user_messages count
        assert SID[:8] in response.text


def test_sessions_sort_by_user_msgs():
    """Sessions page can sort by user message count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=user_msgs&order=desc")
        assert response.status_code == 200
        assert SID[:8] in response.text


def test_sessions_sort_by_asst_msgs():
    """Sessions page can sort by assistant message count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=asst_msgs&order=asc")
        assert response.status_code == 200


def test_session_detail_shows_user_messages():
    """Session detail page displays user messages."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "Hello, help me with tests" in response.text


def test_session_detail_shows_tool_results():
    """Session detail page surfaces tool result content folded under the tool call."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # Both the tool input ("echo hello") and the tool_use_result stdout
        # ("hello") from the fixture must appear inside an agent_tool_call row.
        assert "kind-agent_tool_call" in text
        assert "echo hello" in text
        # tool-section-label markers only exist when the paired result is present
        assert "tool-section-label" in text


def test_session_detail_classifies_message_kinds():
    """Session detail page classifies messages into distinct visual kinds."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # Human prompt is visually distinct from tool results.
        assert "kind-human_prompt" in response.text
        assert "Hello, help me with tests" in response.text
        # Slash command wrapping becomes a divider, not a big box.
        assert "slash-divider" in response.text
        # Filter strip is present with localStorage-backed toggles.
        assert "Hide thinking" in response.text
        assert "Hide tool calls" in response.text
        assert "Only human turns" in response.text
        # Global expand/collapse buttons are present.
        assert "Expand all tools" in response.text
        assert "tools-expand-all" in response.text
        assert "tools-collapse-all" in response.text


def test_session_detail_tool_call_one_liner():
    """Tool calls render as one-liners with a tool-specific hint."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The Bash tool's 'command' input is lifted into the collapsed one-liner.
        assert "tool-call-line" in text
        assert "tool-hint" in text
        assert "echo hello" in text


def test_session_detail_distinguishes_subagent_prompt_from_human():
    """Sidechain user messages render as 'prompt to subagent', not 'you'."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The subagent prompt appears with its own kind class, NOT human_prompt.
        assert "kind-subagent_prompt" in text
        assert "Explore the database layer" in text
        assert "prompt to subagent" in text
        # And it is visually distinct from the real human prompt above it.
        human_idx = text.find("Hello, help me with tests")
        sub_idx = text.find("Explore the database layer")
        assert human_idx != -1
        assert sub_idx != -1
        # The real human prompt is NOT wrapped as a subagent_prompt.
        assert 'class="msg-row kind-subagent_prompt' in text
        assert (
            text[max(0, human_idx - 300) : human_idx].count("kind-subagent_prompt") == 0
        )


def test_session_detail_folds_tool_results_under_calls():
    """tool_result rows are filtered out of the rendered list (folded into calls)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # There should be no standalone kind-tool_result row — tool_results are
        # merged into the matching agent_tool_call row via the tool_calls join.
        assert "kind-tool_result" not in response.text


def test_search_finds_user_content():
    """Search returns results matching user message content."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        assert "help me with tests" in response.text.lower()


def test_search_finds_assistant_content():
    """Search returns results matching assistant message content."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Sure+I+can+help")
        assert response.status_code == 200
        assert "sure" in response.text.lower() or "can help" in response.text.lower()


def test_stats_includes_user_message_totals():
    """Stats page shows user message totals."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert (
            "User Messages" in response.text or "user_messages" in response.text.lower()
        )


def test_sessions_filter_by_model():
    """Sessions page filters by model."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?model=claude-opus-4-6")
        assert response.status_code == 200


def test_sessions_filter_by_branch():
    """Sessions page filters by git branch."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?branch=main")
        assert response.status_code == 200


def test_sessions_empty_page_size_returns_200():
    """Sessions page handles empty page_size param without 422."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?page_size=")
        assert response.status_code == 200


def test_sessions_empty_page_returns_200():
    """Sessions page handles empty page param without 422."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?page=")
        assert response.status_code == 200


def test_sessions_all_empty_params_returns_200():
    """Sessions page handles all empty query params without 422."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(
            "/sessions?page=1&page_size=&sort=asst_msgs&order=desc"
            "&model=&project=&branch="
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "col",
    [
        "started_at",
        "duration",
        "user_msgs",
        "asst_msgs",
        "tool_calls",
        "model",
        "project",
        "branch",
        "cost",
    ],
)
def test_sessions_sort_column(col):
    """Sessions page accepts sort by {{ col }}."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions?sort={col}&order=desc")
        assert response.status_code == 200


def test_sessions_invalid_sort_falls_back():
    """Sessions page falls back to default for invalid sort column."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=nonexistent&order=desc")
        assert response.status_code == 200


# --- Dashboard enrichment tests ---


def test_dashboard_shows_success_rate():
    """Dashboard displays tool success rate percentage."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Success Rate" in response.text


def test_dashboard_shows_project_count():
    """Dashboard displays active project count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Projects" in response.text


def test_dashboard_shows_avg_duration():
    """Dashboard displays average session duration."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Avg Duration" in response.text


def test_dashboard_shows_activity():
    """Dashboard displays today/this week activity counts."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "This week:" in response.text


def test_dashboard_shows_session_titles():
    """Dashboard recent sessions show titles instead of just UUIDs."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Hello, help me with tests" in response.text


# --- Sessions tool count column tests ---


def test_sessions_shows_tools_column():
    """Sessions page has a sortable Tools column."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "tool_calls" in response.text  # sort link param


def test_sessions_sort_by_tool_calls():
    """Sessions page can sort by tool call count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=tool_calls&order=desc")
        assert response.status_code == 200


# --- Session detail enrichment tests ---


def test_session_detail_shows_token_usage():
    """Session detail shows token usage stats."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "Input Tokens" in response.text


def test_session_detail_shows_tool_summary():
    """Session detail shows tool call summary line."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "tool call" in response.text


def test_session_detail_expandable_blocks():
    """Session detail renders tool use blocks with expand support."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # Tool use blocks are rendered (Bash tool call in test data)
        assert "Bash" in response.text


# --- Stats enrichment tests ---


def test_stats_shows_avg_duration():
    """Stats page shows average session duration."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert "Avg Duration" in response.text


def test_stats_shows_avg_tools_per_session():
    """Stats page shows average tool calls per session."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert "Avg Tools/Session" in response.text


def test_stats_shows_distribution_bars():
    """Stats page renders visual bar charts in distribution tables."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert "background:#3b5bdb" in response.text


def test_stats_shows_model_breakdown():
    """Stats page shows per-model breakdown table."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert "Per-Model Breakdown" in response.text
        assert "claude-opus-4-6" in response.text


# --- Search pagination tests ---


def test_search_shows_fts_status():
    """Search page shows FTS availability indicator."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search")
        assert response.status_code == 200
        # Shows either BM25 active or ILIKE fallback depending on FTS availability
        assert "BM25" in response.text or "ILIKE fallback" in response.text


def test_search_pagination_next():
    """Search page shows result count for a query."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Hello&page=1")
        assert response.status_code == 200
        # The results summary always appears when a query is provided
        assert 'result(s) for "Hello"' in response.text


def test_search_pagination_param():
    """Search page accepts page parameter."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Hello&page=2")
        assert response.status_code == 200


# --- Tools pagination and success rate tests ---


def test_tools_pagination():
    """Tools page supports pagination."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/tools?page=1")
        assert response.status_code == 200
        assert "Page 1" in response.text


def test_tools_shows_success_rate():
    """Tools page shows success rate percentage in filter buttons."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/tools")
        assert response.status_code == 200
        assert "100%" in response.text  # our test tool call succeeds


def test_tools_filter_with_pagination():
    """Tools page preserves filters across pagination."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/tools?name=Bash&page=1")
        assert response.status_code == 200


def test_tools_filter_by_session():
    """Tools page filters by session ID."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/tools?session={SID}")
        assert response.status_code == 200
        assert SID[:12] in response.text


def test_sessions_tool_count_links_to_tools():
    """Sessions page tool count links to tools page filtered by session."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert f"/tools?session={SID}" in response.text


# --- MCPs pagination tests ---


def test_mcps_pagination():
    """MCPs page supports pagination."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps?page=1")
        assert response.status_code == 200
        assert "Page 1" in response.text


def test_mcps_filter_with_pagination():
    """MCPs page preserves filters across pagination."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps?server=github&page=1")
        assert response.status_code == 200


# --- Command parsing and filtering tests ---


def test_sessions_shows_commands_column():
    """Sessions page has a Commands column with parsed command badges."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "Commands" in response.text
        assert "/commit" in response.text


def test_sessions_filter_by_command():
    """Sessions page filters by command."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?command=/commit")
        assert response.status_code == 200
        assert SID[:8] in response.text


def test_sessions_filter_by_command_no_results():
    """Sessions page returns no sessions for unknown command."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?command=/nonexistent")
        assert response.status_code == 200
        assert SID[:8] not in response.text


def test_sessions_command_dropdown_populated():
    """Sessions page populates the command filter dropdown."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "All commands" in response.text
        assert "/commit" in response.text


# --- Sessions search tests ---


def test_sessions_search_filters_by_content():
    """Sessions page filters to sessions containing search query."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?q=help+me+with+tests")
        assert response.status_code == 200
        assert SID[:8] in response.text


def test_sessions_search_no_match():
    """Sessions search returns no sessions for unmatched query."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?q=xyznonexistent999")
        assert response.status_code == 200
        assert SID[:8] not in response.text


def test_sessions_search_box_present():
    """Sessions page has a search input box."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert 'placeholder="Search content..."' in response.text


def test_sessions_search_preserves_query():
    """Sessions page preserves the search query in the input."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?q=hello")
        assert response.status_code == 200
        assert 'value="hello"' in response.text


# --- Search results enrichment tests ---


def test_search_results_show_session_info():
    """Search results include session metadata columns."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        # Should show session-level columns
        assert "Project" in response.text
        assert "Branch" in response.text
        assert "Title" in response.text
        assert "Duration" in response.text
        assert "Model" in response.text


def test_search_results_link_to_session():
    """Search results link to the session detail page."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        assert f"/sessions/{SID}" in response.text


def test_clean_title_strips_all_xml_tags():
    """clean_title strips ALL XML tags, not just leading ones."""
    # Leading tag only
    assert clean_title("<foo>bar") == "bar"
    # Wrapping tags (the original bug: command-name pattern)
    assert clean_title("<command-name>/commit</command-name>") == "/commit"
    # Nested / multiple tags
    assert clean_title("<a><b>text</b></a>") == "text"
    # No tags at all
    assert clean_title("plain text") == "plain text"
    # Empty string
    assert clean_title("") == ""
    # Tags with attributes
    assert clean_title('<div class="x">content</div>') == "content"
    # Mixed content
    assert clean_title("before <tag>middle</tag> after") == "before middle after"


def test_clean_title_drops_command_message_that_mirrors_command_name():
    """<command-message> duplicates <command-name>, so it's dropped entirely."""
    # Skill invocation: command-name and command-message carry the same label.
    raw = (
        "<command-name>marimo-pair</command-name>\n"
        "<command-message>/marimo-pair</command-message>\n"
        "<command-args></command-args>"
    )
    assert clean_title(raw) == "marimo-pair"
    # With real args attached — tag boundaries become word separators.
    raw_with_args = (
        "<command-name>commit</command-name>"
        "<command-message>/commit</command-message>"
        "<command-args>fix typo</command-args>"
    )
    assert clean_title(raw_with_args) == "commit fix typo"


# --- Bash page tests ---


@pytest.fixture
def bash_client():
    """Provide a test client with sample data for bash route tests."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        yield client


def test_bash_returns_200(bash_client):
    """Bash page loads without error and shows the test command."""
    response = bash_client.get("/bash")
    assert response.status_code == 200
    assert "Bash Commands" in response.text
    assert "echo hello" in response.text


def test_bash_pagination(bash_client):
    """Bash page supports pagination."""
    response = bash_client.get("/bash?page=1")
    assert response.status_code == 200
    assert "Page 1" in response.text


def test_bash_filter_by_prefix(bash_client):
    """Bash page filters by command prefix and shows matching command."""
    response = bash_client.get("/bash?prefix=echo+hello")
    assert response.status_code == 200
    assert "echo hello" in response.text


def test_bash_filter_by_session(bash_client):
    """Bash page filters by session ID."""
    response = bash_client.get(f"/bash?session={SID}")
    assert response.status_code == 200
    assert SID[:12] in response.text
    assert "echo hello" in response.text


def test_bash_failed_filter(bash_client):
    """Bash page failed filter excludes successful commands."""
    response = bash_client.get("/bash?failed=true")
    assert response.status_code == 200
    # The test fixture's Bash call succeeded, so failed filter should show 0
    assert ">0<" in response.text.replace(" ", "")


def test_bash_shows_project_column(bash_client):
    """Bash page shows the Project column header and project value."""
    response = bash_client.get("/bash")
    assert response.status_code == 200
    assert ">Project<" in response.text


def test_bash_filter_by_project(bash_client):
    """Bash page filters by project name."""
    response = bash_client.get("/bash?project=test")
    assert response.status_code == 200
    assert "echo hello" in response.text


def test_bash_filter_by_project_no_match(bash_client):
    """Bash page project filter with non-matching project returns zero results."""
    response = bash_client.get("/bash?project=nonexistent")
    assert response.status_code == 200
    assert ">0<" in response.text.replace(" ", "")


def test_bash_search_by_command(bash_client):
    """Bash page search matches against command text."""
    response = bash_client.get("/bash?q=echo")
    assert response.status_code == 200
    assert "echo hello" in response.text


def test_bash_search_by_description(bash_client):
    """Bash page search matches against description text."""
    response = bash_client.get("/bash?q=test")
    assert response.status_code == 200
    # The test fixture Bash call has description="test"
    assert "echo hello" in response.text


def test_bash_search_no_match(bash_client):
    """Bash page search with non-matching query returns zero results."""
    response = bash_client.get("/bash?q=zzzznotfound")
    assert response.status_code == 200
    assert ">0<" in response.text.replace(" ", "")


def test_bash_project_dropdown_present(bash_client):
    """Bash page renders the project filter dropdown."""
    response = bash_client.get("/bash")
    assert response.status_code == 200
    assert "All projects" in response.text


def test_bash_search_chip_shown(bash_client):
    """Bash page shows a removable search chip when q is active."""
    response = bash_client.get("/bash?q=echo")
    assert response.status_code == 200
    assert "Search:" in response.text
    assert "&times;" in response.text


# --- Tools project / search tests ---


@pytest.fixture
def tools_client():
    """Provide a test client with sample data for tools route tests."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        yield client


def test_tools_shows_project_column(tools_client):
    """Tools page shows the Project column header."""
    response = tools_client.get("/tools")
    assert response.status_code == 200
    assert ">Project<" in response.text


def test_tools_filter_by_project(tools_client):
    """Tools page filters by project name."""
    response = tools_client.get("/tools?project=test")
    assert response.status_code == 200
    assert "Bash" in response.text


def test_tools_filter_by_project_no_match(tools_client):
    """Tools page project filter with non-matching project returns zero results."""
    response = tools_client.get("/tools?project=nonexistent")
    assert response.status_code == 200
    assert ">0<" in response.text.replace(" ", "")


def test_tools_search_by_description(tools_client):
    """Tools page search matches against description text."""
    # The Bash tool call has description="test"
    response = tools_client.get("/tools?q=test")
    assert response.status_code == 200
    assert "Bash" in response.text


def test_tools_search_by_input(tools_client):
    """Tools page search matches against tool input text."""
    response = tools_client.get("/tools?q=echo")
    assert response.status_code == 200
    assert "Bash" in response.text


def test_tools_search_no_match(tools_client):
    """Tools page search with non-matching query returns zero results."""
    response = tools_client.get("/tools?q=zzzznotfound")
    assert response.status_code == 200
    assert ">0<" in response.text.replace(" ", "")


def test_tools_project_dropdown_present(tools_client):
    """Tools page renders the project filter dropdown."""
    response = tools_client.get("/tools")
    assert response.status_code == 200
    assert "All projects" in response.text


def test_tools_search_chip_shown(tools_client):
    """Tools page shows a removable search chip when q is active."""
    response = tools_client.get("/tools?q=echo")
    assert response.status_code == 200
    assert "Search:" in response.text
    assert "&times;" in response.text


# --- File metrics tests ---


def test_sessions_shows_file_metrics_columns():
    """Sessions page has sortable Read, Edited, Read Only, Outside column headers."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        text = response.text
        assert "files_read" in text  # sort link param
        assert "files_edited" in text  # sort link param
        assert "files_read_only" in text  # sort link param
        assert "files_outside" in text  # sort link param


def test_sessions_sort_by_files_read():
    """Sessions page can sort by files_read count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=files_read&order=desc")
        assert response.status_code == 200
        assert SID[:8] in response.text


def test_sessions_sort_by_files_read_only():
    """Sessions page can sort by files_read_only count."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=files_read_only&order=desc")
        assert response.status_code == 200
        assert SID[:8] in response.text


def test_session_detail_shows_file_metrics():
    """Session detail page shows file metrics line."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The test data has 2 Read calls and 1 Edit call
        assert "Files" in text
        assert "read" in text
        assert "edited" in text


def test_session_detail_shows_read_only_count():
    """Session detail shows read-only file count (read but never written)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # /home/user/other/config.yml is read but never edited → 1 read-only
        assert "1 read-only" in text


def test_session_detail_file_metrics_outside_count():
    """Session detail correctly counts files outside the project directory."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # One file (/home/user/other/config.yml) is outside /tmp/test
        assert "outside project" in text


# --- Cost feature tests (sessions list + cost tab + dedup) ---


def test_sessions_shows_cost_column():
    """Sessions list has a Cost column rendering a $ value."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "Cost" in response.text
        assert "$" in response.text


def test_sessions_cost_links_to_cost_tab():
    """Cost cell wraps the value in a link to the session detail Cost tab."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert f'href="/sessions/{SID}?tab=cost"' in response.text


def test_sessions_sort_by_cost():
    """Sessions list accepts ?sort=cost without erroring."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions?sort=cost&order=desc")
        assert response.status_code == 200


def test_session_detail_has_tab_strip():
    """Session detail renders both tab links; messages is the default."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        assert f"/sessions/{SID}?tab=messages" in text
        assert f"/sessions/{SID}?tab=cost" in text
        # Default tab highlights "Messages" via the bold border style.
        assert "tab-strip" in text


def test_session_detail_cost_tab_renders():
    """Cost tab returns 200 and contains $, model, Read+Created, and SVG chart."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=cost")
        assert response.status_code == 200
        text = response.text
        assert "$" in text
        assert "claude-opus-4-6" in text
        assert "Read" in text
        assert "Created" in text
        assert "<svg" in text


def _dup_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL with two assistant records sharing one message.id."""
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    lines = [
        # Carry a tool_use_result on the seed user message so union_by_name
        # picks the column up when materialising views.
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
            msg_id="msg-dedup-1",
            usage=usage,
        ),
        # Duplicate: same message.id, different uuid, slightly later timestamp
        make_assistant_message(
            session_id,
            "a1-dup",
            "u1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id="msg-dedup-1",
            usage=usage,
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def test_session_cost_dedup():
    """Duplicated message.id rows must collapse to one cost-bearing row."""
    sid = "dedup-session-id-0000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _dup_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            TestClient(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            text = response.text
            # 1M input * $5/M + 1M output * $25/M = $30.00 (single message,
            # not $60 from the duplicated copy).
            assert "$30.00" in text


def _bloat_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL where a Read-tool result is followed by a big cache-write."""
    lines = [
        make_user_message(
            session_id, "u1", None, "2026-04-21T10:00:00.000Z", "please review"
        ),
        # First assistant message just initialises context (small usage)
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "tu-read",
                    "name": "Read",
                    "input": {"file_path": "/repo/src/big_file.py"},
                }
            ],
            model="claude-opus-4-7",
            msg_id="msg-bloat-1",
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
                    "tool_use_id": "tu-read",
                    "content": "x" * 1000,
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "x" * 1000},
            source_tool_uuid="a1",
        ),
        # Second assistant message: parent_uuid points at the user tool_result;
        # the cache-creation tokens are attributed to the preceding Read.
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id="msg-bloat-2",
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
    return write_jsonl(tmp_dir, session_id, lines)


def test_session_cost_bloat_attribution():
    """Bloat table should attribute cache creation to the preceding Read tool."""
    sid = "bloat-session-id-0000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _bloat_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            TestClient(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            text = response.text
            assert "file read" in text
            # basename of the read file should appear in the bloat bucket label
            assert "big_file.py" in text


def _subagent_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL where a sidechain assistant message has cache_creation."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        # Sidechain user prompt (simulating Task subagent dispatch)
        make_user_message(
            session_id,
            "su1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            "subagent: do this",
            is_sidechain=True,
        ),
        # Sidechain assistant response with significant cache_creation
        make_assistant_message(
            session_id,
            "sa1",
            "su1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id="msg-side-1",
            usage={
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_creation_input_tokens": 500_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 500_000,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
            is_sidechain=True,
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def test_session_cost_subagent_attribution():
    """Sidechain rows feed the Subagent column orthogonally to category.

    The fixture's sidechain assistant message has no preceding tool_use_id,
    so it classifies as Conversation/human input — but lands under the
    Subagent agent column rather than collapsing into a flat "Subagent"
    category. That's the orthogonality contract.
    """
    sid = "subagent-session-0000-0000-000000000001"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _subagent_jsonl(tmp, sid)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            TestClient(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            text = response.text
            # Subagent appears as a column header AND in the agent column.
            assert "Subagent" in text
            # The orthogonal categories are still present (no flat "Subagent" cat).
            assert "Read" in text
            assert "Created" in text
            assert "Conversation" in text


def test_fetch_token_usage_dedup():
    """Direct unit test: deduped totals should equal a single message's usage."""
    from introspect.api.handlers._helpers import fetch_token_usage  # noqa: PLC0415
    from introspect.db import get_connection, materialize_views  # noqa: PLC0415

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
