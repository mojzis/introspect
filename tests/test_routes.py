"""Tests for introspect web UI routes."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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
        make_user_message(
            SID,
            "u4",
            "u3",
            "2026-03-27T10:00:06.000Z",
            "<command-name>/commit</command-name>\nCommit my changes",
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
    """MCP streamable-HTTP endpoint is reachable."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # The streamable-http app mounts its route at /mcp internally,
        # so full path is /mcp/mcp. We expect a non-404 response
        # (405 Method Not Allowed or 421 from MCP transport security).
        response = client.get("/mcp/mcp")
        assert response.status_code != 404


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
        # Should show only user records (4 user messages in sample data)
        assert "4 records" in response.text


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
    """Session detail page displays tool results from user messages."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "tool_result" in response.text


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


def test_search_pagination_next():
    """Search page shows page number."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help&page=1")
        assert response.status_code == 200
        assert "Page 1" in response.text


def test_search_pagination_param():
    """Search page accepts page parameter."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help&page=2")
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
