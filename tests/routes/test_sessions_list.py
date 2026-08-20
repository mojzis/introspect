"""Tests for the sessions list page."""

import tempfile
from pathlib import Path

import pytest

from ..conftest import (
    codex_glob_pattern,
    codex_record,
    codex_session_meta,
    codex_turn_context,
    write_codex_rollout,
)
from .conftest import SID, _patched_client

CODEX_SID = "codex-sess-provider-001"


def _write_codex_session(tmp_dir: Path, session_id: str = CODEX_SID) -> Path:
    """Write a minimal single-turn Codex rollout fixture."""
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {"type": "user_message", "message": "please fix", "text_elements": []},
        ),
    ]
    return write_codex_rollout(tmp_dir, session_id, lines)


def test_sessions_returns_200():
    """Sessions page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200


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
        "provider",
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


def test_sessions_tool_count_links_to_tools():
    """Sessions page tool count links to tools page filtered by session."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert f"/tools?session={SID}" in response.text


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


# --- Cost feature tests (sessions list) ---


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


def test_cost_overview_nav_link_present():
    """Nav link to /cost-overview is rendered on the sessions page."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert 'href="/cost-overview"' in response.text
        assert "Cost Overview" in response.text


# --- Provider filter tests ---


def test_sessions_unfiltered_shows_both_providers():
    """Sessions page interleaves Claude and Codex sessions by default."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_codex_session(tmp_path)
        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_CODEX_GLOB": codex_glob_pattern(tmp_path)},
        ) as client:
            response = client.get("/sessions")
            assert response.status_code == 200
            assert SID[:8] in response.text
            assert CODEX_SID in response.text


def test_sessions_filter_by_provider_narrows_to_codex():
    """``?provider=openai`` narrows the list to Codex sessions only."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_codex_session(tmp_path)
        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_CODEX_GLOB": codex_glob_pattern(tmp_path)},
        ) as client:
            response = client.get("/sessions?provider=openai")
            assert response.status_code == 200
            assert CODEX_SID in response.text
            assert SID[:8] not in response.text


def test_sessions_provider_dropdown_populated():
    """Sessions page populates the provider filter dropdown."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_codex_session(tmp_path)
        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_CODEX_GLOB": codex_glob_pattern(tmp_path)},
        ) as client:
            response = client.get("/sessions")
            assert response.status_code == 200
            assert "All providers" in response.text
            assert "openai" in response.text
            assert "anthropic" in response.text


def test_sessions_provider_filter_survives_sort_and_page_change():
    """The ``provider`` filter is preserved in sort/pagination links (qs() macro)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_codex_session(tmp_path)
        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_CODEX_GLOB": codex_glob_pattern(tmp_path)},
        ) as client:
            response = client.get("/sessions?provider=openai&sort=started_at")
            assert response.status_code == 200
            assert "provider=openai" in response.text
