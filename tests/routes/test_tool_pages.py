"""Tests for tools, MCPs, and bash pages."""

import tempfile
from pathlib import Path

import pytest

from .conftest import SID, _patched_client


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
