"""Tests for dashboard and stats pages."""

import tempfile
from pathlib import Path

from .conftest import _patched_client


def test_dashboard_returns_200():
    """Dashboard page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Dashboard" in response.text


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


def test_dashboard_shows_user_message_stats():
    """Dashboard includes user message data in session list."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        # Dashboard shows recent sessions which include user_messages count
        from .conftest import SID  # noqa: PLC0415

        assert SID[:8] in response.text


def test_stats_includes_user_message_totals():
    """Stats page shows user message totals."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200
        assert (
            "User Messages" in response.text or "user_messages" in response.text.lower()
        )


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
