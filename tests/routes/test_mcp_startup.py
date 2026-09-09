"""Embedded MCP retains truthful unlimited-history startup state."""

from introspect.api.main import app
from introspect.mcp.tools import recent_sessions
from introspect.refresh import LoadingPhase

from .conftest import _patched_client


def test_unlimited_embedded_results_are_authoritative(tmp_path):
    with _patched_client(
        tmp_path,
        extra_env={"INTROSPECT_DAYS": "0", "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0"},
    ):
        result = recent_sessions()
        assert "Partial data" not in result
        assert app.state.loading_state.phase is LoadingPhase.READY
        assert app.state.database_label == "authoritative"
        assert app.state.loading_state.candidate_count == 0
