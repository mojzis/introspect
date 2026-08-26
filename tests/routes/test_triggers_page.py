"""Tests for the /triggers first-prompt-triggers page."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from introspect.api.main import app

from ..conftest import (
    glob_pattern,
    make_attachment_message,
    make_user_message,
    write_jsonl,
)
from .conftest import _patched_client


def test_triggers_returns_200():
    """Triggers page loads and renders its heading."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/triggers")
        assert response.status_code == 200
        assert "First-Prompt Triggers" in response.text


def test_triggers_shows_session_and_pagination():
    """The sample session's opening prompt appears, with pagination controls."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/triggers")
        assert response.status_code == 200
        # Sample opener "Hello, help me with tests" is the session title.
        assert "Hello, help me with tests" in response.text
        assert "Page 1" in response.text


def test_triggers_vague_opener_aggregate():
    """The sample opener names no path and triggers no skill, so the vague-opener
    share is 100%."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/triggers")
        assert response.status_code == 200
        assert "Vague openers" in response.text
        assert "100.0%" in response.text


def test_triggers_project_filter():
    """Filtering by project keeps the page rendering (loopback bind)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/triggers?project=test")
        assert response.status_code == 200
        assert "First-Prompt Triggers" in response.text


ASID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@contextmanager
def _client_with_attachments(tmp_path: Path):
    """Yield a TestClient over a session that auto-loaded a nested CLAUDE.md and
    an @-file — so the /triggers auto-load columns render their truthy branch."""
    write_jsonl(
        tmp_path,
        ASID,
        [
            make_user_message(
                ASID,
                "au1",
                None,
                "2026-03-27T10:00:00.000Z",
                "look at @src/db.py",
                tool_use_result={"stdout": "", "stderr": ""},
            ),
            make_attachment_message(
                ASID,
                "aatt1",
                "au1",
                "2026-03-27T10:00:01.000Z",
                {
                    "type": "nested_memory",
                    "path": "/repo/.claude/rules/x.md",
                    "displayPath": ".claude/rules/x.md",
                    "content": "rule",
                },
            ),
            make_attachment_message(
                ASID,
                "aatt2",
                "au1",
                "2026-03-27T10:00:02.000Z",
                {
                    "type": "file",
                    "filename": "/repo/src/db.py",
                    "displayPath": "src/db.py",
                    "content": "code",
                },
            ),
        ],
    )
    db_path = tmp_path / "test.duckdb"
    env = {
        "INTROSPECT_DB_PATH": str(db_path),
        "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
        # Point at a non-existent dir so this test doesn't load real
        # ~/.codex/sessions data from the machine running it.
        "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
        "INTROSPECT_DAYS": "0",
        "INTROSPECT_HOST": "127.0.0.1",
    }
    with patch.dict(os.environ, env), TestClient(app) as client:
        yield client


def test_triggers_renders_auto_load_columns():
    """A session with attachment records shows a 'yes' badge and a non-zero
    CLAUDE.md stat — the headline auto-load surface, exercised through the page."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _client_with_attachments(Path(tmp)) as client,
    ):
        response = client.get("/triggers")
        assert response.status_code == 200
        # nested_memory attachment -> CLAUDE.md "yes" badge in the row.
        assert 'badge-ok">yes' in response.text
        assert "Auto-loaded a CLAUDE.md" in response.text
        # The opener names a path (@src/db.py), so it is NOT a vague opener.
        assert "0.0%" in response.text


def test_triggers_htmx_partial():
    """An HX-Request renders the partial (no full <html> shell)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/triggers", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "First-Prompt Triggers" in response.text
        assert "<nav>" not in response.text
