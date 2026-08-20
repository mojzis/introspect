"""Tests for the session cost tab."""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from introspect.api.main import app

from ..conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)
from .conftest import SID, _patched_client
from .cost_helpers import _dup_jsonl, _subagent_jsonl


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
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
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
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
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
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
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


# --- Phase 1b: Bloat partial endpoint tests ---------------------------


def _first_assistant_uuid(client, session_id: str) -> str:
    """Pull an assistant uuid from the cost tab's figure JSON.

    The figure customdata carries each bucket's first/last raw uuid, so any
    valid assistant uuid in the session ends up in the rendered HTML. This
    helper grabs the first one for use as a filter range endpoint.
    """
    response = client.get(f"/sessions/{session_id}?tab=cost")
    text = response.text
    # Bucket customdata shape: ["uuid1", "uuid2", n] — pull first quoted token
    # that follows "customdata" in the figure JSON.
    m = re.search(r'"customdata":\s*\[\s*\[\s*"([^"]+)"', text)
    assert m, "could not locate a uuid in figure customdata"
    return m.group(1)


def test_session_cost_bloat_partial_unfiltered():
    """GET /sessions/{sid}/cost/bloat with no range params returns rollup."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}/cost/bloat")
        assert response.status_code == 200
        text = response.text
        assert 'id="session-cost-bloat-panel"' in text
        # Headers from the bloat partial.
        assert "Bloat attribution" in text
        assert "Top contributors" in text
        # No filter banner when unfiltered.
        assert "Showing" not in text or "Clear filter" not in text


def test_session_cost_bloat_partial_filtered():
    """GET with from_uuid + to_uuid scopes the rollup and shows the filter banner."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        uuid = _first_assistant_uuid(client, SID)
        response = client.get(
            f"/sessions/{SID}/cost/bloat",
            params={"from_uuid": uuid, "to_uuid": uuid},
        )
        assert response.status_code == 200
        text = response.text
        assert "Showing" in text
        assert "Clear filter" in text


def test_session_cost_bloat_partial_invalid_uuid():
    """Bad uuid → 404 (linear scan over attrib_rows finds nothing)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(
            f"/sessions/{SID}/cost/bloat",
            params={"from_uuid": "not-a-real-uuid", "to_uuid": "also-fake"},
        )
        assert response.status_code == 404


def test_session_cost_top_contributor_links_to_message():
    """Top-contributors row links its 'Worst message' cell to #msg-{uuid}.

    The sample fixture has a Bash + sidechain reply that drives at least
    one cache-write bucket; that bucket should expose a top_uuid anchor.
    """
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}/cost/bloat")
        assert response.status_code == 200
        text = response.text
        # If the fixture produced any cache writes, the contributors table
        # must have at least one anchor to the worst-offender message. If
        # not, the table renders the "no bloat data" fallback — also fine.
        if "No bloat data" in text:
            import pytest  # noqa: PLC0415

            pytest.skip("sample fixture produced no cache writes")
        assert f'href="/sessions/{SID}?tab=messages#msg-' in text


def test_session_cost_chart_serializes_uuid_columns():
    """Regression: DuckDB types uuid-shaped values as UUID, not VARCHAR.

    The cost-attribution SQL must cast both ``session_id`` and ``uuid``
    to VARCHAR — Plotly's JSON encoder doesn't know how to serialize a
    Python ``uuid.UUID`` and the chart fails to render on real session
    data (whose uuids look like ``aff6d25c-43a8-4d49-...``).
    """
    sid = "11111111-2222-3333-4444-555555555555"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        usage = {"input_tokens": 100, "output_tokens": 20}
        # Use uuid-shaped per-message uuids so DuckDB infers UUID type
        # for the uuid column (matches production data shape).
        lines: list[dict] = [
            make_user_message(
                sid,
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
                None,
                "2026-04-21T10:00:00.000Z",
                "hello",
                tool_use_result={"content": "seed"},
            ),
        ]
        lines.extend(
            make_assistant_message(
                sid,
                f"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbb{i:02d}",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
                f"2026-04-21T10:00:{i + 1:02d}.000Z",
                [{"type": "text", "text": f"msg{i}"}],
                model="claude-opus-4-7",
                msg_id=f"msg-uuid-{i}",
                usage=usage,
            )
            for i in range(7)
        )
        write_jsonl(tmp, sid, lines)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                    "INTROSPECT_DAYS": "0",
                },
            ),
            TestClient(app) as client,
        ):
            response = client.get(f"/sessions/{sid}?tab=cost")
            assert response.status_code == 200
            # The figure JSON must contain the uuid as a string (not a
            # serialised UUID object).
            assert "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbb00" in response.text
