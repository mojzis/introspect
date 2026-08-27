"""Tests for the local-only HTTP SQL API (`/api/query`, `/api/schema`)."""

from pathlib import Path

from introspect.api.main import CLIENT_HEADER
from introspect.sql_query import API_SQL_ROW_CAP

from .conftest import _patched_client


def post_query(client, sql: str, **body):
    """POST /api/query with the client header every real caller must send.

    The header forces a CORS preflight that a drive-by cross-origin fetch
    cannot satisfy; omitting it is covered in tests/e2e/test_sql_hardening.py.
    """
    return client.post(
        "/api/query", json={"sql": sql, **body}, headers={CLIENT_HEADER: "1"}
    )


def test_query_happy_path(tmp_path: Path):
    """A valid SELECT returns columns + rows as JSON."""
    with _patched_client(tmp_path) as client:
        resp = post_query(
            client, "SELECT session_id, user_messages FROM logical_sessions"
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["columns"] == ["session_id", "user_messages"]
    assert data["row_count"] == len(data["rows"])
    assert data["row_count"] >= 1
    assert data["truncated"] is False


def test_query_accepts_cte(tmp_path: Path):
    """WITH (CTE) queries are permitted, not just bare SELECT."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "WITH c AS (SELECT 1 AS n) SELECT n FROM c")
    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1]]


def test_query_rejects_write_statement(tmp_path: Path):
    """Non-SELECT statements are rejected with 400."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "DELETE FROM logical_sessions")
    assert resp.status_code == 400
    assert "DELETE" in resp.json()["error"]


def test_query_rejects_attach(tmp_path: Path):
    """ATTACH is blocked by the statement-type guard."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "ATTACH 'evil.db' AS evil")
    assert resp.status_code == 400


def test_query_rejects_multiple_statements(tmp_path: Path):
    """Multi-statement scripts are rejected."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "SELECT 1; SELECT 2")
    assert resp.status_code == 400
    assert "Multiple statements" in resp.json()["error"]


def test_query_surfaces_sql_error(tmp_path: Path):
    """An invalid query (unknown table) returns 400 with the DuckDB error."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "SELECT * FROM no_such_table")
    assert resp.status_code == 400
    assert "SQL error" in resp.json()["error"]


def test_query_enforces_limit(tmp_path: Path):
    """The caller's limit caps returned rows and truncated flags the overflow."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "SELECT * FROM range(0, 50) AS t(n)", limit=5)
    data = resp.json()
    assert data["row_count"] == 5
    assert data["truncated"] is True


def test_query_not_truncated_when_under_limit(tmp_path: Path):
    """A result at exactly the limit with no more rows is not flagged truncated."""
    with _patched_client(tmp_path) as client:
        resp = post_query(client, "SELECT * FROM range(0, 5) AS t(n)", limit=5)
    data = resp.json()
    assert data["row_count"] == 5
    assert data["truncated"] is False


def test_query_clamps_limit_to_cap(tmp_path: Path):
    """A limit above the hard cap is clamped to API_SQL_ROW_CAP."""
    with _patched_client(tmp_path) as client:
        resp = post_query(
            client,
            f"SELECT * FROM range(0, {API_SQL_ROW_CAP + 100}) AS t(n)",
            limit=API_SQL_ROW_CAP + 100,
        )
    data = resp.json()
    assert data["row_count"] == API_SQL_ROW_CAP
    assert data["truncated"] is True


def test_schema_lists_tables(tmp_path: Path):
    """/api/schema returns tables with columns."""
    with _patched_client(tmp_path) as client:
        resp = client.get("/api/schema")
    assert resp.status_code == 200
    tables = resp.json()["tables"]
    assert "logical_sessions" in tables
    cols = {c["column"] for c in tables["logical_sessions"]}
    assert "session_id" in cols


def test_query_disabled_when_not_loopback(tmp_path: Path):
    """Bound to a non-loopback host, the endpoint 404s (fails closed)."""
    with _patched_client(tmp_path, extra_env={"INTROSPECT_HOST": "0.0.0.0"}) as client:
        resp = post_query(client, "SELECT 1 AS n")
        schema_resp = client.get("/api/schema")
    assert resp.status_code == 404
    assert schema_resp.status_code == 404


def test_query_disabled_by_env_toggle(tmp_path: Path):
    """INTROSPECT_SQL_API=off disables the endpoint even on loopback."""
    with _patched_client(
        tmp_path,
        extra_env={"INTROSPECT_HOST": "127.0.0.1", "INTROSPECT_SQL_API": "off"},
    ) as client:
        resp = post_query(client, "SELECT 1 AS n")
    assert resp.status_code == 404
