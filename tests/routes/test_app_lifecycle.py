"""Tests for app lifecycle: DB swap, per-request connections, lifespan validation."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb

from introspect.api.main import app

from ..conftest import glob_pattern, local_client
from .conftest import _patched_client, _write_sample_jsonl


def _setup_marker_swap(db_path: Path, sidecar: Path) -> tuple[str, str]:
    """Add a marker table to ``db_path`` and a sidecar copy with a different row.

    Returns ``(live_marker, sidecar_marker)`` so the caller can assert which
    inode a cursor was reading from.
    """
    import shutil  # noqa: PLC0415

    live_marker, sidecar_marker = "live", "swapped"
    live_conn = duckdb.connect(str(db_path))
    try:
        live_conn.execute("CREATE TABLE swap_marker(label VARCHAR)")
        live_conn.execute("INSERT INTO swap_marker VALUES (?)", [live_marker])
    finally:
        live_conn.close()
    shutil.copy(str(db_path), str(sidecar))
    side_conn = duckdb.connect(str(sidecar))
    try:
        side_conn.execute("DELETE FROM swap_marker")
        side_conn.execute("INSERT INTO swap_marker VALUES (?)", [sidecar_marker])
    finally:
        side_conn.close()
    return live_marker, sidecar_marker


def _run_swap_on_thread(db_path: Path, sidecar: Path) -> None:
    """Run ``_swap_in`` on a thread, mirroring ``asyncio.to_thread`` from the loop."""
    import threading  # noqa: PLC0415

    from introspect.refresh import _swap_in  # noqa: PLC0415

    swap_done = threading.Event()

    def do_swap() -> None:
        _swap_in(db_path, sidecar)
        swap_done.set()

    t = threading.Thread(target=do_swap)
    t.start()
    t.join(timeout=2.0)
    assert swap_done.is_set(), "swap thread did not finish"


def test_swap_during_in_flight_cursor_does_not_500():
    """A swap *while* a cursor is mid-query must not surface as a 500.

    Reproduces the original bug shape: a request opens a connection, starts a
    query, the file is replaced under it, and the cursor must still complete
    successfully because per-request connections hold the old inode open via
    a live file descriptor. After the cursor closes, fresh connections see
    the *new* swapped DB.

    The sidecar is a full copy of the materialized DB plus a marker table
    with a different value, so we can assert that the in-flight cursor
    returned the *pre-swap* row (proving it held the old inode) and that a
    fresh post-swap connection returns the *post-swap* row (proving the
    swap actually replaced the file).
    """
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # Issue one normal request first to ensure middleware/state are warm.
        assert client.get("/sessions").status_code == 200

        db_path = Path(app.state.db_path)
        sidecar = db_path.with_name(db_path.name + ".next")
        live_marker, sidecar_marker = _setup_marker_swap(db_path, sidecar)

        # Open a long-lived read-only connection mimicking what the middleware
        # hands a request, and *start* a query so the cursor holds an open
        # handle on the old inode while the swap fires.
        long_lived = duckdb.connect(str(db_path), read_only=True)
        try:
            cursor = long_lived.execute("SELECT label FROM swap_marker")
            _run_swap_on_thread(db_path, sidecar)
            rows = cursor.fetchall()
            assert rows == [(live_marker,)], (
                f"in-flight cursor saw {rows!r}; expected pre-swap row"
            )
        finally:
            long_lived.close()

        # A fresh connection opened *after* the swap must see the new data,
        # proving the swap actually replaced the file the path resolves to.
        post_swap = duckdb.connect(str(db_path), read_only=True)
        try:
            post_rows = post_swap.execute("SELECT label FROM swap_marker").fetchall()
        finally:
            post_swap.close()
        assert post_rows == [(sidecar_marker,)], (
            f"post-swap connection saw {post_rows!r}; expected swapped row"
        )

        # The middleware path (which opens its own per-request connection)
        # must keep working against the swapped DB without 500ing.
        assert client.get("/sessions").status_code == 200


def test_per_request_connection_no_shared_read_conn():
    """``app.state`` should no longer carry a shared ``read_conn``."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # Issue a request to confirm the middleware path works.
        response = client.get("/sessions")
        assert response.status_code == 200
        assert not hasattr(app.state, "read_conn")
        assert hasattr(app.state, "db_path")


def test_lifespan_rejects_invalid_refresh_window_env(caplog, monkeypatch):
    """An invalid ``INTROSPECT_REFRESH_WINDOW`` env var falls back to default.

    Without validation, a typo would leave ``app.state.refresh_window`` set
    to garbage, breaking the picker (no option renders ``selected``) and
    causing ``_compute_days`` to disagree with ``window_to_days`` about the
    fallback. Lifespan must coerce invalid input to ``DEFAULT_WINDOW``.
    """
    from introspect.refresh import DEFAULT_WINDOW  # noqa: PLC0415

    # The test asserts the picker-window fallback rather than the separate
    # numeric-days override. CI may define the latter for its own test setup.
    monkeypatch.delenv("INTROSPECT_DAYS", raising=False)
    _write_sample_jsonl_path = tempfile.TemporaryDirectory()
    try:
        tmp = Path(_write_sample_jsonl_path.name)
        _write_sample_jsonl(tmp)
        db_path = tmp / "test.duckdb"
        with (
            patch.dict(
                os.environ,
                {
                    "INTROSPECT_DB_PATH": str(db_path),
                    "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                    "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                    "INTROSPECT_REFRESH_WINDOW": "garbage",
                    "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0",
                },
            ),
            caplog.at_level(logging.WARNING, logger="introspect.api.main"),
            local_client(app) as client,
        ):
            assert client.get("/sessions").status_code == 200
            assert app.state.refresh_window == DEFAULT_WINDOW
            assert any(
                "Invalid INTROSPECT_REFRESH_WINDOW" in r.message for r in caplog.records
            )
    finally:
        _write_sample_jsonl_path.cleanup()


def test_lifespan_keeps_numeric_days_in_initial_target():
    """The CLI's numeric days override survives the picker default."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(
            Path(tmp),
            extra_env={
                "INTROSPECT_DAYS": "7",
                "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0",
            },
        ) as client,
    ):
        assert client.get("/sessions").status_code == 200
        assert app.state.refresh_target.days == 7
        assert app.state.refresh_target.window == "7"
        assert app.state.refresh_window == "7"


def test_one_day_startup_schedules_an_authoritative_rebuild():
    """A bounded one-day preview must be followed by the complete target build."""
    refresh_calls: list[dict[str, object]] = []

    async def fake_refresh_loop(*args, **kwargs) -> None:
        refresh_calls.append(kwargs)
        await asyncio.Event().wait()

    with (
        patch("introspect.api.main.refresh_loop", fake_refresh_loop),
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(
            Path(tmp),
            extra_env={
                "INTROSPECT_DAYS": "1",
                "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0",
            },
        ) as client,
    ):
        assert client.get("/sessions").status_code == 200
        assert app.state.refresh_pending is True
        assert len(refresh_calls) == 1
        assert refresh_calls[0]["initial"] is True
        assert refresh_calls[0]["one_shot"] is True


def test_compatible_database_is_published_as_warm_snapshot():
    """A prior complete build serves while the authoritative build runs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_REFRESH_INTERVAL_SECONDS": "0"},
        ) as client:
            assert client.get("/sessions").status_code == 200

        with _patched_client(
            tmp_path,
            extra_env={"INTROSPECT_REFRESH_INTERVAL_SECONDS": "0"},
        ) as client:
            assert client.get("/sessions").status_code == 200
            assert app.state.database_snapshot is True
            assert app.state.database_label == "warm snapshot"
