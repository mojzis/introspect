"""Tests for CLI helpers."""

import socket
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from introspect.cli import _find_available_port, app

runner = CliRunner()

_UVICORN_SHOULD_NOT_RUN = "uvicorn.run should not be called in this test"


def test_materialize_shows_friendly_message_when_db_locked(monkeypatch, mock_locked_db):
    """`introspect materialize` prints a friendly message when the DB is locked."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "introspect.duckdb"
        monkeypatch.setattr("introspect.cli.DEFAULT_DB_PATH", db_path)

        result = runner.invoke(app, ["materialize"])

        assert result.exit_code == 1
        assert "Another Introspect process" in result.output
        assert str(db_path) in result.output


def test_serve_shows_friendly_message_when_db_locked(monkeypatch, mock_locked_db):
    """`introspect serve` aborts with a friendly message when the DB is locked."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "introspect.duckdb"
        monkeypatch.setenv("INTROSPECT_DB_PATH", str(db_path))

        # If uvicorn.run is reached, the test has failed — stub it so any accidental
        # call raises loudly rather than actually starting a server.
        def _fail_run(*args, **kwargs):
            raise AssertionError(_UVICORN_SHOULD_NOT_RUN)

        monkeypatch.setattr("uvicorn.run", _fail_run)

        result = runner.invoke(app, ["serve"])

        assert result.exit_code == 1
        assert "Another Introspect process" in result.output


def test_serve_falls_back_to_next_port_when_requested_port_taken(monkeypatch):
    """`introspect serve` picks the next free port and warns if the default is taken."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "introspect.duckdb"
        monkeypatch.setenv("INTROSPECT_DB_PATH", str(db_path))

        host = "127.0.0.1"
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind((host, 0))
        blocker.listen(1)
        taken_port = blocker.getsockname()[1]

        captured: dict[str, object] = {}

        def _fake_run(_app, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("uvicorn.run", _fake_run)

        try:
            result = runner.invoke(
                app, ["serve", "--port", str(taken_port), "--host", host]
            )
        finally:
            blocker.close()

        assert result.exit_code == 0, result.output
        assert f"Port {taken_port} is in use" in result.output
        assert captured["port"] != taken_port
        assert captured["host"] == host


def test_serve_errors_when_no_port_available(monkeypatch):
    """`introspect serve` exits with a clear message when all probed ports are taken."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "introspect.duckdb"
        monkeypatch.setenv("INTROSPECT_DB_PATH", str(db_path))

        monkeypatch.setattr(
            "introspect.cli._find_available_port", lambda *a, **kw: None
        )

        def _fail_run(*args, **kwargs):
            raise AssertionError(_UVICORN_SHOULD_NOT_RUN)

        monkeypatch.setattr("uvicorn.run", _fail_run)

        result = runner.invoke(app, ["serve"])

        assert result.exit_code == 1
        assert "none were free" in result.output


def test_find_available_port_skips_taken_port():
    """The helper returns the next port when the requested one is bound."""
    host = "127.0.0.1"
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind((host, 0))
    blocker.listen(1)
    taken = blocker.getsockname()[1]

    try:
        available = _find_available_port(host, taken, attempts=5)
    finally:
        blocker.close()

    assert available is not None
    assert available != taken
    assert taken < available < taken + 5


def test_find_available_port_returns_start_port_when_free():
    """The helper returns the start port unchanged when it is free."""
    host = "127.0.0.1"
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((host, 0))
    free_port = probe.getsockname()[1]
    probe.close()

    assert _find_available_port(host, free_port, attempts=1) == free_port
