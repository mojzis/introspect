"""Tests for CLI helpers."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from introspect.cli import app

runner = CliRunner()


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
        fail_msg = "uvicorn.run should not be called when DB is locked"

        def _fail_run(*args, **kwargs):
            raise AssertionError(fail_msg)

        monkeypatch.setattr("uvicorn.run", _fail_run)

        result = runner.invoke(app, ["serve"])

        assert result.exit_code == 1
        assert "Another Introspect process" in result.output
