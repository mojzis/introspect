"""Tests for CLI helpers."""

import asyncio
import contextlib
import json
import socket
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from introspect import version_check as vc
from introspect.cli import (
    CLAUDE_SYSTEM_PROMPT_SUFFIX,
    CODEX_DEVELOPER_INSTRUCTIONS,
    _branch_db_path,
    _find_available_port,
    _finish_connected_session,
    _prune_stale_branch_dbs,
    _sanitize_branch,
    _stop_server,
    app,
)
from introspect.mcp.server import create_mcp_server

from .conftest import TTL_T0, glob_pattern, ttl_turn, write_jsonl

runner = CliRunner()

_UVICORN_SHOULD_NOT_RUN = "uvicorn.run should not be called in this test"
_BRANCH_DETECTION_SHOULD_SKIP = "branch detection should be skipped"


def _patch_cli_paths(monkeypatch, tmp: str) -> Path:
    """Point the CLI's DB/JSONL/Codex defaults at non-existent paths under
    ``tmp``, so tests never touch the real ``~/.claude`` or ``~/.codex``
    trees on the machine running them. Returns the patched DB path."""
    db_path = Path(tmp) / "introspect.duckdb"
    monkeypatch.setattr("introspect.cli.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_JSONL_GLOB",
        str(Path(tmp) / "claude" / "**" / "*.jsonl"),
    )
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_CODEX_GLOB",
        str(Path(tmp) / "codex" / "**" / "*.jsonl"),
    )
    return db_path


# Read commands that should work end-to-end against an empty DB. ``materialize``
# is exercised explicitly elsewhere; ``serve`` / ``devserve`` / ``mcp`` start
# long-running processes; ``query`` requires a SQL string. The remaining
# commands all run a default query and exercise different views.
_EMPTY_DB_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("sessions",),
    ("tools",),
    ("stats",),
    ("raw",),
    ("tables",),
    ("search", "anything"),
    ("refresh",),
    ("cache-ttl",),
    ("cache-ttl", "--verify"),
    ("cache-ttl", "--subagents"),
)


@pytest.mark.parametrize("command", _EMPTY_DB_COMMANDS)
def test_cli_command_works_on_empty_db(monkeypatch, command):
    """CLI read commands succeed when no DB and no JSONL files exist yet.

    Auto-materialization must build empty stub tables instead of crashing on
    ``read_json_auto``'s "no files found" error, and every command must print
    the "Last materialized" banner so users see when the data was last built.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Glob points at a non-existent directory; nothing matches.
        db_path = _patch_cli_paths(monkeypatch, tmp)

        result = runner.invoke(app, list(command))

        assert result.exit_code == 0, result.output
        # ``refresh`` deliberately skips the banner — it rebuilds the index
        # using its own writable connection rather than going through ``_db``.
        if command != ("refresh",):
            assert "Last materialized" in result.output, result.output
        # The DB file should now exist and contain the materialize_meta stamp.
        assert db_path.exists()


def test_cache_ttl_recommends_5m_when_there_are_no_gaps(monkeypatch):
    """No pauses → 1h's 2x write surcharge is pure loss, and it says so."""
    sid = "77777777-7777-7777-7777-777777777777"
    lines = ttl_turn(sid, 1, TTL_T0, read=0, create=50_000)
    for n in range(2, 6):
        lines += ttl_turn(
            sid,
            n,
            TTL_T0 + timedelta(seconds=20 * (n - 1)),
            read=50_000 * (n - 1),
            create=50_000,
        )
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)
        claude = Path(tmp) / "claude"
        write_jsonl(claude, sid, lines)
        monkeypatch.setattr("introspect.cli.DEFAULT_JSONL_GLOB", glob_pattern(claude))

        result = runner.invoke(app, ["cache-ttl"])

        assert result.exit_code == 0, result.output
        assert "Main conversation" in result.output
        assert "5m saves" in result.output
        # Every gap is short, so nothing is recoverable and nothing is a break.
        assert "0 recoverable gap(s)" in result.output


def test_cache_ttl_verify_reports_zero_residual_on_uniform_session(monkeypatch):
    """The gate: simulating the billed TTL reproduces the billed cost."""
    sid = "88888888-8888-8888-8888-888888888888"
    lines = ttl_turn(sid, 1, TTL_T0, read=0, create=10_000, ttl="1h")
    lines += ttl_turn(
        sid, 2, TTL_T0 + timedelta(minutes=2), read=10_000, create=10_000, ttl="1h"
    )
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)
        claude = Path(tmp) / "claude"
        write_jsonl(claude, sid, lines)
        monkeypatch.setattr("introspect.cli.DEFAULT_JSONL_GLOB", glob_pattern(claude))

        result = runner.invoke(app, ["cache-ttl", "--verify"])

        assert result.exit_code == 0, result.output
        assert "Worst residual: 0.000%" in result.output
        assert "0 (0.0%)" in result.output  # split present on every row


_MATERIALIZED_BANNER_PREFIX = "Last materialized: "


def _extract_banner_timestamp(output: str) -> str:
    """Pull the ``YYYY-MM-DD HH:MM:SS`` field out of the banner line."""
    for line in output.splitlines():
        if _MATERIALIZED_BANNER_PREFIX in line:
            tail = line.split(_MATERIALIZED_BANNER_PREFIX, 1)[1]
            # The banner is "<iso> (<relative>)" — slice off everything after the
            # timestamp so the relative-time portion (which moves with wall clock)
            # doesn't make the comparison flaky.
            return tail.split(" (", 1)[0].strip()
    pytest.fail(f"banner not present in CLI output: {output!r}")


def test_cli_reuses_existing_materialized_db(monkeypatch):
    """A second CLI invocation prints the prior timestamp instead of rebuilding."""
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)

        first = runner.invoke(app, ["sessions"])
        assert first.exit_code == 0, first.output
        first_ts = _extract_banner_timestamp(first.output)

        second = runner.invoke(app, ["sessions"])
        assert second.exit_code == 0, second.output
        second_ts = _extract_banner_timestamp(second.output)

        assert first_ts == second_ts, (
            "second invocation should reuse the existing materialized DB; "
            f"banner went from {first_ts!r} to {second_ts!r}"
        )


def _stub_behind_cache(monkeypatch, tmp, *, latest="9.9.9", current="0.0.1"):
    """Point the version-check cache at ``tmp`` with a fresh 'we're behind' entry
    and force the eligible-install / TTY predicates on."""
    monkeypatch.setenv("INTROSPECT_DB_PATH", str(Path(tmp) / "introspect.duckdb"))
    for name in (*vc._CI_ENV_VARS, vc.ENV_ENABLED, vc.ENV_INTERVAL):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vc, "_current_version", lambda: current)
    monkeypatch.setattr(vc, "_is_editable_install", lambda: False)
    monkeypatch.setattr(vc, "_stderr_is_tty", lambda: True)
    # Far-future timestamp keeps the cache "fresh" without coupling to the clock.
    vc._write_cache(
        Path(tmp) / "version_check.json",
        vc.VersionCache(checked_at=9_999_999_999.0, latest=latest),
    )


def test_nag_prints_to_stderr_not_stdout_when_behind(monkeypatch):
    """An eligible command prints the one-line nag to stderr only, after output."""
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)
        _stub_behind_cache(monkeypatch, tmp)

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0, result.output
        assert "9.9.9 is available" in result.stderr
        assert "uvx introspy@latest" in result.stderr
        assert "is available" not in result.stdout


def test_no_nag_when_up_to_date_cli(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)
        _stub_behind_cache(monkeypatch, tmp, latest="0.0.1", current="0.0.1")

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0, result.output
        assert "is available" not in result.stderr
        assert "is available" not in result.stdout


def test_opt_out_env_silences_nag_cli(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _patch_cli_paths(monkeypatch, tmp)
        _stub_behind_cache(monkeypatch, tmp)
        monkeypatch.setenv("INTROSPECT_VERSION_CHECK", "off")

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0, result.output
        assert "is available" not in result.stderr


def test_mcp_command_never_nags(monkeypatch):
    """The ``mcp`` (stdio) command must never emit the nag, even when behind."""
    with tempfile.TemporaryDirectory() as tmp:
        _stub_behind_cache(monkeypatch, tmp)

        class _StubServer:
            def run(self, transport):
                return None

        monkeypatch.setattr("introspect.mcp.server.create_mcp_server", _StubServer)

        result = runner.invoke(app, ["mcp"])

        assert result.exit_code == 0, result.output
        assert "is available" not in result.stderr
        assert "is available" not in result.stdout


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


def test_sanitize_branch_replaces_unsafe_chars():
    assert _sanitize_branch("feat/new-thing") == "feat-new-thing"
    assert _sanitize_branch("main") == "main"


def test_branch_db_path_namespaces_per_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_DB_PATH", tmp_path / "introspect.duckdb"
    )
    path = _branch_db_path("feat/x")
    assert path == tmp_path / "introspect-feat-x.duckdb"


def test_prune_stale_branch_dbs_removes_only_dead_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_DB_PATH", tmp_path / "introspect.duckdb"
    )
    keep = tmp_path / "introspect-current.duckdb"
    live = tmp_path / "introspect-main.duckdb"
    dead = tmp_path / "introspect-old.duckdb"
    shared = tmp_path / "introspect.duckdb"  # default DB, must be untouched
    for f in (keep, live, dead, shared):
        f.touch()
    dead.with_name(dead.name + ".wal").touch()

    # Prune only runs when launched from the introspect repo itself.
    monkeypatch.setattr("introspect.cli._git_toplevel", lambda _cwd: tmp_path)

    def _fake_run(*_a, **_k):
        return subprocess.CompletedProcess([], 0, stdout="current\nmain\n", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    removed = _prune_stale_branch_dbs(keep=keep)

    assert removed == [dead]
    assert keep.exists()
    assert live.exists()
    assert shared.exists()
    assert not dead.exists()
    assert not dead.with_name(dead.name + ".wal").exists()


def test_prune_stale_branch_dbs_skips_when_not_in_introspect_repo(
    monkeypatch, tmp_path
):
    """Launched from an unrelated repo, prune must not delete introspect's DBs."""
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_DB_PATH", tmp_path / "introspect.duckdb"
    )
    dead = tmp_path / "introspect-old.duckdb"
    dead.touch()

    # cwd repo (first call) differs from the introspect source repo (second call).
    tops = iter([tmp_path / "other-repo", tmp_path / "introspect-src"])
    monkeypatch.setattr("introspect.cli._git_toplevel", lambda _cwd: next(tops))

    removed = _prune_stale_branch_dbs(keep=tmp_path / "introspect-current.duckdb")

    assert removed == []
    assert dead.exists()


def test_devserve_uses_shared_db_on_detached_head(monkeypatch, tmp_path):
    """Detached HEAD → no branch → leave INTROSPECT_DB_PATH unset (shared default)."""
    monkeypatch.delenv("INTROSPECT_DB_PATH", raising=False)
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_DB_PATH", tmp_path / "introspect.duckdb"
    )
    monkeypatch.setattr("introspect.cli._current_git_branch", lambda: None)

    captured: dict[str, object] = {}

    def _fake_run(_app, **kwargs):
        import os  # noqa: PLC0415

        captured["db"] = os.environ.get("INTROSPECT_DB_PATH")

    monkeypatch.setattr("uvicorn.run", _fake_run)

    result = runner.invoke(app, ["devserve"])

    assert result.exit_code == 0, result.output
    assert captured["db"] is None


def test_devserve_sets_branch_db_path(monkeypatch, tmp_path):
    monkeypatch.delenv("INTROSPECT_DB_PATH", raising=False)
    monkeypatch.setattr(
        "introspect.cli.DEFAULT_DB_PATH", tmp_path / "introspect.duckdb"
    )
    monkeypatch.setattr("introspect.cli._current_git_branch", lambda: "feat/x")
    monkeypatch.setattr("introspect.cli._prune_stale_branch_dbs", lambda keep: [])

    captured: dict[str, object] = {}

    def _fake_run(_app, **kwargs):
        import os  # noqa: PLC0415

        captured["db"] = os.environ.get("INTROSPECT_DB_PATH")

    monkeypatch.setattr("uvicorn.run", _fake_run)

    result = runner.invoke(app, ["devserve"])

    assert result.exit_code == 0, result.output
    assert captured["db"] == str(tmp_path / "introspect-feat-x.duckdb")


def test_devserve_respects_explicit_db_path(monkeypatch, tmp_path):
    explicit = str(tmp_path / "custom.duckdb")
    monkeypatch.setenv("INTROSPECT_DB_PATH", explicit)

    def _boom():
        raise AssertionError(_BRANCH_DETECTION_SHOULD_SKIP)

    monkeypatch.setattr("introspect.cli._current_git_branch", _boom)

    captured: dict[str, object] = {}

    def _fake_run(_app, **kwargs):
        import os  # noqa: PLC0415

        captured["db"] = os.environ.get("INTROSPECT_DB_PATH")

    monkeypatch.setattr("uvicorn.run", _fake_run)

    result = runner.invoke(app, ["devserve"])

    assert result.exit_code == 0, result.output
    assert captured["db"] == explicit


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


def test_claude_errors_when_cli_not_installed(monkeypatch):
    """`claude` exits with a helpful error when the claude binary is missing."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_claude_passes_inline_mcp_config(monkeypatch):
    """`claude` launches the binary with an inline --mcp-config pointing at /mcp."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 0)

    result = runner.invoke(app, ["claude", "--port", "3000"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "/usr/bin/claude"
    assert argv[1] == "--mcp-config"
    config = json.loads(argv[2])
    server = config["mcpServers"]["introspect"]
    assert server == {"type": "http", "url": "http://127.0.0.1:3000/mcp"}


def test_codex_passes_session_only_mcp_config_and_instructions(monkeypatch):
    """`codex` uses overrides, never a persistent MCP registration."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 0)

    result = runner.invoke(app, ["codex", "--port", "3000"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "/usr/bin/codex"
    config_values = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--config"]
    assert 'mcp_servers.introspect.url="http://127.0.0.1:3000/mcp"' in config_values
    developer_config = next(
        value for value in config_values if value.startswith("developer_instructions=")
    )
    assert json.loads(developer_config.split("=", 1)[1]) == CODEX_DEVELOPER_INSTRUCTIONS


def test_claude_steers_session_toward_mcp_tools(monkeypatch):
    """`claude` appends a dedicated system prompt and pre-allows the MCP tools."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 0)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 0, result.output
    argv = calls[0]
    prompt = argv[argv.index("--append-system-prompt") + 1]
    assert "mcp__introspect__" in prompt
    assert argv[argv.index("--allowedTools") + 1] == "mcp__introspect"


def test_claude_forwards_extra_args(monkeypatch):
    """Extra args after `--` are appended verbatim to the claude invocation."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 0)

    result = runner.invoke(
        app, ["claude", "--port", "3000", "--", "--model", "opus", "--resume"]
    )

    assert result.exit_code == 0, result.output
    argv = calls[0]
    assert argv[-3:] == ["--model", "opus", "--resume"]
    # introspect's own --port is consumed, not forwarded
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert config["mcpServers"]["introspect"]["url"].endswith(":3000/mcp")


def test_codex_forwards_extra_args(monkeypatch):
    """Extra args after `--` are appended verbatim to the Codex invocation."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 0)

    result = runner.invoke(app, ["codex", "--port", "3000", "--", "--model", "gpt-5.4"])

    assert result.exit_code == 0, result.output
    assert calls[0][-2:] == ["--model", "gpt-5.4"]


def test_claude_system_prompt_lists_every_registered_tool():
    """The hand-written tool list in the system prompt must not drift.

    `CLAUDE_SYSTEM_PROMPT_SUFFIX` enumerates the MCP tools by name; when a
    tool is added or renamed in the registry, the prompt must be updated too.
    """
    registered = {tool.name for tool in asyncio.run(create_mcp_server().list_tools())}
    missing = {name for name in registered if name not in CLAUDE_SYSTEM_PROMPT_SUFFIX}
    assert not missing, f"system prompt omits registered MCP tools: {missing}"


def test_codex_developer_instructions_list_every_registered_tool():
    """Codex's dedicated-session instructions must not drift from the registry."""
    registered = {tool.name for tool in asyncio.run(create_mcp_server().list_tools())}
    missing = {name for name in registered if name not in CODEX_DEVELOPER_INSTRUCTIONS}
    assert not missing, f"developer instructions omit registered MCP tools: {missing}"


def test_claude_propagates_claude_exit_code(monkeypatch):
    """The claude binary's exit code becomes the command's exit code."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: 3)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 3


def test_codex_propagates_codex_exit_code(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(subprocess, "call", lambda argv: 3)

    result = runner.invoke(app, ["codex"])

    assert result.exit_code == 3


class FakeServerProc:
    """Configurable stand-in for the spawned `introspy serve` Popen.

    ``poll()`` returns ``returncode`` (``None`` = still running).  With
    ``hang_on_term=True``, timed ``wait()`` calls raise ``TimeoutExpired``
    until ``kill()`` is called, simulating a server that ignores SIGTERM.
    """

    pid = 1234

    def __init__(self, returncode=None, hang_on_term=False):
        self.returncode = returncode
        self.hang_on_term = hang_on_term
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.hang_on_term and timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired(cmd="serve", timeout=timeout)
        return 0


def test_claude_starts_server_when_not_running(monkeypatch, tmp_path):
    """When nothing is listening, `claude` spawns the server and waits for it."""
    _proc, popen_calls = _wire_autospawned_server_fakes(monkeypatch, tmp_path)

    result = runner.invoke(app, ["claude", "--port", "3000"])

    assert result.exit_code == 0, result.output
    assert "Server ready" in result.output
    assert len(popen_calls) == 1
    argv = popen_calls[0]
    assert "-m" in argv
    assert "introspect.cli" in argv
    assert "serve" in argv
    assert "--port" in argv
    assert "3000" in argv
    assert "--host" in argv


def test_claude_skips_start_when_server_running(monkeypatch):
    """When the server is already listening, `claude` does not spawn a new one."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: contextlib.nullcontext()
    )

    def _should_not_popen(*args, **kwargs):
        pytest.fail("subprocess.Popen should not be called when server is running")

    monkeypatch.setattr(subprocess, "Popen", _should_not_popen)
    monkeypatch.setattr(subprocess, "call", lambda argv: 0)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 0, result.output
    # The pre-existing server must be left untouched on exit.
    assert "Stopping introspect server" not in result.output


def _wire_autospawned_server_fakes(monkeypatch, tmp_path, claude_exit_code=0):
    """Make `claude` auto-spawn a fake server; return (fake proc, popen argvs).

    First connection probe refuses (no server), subsequent ones succeed
    (readiness poll passes).
    """
    call_count = 0

    def _fake_create_connection(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionRefusedError
        return contextlib.nullcontext()

    proc = FakeServerProc()
    popen_calls = []

    def _fake_popen(argv, **kwargs):
        popen_calls.append(argv)
        return proc

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(socket, "create_connection", _fake_create_connection)
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(subprocess, "call", lambda argv: claude_exit_code)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(
        "introspect.cli._serve_log_path", lambda: tmp_path / "serve.log"
    )
    return proc, popen_calls


def test_claude_stops_spawned_server_on_exit(monkeypatch, tmp_path):
    """A server we auto-started is terminated once Claude Code exits."""
    proc, _ = _wire_autospawned_server_fakes(monkeypatch, tmp_path)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 0, result.output
    assert proc.terminated
    assert proc.wait_calls == 1
    assert "Stopping introspect server" in result.output


def test_claude_stops_spawned_server_even_when_claude_fails(monkeypatch, tmp_path):
    """Server cleanup runs regardless of Claude Code's exit code."""
    proc, _ = _wire_autospawned_server_fakes(monkeypatch, tmp_path, claude_exit_code=3)

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 3
    assert proc.terminated


def test_claude_keep_server_leaves_spawned_server_running(monkeypatch, tmp_path):
    """--keep-server skips the shutdown and tells the user where it lives."""
    proc, _ = _wire_autospawned_server_fakes(monkeypatch, tmp_path)

    result = runner.invoke(app, ["claude", "--keep-server"])

    assert result.exit_code == 0, result.output
    assert not proc.terminated
    flat_output = result.output.replace("\n", "")
    assert "left running" in flat_output
    assert "1234" in flat_output


def test_stop_server_escalates_to_kill_on_hang():
    """A server that ignores SIGTERM gets SIGKILL after the grace period."""
    proc = FakeServerProc(hang_on_term=True)

    _stop_server(cast("subprocess.Popen[bytes]", proc))

    assert proc.killed
    assert proc.wait_calls == 2  # graceful wait, then post-kill reap


def test_stop_server_noop_when_already_exited():
    """No signals are sent if the spawned server already exited."""
    proc = FakeServerProc(returncode=0)

    _stop_server(cast("subprocess.Popen[bytes]", proc))

    assert not proc.terminated
    assert not proc.killed


def test_finish_connected_session_leaves_existing_server_alone():
    _finish_connected_session(None, host="127.0.0.1", port=8347, keep_server=False)


def test_claude_errors_when_server_exits_during_start(monkeypatch, tmp_path):
    """If the spawned server process exits before the port opens, exit code is 1."""

    def _refuse(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: FakeServerProc(returncode=1)
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(
        "introspect.cli._serve_log_path", lambda: tmp_path / "serve.log"
    )

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 1
    # Rich may wrap long paths across lines; collapse whitespace before checking.
    flat_output = result.output.replace("\n", "")
    assert "serve.log" in flat_output


def test_claude_errors_when_server_start_times_out(monkeypatch, tmp_path):
    """If the server never becomes connectable within the timeout, exit code is 1."""

    def _refuse(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeServerProc())
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("introspect.cli.SERVER_START_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr("introspect.cli.SERVER_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(
        "introspect.cli._serve_log_path", lambda: tmp_path / "serve.log"
    )

    result = runner.invoke(app, ["claude"])

    assert result.exit_code == 1
    # Rich may wrap long paths across lines; collapse whitespace before checking.
    flat_output = result.output.replace("\n", "")
    assert "serve.log" in flat_output
    # Message should mention the server was left running
    assert "left running" in result.output
