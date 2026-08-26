"""Shared test fixtures and helpers."""

import json
from pathlib import Path

import duckdb
import pytest

LOCK_ERROR_MESSAGE = (
    'IO Error: Could not set lock on file "/tmp/fake.duckdb": '
    "Conflicting lock is held in /tmp/other_proc."
)


@pytest.fixture
def mock_locked_db(monkeypatch):
    """Patch duckdb.connect to simulate a 'DB locked by another process' error."""

    def _raise_lock(*args, **kwargs):
        raise duckdb.IOException(LOCK_ERROR_MESSAGE)

    monkeypatch.setattr("introspect.db.duckdb.connect", _raise_lock)
    return _raise_lock


@pytest.fixture(scope="session", autouse=True)
def _prewarm_fts_cache():
    """Detect FTS availability once per session.

    ``introspect.search.fts_available`` calls ``INSTALL fts``, which contacts
    ``extensions.duckdb.org``. In offline/sandboxed environments each attempt
    takes ~80s to fail due to DNS timeouts. Detecting availability once per
    session (instead of once per test) keeps the suite fast.

    Uses a localhost custom repository for the INSTALL probe so the fallback
    fails immediately (no network wait) if the extension isn't already on disk.
    """
    from introspect.search import _fts_cache  # noqa: PLC0415

    conn = duckdb.connect(":memory:")
    try:
        # Fast path: extension already on disk, LOAD succeeds without network
        already_loaded = False
        try:
            conn.execute("LOAD fts")
            already_loaded = True
        except duckdb.IOException:
            pass
        if already_loaded:
            _fts_cache["available"] = True
            return
        # Fallback: try INSTALL with a localhost repo so it fails fast offline
        conn.execute("SET custom_extension_repository = 'http://127.0.0.1:1'")
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            _fts_cache["available"] = True
        except (duckdb.IOException, duckdb.CatalogException, duckdb.HTTPException):
            _fts_cache["available"] = False
    finally:
        conn.close()


def make_user_message(
    session_id: str,
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content,
    *,
    tool_use_result=None,
    source_tool_uuid: str | None = None,
    is_sidechain: bool = False,
) -> dict:
    """Build a user-type JSONL record."""
    record = {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": is_sidechain,
        "cwd": "/tmp/test",
        "version": "2.1.0",
        "entrypoint": "cli",
        "gitBranch": "main",
        "message": {"role": "user", "content": content},
    }
    if tool_use_result is not None:
        record["toolUseResult"] = tool_use_result
    if source_tool_uuid is not None:
        record["sourceToolAssistantUUID"] = source_tool_uuid
    return record


def make_assistant_message(
    session_id: str,
    uuid: str,
    parent_uuid: str,
    timestamp: str,
    content: list,
    *,
    model: str = "claude-opus-4-6",
    msg_id: str = "msg1",
    usage: dict | None = None,
    is_sidechain: bool = False,
) -> dict:
    """Build an assistant-type JSONL record."""
    message: dict = {
        "role": "assistant",
        "model": model,
        "id": msg_id,
        "content": content,
    }
    if usage is not None:
        message["usage"] = usage
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": is_sidechain,
        "cwd": "/tmp/test",
        "version": "2.1.0",
        "entrypoint": "cli",
        "gitBranch": "main",
        "requestId": f"req-{uuid}",
        "message": message,
    }


def make_attachment_message(
    session_id: str,
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    attachment: dict,
) -> dict:
    """Build an ``type='attachment'`` JSONL record.

    These carry harness-injected context (CLAUDE.md auto-load, @-file
    expansions, the skill menu, MCP instructions, hook output) in the
    ``attachment`` object; ``session_context_loads`` reads them from
    ``raw_data``.
    """
    return {
        "type": "attachment",
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "cwd": "/tmp/test",
        "version": "2.1.0",
        "entrypoint": "cli",
        "gitBranch": "main",
        "attachment": attachment,
    }


def write_jsonl(tmp_dir: Path, session_id: str, lines: list[dict]) -> Path:
    """Write JSONL records to a test file and return the path."""
    jsonl_path = tmp_dir / "projects" / "test-project" / f"{session_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return jsonl_path


def glob_pattern(tmp_dir: Path) -> str:
    """Return the JSONL glob pattern for a test temp directory."""
    return str(tmp_dir / "projects" / "**" / "*.jsonl")


def write_codex_rollout(tmp_dir: Path, session_id: str, lines: list[dict]) -> Path:
    """Write Codex rollout JSONL records (Codex's directory layout, not Claude's)."""
    jsonl_path = (
        tmp_dir / "sessions" / "2026" / "08" / "20" / f"rollout-{session_id}.jsonl"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return jsonl_path


def codex_glob_pattern(tmp_dir: Path) -> str:
    """Return the Codex rollout glob pattern for a test temp directory."""
    return str(tmp_dir / "sessions" / "**" / "*.jsonl")


def codex_record(
    record_type: str, payload: dict, timestamp: str = "2026-08-20T10:00:00Z"
) -> dict:
    """Build one ``{timestamp, type, payload}`` Codex rollout record."""
    return {"timestamp": timestamp, "type": record_type, "payload": payload}


def codex_session_meta(
    session_id: str,
    *,
    cwd: str = "/tmp/codex-test",
    cli_version: str = "0.145.0",
    originator: str = "codex-tui",
    model_provider: str = "openai",
    thread_source: str = "user",
    git_branch: str = "main",
) -> dict:
    """Build a Codex ``session_meta`` payload (always line 1 of a rollout)."""
    return {
        "session_id": session_id,
        "id": session_id,
        "cwd": cwd,
        "originator": originator,
        "cli_version": cli_version,
        "source": "cli",
        "thread_source": thread_source,
        "model_provider": model_provider,
        "git": {"commit_hash": "abc123", "branch": git_branch, "repository_url": ""},
    }


def codex_turn_context(turn_id: str, model: str = "gpt-5.6-terra") -> dict:
    """Build a Codex ``turn_context`` payload."""
    return {"turn_id": turn_id, "model": model, "cwd": "/tmp/codex-test"}


def write_codex_session(tmp_dir: Path, session_id: str) -> Path:
    """Write a minimal single-turn Codex rollout fixture."""
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {"type": "user_message", "message": "please fix", "text_elements": []},
        ),
    ]
    return write_codex_rollout(tmp_dir, session_id, lines)
