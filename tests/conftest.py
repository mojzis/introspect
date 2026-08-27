"""Shared test fixtures and helpers."""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from introspect.db import materialize_views

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


# ---------------------------------------------------------------------------
# Cache-TTL fixture builders
#
# Shared by ``test_cache_ttl``, ``test_cli`` and ``test_mcp_tools`` — token
# counts and gaps are round numbers so an expected cost can be worked out by
# hand from ``pricing._PRICING`` instead of snapshotted from the code.
# ---------------------------------------------------------------------------

TTL_MODEL = "claude-opus-4-6"
TTL_T0 = datetime.fromisoformat("2026-04-21T09:00:00+00:00")


def ttl_ts(moment: datetime) -> str:
    """Render a datetime the way Claude Code stamps its JSONL records."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def ttl_usage(
    *, read: int, create: int, ttl: str | None = "5m", inp: int = 10, out: int = 20
) -> dict:
    """Usage block with an optional nested 5m/1h split.

    ``ttl=None`` reproduces the legacy schema — ``cache_creation_input_tokens``
    with no ``cache_creation`` sub-object — which bills at the 5m rate.
    """
    usage: dict = {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
    }
    if ttl is not None:
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": create if ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": create if ttl == "1h" else 0,
        }
    return usage


def ttl_turn(
    session_id: str,
    n: int,
    at: datetime,
    *,
    prompt: str = "go",
    read: int,
    create: int,
    ttl: str | None = "5m",
    inp: int = 10,
    out: int = 20,
) -> list[dict]:
    """One user prompt plus the assistant reply it triggered.

    The first prompt carries a ``toolUseResult`` so ``read_json_auto`` infers
    that column; without it the raw-messages load fails to bind.
    """
    return [
        make_user_message(
            session_id,
            f"u{n}",
            f"a{n - 1}" if n > 1 else None,
            ttl_ts(at),
            prompt,
            tool_use_result={"content": "seed"} if n == 1 else None,
        ),
        make_assistant_message(
            session_id,
            f"a{n}",
            f"u{n}",
            ttl_ts(at + timedelta(seconds=1)),
            [{"type": "text", "text": f"reply {n}"}],
            model=TTL_MODEL,
            msg_id=f"msg{n}",
            usage=ttl_usage(read=read, create=create, ttl=ttl, inp=inp, out=out),
        ),
    ]


@contextmanager
def ttl_materialized(tmp_dir: Path, session_id: str, lines: list[dict]):
    """Write ``lines`` and yield an in-memory DB materialized over them."""
    write_jsonl(tmp_dir, session_id, lines)
    conn = duckdb.connect(":memory:")
    materialize_views(conn, glob_pattern(tmp_dir), 0, resolve_projects=False)
    try:
        yield conn
    finally:
        conn.close()


# ``host_guard`` (see introspect.api.main) only accepts loopback Host headers,
# which is what makes DNS rebinding against the local server fail.
# TestClient's default base URL is ``http://testserver``, so every test client
# has to speak as a local browser would or it gets a 400 before reaching a
# route.
LOOPBACK_BASE_URL = "http://127.0.0.1:8347"


def local_client(app) -> TestClient:
    """``TestClient`` with a loopback Host header."""
    return TestClient(app, base_url=LOOPBACK_BASE_URL)
