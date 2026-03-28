"""Shared test fixtures and helpers."""

import json
from pathlib import Path


def make_user_message(
    session_id: str,
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content,
    *,
    tool_use_result=None,
    source_tool_uuid: str | None = None,
) -> dict:
    """Build a user-type JSONL record."""
    record = {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": False,
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
        "isSidechain": False,
        "cwd": "/tmp/test",
        "version": "2.1.0",
        "entrypoint": "cli",
        "gitBranch": "main",
        "requestId": f"req-{uuid}",
        "message": message,
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
