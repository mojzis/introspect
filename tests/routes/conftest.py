"""Shared fixtures for tests/routes/ package."""

import os
from contextlib import contextmanager
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

SID = "01234567-abcd-abcd-abcd-0123456789ab"


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing."""
    lines = [
        make_user_message(
            SID,
            "u1",
            None,
            "2026-03-27T10:00:00.000Z",
            "Hello, help me with tests",
        ),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-03-27T10:00:01.000Z",
            [{"type": "text", "text": "Sure, I can help!"}],
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
        make_assistant_message(
            SID,
            "a2",
            "a1",
            "2026-03-27T10:00:02.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_test1",
                    "name": "Bash",
                    "input": {"command": "echo hello", "description": "test"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u2",
            "a2",
            "2026-03-27T10:00:03.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_test1",
                    "content": "hello\n",
                    "is_error": False,
                }
            ],
            tool_use_result={
                "stdout": "hello\n",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
            source_tool_uuid="a2",
        ),
        make_assistant_message(
            SID,
            "a3",
            "u2",
            "2026-03-27T10:00:04.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp1",
                    "name": "mcp__github__get_me",
                    "input": {},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3",
            "a3",
            "2026-03-27T10:00:05.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_mcp1",
                    "content": '{"login": "test"}',
                    "is_error": False,
                }
            ],
            tool_use_result={},
            source_tool_uuid="a3",
        ),
        # Read tool call (file inside project)
        make_assistant_message(
            SID,
            "a4",
            "u3",
            "2026-03-27T10:00:05.500Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/test/src/main.py"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3b",
            "a4",
            "2026-03-27T10:00:05.600Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read1",
                    "content": "file contents here",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "file contents here"},
            source_tool_uuid="a4",
        ),
        # Read tool call (file OUTSIDE project)
        make_assistant_message(
            SID,
            "a5",
            "u3b",
            "2026-03-27T10:00:05.700Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read2",
                    "name": "Read",
                    "input": {"file_path": "/home/user/other/config.yml"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3c",
            "a5",
            "2026-03-27T10:00:05.800Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read2",
                    "content": "config data",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "config data"},
            source_tool_uuid="a5",
        ),
        # Edit tool call
        make_assistant_message(
            SID,
            "a6",
            "u3c",
            "2026-03-27T10:00:05.900Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_edit1",
                    "name": "Edit",
                    "input": {
                        "file_path": "/tmp/test/src/main.py",
                        "old_string": "old",
                        "new_string": "new",
                    },
                }
            ],
        ),
        make_user_message(
            SID,
            "u3d",
            "a6",
            "2026-03-27T10:00:05.950Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_edit1",
                    "content": "ok",
                    "is_error": False,
                }
            ],
            tool_use_result={"content": "ok"},
            source_tool_uuid="a6",
        ),
        make_user_message(
            SID,
            "u4",
            "u3d",
            "2026-03-27T10:00:06.000Z",
            "<command-name>/commit</command-name>\nCommit my changes",
        ),
        # Sidechain user message — this is the prompt the main agent passed
        # to a subagent via the Task/Agent tool, NOT a human-typed prompt.
        make_user_message(
            SID,
            "s1",
            "a3",
            "2026-03-27T10:00:07.000Z",
            "Explore the database layer and report back findings.",
            is_sidechain=True,
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


@contextmanager
def _patched_client(tmp_path: Path, extra_env: dict[str, str] | None = None):
    """Context manager that yields a TestClient with materialized test data.

    ``extra_env`` overlays additional environment variables read by the app
    lifespan (e.g. ``INTROSPECT_HOST`` to exercise the SQL API bind gate).
    """
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"

    env = {
        "INTROSPECT_DB_PATH": str(db_path),
        "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
        # Point at a non-existent dir so route tests don't load real
        # ~/.codex/sessions data from the machine running them.
        "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
        "INTROSPECT_DAYS": "0",
        # The SQL API fails closed on an unset host; tests run as a loopback
        # bind unless a case overrides it via ``extra_env``.
        "INTROSPECT_HOST": "127.0.0.1",
        **(extra_env or {}),
    }
    with (
        patch.dict(os.environ, env),
        TestClient(app) as client,
    ):
        yield client
