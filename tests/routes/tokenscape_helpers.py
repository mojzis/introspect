"""Shared tokenscape session builders used by both tokenscape test modules."""

from pathlib import Path

from ..conftest import make_assistant_message, make_user_message, write_jsonl


def _tokenscape_session_jsonl(
    tmp_dir: Path, session_id: str, model: str = "claude-sonnet-4-6"
) -> Path:
    """3-turn session with realistic cache usage for cost tie-out tests."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "first prompt",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_ts1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/big.py"},
                }
            ],
            model=model,
            msg_id="msg-ts-1",
            usage={
                "input_tokens": 4,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 10_000,
            },
        ),
        make_user_message(
            session_id,
            "u2",
            "a1",
            "2026-04-21T10:00:02.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ts1",
                    "content": "x" * 8_000,
                }
            ],
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:03.000Z",
            [{"type": "text", "text": "done reading"}],
            model=model,
            msg_id="msg-ts-2",
            usage={
                "input_tokens": 4,
                "output_tokens": 50,
                "cache_read_input_tokens": 10_000,
                "cache_creation_input_tokens": 2_100,
            },
        ),
        make_assistant_message(
            session_id,
            "a3",
            "u2",
            "2026-04-21T10:00:04.000Z",
            [{"type": "text", "text": "still here"}],
            model=model,
            msg_id="msg-ts-3",
            usage={
                "input_tokens": 4,
                "output_tokens": 10,
                "cache_read_input_tokens": 12_100,
                "cache_creation_input_tokens": 50,
            },
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def _tokenscape_subagent_session_jsonl(
    tmp_dir: Path, session_id: str, model: str = "claude-sonnet-4-6"
) -> Path:
    """Main chain launching one Task, plus the agent's sidechain thread."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "first prompt",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_task1",
                    "name": "Task",
                    "input": {
                        "description": "scan docs",
                        "prompt": "analyze the docs",
                        "subagent_type": "explore",
                    },
                }
            ],
            model=model,
            msg_id="msg-sub-1",
            usage={
                "input_tokens": 4,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 10_000,
            },
        ),
        # Sidechain thread: root prompt (Task prompt verbatim), a Read,
        # its 8k-char result, and a closing assistant turn.
        make_user_message(
            session_id,
            "sc-u1",
            None,
            "2026-04-21T10:00:02.000Z",
            "analyze the docs",
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sc-a1",
            "sc-u1",
            "2026-04-21T10:00:03.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_sc_read1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/doc.md"},
                }
            ],
            model=model,
            msg_id="msg-sub-sc1",
            usage={
                "input_tokens": 4,
                "output_tokens": 1_000,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 50_000,
            },
            is_sidechain=True,
        ),
        make_user_message(
            session_id,
            "sc-u2",
            "sc-a1",
            "2026-04-21T10:00:04.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_sc_read1",
                    "content": "x" * 8_000,
                }
            ],
            is_sidechain=True,
        ),
        make_assistant_message(
            session_id,
            "sc-a2",
            "sc-u2",
            "2026-04-21T10:00:05.000Z",
            [{"type": "text", "text": "docs analyzed"}],
            model=model,
            msg_id="msg-sub-sc2",
            usage={
                "input_tokens": 4,
                "output_tokens": 500,
                "cache_read_input_tokens": 50_000,
                "cache_creation_input_tokens": 21_000,
            },
            is_sidechain=True,
        ),
        make_user_message(
            session_id,
            "u2",
            "a1",
            "2026-04-21T10:00:06.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_task1",
                    "content": "agent finished",
                }
            ],
        ),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            "2026-04-21T10:00:07.000Z",
            [{"type": "text", "text": "all done"}],
            model=model,
            msg_id="msg-sub-2",
            usage={
                "input_tokens": 4,
                "output_tokens": 50,
                "cache_read_input_tokens": 10_000,
                "cache_creation_input_tokens": 500,
            },
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)
