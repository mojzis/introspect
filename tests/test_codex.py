"""Tests for the Codex rollout adapter (src/introspect/codex.py)."""

import json

import pytest

from introspect.codex import _alias_tool_call, transcode_rollout

from .conftest import (
    codex_record,
    codex_session_meta,
    codex_turn_context,
    write_codex_rollout,
)


def _tool_uses(rows: list[dict], name: str | None = None) -> list[dict]:
    """``tool_use`` content blocks across all assistant rows, optionally
    filtered to one tool ``name``."""
    return [
        b
        for r in rows
        if r["type"] == "assistant"
        for b in r["message"]["content"]
        if b.get("type") == "tool_use" and (name is None or b.get("name") == name)
    ]


def _exec_call(call_id: str, cmd: str, turn_id: str = "turn-1") -> dict:
    return codex_record(
        "response_item",
        {
            "type": "custom_tool_call",
            "id": f"item-{call_id}",
            "call_id": call_id,
            "name": "exec",
            "input": (
                f"const r = await tools.exec_command("
                f'{{"cmd":"{cmd}","workdir":"/tmp"}});'
            ),
            "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
        },
    )


def _exec_output(call_id: str, text: str = "done") -> dict:
    return codex_record(
        "response_item",
        {
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": [{"type": "input_text", "text": text}],
        },
    )


def _token_count(last: dict, total_tokens: int) -> dict:
    return codex_record(
        "event_msg",
        {
            "type": "token_count",
            "info": {
                "last_token_usage": last,
                "total_token_usage": {"total_tokens": total_tokens},
            },
        },
    )


def test_pre_0_147_sidecars(tmp_path):
    """<= 0.145 era: patch_apply_end / mcp_tool_call_end / web_search_end sidecars."""
    session_id = "sess-old"
    lines = [
        codex_record(
            "session_meta", codex_session_meta(session_id, cli_version="0.145.0")
        ),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {
                "type": "user_message",
                "message": "please fix the bug",
                "text_elements": [],
            },
        ),
        _exec_call("call-1", "cat src/foo.py"),
        _exec_output("call-1"),
        codex_record(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "item-patch",
                "call_id": "call-patch",
                "name": "exec",
                "input": (
                    'const r = await tools.apply_patch({"input":"*** Begin Patch\\n"});'
                ),
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "patch_apply_end",
                "call_id": "call-patch",
                "changes": {"src/foo.py": {"type": "update"}},
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "mcp_tool_call_end",
                "call_id": "call-mcp",
                "server": "github",
                "tool": "search",
                "arguments": {"q": "bug"},
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "web_search_end",
                "call_id": "call-web",
                "query": "duckdb json_extract",
                "results": [],
            },
        ),
        _token_count(
            {
                "input_tokens": 1000,
                "cached_input_tokens": 200,
                "cache_write_input_tokens": 0,
                "output_tokens": 50,
            },
            1050,
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    assert rows, "expected at least one row"
    assert all(set(r) >= {"provider", "harness", "message"} for r in rows)
    assert all(r["provider"] == "openai" and r["harness"] == "codex" for r in rows)

    bash_calls = _tool_uses(rows, "Bash")
    assert bash_calls
    assert bash_calls[0]["input"] == {"command": "cat src/foo.py"}

    # heuristic file-read enrichment for `cat` on <= 0.145
    read_calls = _tool_uses(rows, "Read")
    assert any(b["input"].get("file_path") == "src/foo.py" for b in read_calls)

    # apply_patch sidecar -> Edit alias with real file path
    edit_calls = _tool_uses(rows, "Edit")
    assert edit_calls
    assert edit_calls[0]["input"] == {"file_path": "src/foo.py"}

    # message.id non-null and unique per assistant row
    assistant_rows = [r for r in rows if r["type"] == "assistant"]
    ids = [r["message"]["id"] for r in assistant_rows]
    assert all(i is not None for i in ids)
    assert len(ids) == len(set(ids))

    # token usage attached to the first buffered assistant row
    usage_rows = [r for r in assistant_rows if "usage" in r["message"]]
    assert len(usage_rows) == 1
    usage = usage_rows[0]["message"]["usage"]
    assert usage["input_tokens"] == 800
    assert usage["cache_read_input_tokens"] == 200
    assert usage["output_tokens"] == 50

    # token reconciliation: attributed usage reconstructs the fixture's total
    reconstructed = (
        usage["input_tokens"]
        + usage["cache_read_input_tokens"]
        + usage["output_tokens"]
    )
    assert reconstructed == 1050


def test_0_147_item_completed_enrichment(tmp_path):
    """0.147+ era: item_completed.CommandExecution.parsed_cmd read entries."""
    session_id = "sess-new"
    lines = [
        codex_record(
            "session_meta", codex_session_meta(session_id, cli_version="0.147.0")
        ),
        codex_record("turn_context", codex_turn_context("turn-1")),
        _exec_call("call-1", "sed -n '1,20p' src/bar.py"),
        codex_record(
            "event_msg",
            {
                "type": "item_completed",
                "item": {
                    "item_type": "CommandExecution",
                    "command": ["sed", "-n", "1,20p", "src/bar.py"],
                    "parsed_cmd": [{"type": "read", "path": "src/bar.py"}],
                },
            },
        ),
        _exec_output("call-1"),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    read_calls = _tool_uses(rows, "Read")
    assert any(b["input"].get("file_path") == "src/bar.py" for b in read_calls)


def test_promise_all_multi_command_batch(tmp_path):
    """A single custom_tool_call batching several tools.exec_command via Promise.all."""
    session_id = "sess-batch"
    js = (
        "const r = await Promise.all(["
        'tools.exec_command({"cmd":"ls src"}),'
        'tools.exec_command({"cmd":"pytest -q"})'
        "]);"
    )
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "item-batch",
                "call_id": "call-batch",
                "name": "exec",
                "input": js,
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        _exec_output("call-batch"),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    bash_blocks = _tool_uses(rows, "Bash")
    assert len(bash_blocks) == 2
    assert bash_blocks[0]["id"] == "call-batch#0"
    assert bash_blocks[1]["id"] == "call-batch#1"

    # only the first call site gets a tool_result — known lossiness
    tool_results = [
        b
        for r in rows
        if r["type"] == "user"
        for b in r["message"]["content"]
        if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "call-batch#0"


def test_non_monotonic_token_count_guard(tmp_path):
    """A total_token_usage.total_tokens that doesn't strictly increase is skipped."""
    session_id = "sess-guard"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        _exec_call("call-1", "echo one"),
        _token_count(
            {
                "input_tokens": 500,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 10,
            },
            510,
        ),
        _exec_call("call-2", "echo two"),
        # rolled-back / duplicated total — must be skipped, not attributed
        _token_count(
            {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 5,
            },
            400,
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    assistant_rows = [r for r in rows if r["type"] == "assistant"]
    usage_rows = [r for r in assistant_rows if "usage" in r["message"]]
    # only the first (monotonic) token_count attributed usage
    assert len(usage_rows) == 1
    assert usage_rows[0]["message"]["usage"]["output_tokens"] == 10

    # the second call's row is still buffered/unattributed, not silently dropped
    second_call_row = next(
        r
        for r in assistant_rows
        for b in r["message"]["content"]
        if b.get("type") == "tool_use" and b["input"].get("command") == "echo two"
    )
    assert "usage" not in second_call_row["message"]


def test_js_arg_parse_failure_emits_empty_input_and_logs(tmp_path, caplog):
    """A JS arg literal that fails to parse still emits tool_use with empty input."""
    session_id = "sess-badjs"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "item-bad",
                "call_id": "call-bad",
                "name": "exec",
                "input": "const r = await tools.exec_command(someVariable);",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    with caplog.at_level("WARNING", logger="introspect.codex"):
        rows = transcode_rollout(path)

    tool_use_blocks = _tool_uses(rows)
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Bash"
    assert tool_use_blocks[0]["input"] == {"command": ""}
    assert any("parse" in rec.message.lower() for rec in caplog.records)


def test_function_call_arg_parse_failure_emits_empty_input_and_logs(tmp_path, caplog):
    """A malformed function_call.arguments string still emits tool_use, empty input."""
    session_id = "sess-badfnargs"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "function_call",
                "id": "item-bad-fn",
                "call_id": "call-bad-fn",
                "name": "shell",
                "arguments": "{not valid json",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    with caplog.at_level("WARNING", logger="introspect.codex"):
        rows = transcode_rollout(path)

    tool_use_blocks = _tool_uses(rows)
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["input"] == {}
    assert any("parse" in rec.message.lower() for rec in caplog.records)


def test_message_id_non_null_and_unique(tmp_path):
    """Assistant message.id falls back to the synthesized uuid when absent."""
    session_id = "sess-ids"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
                # no "id" field
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        codex_record(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "id": "resp-42",
                "content": [{"type": "output_text", "text": "bye"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    ids = [r["message"]["id"] for r in rows if r["type"] == "assistant"]
    assert all(i is not None for i in ids)
    assert len(ids) == len(set(ids))
    assert "resp-42" in ids


def test_skips_environment_and_developer_messages(tmp_path):
    session_id = "sess-skip"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>"}],
            },
        ),
        codex_record(
            "response_item",
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "permissions..."}],
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    assert rows == []


def test_row_shape_matches_raw_messages_columns(tmp_path):
    session_id = "sess-shape"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {"type": "user_message", "message": "hello", "text_elements": []},
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    expected_keys = {
        "file_path",
        "type",
        "timestamp",
        "session_id",
        "uuid",
        "parent_uuid",
        "is_sidechain",
        "cwd",
        "version",
        "entrypoint",
        "git_branch",
        "role",
        "model",
        "message",
        "tool_use_result",
        "provider",
        "harness",
    }
    assert set(rows[0]) == expected_keys
    # message payload must be JSON-serializable (Anthropic-shaped)
    json.dumps(rows[0]["message"])


def test_provider_and_harness_read_from_session_meta(tmp_path):
    """provider/harness come from session_meta, not a blind hardcoded constant."""
    session_id = "sess-azure"
    lines = [
        codex_record(
            "session_meta",
            codex_session_meta(session_id, model_provider="azure"),
        ),
        codex_record(
            "event_msg",
            {"type": "user_message", "message": "hello", "text_elements": []},
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    assert rows
    assert all(r["provider"] == "azure" for r in rows)
    assert all(r["harness"] == "codex" for r in rows)


def test_read_enrichment_batch_gets_distinct_uuids(tmp_path):
    """Multiple read enrichments from one source line don't collide on uuid/id."""
    session_id = "sess-multiread"
    js = (
        "const r = await Promise.all(["
        'tools.exec_command({"cmd":"cat a.py b.py"}),'
        'tools.exec_command({"cmd":"cat c.py"})'
        "]);"
    )
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "item-batch",
                "call_id": "call-batch",
                "name": "exec",
                "input": js,
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    read_calls = _tool_uses(rows, "Read")
    read_paths = {b["input"]["file_path"] for b in read_calls}
    assert read_paths == {"a.py", "b.py", "c.py"}

    ids = [r["message"]["id"] for r in rows if r["type"] == "assistant"]
    assert len(ids) == len(set(ids))
    uuids = [r["uuid"] for r in rows]
    assert len(uuids) == len(set(uuids))
    tool_use_ids = [b["id"] for b in read_calls]
    assert len(tool_use_ids) == len(set(tool_use_ids))


def test_extract_paths_from_cmd_ignores_non_path_tokens(tmp_path):
    """Pipe/redirect targets and non-path tokens aren't emitted as file_reads."""
    session_id = "sess-pipe"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        _exec_call("call-1", "cat foo.py | head -5"),
        _exec_call("call-2", "head -20 x.txt > out"),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    read_paths = {b["input"]["file_path"] for b in _tool_uses(rows, "Read")}
    assert read_paths == {"foo.py", "x.txt"}


def test_item_completed_file_change_emits_edit(tmp_path):
    """0.147+ FileChange item_completed enrichment, alternative to patch_apply_end."""
    session_id = "sess-filechange"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {
                "type": "item_completed",
                "item": {
                    "item_type": "FileChange",
                    "call_id": "call-fc",
                    "changes": {"src/baz.py": {"type": "update"}},
                },
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    edit_calls = _tool_uses(rows, "Edit")
    assert edit_calls
    assert edit_calls[0]["input"] == {"file_path": "src/baz.py"}


def test_apply_patch_call_site_not_double_counted(tmp_path):
    """tools.apply_patch call site is suppressed; only the sidecar Edit row appears."""
    session_id = "sess-nodup"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "item-patch",
                "call_id": "call-patch",
                "name": "exec",
                "input": (
                    'const r = await tools.apply_patch({"input":"*** Begin Patch\\n"});'
                ),
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "patch_apply_end",
                "call_id": "call-patch",
                "changes": {"src/foo.py": {"type": "update"}},
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    tool_names = [b["name"] for b in _tool_uses(rows)]
    assert tool_names.count("apply_patch") == 0
    assert tool_names.count("Edit") == 1


def test_command_prefix_synthesized_from_dollar_placeholder(tmp_path):
    session_id = "sess-cmd"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record(
            "event_msg",
            {
                "type": "user_message",
                "message": "run the jira create skill",
                "text_elements": [{"placeholder": "$jira-create"}],
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    assert len(rows) == 1
    text = rows[0]["message"]["content"][0]["text"]
    assert text.startswith("<command-name>/jira-create</command-name>")


@pytest.mark.parametrize(
    ("name", "args", "expected_tool_name", "expected_tool_input"),
    [
        ("exec_command", {"cmd": "ls -la"}, "Bash", {"command": "ls -la"}),
        ("web__run", {"query": "duckdb json"}, "WebSearch", {"query": "duckdb json"}),
        ("web__run", {"q": "duckdb json"}, "WebSearch", {"query": "duckdb json"}),
        (
            "view_image",
            {"path": "diagram.png"},
            "Read",
            {"file_path": "diagram.png"},
        ),
        (
            "view_image",
            {"image_path": "diagram.png"},
            "Read",
            {"file_path": "diagram.png"},
        ),
        (
            "update_plan",
            {"plan": [{"step": "1", "status": "in_progress"}]},
            "TodoWrite",
            {"plan": [{"step": "1", "status": "in_progress"}]},
        ),
        (
            "mcp__github__search",
            {"q": "bug"},
            "mcp__github__search",
            {"q": "bug"},
        ),
        ("wait", {"seconds": 5}, "wait", {"seconds": 5}),
    ],
)
def test_alias_tool_call(name, args, expected_tool_name, expected_tool_input):
    """Every Codex ``tools.X`` call site maps to the documented Claude
    tool_name/tool_input pair; unrecognized names (e.g. ``wait``, observed in
    the corpus) fall through unchanged rather than crashing."""
    tool_name, tool_input = _alias_tool_call(name, args)
    assert (tool_name, tool_input) == (expected_tool_name, expected_tool_input)


def test_reasoning_and_function_call_pair(tmp_path):
    """A ``reasoning`` item plus a ``function_call``/``function_call_output``
    pair both round-trip: reasoning becomes a thinking block, the agent
    function call is aliased to Task, and its output attaches as a tool_result."""
    session_id = "sess-reasoning"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "response_item",
            {
                "type": "reasoning",
                "id": "item-reasoning",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        codex_record(
            "response_item",
            {
                "type": "function_call",
                "id": "item-fn",
                "call_id": "call-fn",
                "name": "spawn_agent",
                "arguments": json.dumps({"task": "review the diff"}),
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"},
            },
        ),
        codex_record(
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "call-fn",
                "output": "agent finished",
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    thinking_blocks = [
        b
        for r in rows
        if r["type"] == "assistant"
        for b in r["message"]["content"]
        if b.get("type") == "thinking"
    ]
    assert thinking_blocks

    task_calls = _tool_uses(rows, "Task")
    assert len(task_calls) == 1
    assert task_calls[0]["id"] == "call-fn"

    tool_results = [
        b
        for r in rows
        if r["type"] == "user"
        for b in r["message"]["content"]
        if b.get("type") == "tool_result"
    ]
    assert any(b["tool_use_id"] == "call-fn" for b in tool_results)


def test_patch_apply_end_and_file_change_same_call_id_not_double_counted(tmp_path):
    """``patch_apply_end`` and ``item_completed.FileChange`` describing the
    same ``call_id`` must dedupe to exactly one Edit block, not two."""
    session_id = "sess-patch-dedup"
    lines = [
        codex_record("session_meta", codex_session_meta(session_id)),
        codex_record("turn_context", codex_turn_context("turn-1")),
        codex_record(
            "event_msg",
            {
                "type": "patch_apply_end",
                "call_id": "call-shared",
                "changes": {"src/foo.py": {"type": "update"}},
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "item_completed",
                "item": {
                    "item_type": "FileChange",
                    "call_id": "call-shared",
                    "changes": {"src/foo.py": {"type": "update"}},
                },
            },
        ),
    ]
    path = write_codex_rollout(tmp_path, session_id, lines)

    rows = transcode_rollout(path)

    edit_calls = _tool_uses(rows, "Edit")
    assert len(edit_calls) == 1
