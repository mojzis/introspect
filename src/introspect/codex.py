"""Transcode OpenAI Codex CLI rollout JSONL into ``raw_messages``-shaped rows.

Pure leaf module: no FastAPI, no DB, no handler imports. Mirrors the posture
of ``sql_fragments.py`` / ``projects.py`` so ``db.py`` can import it without
inverting layering.

A rollout file is a sequence of ``{timestamp, type, payload}`` records
(``session_meta``, ``turn_context``, ``response_item``, ``event_msg``).
:func:`transcode_rollout` turns one file into a list of dicts shaped like the
15 columns of ``db._RAW_MESSAGES_COLUMNS`` plus ``provider`` / ``harness``,
with an Anthropic-shaped ``message`` JSON payload so every downstream view
(``tool_calls``, ``file_reads``, ``assistant_message_costs``, ...) works
unmodified.

Known lossiness, by design, not bugs to fix later:

1. **``Bash`` alias for a JS sandbox.** Codex's ``exec_command`` runs inside a
   JavaScript harness, not a real shell tool. Aliasing it to ``Bash`` lets
   every Bash-shaped rollup (``bash.py``, ``trajectory.py``) work, but the
   ``command`` string is JS-embedded shell text, not a first-class tool
   call the way Claude's ``Bash`` is.
2. **One ``tool_result`` per batched ``Promise.all`` exec.** A single
   ``custom_tool_call`` can run several ``tools.exec_command(...)`` calls in
   parallel via ``await Promise.all([...])``. Each call site gets its own
   synthesized ``tool_use`` block (``{call_id}#0``, ``{call_id}#1``, ...),
   but the script returns one output blob, so only ``#0`` receives a
   ``tool_result`` — the rest are permanently unmatched.
3. **Empty ``thinking`` text.** Codex ``reasoning`` items don't expose their
   reasoning content in the rollout the way Claude's extended-thinking blocks
   do; the emitted ``thinking`` block always has empty text.
4. **``files_read`` era asymmetry.** 0.147+ rollouts carry
   ``item_completed.CommandExecution.parsed_cmd`` with typed ``read`` entries
   and a resolved ``path`` — reliable. <= 0.145 rollouts have no such
   structure; file reads are inferred from a shell-command heuristic
   (``sed -n`` / ``cat`` / ``head`` / ``tail``) ported from
   ``trajectory.py``'s Bash classifier, which is best-effort and will
   undercount relative to 0.147+ sessions.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROVIDER = "openai"
HARNESS = "codex"

_RAW_MESSAGES_KEYS = (
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
)

# Codex function_call names that all mean "talk to a subagent" — aliased to
# Claude's Task tool so /subagents-adjacent rollups don't crash on them.
_AGENT_FUNCTIONS = {
    "spawn_agent",
    "wait_agent",
    "send_message",
    "list_agents",
    "interrupt_agent",
    "followup_task",
}

# tools.X(...) call-site name -> (Claude tool_name, arg-mapping)
_TOOL_ALIAS_CALL_SITE = re.compile(r"tools\.([a-zA-Z0-9_]+)\s*\(")

# Shell-command heuristic for <= 0.145 rollouts (no parsed_cmd). Ordered,
# first match wins — mirrors trajectory.py's BASH_PATTERNS shape.
_SHELL_READ_CMD = re.compile(r"^\s*(sed -n|cat|head|tail)\b")
_SHELL_SEARCH_CMD = re.compile(r"^\s*(rg|grep)\b")
_SHELL_LIST_CMD = re.compile(r"^\s*ls\b")


class _ParseStats:
    """Per-file counters, surfaced via WARNING logs, never silent."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.js_arg_parse_failures = 0

    def log_summary(self) -> None:
        if self.js_arg_parse_failures:
            log.warning(
                "codex adapter: %d JS argument parse failure(s) in %s",
                self.js_arg_parse_failures,
                self.file_path,
            )


def _balanced_brace_scan(text: str, start: int) -> str | None:
    """Scan a balanced ``{...}`` (or ``(...)``-wrapped call arg) from ``start``.

    ``start`` points at the opening paren of a ``tools.name(...)`` call.
    Returns the raw text of the first top-level argument literal, or
    ``None`` if braces never balance.
    """
    depth = 0
    i = start
    n = len(text)
    began = False
    arg_start = None
    while i < n:
        ch = text[i]
        if ch in "({[":
            if not began and ch == "(":
                began = True
                i += 1
                # skip whitespace to the first real argument character
                while i < n and text[i].isspace():
                    i += 1
                arg_start = i
                continue
            depth += 1
        elif ch in ")}]":
            if ch == ")" and depth == 0 and began:
                if arg_start is None:
                    return None
                return text[arg_start:i].rstrip().rstrip(",").rstrip()
            depth -= 1
        i += 1
    return None


def _js_object_to_dict(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of a JS object literal (unquoted keys allowed).

    Not strict JSON: ``{workdir: "/x", cmd: "ls"}`` is valid input. Quotes
    bare identifier keys, then falls back to ``json.loads``.
    """
    raw = raw.strip()
    if not raw:
        return None
    quoted = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', raw)
    try:
        parsed = json.loads(quoted)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_call_sites(js_source: str) -> list[tuple[str, dict[str, Any] | None]]:
    """Extract ``tools.X(...)`` call sites: (tool name, parsed args or None)."""
    sites = []
    for m in _TOOL_ALIAS_CALL_SITE.finditer(js_source):
        name = m.group(1)
        raw_arg = _balanced_brace_scan(js_source, m.end() - 1)
        parsed = _js_object_to_dict(raw_arg) if raw_arg else None
        sites.append((name, parsed))
    return sites


def _alias_tool_call(
    name: str, args: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    """Map one ``tools.X`` call site onto a Claude tool_name / tool_input pair."""
    args = args or {}
    if name == "exec_command":
        return "Bash", {"command": args.get("cmd", "")}
    if name == "web__run":
        return "WebSearch", {"query": args.get("query") or args.get("q") or ""}
    if name == "view_image":
        return "Read", {"file_path": args.get("path") or args.get("image_path") or ""}
    if name == "update_plan":
        return "TodoWrite", args
    if name.startswith("mcp__"):
        return name, args
    # apply_patch and any unrecognized tool: pass through with raw args.
    return name, args


def _classify_shell_cmd(cmd: str) -> str | None:
    """Ported shell heuristic for <= 0.145 file-read/search/list classification."""
    if _SHELL_READ_CMD.match(cmd):
        return "read"
    if _SHELL_SEARCH_CMD.match(cmd):
        return "search"
    if _SHELL_LIST_CMD.match(cmd):
        return "list"
    return None


def _extract_path_from_cmd(cmd: str) -> str | None:
    """Best-effort last-token path extraction from a classified shell command."""
    tokens = [t.strip("'\"") for t in cmd.split() if not t.startswith("-")]
    return tokens[-1] if len(tokens) > 1 else None


def _new_row(  # noqa: PLR0913 -- building the full 15-column row is inherently wide
    *,
    row_type: str,
    role: str,
    content: list[dict[str, Any]],
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    session_id: str,
    file_path: str,
    is_sidechain: bool,
    cwd: str,
    version: str,
    entrypoint: str,
    git_branch: str,
    model: str | None,
    message_id: str | None,
    tool_use_result: Any = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": role, "content": content}
    if row_type == "assistant":
        message["id"] = message_id or uuid
        message["model"] = model
    return {
        "file_path": file_path,
        "type": row_type,
        "timestamp": timestamp,
        "session_id": session_id,
        "uuid": uuid,
        "parent_uuid": parent_uuid,
        "is_sidechain": is_sidechain,
        "cwd": cwd,
        "version": version,
        "entrypoint": entrypoint,
        "git_branch": git_branch,
        "role": role,
        "model": model if row_type == "assistant" else None,
        "message": message,
        "tool_use_result": tool_use_result,
        "provider": PROVIDER,
        "harness": HARNESS,
    }


class _FileState:
    """Mutable per-file state threaded through record handling."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.session_id: str = ""
        self.cwd: str = ""
        self.version: str = ""
        self.entrypoint: str = ""
        self.git_branch: str = ""
        self.is_sidechain: bool = False
        self.turn_model: dict[str, str] = {}
        self.last_model: str | None = None
        self.prev_uuid: str | None = None
        self.token_buffer: list[dict[str, Any]] = []
        self.last_total_tokens: int | None = None
        self.rows: list[dict[str, Any]] = []
        self.stats = _ParseStats(file_path)

    def next_uuid(self, line_index: int, sub_index: int = 0) -> str:
        base = f"{self.session_id}:{line_index}"
        return base if sub_index == 0 else f"{base}:{sub_index}"

    def resolve_model(self, turn_id: str | None) -> str | None:
        if turn_id and turn_id in self.turn_model:
            return self.turn_model[turn_id]
        return self.last_model

    def emit(self, row: dict[str, Any]) -> None:
        row["parent_uuid"] = self.prev_uuid
        self.prev_uuid = row["uuid"]
        self.rows.append(row)
        if row["type"] == "assistant":
            self.token_buffer.append(row)

    def attach_usage(self, last_usage: dict[str, Any]) -> None:
        if not self.token_buffer:
            return
        target = self.token_buffer[0]
        input_total = int(last_usage.get("input_tokens", 0) or 0)
        cached = int(last_usage.get("cached_input_tokens", 0) or 0)
        target["message"]["usage"] = {
            "input_tokens": input_total - cached,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": int(
                last_usage.get("cache_write_input_tokens", 0) or 0
            ),
            "output_tokens": int(last_usage.get("output_tokens", 0) or 0),
        }
        self.token_buffer.clear()


def _handle_session_meta(state: _FileState, payload: dict[str, Any]) -> None:
    state.session_id = payload.get("session_id", "")
    state.cwd = payload.get("cwd", "")
    state.version = payload.get("cli_version", "")
    state.entrypoint = payload.get("originator", "")
    git = payload.get("git") or {}
    state.git_branch = git.get("branch", "")
    state.is_sidechain = payload.get("thread_source") == "subagent"


def _handle_turn_context(state: _FileState, payload: dict[str, Any]) -> None:
    model = payload.get("model")
    turn_id = payload.get("turn_id")
    if model:
        state.last_model = model
        if turn_id:
            state.turn_model[turn_id] = model


def _item_turn_id(item: dict[str, Any]) -> str | None:
    meta = item.get("internal_chat_message_metadata_passthrough") or {}
    return meta.get("turn_id")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "".join(parts)


def _handle_message_item(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    role = item.get("role")
    if role in ("user", "developer"):
        return  # harness injection / mode boilerplate, not human input
    if role != "assistant":
        return
    model = state.resolve_model(_item_turn_id(item))
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=[{"type": "text", "text": _message_text(item.get("content"))}],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=model,
        message_id=item.get("id"),
    )
    state.emit(row)


def _handle_reasoning_item(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    model = state.resolve_model(_item_turn_id(item))
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=[{"type": "thinking", "thinking": ""}],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=model,
        message_id=item.get("id"),
    )
    state.emit(row)


def _handle_custom_tool_call(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    call_id = item.get("call_id") or item.get("id") or ""
    js_source = item.get("input", "")
    sites = _extract_call_sites(js_source)
    model = state.resolve_model(_item_turn_id(item))
    blocks = []
    for idx, (name, args) in enumerate(sites):
        if args is None:
            state.stats.js_arg_parse_failures += 1
            log.warning(
                "codex adapter: failed to parse JS args for tools.%s(...) in %s",
                name,
                state.file_path,
            )
        tool_name, tool_input = _alias_tool_call(name, args)
        blocks.append(
            {
                "type": "tool_use",
                "id": f"{call_id}#{idx}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    if not blocks:
        return
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=blocks,
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=model,
        message_id=item.get("id"),
    )
    state.emit(row)

    # <= 0.145 file-read classification: heuristic over the exec_command's cmd.
    for name, args in sites:
        if name != "exec_command" or not args:
            continue
        cmd = str(args.get("cmd", ""))
        if _classify_shell_cmd(cmd) != "read":
            continue
        path = _extract_path_from_cmd(cmd)
        if not path:
            continue
        _emit_read_enrichment(state, path, timestamp, line_index)


def _emit_read_enrichment(
    state: _FileState, path: str, timestamp: str, line_index: int
) -> None:
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": f"{state.next_uuid(line_index)}-read",
                "name": "Read",
                "input": {"file_path": path},
            }
        ],
        uuid=state.next_uuid(line_index, sub_index=1),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=state.last_model,
        message_id=None,
    )
    state.emit(row)


def _handle_custom_tool_call_output(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    call_id = item.get("call_id") or ""
    output = item.get("output")
    text = _message_text(output) if isinstance(output, list) else str(output or "")
    row = _new_row(
        row_type="user",
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": f"{call_id}#0",
            }
        ],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=None,
        message_id=None,
        tool_use_result=text,
    )
    state.emit(row)


def _handle_function_call(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    name = item.get("name", "")
    call_id = item.get("call_id") or item.get("id") or ""
    raw_args = item.get("arguments")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except (json.JSONDecodeError, ValueError):
        args = {}
    if name in _AGENT_FUNCTIONS:
        tool_name, tool_input = "Task", {"description": json.dumps(args)}
    else:
        tool_name, tool_input = name, args
    model = state.resolve_model(_item_turn_id(item))
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": call_id,
                "name": tool_name,
                "input": tool_input,
            }
        ],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=model,
        message_id=item.get("id"),
    )
    state.emit(row)


def _handle_function_call_output(
    state: _FileState, item: dict[str, Any], timestamp: str, line_index: int
) -> None:
    call_id = item.get("call_id") or ""
    output = item.get("output")
    row = _new_row(
        row_type="user",
        role="user",
        content=[{"type": "tool_result", "tool_use_id": call_id}],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=None,
        message_id=None,
        tool_use_result=output,
    )
    state.emit(row)


_RESPONSE_ITEM_HANDLERS = {
    "message": _handle_message_item,
    "reasoning": _handle_reasoning_item,
    "custom_tool_call": _handle_custom_tool_call,
    "custom_tool_call_output": _handle_custom_tool_call_output,
    "function_call": _handle_function_call,
    "function_call_output": _handle_function_call_output,
}


def _handle_response_item(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    handler = _RESPONSE_ITEM_HANDLERS.get(payload.get("type", ""))
    if handler:
        handler(state, payload, timestamp, line_index)


_PATCH_CHANGE_TOOL = {
    "add": "Write",
    "update": "Edit",
    "move": "Edit",
    "delete": "Edit",
}


def _handle_patch_apply_end(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    call_id = payload.get("call_id", "")
    changes = payload.get("changes") or {}
    blocks = []
    for idx, (path, change) in enumerate(changes.items()):
        change_type = (
            change.get("type", "update") if isinstance(change, dict) else "update"
        )
        tool_name = _PATCH_CHANGE_TOOL.get(change_type, "Edit")
        blocks.append(
            {
                "type": "tool_use",
                "id": f"{call_id}#{idx}",
                "name": tool_name,
                "input": {"file_path": path},
            }
        )
    if not blocks:
        return
    row = _new_row(
        row_type="assistant",
        role="assistant",
        content=blocks,
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=state.last_model,
        message_id=None,
    )
    state.emit(row)


def _handle_item_completed(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    """0.147+ enrichment only: extract typed file-read entries, never duplicate."""
    item = payload.get("item") or {}
    if item.get("item_type") != "CommandExecution":
        return
    for entry in item.get("parsed_cmd") or []:
        if not isinstance(entry, dict) or entry.get("type") != "read":
            continue
        path = entry.get("path")
        if path:
            _emit_read_enrichment(state, path, timestamp, line_index)


def _handle_user_message(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    text = payload.get("message", "")
    command = None
    for elem in payload.get("text_elements") or []:
        placeholder = elem.get("placeholder", "") if isinstance(elem, dict) else ""
        if placeholder.startswith("$"):
            command = "/" + placeholder[1:]
            break
    if command:
        text = f"<command-name>{command}</command-name>\n{text}"
    row = _new_row(
        row_type="user",
        role="user",
        content=[{"type": "text", "text": text}],
        uuid=state.next_uuid(line_index),
        parent_uuid=state.prev_uuid,
        timestamp=timestamp,
        session_id=state.session_id,
        file_path=state.file_path,
        is_sidechain=state.is_sidechain,
        cwd=state.cwd,
        version=state.version,
        entrypoint=state.entrypoint,
        git_branch=state.git_branch,
        model=None,
        message_id=None,
    )
    state.emit(row)


def _handle_token_count(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    info = payload.get("info") or {}
    total = info.get("total_token_usage") or {}
    total_tokens = total.get("total_tokens")
    if total_tokens is not None:
        if (
            state.last_total_tokens is not None
            and total_tokens <= state.last_total_tokens
        ):
            return  # non-monotonic guard: skip a duplicated/rolled-back emission
        state.last_total_tokens = total_tokens
    last_usage = info.get("last_token_usage") or {}
    state.attach_usage(last_usage)


_EVENT_MSG_HANDLERS = {
    "user_message": _handle_user_message,
    "token_count": _handle_token_count,
    "patch_apply_end": _handle_patch_apply_end,
    "item_completed": _handle_item_completed,
}


def _handle_event_msg(
    state: _FileState, payload: dict[str, Any], timestamp: str, line_index: int
) -> None:
    handler = _EVENT_MSG_HANDLERS.get(payload.get("type", ""))
    if handler:
        handler(state, payload, timestamp, line_index)


_TOP_LEVEL_HANDLERS = {
    "session_meta": _handle_session_meta,
    "turn_context": _handle_turn_context,
}


def transcode_rollout(path: str | Path) -> list[dict[str, Any]]:
    """Transcode one Codex rollout JSONL file into ``raw_messages``-shaped rows.

    Single pass over the file; no whole-corpus buffering. Returns a list of
    dicts with the 15 keys of ``db._RAW_MESSAGES_COLUMNS`` plus ``provider``
    and ``harness``.
    """
    path = Path(path)
    state = _FileState(str(path))
    with path.open() as f:
        for line_index, raw_line in enumerate(f):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            timestamp = record.get("timestamp", "")
            record_type = record.get("type")
            payload = record.get("payload") or {}
            if record_type in _TOP_LEVEL_HANDLERS:
                _TOP_LEVEL_HANDLERS[record_type](state, payload)
            elif record_type == "response_item":
                _handle_response_item(state, payload, timestamp, line_index)
            elif record_type == "event_msg":
                _handle_event_msg(state, payload, timestamp, line_index)
    state.stats.log_summary()
    return state.rows
