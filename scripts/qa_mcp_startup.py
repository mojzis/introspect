"""Black-box smoke test for progressive standalone MCP startup.

The route is intentionally independent of pytest: it creates synthetic logs,
talks MCP over the real stdio transport, and removes the temporary workspace
when both cold and warm runs finish.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _message(session_id: str, uuid: str, parent: str | None, timestamp: str, role: str):
    content = (
        "synthetic recent session"
        if session_id.endswith("recent")
        else "synthetic older session"
    )
    message = {
        "role": role,
        "content": (content if role == "user" else [{"type": "text", "text": content}]),
    }
    if role == "assistant":
        message.update({"model": "claude-opus-4-6", "id": f"msg-{uuid}"})
    return {
        "type": role,
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": uuid,
        "parentUuid": parent,
        "isSidechain": False,
        "cwd": "/synthetic/project",
        "version": "2.1.0",
        "entrypoint": "cli",
        "gitBranch": "main",
        "toolUseResult": None,
        "message": message,
    }


def _write_fixture(root: Path) -> str:
    now = datetime.now(UTC).replace(microsecond=0)
    records: list[dict] = []
    for session_id, timestamp in (
        ("synthetic-recent", now),
        ("synthetic-older", now - timedelta(days=5)),
    ):
        user_id = f"{session_id}-u"
        records.extend(
            [
                _message(session_id, user_id, None, timestamp.isoformat(), "user"),
                _message(
                    session_id,
                    f"{session_id}-a",
                    user_id,
                    (timestamp + timedelta(seconds=1)).isoformat(),
                    "assistant",
                ),
            ]
        )
    log_path = root / "claude" / "projects" / "synthetic" / "sessions.jsonl"
    log_path.parent.mkdir(parents=True)
    with log_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return str(root / "claude" / "projects" / "**" / "*.jsonl")


def _result_text(result) -> str:
    """Extract text from the first MCP content block."""
    return str(getattr(result.content[0], "text", ""))


async def _run_once(root: Path, jsonl_glob: str) -> dict[str, object]:
    db_path = root / "introspect.duckdb"
    env = os.environ.copy()
    env.update(
        {
            "INTROSPECT_DB_PATH": str(db_path),
            "INTROSPECT_JSONL_GLOB": jsonl_glob,
            "INTROSPECT_CODEX_GLOB": str(root / "codex" / "**" / "*.jsonl"),
            "INTROSPECT_DAYS": "30",
            "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0",
            "INTROSPECT_VERSION_CHECK": "off",
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "introspect.cli", "mcp"],
        env=env,
    )
    started = time.perf_counter()
    with Path(os.devnull).open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                initialize_ms = round((time.perf_counter() - started) * 1000, 1)
                listed = await session.list_tools()
                list_ms = round((time.perf_counter() - started) * 1000, 1)
                first = await asyncio.wait_for(
                    session.call_tool("recent_sessions", {"n": 20}), timeout=5
                )
                first_text = _result_text(first)
                partial = "Partial data" in first_text
                loading = "Data loading" in first_text
                observed_loading = loading
                observed_partial = partial
                deadline = time.perf_counter() + 30
                final_text = first_text
                while (
                    "synthetic-older" not in final_text
                    and time.perf_counter() < deadline
                ):
                    await asyncio.sleep(0.05)
                    result = await session.call_tool("recent_sessions", {"n": 20})
                    final_text = _result_text(result)
                    observed_loading |= "Data loading" in final_text
                    observed_partial |= "Partial data" in final_text
                status = await session.call_tool("refresh_data", {})
                status_text = _result_text(status)
                return {
                    "initialize_ms": initialize_ms,
                    "tools_list_ms": list_ms,
                    "tool_count": len(listed.tools),
                    "first_call_loading": loading,
                    "first_call_partial": partial,
                    "observed_loading": observed_loading,
                    "observed_partial": observed_partial,
                    "final_contains_recent": "synthetic-recent" in final_text,
                    "final_contains_older": "synthetic-older" in final_text,
                    "final_phase_ready": "phase=ready" in status_text,
                }


async def _run_failed_startup(root: Path, jsonl_glob: str) -> dict[str, object]:
    """Exercise the terminal cold-start failure contract over real stdio."""
    db_path = root / "failed.duckdb"
    db_path.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "INTROSPECT_DB_PATH": str(db_path),
            "INTROSPECT_JSONL_GLOB": jsonl_glob,
            "INTROSPECT_CODEX_GLOB": str(root / "codex" / "**" / "*.jsonl"),
            "INTROSPECT_DAYS": "30",
            "INTROSPECT_REFRESH_INTERVAL_SECONDS": "0",
            "INTROSPECT_VERSION_CHECK": "off",
        }
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "introspect.cli", "mcp"],
        env=env,
    )
    with Path(os.devnull).open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                deadline = time.perf_counter() + 10
                data_text = ""
                while "Data unavailable" not in data_text:
                    result = await session.call_tool("recent_sessions", {"n": 20})
                    data_text = _result_text(result)
                    if time.perf_counter() >= deadline:
                        break
                    await asyncio.sleep(0.05)
                refresh_result = await session.call_tool("refresh_data", {})
                refresh_text = _result_text(refresh_result)
                return {
                    "tool_count": len(listed.tools),
                    "data_unavailable": "Data unavailable" in data_text,
                    "data_error_detail": "startup preview failed" in data_text,
                    "refresh_unavailable": "Data unavailable" in refresh_text,
                    "refresh_error_detail": "startup preview failed" in refresh_text,
                }


def _require_lifecycle(results: dict[str, object], *, warm: bool) -> None:
    """Fail the consumer route when a claimed lifecycle state was not seen."""
    label = "warm" if warm else "cold"
    required = {
        "tools_list": bool(results["tool_count"]),
        "partial": results["observed_partial"],
        "recent result": results["final_contains_recent"],
        "authoritative result": results["final_contains_older"],
        "ready status": results["final_phase_ready"],
    }
    if not warm:
        required["cold loading or partial state"] = (
            results["observed_loading"] or results["observed_partial"]
        )
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(  # noqa: TRY003
            f"{label} lifecycle missing: {', '.join(missing)}"
        )


def _require_failed_startup(results: dict[str, object]) -> None:
    """Fail the consumer route when terminal failure is hidden or retryable."""
    required = {
        "tools_list": bool(results["tool_count"]),
        "data unavailable": results["data_unavailable"],
        "data error detail": results["data_error_detail"],
        "refresh unavailable": results["refresh_unavailable"],
        "refresh error detail": results["refresh_error_detail"],
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(  # noqa: TRY003
            f"failed-startup contract missing: {', '.join(missing)}"
        )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="introspect-mcp-qa-") as directory:
        root = Path(directory)
        jsonl_glob = _write_fixture(root)
        cold = await _run_once(root, jsonl_glob)
        warm = await _run_once(root, jsonl_glob)
        failure = await _run_failed_startup(root, jsonl_glob)
        report = {"cold": cold, "failure": failure, "warm": warm}
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        _require_lifecycle(cold, warm=False)
        _require_lifecycle(warm, warm=True)
        _require_failed_startup(failure)


if __name__ == "__main__":
    asyncio.run(main())
