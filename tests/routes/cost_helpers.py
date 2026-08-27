"""Shared cost-test helpers used by more than one test module."""

from datetime import datetime, timedelta
from pathlib import Path

from ..conftest import (
    local_client,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)


def _cache_loss_session_lines(
    session_id: str,
    *,
    gap_minutes: int,
    timestamp_day: str = "2026-04-21",
) -> list[dict]:
    """Build a 4-message JSONL with a single cache-loss event.

    First turn warms the cache (``cache_creation_input_tokens=8000``). The
    second user prompt arrives ``gap_minutes`` after the first assistant
    reply; the second assistant reply rebuilds the cache (cache_creation
    8500 > cache_read 500).
    """
    t0 = datetime.fromisoformat(f"{timestamp_day}T09:30:00+00:00")
    t1 = t0 + timedelta(seconds=2)
    t2 = t1 + timedelta(minutes=gap_minutes)
    t3 = t2 + timedelta(seconds=2)

    def _ts(d: datetime) -> str:
        return d.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    return [
        make_user_message(
            session_id,
            "u1",
            None,
            _ts(t0),
            "first prompt",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            _ts(t1),
            [{"type": "text", "text": "first reply"}],
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 8000,
            },
        ),
        make_user_message(session_id, "u2", "a1", _ts(t2), "second prompt"),
        make_assistant_message(
            session_id,
            "a2",
            "u2",
            _ts(t3),
            [{"type": "text", "text": "second reply"}],
            msg_id="msg2",
            usage={
                "input_tokens": 120,
                "output_tokens": 40,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 8500,
            },
        ),
    ]


def _dup_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL with two assistant records sharing one message.id."""
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    lines = [
        # Carry a tool_use_result on the seed user message so union_by_name
        # picks the column up when materialising views.
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "hi",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id="msg-dedup-1",
            usage=usage,
        ),
        # Duplicate: same message.id, different uuid, slightly later timestamp
        make_assistant_message(
            session_id,
            "a1-dup",
            "u1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id="msg-dedup-1",
            usage=usage,
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def _legacy_and_modern_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """One model, one modern cache_creation record and one legacy record.

    The legacy record predates the 5m/1h split: it carries only
    ``cache_creation_input_tokens``. Billing has to fall back to the 5m rate
    for it, and that decision has to be made per row — a per-model aggregate
    sees a non-zero 5m sum from the modern record and skips the fallback,
    dropping the legacy tokens from the bill entirely.
    """
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "hi",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            [{"type": "text", "text": "modern"}],
            model="claude-opus-4-7",
            msg_id="msg-modern",
            usage={
                "cache_creation_input_tokens": 100,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 100,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
        ),
        make_assistant_message(
            session_id,
            "a2",
            "a1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "legacy"}],
            model="claude-opus-4-7",
            msg_id="msg-legacy",
            usage={"cache_creation_input_tokens": 1_000_000},
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def _subagent_jsonl(tmp_dir: Path, session_id: str) -> Path:
    """Build a JSONL where a sidechain assistant message has cache_creation."""
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            "2026-04-21T10:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        # Sidechain user prompt (simulating Task subagent dispatch)
        make_user_message(
            session_id,
            "su1",
            "u1",
            "2026-04-21T10:00:01.000Z",
            "subagent: do this",
            is_sidechain=True,
        ),
        # Sidechain assistant response with significant cache_creation
        make_assistant_message(
            session_id,
            "sa1",
            "su1",
            "2026-04-21T10:00:02.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id="msg-side-1",
            usage={
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_creation_input_tokens": 500_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 500_000,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
            is_sidechain=True,
        ),
    ]
    return write_jsonl(tmp_dir, session_id, lines)


def _cost_overview_setup(tmp_dir: Path, specs: list[tuple[str, list[dict]]]) -> None:
    """Write one JSONL per (session_id, lines) tuple.

    Helper for the Cost Overview fixtures which need to synthesise multiple
    distinct sessions with known-cost profiles.
    """
    for session_id, lines in specs:
        write_jsonl(tmp_dir, session_id, lines)


def _session_at_cost(
    session_id: str,
    input_tokens: int,
    *,
    timestamp_day: str = "2026-04-21",
    timestamp_hour: str = "10",
) -> list[dict]:
    """Build a minimal two-message JSONL that costs exactly
    ``input_tokens * $5 / 1_000_000`` at claude-opus-4-7 pricing.
    """
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            f"{timestamp_day}T{timestamp_hour}:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            f"{timestamp_day}T{timestamp_hour}:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{session_id}-a1",
            usage={"input_tokens": input_tokens, "output_tokens": 0},
        ),
    ]
    return lines


def _run_with_client(tmp: Path, fn):
    """Run ``fn(client)`` with an initialised TestClient for this tmp dir."""
    import os  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    from introspect.api.main import app  # noqa: PLC0415

    from ..conftest import glob_pattern  # noqa: PLC0415

    db_path = tmp / "test.duckdb"
    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp),
                "INTROSPECT_CODEX_GLOB": str(tmp / "codex" / "**" / "*.jsonl"),
                "INTROSPECT_DAYS": "0",
            },
        ),
        local_client(app) as client,
    ):
        return fn(client)


def _materialize_and_run(tmp: Path, fn):
    """Materialize views and run ``fn(conn)`` against a standalone connection.

    Avoids the TestClient fixture's writable DB lock so tests can call the
    helper functions directly and assert on structured output.
    """
    from introspect.db import (  # noqa: PLC0415
        get_connection,
        materialize_views,
    )

    from ..conftest import glob_pattern  # noqa: PLC0415

    db_path = tmp / "test.duckdb"
    conn = get_connection(db_path, glob_pattern(tmp))
    try:
        materialize_views(conn, glob_pattern(tmp), 0, resolve_projects=False)
        return fn(conn)
    finally:
        conn.close()


def _multi_day_specs() -> list[tuple[str, list[dict]]]:
    """Three sessions on three distinct days, each with a known cost.

    Day 2026-04-21 → $20 (4M tokens)
    Day 2026-04-22 → $10 (2M tokens)
    Day 2026-04-23 →  $5 (1M tokens)
    Total = $35; costs computed at $5/M claude-opus-4-7 input pricing.
    """
    return [
        (
            "sess-day-21-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost(
                "sess-day-21-aaaa-aaaa-aaaaaaaaaaaa",
                4_000_000,
                timestamp_day="2026-04-21",
            ),
        ),
        (
            "sess-day-22-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost(
                "sess-day-22-aaaa-aaaa-aaaaaaaaaaaa",
                2_000_000,
                timestamp_day="2026-04-22",
            ),
        ),
        (
            "sess-day-23-aaaa-aaaa-aaaaaaaaaaaa",
            _session_at_cost(
                "sess-day-23-aaaa-aaaa-aaaaaaaaaaaa",
                1_000_000,
                timestamp_day="2026-04-23",
            ),
        ),
    ]


def _multi_model_specs() -> list[tuple[str, list[dict]]]:
    """Two same-day sessions on different models so by-model has 2 traces."""
    sid_opus = "sess-mm-opus-aaaa-aaaa-aaaaaaaaaaaa"
    sid_son = "sess-mm-sonnet-aaa-aaaa-aaaaaaaaaaaa"
    return [
        (
            sid_opus,
            _session_at_cost(sid_opus, 2_000_000, timestamp_day="2026-04-21"),
        ),
        (
            sid_son,
            [
                make_user_message(
                    sid_son,
                    "u1",
                    None,
                    "2026-04-21T11:00:00.000Z",
                    "go",
                    tool_use_result={"content": "seed"},
                ),
                make_assistant_message(
                    sid_son,
                    "a1",
                    "u1",
                    "2026-04-21T11:00:01.000Z",
                    [{"type": "text", "text": "ok"}],
                    model="claude-sonnet-4-6",
                    msg_id=f"msg-{sid_son}-a1",
                    usage={"input_tokens": 1_000_000, "output_tokens": 0},
                ),
            ],
        ),
    ]
