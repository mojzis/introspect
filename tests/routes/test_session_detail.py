"""Tests for the session detail page."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import duckdb

from introspect.api.main import app

from ..conftest import (
    codex_glob_pattern,
    glob_pattern,
    local_client,
    make_assistant_message,
    make_user_message,
    write_codex_parent_nested_replay,
    write_jsonl,
)
from .conftest import SID, _patched_client
from .cost_helpers import _cache_loss_session_lines

_TWEAK_SID = "01234567-abcd-abcd-abcd-fedcba987654"

_CACHE_LOSS_SID = "deadbeef-aaaa-bbbb-cccc-fedcba987654"

_CODEX_REPLAY_SID = "codex-replay-route"


def _cache_loss_session_jsonl(tmp_dir: Path, *, gap_minutes: int = 6) -> Path:
    """Write a single-session cache-loss fixture to ``tmp_dir``."""
    lines = _cache_loss_session_lines(
        _CACHE_LOSS_SID, gap_minutes=gap_minutes, timestamp_day="2026-04-15"
    )
    return write_jsonl(tmp_dir, _CACHE_LOSS_SID, lines)


@contextmanager
def _cache_loss_client(tmp_path: Path, *, gap_minutes: int = 6):
    _cache_loss_session_jsonl(tmp_path, gap_minutes=gap_minutes)
    db_path = tmp_path / "test.duckdb"
    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
                "INTROSPECT_DAYS": "0",
            },
        ),
        local_client(app) as client,
    ):
        yield client


def test_session_detail_marks_cache_loss_event():
    """A >5min gap with cache rebuild surfaces a divider + header chip."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _cache_loss_client(tmp, gap_minutes=6) as client:
            response = client.get(f"/sessions/{_CACHE_LOSS_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            assert 'class="cache-loss-divider"' in text
            assert "6 min gap" in text
            assert "cache lost" in text
            # Header chip on session_detail.html.
            assert "Cache losses" in text
            assert "1 ·" in text
            # opus-4-6 5m write premium is (6.25 - 0.50)/1M per token, over
            # the 8000-token prefix the warm cache would have read (not the
            # 8500 written — the 500 the new prompt added get written under
            # any TTL) ≈ $0.046 → format_cost rounds to "$0.05".
            assert "~$0.05 wasted" in text


def test_cache_loss_divider_survives_the_message_filters():
    """The divider sits outside the row, so ``x-show`` can't swallow it.

    It anchors on the reply that paid, and a reply routinely opens with a
    thinking or tool_use block — both of which live in rows the "hide
    thinking" / "hide tools" toggles hide. Nesting the divider inside such a
    row would make it vanish exactly when the reply is collapsed.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _cache_loss_client(tmp, gap_minutes=6) as client:
            response = client.get(f"/sessions/{_CACHE_LOSS_SID}?tab=messages")
            assert response.status_code == 200
            divider = response.text.index('class="cache-loss-divider"')
            row_open = response.text.rindex("<div ", 0, divider)
            # The element opened immediately before the divider must not be
            # the visibility-gated msg-row wrapper.
            assert "msg-row" not in response.text[row_open:divider]
            assert "x-show" not in response.text[row_open:divider]


def test_session_detail_marks_long_gap_as_unrecoverable_break():
    """A 90-min gap is labelled a break, not waste — no TTL recovers it."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _cache_loss_client(tmp, gap_minutes=90) as client:
            response = client.get(f"/sessions/{_CACHE_LOSS_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            assert 'class="cache-loss-divider"' in text
            assert "no TTL recovers this" in text
            assert "cache lost" not in text
            # Header splits the two counts apart.
            assert "Cache breaks" in text
            assert "Cache losses" not in text


def test_session_detail_no_cache_loss_under_threshold():
    """Gap under 5 min should not produce a divider or header chip."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _cache_loss_client(tmp, gap_minutes=4) as client:
            response = client.get(f"/sessions/{_CACHE_LOSS_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            # The CSS class is defined in <style>; check the rendered element.
            assert 'class="cache-loss-divider"' not in text
            assert "Cache losses" not in text


def _tweak_session_jsonl(tmp_dir: Path) -> Path:
    """Session with long assistant text + short human prompt + token usage."""
    long_body = "\n".join(f"line {i}" for i in range(1, 11))  # 10 lines
    lines = [
        # Give the first user row a ``toolUseResult`` key so DuckDB's schema
        # inference sees the column (raw_messages SELECTs it). The field is
        # otherwise irrelevant to the tests.
        make_user_message(
            _TWEAK_SID,
            "u1",
            None,
            "2026-04-15T09:30:45.000Z",
            "short hi",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            _TWEAK_SID,
            "a1",
            "u1",
            "2026-04-15T09:30:47.000Z",
            [{"type": "text", "text": long_body}],
            usage={
                "input_tokens": 2134,
                "output_tokens": 180,
                "cache_read_input_tokens": 340,
                "cache_creation_input_tokens": 5120,
            },
        ),
    ]
    return write_jsonl(tmp_dir, _TWEAK_SID, lines)


@contextmanager
def _tweak_client(tmp_path: Path):
    """TestClient wired to the tweak-session fixture (long assistant body)."""
    _tweak_session_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"
    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
                "INTROSPECT_DAYS": "0",
            },
        ),
        local_client(app) as client,
    ):
        yield client


def test_session_detail_returns_200():
    """Session detail page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions/01234567-abcd-abcd-abcd-0123456789ab")
        assert response.status_code == 200


def test_session_detail_tolerates_a_database_missing_cache_requests():
    """Older materialized databases should not turn every detail page into a 500."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        with duckdb.connect(str(app.state.db_path)) as db:
            db.execute("DROP TABLE cache_requests")

        response = client.get(f"/sessions/{SID}")

        assert response.status_code == 200
        assert "Cache losses" not in response.text


def test_session_detail_shows_user_messages():
    """Session detail page displays user messages."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "Hello, help me with tests" in response.text
        assert (
            '<span class="meta-label">Title</span> Hello, help me with tests'
            in response.text
        )


def test_session_detail_omits_codex_parent_replay_but_keeps_subagent():
    """Messages renders one parent response and the unique nested response."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        write_codex_parent_nested_replay(tmp, _CODEX_REPLAY_SID)
        with _patched_client(
            tmp, {"INTROSPECT_CODEX_GLOB": codex_glob_pattern(tmp)}
        ) as client:
            response = client.get(f"/sessions/{_CODEX_REPLAY_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            assert text.count("parent answer") == 1
            assert text.count("subagent answer") == 1
            assert "kind-subagent_prompt" in text


def test_session_detail_shows_tool_results():
    """Session detail page surfaces tool result content folded under the tool call."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # Both the tool input ("echo hello") and the tool_use_result stdout
        # ("hello") from the fixture must appear inside an agent_tool_call row.
        assert "kind-agent_tool_call" in text
        assert "echo hello" in text
        # tool-section-label markers only exist when the paired result is present
        assert "tool-section-label" in text


def test_session_detail_classifies_message_kinds():
    """Session detail page classifies messages into distinct visual kinds."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # Human prompt is visually distinct from tool results.
        assert "kind-human_prompt" in response.text
        assert "Hello, help me with tests" in response.text
        # Slash command wrapping becomes a divider, not a big box.
        assert "slash-divider" in response.text
        # Filter strip is present with localStorage-backed toggles.
        assert "Hide thinking" in response.text
        assert "Hide tool calls" in response.text
        assert "Only human turns" in response.text
        # Global expand/collapse buttons are present.
        assert "Expand all tools" in response.text
        assert "tools-expand-all" in response.text
        assert "tools-collapse-all" in response.text


def test_session_detail_tool_call_one_liner():
    """Tool calls render as one-liners with a tool-specific hint."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The Bash tool's 'command' input is lifted into the collapsed one-liner.
        assert "tool-call-line" in text
        assert "tool-hint" in text
        assert "echo hello" in text


def test_session_detail_distinguishes_subagent_prompt_from_human():
    """Sidechain user messages render as 'prompt to subagent', not 'you'."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The subagent prompt appears with its own kind class, NOT human_prompt.
        assert "kind-subagent_prompt" in text
        assert "Explore the database layer" in text
        assert "prompt to subagent" in text
        # And it is visually distinct from the real human prompt above it.
        human_idx = text.find("Hello, help me with tests")
        sub_idx = text.find("Explore the database layer")
        assert human_idx != -1
        assert sub_idx != -1
        # The real human prompt is NOT wrapped as a subagent_prompt.
        assert 'class="msg-row kind-subagent_prompt' in text
        assert (
            text[max(0, human_idx - 300) : human_idx].count("kind-subagent_prompt") == 0
        )


def test_session_detail_folds_tool_results_under_calls():
    """tool_result rows are filtered out of the rendered list (folded into calls)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # There should be no standalone kind-tool_result row — tool_results are
        # merged into the matching agent_tool_call row via the tool_calls join.
        assert "kind-tool_result" not in response.text


def test_session_detail_shows_token_usage():
    """Session detail shows token usage stats."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "Input Tokens" in response.text


def test_session_detail_shows_tool_summary():
    """Session detail shows tool call summary line."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        assert "tool call" in response.text


def test_session_detail_expandable_blocks():
    """Session detail renders tool use blocks with expand support."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        # Tool use blocks are rendered (Bash tool call in test data)
        assert "Bash" in response.text


def test_session_detail_long_body_collapses():
    """Long assistant text bodies render with clamp + meta indicator."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _tweak_client(tmp) as client:
            response = client.get(f"/sessions/{_TWEAK_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            assert "msg-text-meta" in text
            assert "click to expand" in text
            # Body content is present.
            assert "line 10" in text
            # The <pre> surrounding the long body carries the Alpine class
            # binding that toggles the clamp. Look in a 400-char window
            # before "line 10" for that binding; the body should NOT render
            # as a plain unclamped <pre class="msg-text"> elsewhere.
            idx = text.find("line 10")
            window_before = text[max(0, idx - 400) : idx]
            assert "'msg-text-clamp': !expanded" in window_before
            # Single-render design: the long body should appear only once in
            # the rendered HTML (previous implementation rendered it twice,
            # once per x-show branch).
            assert text.count("line 10") == 1


def test_session_detail_short_body_has_no_clamp():
    """Short bodies (<=3 lines, <=240 chars) render without the clamp wrapper."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=messages")
        assert response.status_code == 200
        text = response.text
        # Sample fixture has short bodies ("Sure, I can help!", "Hello, help me
        # with tests") — none should trigger clamp.
        # Confirm the short assistant reply is rendered as a plain msg-text.
        assert "Sure, I can help!" in text
        # And the short human prompt does NOT pull in a clamp wrapper.
        human_idx = text.find("Hello, help me with tests")
        assert human_idx != -1
        window = text[max(0, human_idx - 200) : human_idx + 200]
        assert "msg-text-clamp" not in window


def test_session_detail_time_only_timestamp_with_title():
    """Per-entry timestamps show HH:MM:SS with the full datetime in a title tip."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _tweak_client(tmp) as client:
            response = client.get(f"/sessions/{_TWEAK_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            # Full datetime lives in the tooltip.
            assert 'title="2026-04-15 09:30:47"' in text
            # Visible text on the msg-time span is time-only.
            assert ">09:30:47<" in text
            # The visible span must not itself contain the date — only the
            # adjacent title attribute should.
            assert ">2026-04-15 09:30:47<" not in text


def test_session_detail_assistant_token_badge():
    """First-block assistant entries with usage render a token-badge + title."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with _tweak_client(tmp) as client:
            response = client.get(f"/sessions/{_TWEAK_SID}?tab=messages")
            assert response.status_code == 200
            text = response.text
            assert "token-badge" in text
            # Compact form: omit zero components, use arrow + ⚡/✎ glyphs.
            assert "↓" in text
            assert "↑" in text
            assert "⚡" in text
            assert "✎" in text
            # Detailed tooltip includes raw counts.
            assert "Input:" in text
            assert "Output:" in text
            assert "2,134" in text
            assert "180" in text
            assert "5,120" in text


def test_session_detail_shows_file_metrics():
    """Session detail page shows file metrics line."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # The test data has 2 Read calls and 1 Edit call
        assert "Files" in text
        assert "read" in text
        assert "edited" in text


def test_session_detail_shows_read_only_count():
    """Session detail shows read-only file count (read but never written)."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # /home/user/other/config.yml is read but never edited → 1 read-only
        assert "1 read-only" in text


def test_session_detail_file_metrics_outside_count():
    """Session detail correctly counts files outside the project directory."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}")
        assert response.status_code == 200
        text = response.text
        # One file (/home/user/other/config.yml) is outside /tmp/test
        assert "outside project" in text


def test_session_messages_have_uuid_anchors():
    """Messages tab rows carry id=\"msg-{uuid}\" anchors."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=messages")
        assert response.status_code == 200
        assert 'id="msg-' in response.text
