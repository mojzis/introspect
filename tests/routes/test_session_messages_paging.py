"""Tests for server-side paging of the session detail Messages tab."""

import os
from unittest.mock import patch

import pytest

from introspect.api.main import app

from ..conftest import (
    glob_pattern,
    local_client,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

_BIG_SID = "aabbccdd-1111-2222-3333-444455556666"

# Each turn renders three blocks — human prompt, assistant text, tool_use —
# and the tool_result row never renders. 200 turns is 600 blocks: three pages
# at the smallest allowed size (250), two at 500 with a turn straddling the
# boundary.
_TURNS = 200
_BLOCKS = _TURNS * 3


def _big_session_lines() -> list[dict]:
    """A session long enough to span several pages at ``page_size=250``."""
    lines: list[dict] = []
    parent: str | None = None
    for i in range(_TURNS):
        minute = f"{i // 60:02d}:{i % 60:02d}"
        user_uuid = f"u{i}"
        lines.append(
            make_user_message(
                _BIG_SID,
                user_uuid,
                parent,
                f"2026-05-01T{minute}:00.000Z",
                f"question number {i}",
            )
        )
        asst_uuid = f"a{i}"
        lines.append(
            make_assistant_message(
                _BIG_SID,
                asst_uuid,
                user_uuid,
                f"2026-05-01T{minute}:01.000Z",
                [
                    {"type": "text", "text": f"answer number {i}"},
                    {
                        "type": "tool_use",
                        "id": f"toolu_{i}",
                        "name": "Bash",
                        "input": {"command": f"echo {i}"},
                    },
                ],
                usage={"input_tokens": 100, "output_tokens": 20},
            )
        )
        lines.append(
            make_user_message(
                _BIG_SID,
                f"r{i}",
                asst_uuid,
                f"2026-05-01T{minute}:02.000Z",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{i}",
                        "content": f"{i}\n",
                        "is_error": False,
                    }
                ],
                tool_use_result={"stdout": f"{i}\n"},
                source_tool_uuid=asst_uuid,
            )
        )
        parent = f"r{i}"
    return lines


@pytest.fixture(scope="module")
def big_client(tmp_path_factory):
    """One materialized 600-block session, shared by the whole module."""
    tmp_path = tmp_path_factory.mktemp("big-session")
    write_jsonl(tmp_path, _BIG_SID, _big_session_lines())
    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(tmp_path / "test.duckdb"),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_CODEX_GLOB": str(tmp_path / "codex" / "**" / "*.jsonl"),
                "INTROSPECT_DAYS": "0",
                "INTROSPECT_HOST": "127.0.0.1",
            },
        ),
        local_client(app) as client,
    ):
        yield client


def test_messages_tab_pages_server_side(big_client):
    """Only one page of blocks is rendered, with a pager and a total."""
    response = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&page_size=250")
    assert response.status_code == 200
    text = response.text
    assert f"Messages ({_BLOCKS:,})" in text
    assert "showing 1\u2013250" in text  # en-dash range
    assert "Page 1 of 3" in text
    assert "&page=2&" in text
    # Page 1 stops at block 250 — turn 83's prompt is the last one in,
    # and its reply (blocks 251/252) is not.
    assert "question number 83" in text
    assert "answer number 83" not in text


def test_messages_tab_second_page_holds_the_tail(big_client):
    """?page=2 renders the blocks page 1 cut off, and offers Prev + Next."""
    response = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&page=2&page_size=250")
    assert response.status_code == 200
    text = response.text
    assert "Page 2 of 3" in text
    assert ">Prev<" in text
    assert ">Next<" in text
    assert "answer number 83" in text
    assert "question number 100" in text
    assert "question number 170" not in text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # A page past the end lands on the last page rather than empty.
        ("page=99&page_size=250", "Page 3 of 3"),
        # Unsupported sizes snap to the 1000 default — one page for 600 blocks.
        ("page=1&page_size=7", "Page 1 of 1"),
        # Route-level coercion: non-numeric, negative and blank all read as
        # page 1 / the default size.
        ("page=abc&page_size=250", "Page 1 of 3"),
        ("page=-1&page_size=250", "Page 1 of 3"),
        ("page=&page_size=", "Page 1 of 1"),
        ("page=2&page_size=nonsense", "Page 1 of 1"),
    ],
)
def test_messages_page_params_are_coerced(big_client, query, expected):
    """Junk in the query string degrades to a sane page, never a 500."""
    response = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&{query}")
    assert response.status_code == 200
    assert expected in response.text
    # The block total is stated whether or not the list spans pages.
    assert f"Messages ({_BLOCKS:,})" in response.text


def test_size_picker_survives_a_single_page(big_client):
    """Picking 2000 must not hide the control that gets you back to 250."""
    response = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&page_size=2000")
    assert "Page 1 of 1" in response.text
    assert "page_size=250" in response.text


def test_pager_links_drop_focus(big_client):
    """Paging by hand off a deep link must not snap back to the anchor."""
    response = big_client.get(
        f"/sessions/{_BIG_SID}?tab=messages&page_size=250&focus=a120"
    )
    assert "Page 2 of 3" in response.text
    assert "focus=" not in response.text


def test_focus_uuid_selects_the_page_holding_that_block(big_client):
    """A ``?focus=<uuid>`` deep link opens on the page with that message."""
    response = big_client.get(
        f"/sessions/{_BIG_SID}?tab=messages&page_size=250&focus=a120"
    )
    assert response.status_code == 200
    assert "Page 2 of 3" in response.text
    assert 'id="msg-a120"' in response.text


def test_focus_tool_use_id_selects_the_page_holding_that_call(big_client):
    """The tools/bash pages link by tool_use_id — that resolves too."""
    response = big_client.get(f"/sessions/{_BIG_SID}?page_size=250&focus=toolu_120")
    assert response.status_code == 200
    assert "Page 2 of 3" in response.text
    assert 'id="tc-toolu_120"' in response.text


def test_unknown_focus_falls_back_to_first_page(big_client):
    """An id from another session doesn't 500 or empty the tab."""
    response = big_client.get(
        f"/sessions/{_BIG_SID}?page_size=250&focus=not-a-real-uuid"
    )
    assert response.status_code == 200
    assert "Page 1 of 3" in response.text


def test_turn_split_across_pages_keeps_one_anchor(big_client):
    """``block_rank`` comes from SQL, so a split turn re-anchors on neither.

    Turn 166's assistant reply straddles the 500-block boundary: its text
    block ends page 1 and its tool_use opens page 2. The anchor id belongs
    to the text block only.
    """
    first = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&page_size=500")
    second = big_client.get(f"/sessions/{_BIG_SID}?tab=messages&page=2&page_size=500")
    assert 'id="msg-a166"' in first.text
    assert 'id="msg-a166"' not in second.text
    # The continuation block still renders, just unanchored.
    assert 'id="tc-toolu_166"' in second.text
