"""Tests for the search page."""

import tempfile
from pathlib import Path

from introspect.api.handlers._helpers import clean_title

from .conftest import SID, _patched_client


def test_search_returns_200():
    """Search page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search")
        assert response.status_code == 200


def test_search_finds_user_content():
    """Search returns results matching user message content."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        assert "help me with tests" in response.text.lower()


def test_search_finds_assistant_content():
    """Search returns results matching assistant message content."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Sure+I+can+help")
        assert response.status_code == 200
        assert "sure" in response.text.lower() or "can help" in response.text.lower()


def test_search_shows_fts_status():
    """Search page shows FTS availability indicator."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search")
        assert response.status_code == 200
        # Shows either BM25 active or ILIKE fallback depending on FTS availability
        assert "BM25" in response.text or "ILIKE fallback" in response.text


def test_search_pagination_next():
    """Search page shows result count for a query."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Hello&page=1")
        assert response.status_code == 200
        # The results summary always appears when a query is provided
        assert 'result(s) for "Hello"' in response.text


def test_search_pagination_param():
    """Search page accepts page parameter."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=Hello&page=2")
        assert response.status_code == 200


# --- Search results enrichment tests ---


def test_search_results_show_session_info():
    """Search results include session metadata columns."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        # Should show session-level columns
        assert "Project" in response.text
        assert "Branch" in response.text
        assert "Title" in response.text
        assert "Duration" in response.text
        assert "Model" in response.text


def test_search_results_link_to_session():
    """Search results link to the session detail page."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search?q=help+me+with+tests")
        assert response.status_code == 200
        assert f"/sessions/{SID}" in response.text


def test_clean_title_strips_all_xml_tags():
    """clean_title strips ALL XML tags, not just leading ones."""
    # Leading tag only
    assert clean_title("<foo>bar") == "bar"
    # Wrapping tags (the original bug: command-name pattern)
    assert clean_title("<command-name>/commit</command-name>") == "/commit"
    # Nested / multiple tags
    assert clean_title("<a><b>text</b></a>") == "text"
    # No tags at all
    assert clean_title("plain text") == "plain text"
    # Empty string
    assert clean_title("") == ""
    # Tags with attributes
    assert clean_title('<div class="x">content</div>') == "content"
    # Mixed content
    assert clean_title("before <tag>middle</tag> after") == "before middle after"


def test_clean_title_drops_command_message_that_mirrors_command_name():
    """<command-message> duplicates <command-name>, so it's dropped entirely."""
    # Skill invocation: command-name and command-message carry the same label.
    raw = (
        "<command-name>marimo-pair</command-name>\n"
        "<command-message>/marimo-pair</command-message>\n"
        "<command-args></command-args>"
    )
    assert clean_title(raw) == "marimo-pair"
    # With real args attached — tag boundaries become word separators.
    raw_with_args = (
        "<command-name>commit</command-name>"
        "<command-message>/commit</command-message>"
        "<command-args>fix typo</command-args>"
    )
    assert clean_title(raw_with_args) == "commit fix typo"
