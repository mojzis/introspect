"""Shared helpers, constants, and template setup for route handlers."""

import json
import logging
import re
from pathlib import Path

import duckdb
from fastapi import Request
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE_DEFAULT = 50
SESSIONS_PAGE_SIZES = [25, 50, 100, 200]

# Allowed sort columns for sessions page
SESSIONS_SORT_COLS = {
    "started_at": "ls.started_at",
    "duration": "ls.duration",
    "user_msgs": "ls.user_messages",
    "asst_msgs": "ls.assistant_messages",
    "tool_calls": "tc.tool_count",
    "model": "ls.model",
    "project": "ls.cwd",
    "branch": "ls.git_branch",
}
SESSIONS_SORT_DEFAULT = "started_at"

RAW_PER_PAGE = 20

_XML_TAG_PREFIX_RE = re.compile(r"^<[^>]+>")

# Raised from 500 to allow expand/collapse UI in session_detail to show
# meaningful content.  The template truncates at 200 chars for the collapsed
# view; the full value is hidden behind an Alpine.js toggle.
_CONTENT_PREVIEW_MAX = 5000


def clean_title(raw: str) -> str:
    """Strip leading XML-style tags from session titles."""
    return _XML_TAG_PREFIX_RE.sub("", raw).strip()


def parent(request: Request) -> str:
    """Return the base template: full page for normal requests, partial for HTMX."""
    if request.headers.get("HX-Request"):
        return "partial.html"
    return "base.html"


def parse_content_block(block) -> dict:  # noqa: PLR0911
    """Parse a single content block from a message."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)[:_CONTENT_PREVIEW_MAX]}

    block_type = block.get("type", "text")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "tool_use":
        input_str = block.get("input", "")
        if isinstance(input_str, dict):
            input_str = json.dumps(input_str, indent=2)
        return {
            "type": "tool_use",
            "name": block.get("name", ""),
            "tool_use_id": block.get("id", ""),
            "input": str(input_str)[:_CONTENT_PREVIEW_MAX],
        }
    if block_type == "tool_result":
        result_content = block.get("content", "")
        if isinstance(result_content, list):
            result_content = json.dumps(result_content)
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", ""),
            "content": str(result_content)[:_CONTENT_PREVIEW_MAX],
            "is_error": block.get("is_error", False),
        }
    if block_type == "thinking":
        return {"type": "thinking", "text": block.get("thinking", "")}
    return {"type": "text", "text": str(block)[:_CONTENT_PREVIEW_MAX]}


def conn(request: Request):
    """Get the DuckDB connection from request state."""
    return request.state.conn


DEFAULT_PAGE_SIZE = 50

# Reusable SQL fragment for per-session tool counts.
TOOL_COUNTS_SUBQUERY = """(
    SELECT session_id, COUNT(*) AS tool_count
    FROM tool_calls GROUP BY session_id
) tc"""

COMMAND_LIST_SUBQUERY = """(
    SELECT session_id, string_agg(DISTINCT command, ', ' ORDER BY command) AS commands
    FROM message_commands GROUP BY session_id
) cmd"""

TOOL_COUNTS_WITH_ERRORS_SUBQUERY = """(
    SELECT session_id,
           COUNT(*) AS tool_count,
           COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
    FROM tool_calls GROUP BY session_id
) tc"""


def format_duration(total_seconds: float) -> str:
    """Format seconds as M:SS string."""
    secs = int(total_seconds)
    return f"{secs // 60}:{secs % 60:02d}"


def fetch_token_usage(
    db: duckdb.DuckDBPyConnection,
    *,
    session_id: str | None = None,
    include_cache: bool = False,
) -> tuple | None:
    """Fetch token usage sums. Returns None on error.

    Without session_id: returns (input_tokens, output_tokens).
    With include_cache: appends (cache_creation_tokens, cache_read_tokens).
    """
    cache_cols = ""
    if include_cache:
        cache_cols = """,
            SUM(CAST(json_extract(
                message, '$.usage.cache_creation_input_tokens'
            ) AS BIGINT)),
            SUM(CAST(json_extract(
                message, '$.usage.cache_read_input_tokens'
            ) AS BIGINT))"""

    session_filter = ""
    params: list[str] = []
    if session_id is not None:
        session_filter = "AND session_id = ?"
        params.append(session_id)

    try:
        return db.execute(
            f"""
            SELECT
                SUM(CAST(json_extract(message, '$.usage.input_tokens') AS BIGINT)),
                SUM(CAST(json_extract(message, '$.usage.output_tokens') AS BIGINT))
                {cache_cols}
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract(message, '$.usage.input_tokens') IS NOT NULL
              {session_filter}
        """,  # nosec B608
            params,
        ).fetchone()
    except Exception:
        log.debug("token usage query failed", exc_info=True)
        return None
