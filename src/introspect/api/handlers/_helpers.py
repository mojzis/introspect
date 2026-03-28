"""Shared helpers, constants, and template setup for route handlers."""

import json
import re
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

SESSIONS_PER_PAGE_DEFAULT = 50
SESSIONS_PAGE_SIZES = [25, 50, 100, 200]

# Allowed sort columns for sessions page
_SESSIONS_SORT_COLS = {
    "started_at": "ls.started_at",
    "duration": "ls.duration",
    "user_msgs": "ls.user_messages",
    "asst_msgs": "ls.assistant_messages",
    "model": "ls.model",
    "project": "ls.cwd",
    "branch": "ls.git_branch",
}
_SESSIONS_SORT_DEFAULT = "started_at"

RAW_PER_PAGE = 20

_XML_TAG_PREFIX_RE = re.compile(r"^<[^>]+>")


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
        return {"type": "text", "text": str(block)[:500]}

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
            "input": str(input_str)[:500],
        }
    if block_type == "tool_result":
        result_content = block.get("content", "")
        if isinstance(result_content, list):
            result_content = json.dumps(result_content)
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", ""),
            "content": str(result_content)[:500],
            "is_error": block.get("is_error", False),
        }
    if block_type == "thinking":
        return {"type": "thinking", "text": block.get("thinking", "")}
    return {"type": "text", "text": str(block)[:500]}


def conn(request: Request):
    """Get the DuckDB connection from request state."""
    return request.state.conn
