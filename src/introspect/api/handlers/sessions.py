"""Session-related route handlers."""

import json
import logging
import math
import re
from pathlib import PurePosixPath

import nolegend
import plotly.graph_objects as go
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from introspect.pricing import compute_cost_usd
from introspect.search import ensure_search_corpus, fts_available

from ._helpers import (
    OBVIOUS_COMMANDS_SQL,
    SESSION_INFO_SELECT,
    SESSIONS_PAGE_SIZES,
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_COLS,
    SESSIONS_SORT_DEFAULT,
    build_cost_attribution_sql,
    conn,
    fetch_token_usage,
    format_cost,
    parent,
    session_row_to_dict,
    templates,
)

logger = logging.getLogger(__name__)

_MESSAGE_HARD_CAP = 5000
_THINKING_PREVIEW_MAX = 200
_TOOL_HINT_MAX = 120
_BODY_COLLAPSE_LINES = 3
_BODY_COLLAPSE_CHARS = 240
_COMMAND_NAME_RE = re.compile(r"<command-name>([^<]+)</command-name>")

# Per-tool preferred input keys for the one-line collapsed summary.
# First matching key in order wins; missing tools fall back to the first value.
_TOOL_HINT_KEYS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "MultiEdit": ("file_path",),
    "Write": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Bash": ("command",),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "Task": ("description", "subagent_type"),
    "Agent": ("description", "subagent_type"),
    "TaskCreate": ("subject",),
    "TaskUpdate": ("taskId", "status"),
    "Skill": ("skill", "args"),
    "ScheduleWakeup": ("reason",),
}


_SECONDS_PER_MINUTE = 60
_MIN_SENTENCE_LEN = 20


def _format_exec_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), _SECONDS_PER_MINUTE)
    return f"{mins}m {secs}s"


def _cap(value: str | None, limit: int = _MESSAGE_HARD_CAP) -> str:
    """Cap long strings with a visible truncation marker."""
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n… [truncated]"


def _pretty_tool_input(raw: str | None) -> str:
    """Pretty-print a tool input string. Falls back to the raw value on failure."""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return _cap(raw)
    try:
        return _cap(json.dumps(parsed, indent=2, ensure_ascii=False))
    except (TypeError, ValueError):
        return _cap(raw)


def _slash_command_label(text: str | None) -> str:
    """Extract the command name from a <command-name>…</command-name> wrapper."""
    if not text:
        return ""
    match = _COMMAND_NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip().splitlines()[0][:80]


def _thinking_preview(text: str | None) -> tuple[str, bool]:
    """Return (preview, has_more) for a thinking block.

    Preview is the first sentence if reasonably short, otherwise a hard cap.
    """
    if not text:
        return "", False
    stripped = text.strip()
    first_period = stripped.find(". ")
    if _MIN_SENTENCE_LEN < first_period < _THINKING_PREVIEW_MAX:
        preview = stripped[: first_period + 1]
    else:
        preview = stripped[:_THINKING_PREVIEW_MAX]
    has_more = len(stripped) > len(preview)
    return preview, has_more


def _tool_hint(tool_name: str, raw_input: str | None) -> str:
    """One-line summary of a tool call's input for the collapsed view."""
    if not raw_input:
        return ""
    try:
        parsed = json.loads(raw_input)
    except (ValueError, TypeError):
        return _single_line(raw_input, _TOOL_HINT_MAX)
    if not isinstance(parsed, dict):
        return _single_line(str(parsed), _TOOL_HINT_MAX)
    keys = _TOOL_HINT_KEYS.get(tool_name, ())
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return _single_line(str(parsed[key]), _TOOL_HINT_MAX)
    for value in parsed.values():
        if value not in (None, ""):
            return _single_line(str(value), _TOOL_HINT_MAX)
    return ""


def _single_line(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


_TOKEN_COMPACT_K = 1_000
_TOKEN_COMPACT_M = 1_000_000
_TOKEN_COMPACT_SINGLE_DIGIT = 10


def _format_tokens_compact(count: int) -> str:
    """Compact human-readable token count, e.g. 2134 -> '2.1k', 1_250_000 -> '1.2M'."""
    if count < _TOKEN_COMPACT_K:
        return str(count)
    if count < _TOKEN_COMPACT_M:
        value = count / _TOKEN_COMPACT_K
        return (
            f"{value:.1f}k"
            if value < _TOKEN_COMPACT_SINGLE_DIGIT
            else f"{round(value)}k"
        )
    value = count / _TOKEN_COMPACT_M
    return (
        f"{value:.1f}M" if value < _TOKEN_COMPACT_SINGLE_DIGIT else f"{round(value)}M"
    )


def _token_badge_strings(
    tokens_in: int, tokens_out: int, cache_read: int, cache_create: int
) -> tuple[str, str]:
    """Return ``(compact_summary, detailed_title)`` for the per-turn token badge.

    ``compact_summary`` is empty when all counts are zero. Zero components are
    omitted from the compact form; the tooltip always lists the full breakdown.
    """
    if not (tokens_in or tokens_out or cache_read or cache_create):
        return "", ""
    parts: list[str] = []
    if tokens_in:
        parts.append(f"↓{_format_tokens_compact(tokens_in)}")
    if tokens_out:
        parts.append(f"↑{_format_tokens_compact(tokens_out)}")
    if cache_read:
        parts.append(f"⚡{_format_tokens_compact(cache_read)}")
    if cache_create:
        parts.append(f"✎{_format_tokens_compact(cache_create)}")
    title = (
        f"Input: {tokens_in:,} · Output: {tokens_out:,} "
        f"· Cache read: {cache_read:,} · Cache create: {cache_create:,}"
    )
    return " ".join(parts), title


def _collapse_info(
    text: object,
    *,
    lines: int = _BODY_COLLAPSE_LINES,
    chars: int = _BODY_COLLAPSE_CHARS,
) -> tuple[int, int, bool]:
    """Return ``(line_count, char_count, needs_collapse)`` for a text body.

    Accepts non-string input (e.g. dicts yielded by the ``tool_calls`` join
    for tool-result rows whose payload is JSON, not a string) by coercing to
    ``str`` first — matches what Jinja will render downstream.
    """
    if not text:
        return 0, 0, False
    s = text if isinstance(text, str) else str(text)
    char_count = len(s)
    line_count = s.count("\n") + 1
    needs = line_count > lines or char_count > chars
    return line_count, char_count, needs


def _coerce_bool(value: object) -> bool:
    """Normalize DuckDB/JSON boolean-ish values to a Python bool."""
    if value is None or value is False:
        return False
    if value is True:
        return True
    return str(value).strip().lower() == "true"


async def sessions(  # noqa: PLR0913
    request: Request,
    page: int,
    page_size: int,
    sort: str,
    order: str,
    model: str,
    project: str,
    branch: str,
    command: str,
    q: str,
) -> HTMLResponse:
    """Paginated session list with filtering and sorting."""
    db = conn(request)

    # Clamp page_size to allowed values
    if page_size not in SESSIONS_PAGE_SIZES:
        page_size = SESSIONS_PER_PAGE_DEFAULT

    # Build WHERE clause from filters
    where_clauses: list[str] = []
    params: list[str | int] = []
    if model.strip():
        where_clauses.append("ss.model = ?")
        params.append(model.strip())
    if project.strip():
        where_clauses.append("ss.project = ?")
        params.append(project.strip())
    if branch.strip():
        where_clauses.append("ss.git_branch = ?")
        params.append(branch.strip())
    if command.strip():
        where_clauses.append(
            "EXISTS (SELECT 1 FROM message_commands mc"
            " WHERE mc.session_id = ss.session_id AND mc.command = ?)"
        )
        params.append(command.strip())
    search_query = q.strip()
    if search_query:
        ensure_search_corpus(db)
        if fts_available(db):
            where_clauses.append(
                "EXISTS (SELECT 1 FROM (SELECT *, "
                "fts_main_search_corpus.match_bm25(rowid, ?) AS score "
                "FROM search_corpus) sc "
                "WHERE sc.session_id = ss.session_id AND sc.score IS NOT NULL)"
            )
            params.append(search_query)
        else:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM search_corpus sc"
                " WHERE sc.session_id = ss.session_id"
                " AND sc.content_text ILIKE ?)"
            )
            params.append(f"%{search_query}%")

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Count with filters
    total = db.execute(
        f"SELECT COUNT(*) FROM session_stats ss {where}",  # noqa: S608
        params,
    ).fetchone()[0]
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    # Resolve sort column
    default_col = SESSIONS_SORT_COLS[SESSIONS_SORT_DEFAULT]
    sort_col = SESSIONS_SORT_COLS.get(sort, default_col)
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"
    nulls = "NULLS LAST" if sort_dir == "DESC" else "NULLS FIRST"

    rows = db.execute(
        f"""
        SELECT {SESSION_INFO_SELECT}
        FROM session_stats ss
        {where}
        ORDER BY {sort_col} {sort_dir} {nulls}
        LIMIT ? OFFSET ?
    """,  # noqa: S608
        [*params, page_size, offset],
    ).fetchall()

    session_list = [session_row_to_dict(row) for row in rows]

    # Get distinct values for filter dropdowns
    models = db.execute("""
        SELECT DISTINCT model FROM logical_sessions
        WHERE model IS NOT NULL ORDER BY model
    """).fetchall()
    projects = db.execute("""
        SELECT DISTINCT project AS proj
        FROM logical_sessions
        WHERE project IS NOT NULL
        ORDER BY proj
    """).fetchall()
    branches = db.execute("""
        SELECT DISTINCT git_branch FROM logical_sessions
        WHERE git_branch IS NOT NULL ORDER BY git_branch
    """).fetchall()
    commands_list = db.execute(f"""
        SELECT DISTINCT command FROM message_commands
        WHERE command NOT IN {OBVIOUS_COMMANDS_SQL}
        ORDER BY command
    """).fetchall()  # noqa: S608

    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "parent": parent(request),
            "sessions": session_list,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_size": page_size,
            "page_sizes": SESSIONS_PAGE_SIZES,
            "sort": sort,
            "order": order.lower(),
            "filter_model": model,
            "filter_project": project,
            "filter_branch": branch,
            "filter_command": command,
            "filter_q": search_query,
            "models": [r[0] for r in models],
            "projects": [r[0] for r in projects],
            "branches": [r[0] for r in branches],
            "commands_list": [r[0] for r in commands_list],
        },
    )


def _build_messages_context(db, session_id: str) -> list[dict]:
    """Build the parsed message list shown in the Messages tab."""
    cur = db.execute(
        """
        SELECT
            e.uuid,
            e.timestamp,
            e.kind,
            e.is_sidechain,
            e.text,
            e.thinking_text,
            e.tool_name,
            e.tool_input,
            e.tool_use_id,
            tc.tool_use_result,
            tc.is_error,
            tc.execution_time,
            amc.input_tokens,
            amc.output_tokens,
            amc.cache_read_tokens,
            amc.cache_creation_tokens
        FROM session_messages_enriched e
        LEFT JOIN tool_calls tc ON tc.tool_use_id = e.tool_use_id
        LEFT JOIN assistant_message_costs amc ON amc.uuid = e.uuid
        WHERE e.session_id = ?
          AND e.kind <> 'tool_result'
        ORDER BY e.timestamp ASC, e.block_idx ASC
        """,
        [session_id],
    )
    col_names = [d[0] for d in cur.description]
    parsed_messages: list[dict] = []
    seen_uuids: set[str] = set()
    for row in cur.fetchall():
        rec = dict(zip(col_names, row, strict=True))
        kind = rec["kind"]
        exec_secs = (
            rec["execution_time"].total_seconds()
            if rec["execution_time"] is not None
            else None
        )
        thinking_preview, thinking_has_more = _thinking_preview(rec["thinking_text"])
        # Assistant turns expand to one row per content block (text, thinking,
        # tool_use, etc.), all sharing one uuid. Only the first block in each
        # turn gets the anchor id so DOM ids stay unique.
        uuid_val = rec["uuid"] or ""
        is_first = bool(uuid_val) and uuid_val not in seen_uuids
        if uuid_val:
            seen_uuids.add(uuid_val)

        capped_text = _cap(rec["text"])
        pretty_tool_input = _pretty_tool_input(rec["tool_input"])
        capped_tool_result = _cap(rec["tool_use_result"])

        text_lines, text_chars, text_needs_collapse = _collapse_info(capped_text)
        tool_input_lines, tool_input_chars, tool_input_needs_collapse = _collapse_info(
            pretty_tool_input
        )
        tool_result_lines, tool_result_chars, tool_result_needs_collapse = (
            _collapse_info(capped_tool_result)
        )

        is_assistant_entry = kind in ("agent_text", "agent_thinking")
        if is_first and is_assistant_entry:
            tokens_summary, tokens_title = _token_badge_strings(
                int(rec["input_tokens"] or 0),
                int(rec["output_tokens"] or 0),
                int(rec["cache_read_tokens"] or 0),
                int(rec["cache_creation_tokens"] or 0),
            )
        else:
            tokens_summary, tokens_title = "", ""

        ts = rec["timestamp"]
        if ts is not None and hasattr(ts, "strftime"):
            timestamp_full = ts.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_time = ts.strftime("%H:%M:%S")
        elif ts:
            # Defensive path: driver handed us a string. Slice the canonical
            # ``YYYY-MM-DD HH:MM:SS`` / ``YYYY-MM-DDTHH:MM:SS`` layout.
            ts_str = str(ts)
            timestamp_full = ts_str[:19]
            timestamp_time = ts_str[11:19]
        else:
            timestamp_full = ""
            timestamp_time = ""

        parsed_messages.append(
            {
                "uuid": uuid_val,
                "is_first_block": is_first,
                "timestamp": timestamp_time,
                "timestamp_full": timestamp_full,
                "kind": kind,
                "is_sidechain": bool(rec["is_sidechain"]),
                "text": capped_text,
                "text_line_count": text_lines,
                "text_char_count": text_chars,
                "text_needs_collapse": text_needs_collapse,
                "thinking_text": _cap(rec["thinking_text"]),
                "thinking_preview": thinking_preview,
                "thinking_has_more": thinking_has_more,
                "command_label": (
                    _slash_command_label(rec["text"]) if kind == "slash_command" else ""
                ),
                "tool_name": rec["tool_name"] or "",
                "tool_hint": _tool_hint(rec["tool_name"] or "", rec["tool_input"]),
                "tool_input": pretty_tool_input,
                "tool_input_line_count": tool_input_lines,
                "tool_input_char_count": tool_input_chars,
                "tool_input_needs_collapse": tool_input_needs_collapse,
                "tool_result": capped_tool_result,
                "tool_result_line_count": tool_result_lines,
                "tool_result_char_count": tool_result_chars,
                "tool_result_needs_collapse": tool_result_needs_collapse,
                "is_error": _coerce_bool(rec["is_error"]),
                "exec_time": (
                    _format_exec_time(exec_secs) if exec_secs is not None else ""
                ),
                "tool_use_id": rec["tool_use_id"] or "",
                "tokens_summary": tokens_summary,
                "tokens_title": tokens_title,
            }
        )
    return parsed_messages


_CUMULATIVE_MAX_BUCKETS = 120
_CUMULATIVE_RAW_THRESHOLD = 200
_BLOAT_TOP_N = 20
_BLOAT_BASH_FIRST_WORD_MAX = 32
# Categories describe what the new context tokens *are* (orthogonal to the
# main-vs-subagent split, which is tracked separately — subagents do reads,
# writes, and conversation too).
_BLOAT_CATEGORIES = ("Read", "Created", "Conversation")
_BLOAT_AGENTS = ("Main", "Subagent")

# Colors cycled through for per-invocation chart series. Main agent always
# gets _MAIN_AGENT_COLOR; the top-N most expensive subagent invocations get
# the palette entries in cost-descending order, so the expensive "bad apples"
# land on the most visually distinct colors.  Palette size caps how many
# invocations appear individually — the rest roll into an "Other" series.
_MAIN_AGENT_COLOR = "#3b5bdb"
_INVOCATION_COLOR_PALETTE = (
    "#c62828",  # red — reserved for the single most expensive call
    "#c86b1a",
    "#c0a000",
    "#2e8b57",
    "#0e7490",
    "#5a6e9a",
    "#8a5ad0",
    "#a0429e",
    "#6b4f9e",
    "#3f7d50",
    "#a26a3f",
    "#7b3b8a",
)
_INVOCATION_OTHER_COLOR = "#888888"
_INVOCATION_TOP_N = len(_INVOCATION_COLOR_PALETTE)


def _basename(path: str | None) -> str:
    """Best-effort POSIX basename of an arbitrary path string."""
    if not path:
        return "(unknown)"
    return PurePosixPath(path).name or path


def _safe_json(raw: str | None) -> dict:
    """Parse JSON, returning an empty dict on failure."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


_BLOAT_READ_TOOLS = {"Grep": "grep", "Glob": "glob"}
_BLOAT_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Bash dispatchers whose first word is uninteresting on its own; show two words
# so e.g. `uv run poe check` and `uv run poe test` get separate buckets.
_BLOAT_BASH_DISPATCHERS = frozenset(
    {"uv", "git", "npm", "cargo", "make", "pnpm", "yarn", "poetry", "uvx", "pipx"}
)


def _classify_bucket(  # noqa: PLR0911
    *,
    tool_name: str | None,
    tool_input_raw: str | None,
    user_block_type: str | None,
) -> tuple[str, str]:
    """Return (bucket_label, category) for one bloat row.

    Independent of main-vs-subagent: subagents do reads, writes, and
    conversation too. The agent dimension is added by the caller.
    """
    if tool_name:
        if tool_name == "Read":
            d = _safe_json(tool_input_raw)
            return (f"file read: {_basename(d.get('file_path'))}", "Read")
        if tool_name in _BLOAT_READ_TOOLS:
            return (_BLOAT_READ_TOOLS[tool_name], "Read")
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_name, "Read")
        if tool_name == "Bash":
            d = _safe_json(tool_input_raw)
            cmd_str = (d.get("command") or "").strip()
            words = cmd_str.split()
            if not words:
                return ("bash: (empty)", "Read")
            head = words[0]
            if head in _BLOAT_BASH_DISPATCHERS and len(words) > 1:
                head = f"{head} {words[1]}"
            head = head[:_BLOAT_BASH_FIRST_WORD_MAX]
            return (f"bash: {head}", "Read")
        if tool_name in _BLOAT_WRITE_TOOLS:
            d = _safe_json(tool_input_raw)
            fname = d.get("file_path") or d.get("notebook_path")
            return (f"file write: {_basename(fname)}", "Created")
        if tool_name.startswith("mcp__"):
            return (f"mcp: {tool_name[len('mcp__') :]}", "Read")
        return (f"tool: {tool_name}", "Read")
    if user_block_type == "tool_result":
        return ("tool result (unknown)", "Read")
    if user_block_type in (None, "text"):
        return ("human input", "Conversation")
    return ("agent context", "Conversation")


def _aggregate_per_model(rows: list[tuple]) -> tuple[list[dict], float]:
    """Group rows by model, returning (per_model, total_cost)."""
    by_model: dict[str, dict] = {}
    running = 0.0
    for cost_row in rows:
        _ts, _is_side, model = cost_row[0], cost_row[1], cost_row[2]
        in_tok, out_tok, cr_tok, cc_tok, cc_5m, cc_1h = (
            int(v or 0) for v in cost_row[3:9]
        )
        eff_5m = cc_tok if (cc_5m == 0 and cc_1h == 0 and cc_tok > 0) else cc_5m
        cost = compute_cost_usd(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=cr_tok,
            cache_creation_5m=eff_5m,
            cache_creation_1h=cc_1h,
        )
        running += cost
        key = model or "(unknown)"
        bucket = by_model.setdefault(
            key,
            {
                "model": key,
                "messages": 0,
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_creation": 0,
                "cache_creation_5m": 0,
                "cache_creation_1h": 0,
                "cost_usd": 0.0,
            },
        )
        bucket["messages"] += 1
        bucket["input"] += in_tok
        bucket["output"] += out_tok
        bucket["cache_read"] += cr_tok
        bucket["cache_creation"] += cc_tok
        bucket["cache_creation_5m"] += cc_5m
        bucket["cache_creation_1h"] += cc_1h
        bucket["cost_usd"] += cost
    per_model = sorted(by_model.values(), key=lambda d: d["cost_usd"], reverse=True)
    for entry in per_model:
        entry["cost"] = format_cost(entry["cost_usd"])
    return per_model, running


def _aggregate_bloat(rows: list[tuple]) -> tuple[dict, dict]:
    """Aggregate bloat rows into bucket_totals and category_totals dicts.

    Both dicts are keyed by (label, agent) so the main vs. subagent split is
    preserved as an orthogonal dimension, not collapsed into one category.

    Each bucket entry also carries ``top_uuid`` / ``top_cost_usd`` — the
    single message that contributed the most cache-write cost to that
    bucket. That's the worst-offender users want to jump to from the
    contributors table.
    """
    bucket_totals: dict[tuple[str, str], dict] = {}
    category_totals: dict[tuple[str, str], dict] = {
        (c, a): {"category": c, "agent": a, "tokens": 0, "cost_usd": 0.0}
        for c in _BLOAT_CATEGORIES
        for a in _BLOAT_AGENTS
    }
    for bloat_row in rows:
        (
            uuid,
            is_side,
            model,
            cc_total,
            cc_5m,
            cc_1h,
            user_block_type,
            tool_name,
            tool_input_raw,
        ) = bloat_row
        cc_total = int(cc_total or 0)
        cc_5m = int(cc_5m or 0)
        cc_1h = int(cc_1h or 0)
        if cc_total <= 0:
            continue
        eff_5m, eff_1h = cc_5m, cc_1h
        if cc_5m == 0 and cc_1h == 0:
            eff_5m = cc_total
        cost = compute_cost_usd(
            model=model,
            cache_creation_5m=eff_5m,
            cache_creation_1h=eff_1h,
        )
        bucket, category = _classify_bucket(
            tool_name=tool_name,
            tool_input_raw=tool_input_raw,
            user_block_type=user_block_type,
        )
        agent = "Subagent" if is_side else "Main"
        cat_entry = category_totals[(category, agent)]
        cat_entry["tokens"] += cc_total
        cat_entry["cost_usd"] += cost
        bucket_entry = bucket_totals.setdefault(
            (bucket, agent),
            {
                "bucket": bucket,
                "category": category,
                "agent": agent,
                "tokens": 0,
                "cost_usd": 0.0,
                "top_uuid": uuid,
                "top_cost_usd": cost,
            },
        )
        bucket_entry["tokens"] += cc_total
        bucket_entry["cost_usd"] += cost
        if cost > bucket_entry["top_cost_usd"]:
            bucket_entry["top_uuid"] = uuid
            bucket_entry["top_cost_usd"] = cost
    return bucket_totals, category_totals


def _resolve_uuid_range(
    attrib_rows: list[tuple], from_uuid: str, to_uuid: str
) -> tuple[int, int]:
    """Look up the inclusive index range that ``from_uuid``..``to_uuid`` spans.

    Linear scan over already-fetched rows. Both ends inclusive; missing
    uuid raises 404. If ``from_uuid`` lands later than ``to_uuid`` (the
    user dragged right-to-left), swap them so the slice stays sane.
    """
    lo: int | None = None
    hi: int | None = None
    for i, row in enumerate(attrib_rows):
        uuid = row[1]
        if uuid == from_uuid and lo is None:
            lo = i
        if uuid == to_uuid:
            hi = i
    if lo is None or hi is None:
        raise HTTPException(status_code=404, detail="uuid not in session")
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _build_cost_context(
    db,
    session_id: str,
    *,
    range_filter: tuple[str, str] | None = None,
) -> dict:
    """Build the data structures the Cost tab template needs.

    When ``range_filter`` is set, the bloat tables (rollup + top
    contributors) are scoped to that uuid range. The chart context still
    covers the whole session — the chart is the user's navigation surface,
    not a filtered view.
    """
    # Single query drives three consumers: per-model rollup, bloat
    # aggregator, *and* the per-message cost chart.  The parent-user-message
    # join unnests `u.message.content` (an array of blocks — parallel-tool
    # calls produce one tool_result per block), then prefers a tool_result
    # block over text.  Without the explicit unnest, the outer LEFT JOIN's
    # ON clause would silently drop the tool_use_id whenever the result
    # lived at index >= 1, mis-attributing legitimate file reads as "human
    # input".
    attrib_rows = db.execute(
        build_cost_attribution_sql("WHERE session_id = ?"),
        [session_id],
    ).fetchall()

    # Columns from build_cost_attribution_sql (0-indexed):
    #   0 session_id, 1 uuid, 2 timestamp, 3 is_sidechain, 4 model,
    #   5 input_tokens, 6 output_tokens, 7 cache_read_tokens, 8 cc_total,
    #   9 cache_creation_5m, 10 cache_creation_1h,
    #   11 user_block_type, 12 tool_name, 13 tool_input.
    #
    # _aggregate_per_model reads 9 columns starting from timestamp (index 2),
    # so the slice rows[2:11] gives it exactly what it needs.
    per_model_rows = [row[2:11] for row in attrib_rows]
    per_model, total_cost = _aggregate_per_model(per_model_rows)

    if range_filter is not None:
        lo, hi = _resolve_uuid_range(attrib_rows, *range_filter)
        bloat_attrib_rows = attrib_rows[lo : hi + 1]
    else:
        bloat_attrib_rows = attrib_rows

    bloat_rows = [
        (
            row[1],  # uuid
            row[3],  # is_sidechain
            row[4],  # model
            row[8],  # cc_total
            row[9],  # cache_creation_5m
            row[10],  # cache_creation_1h
            row[11],  # user_block_type
            row[12],  # tool_name
            row[13],  # tool_input
        )
        for row in bloat_attrib_rows
    ]
    bucket_totals, category_totals = _aggregate_bloat(bloat_rows)

    # Task/Agent invocations give us the subagent_type for each sidechain run.
    # Ordered by called_at so the chart builder can sweep through them in one
    # pass alongside the (already-sorted) attrib_rows.
    subagent_type_timeline = db.execute(
        """
        SELECT called_at,
               json_extract_string(tool_input, '$.subagent_type')
        FROM tool_calls
        WHERE session_id = ? AND tool_name IN ('Task', 'Agent')
        ORDER BY called_at
        """,
        [session_id],
    ).fetchall()
    # Skip Task calls without a subagent_type (general Agent tool uses without
    # a named type) so they don't overwrite the current type with None mid-run.
    subagent_type_timeline = [(t, s) for t, s in subagent_type_timeline if s]

    # Chart construction touches Plotly + DuckDB-typed data; isolate
    # failures so the rest of the cost tab (per-model rollup, bloat
    # tables) still renders for the user. The template handles the
    # ``chart_error`` key by replacing the plot with an inline notice.
    try:
        chart = _build_chart_from_attrib(attrib_rows, subagent_type_timeline)
        chart_error: str | None = None
    except Exception as exc:
        logger.exception("Failed to build session cost chart for %s", session_id)
        chart = {
            "messages": 0,
            "points": 0,
            "bucket_size": 1,
            "figure_json": "",
            "view_map": {"total": [], "agent": [], "category": [], "invocations": []},
            "annotation_view_map": {
                "total": [],
                "agent": [],
                "category": [],
                "invocations": [],
            },
            "marker_trace": -1,
            "invocation_series": [],
            "invocation_summary": [],
            "has_subagents": False,
        }
        chart_error = f"{type(exc).__name__}: {exc}"
    raw_tokens = sum(c["tokens"] for c in category_totals.values())
    total_bloat_tokens = raw_tokens or 1
    total_bloat_cost = sum(c["cost_usd"] for c in category_totals.values())

    def _cell(category: str, agent: str) -> dict:
        entry = category_totals[(category, agent)]
        return {
            "tokens": entry["tokens"],
            "cost_usd": entry["cost_usd"],
            "cost": format_cost(entry["cost_usd"]),
            "pct": 100.0 * entry["tokens"] / total_bloat_tokens,
        }

    rollup = []
    for category in _BLOAT_CATEGORIES:
        main = _cell(category, "Main")
        sub = _cell(category, "Subagent")
        total_tokens = main["tokens"] + sub["tokens"]
        total_cost_cat = main["cost_usd"] + sub["cost_usd"]
        rollup.append(
            {
                "category": category,
                "main": main,
                "subagent": sub,
                "total": {
                    "tokens": total_tokens,
                    "cost_usd": total_cost_cat,
                    "cost": format_cost(total_cost_cat),
                    "pct": 100.0 * total_tokens / total_bloat_tokens,
                },
            }
        )

    agent_totals = {
        agent: {
            "tokens": sum(
                category_totals[(c, agent)]["tokens"] for c in _BLOAT_CATEGORIES
            ),
            "cost_usd": sum(
                category_totals[(c, agent)]["cost_usd"] for c in _BLOAT_CATEGORIES
            ),
        }
        for agent in _BLOAT_AGENTS
    }
    rollup_totals = {
        "main": {
            "tokens": agent_totals["Main"]["tokens"],
            "cost": format_cost(agent_totals["Main"]["cost_usd"]),
            "pct": 100.0 * agent_totals["Main"]["tokens"] / total_bloat_tokens,
        },
        "subagent": {
            "tokens": agent_totals["Subagent"]["tokens"],
            "cost": format_cost(agent_totals["Subagent"]["cost_usd"]),
            "pct": 100.0 * agent_totals["Subagent"]["tokens"] / total_bloat_tokens,
        },
        "total": {
            "tokens": raw_tokens,
            "cost": format_cost(total_bloat_cost),
        },
    }

    sorted_buckets = sorted(
        bucket_totals.values(), key=lambda d: d["tokens"], reverse=True
    )
    extra_count = max(0, len(sorted_buckets) - _BLOAT_TOP_N)
    top_buckets_view = [
        {
            "bucket": bucket["bucket"],
            "category": bucket["category"],
            "agent": bucket["agent"],
            "tokens": bucket["tokens"],
            "cost_usd": bucket["cost_usd"],
            "cost": format_cost(bucket["cost_usd"]),
            "pct": 100.0 * bucket["tokens"] / total_bloat_tokens,
            "top_uuid": bucket["top_uuid"],
            "top_cost": format_cost(bucket["top_cost_usd"]),
        }
        for bucket in sorted_buckets[:_BLOAT_TOP_N]
    ]

    return {
        "per_model": per_model,
        "total_cost_usd": total_cost,
        "total_cost": format_cost(total_cost),
        "chart": chart,
        "bloat_rollup": rollup,
        "bloat_rollup_totals": rollup_totals,
        "bloat_top_buckets": top_buckets_view,
        "bloat_extra_count": extra_count,
        "bloat_total_tokens": raw_tokens,
        "bloat_total_cost": format_cost(total_bloat_cost),
        "has_data": bool(attrib_rows),
        "bloat_filter_active": range_filter is not None,
        "bloat_filter_summary": (
            f"Showing {len(bloat_attrib_rows)} of {len(attrib_rows)} messages"
            if range_filter is not None
            else ""
        ),
        "bloat_filter_count": len(bloat_attrib_rows) if range_filter is not None else 0,
        "bloat_filter_total": len(attrib_rows),
        "chart_error": chart_error,
    }


_SPIKE_MIN_N = 6
_SLOPE_MIN_N = 10
_SPIKE_TOP_N = 6
_SLOPE_TOP_N = 8
_SPIKE_ABS_FLOOR = 0.02
_SPIKE_MEDIAN_MULTIPLIER = 1.5
_SLOPE_SIGMA_MULTIPLIER = 1.5
_SLOPE_WINDOW = 5
_SLOPE_DEDUPE_RADIUS = 5
# Slope detection needs ≥ 2 positive deltas to compute variance — 1 value
# has zero spread and would collapse the threshold to 0.
_SLOPE_MIN_POSITIVE_DELTAS = 2


def _detect_inflection_points(
    uuids: list[str],
    inc_usd: list[float],
    cum: list[float],
) -> list[dict]:
    """Spike + slope inflection detection on the raw per-message arrays.

    Thresholds:
      * Spike: inc_usd[i] >= max($0.02, 1.5 * median(inc_usd)); top-6.
      * Slope: delta(i) = cum[i] - cum[i-W] (W=5) >= 1.5 stdev of positive
        deltas; top-8; de-duped +/-5 against spike indices.
      * Minimum N: 10 for slope, 6 for spike.
    """
    n = len(inc_usd)
    markers: list[dict] = []
    spike_indices: set[int] = set()

    if n >= _SPIKE_MIN_N:
        sorted_inc = sorted(inc_usd)
        mid = n // 2
        median = (
            sorted_inc[mid]
            if n % 2 == 1
            else 0.5 * (sorted_inc[mid - 1] + sorted_inc[mid])
        )
        threshold = max(_SPIKE_ABS_FLOOR, _SPIKE_MEDIAN_MULTIPLIER * median)
        candidates = [(i, inc_usd[i]) for i in range(n) if inc_usd[i] >= threshold]
        candidates.sort(key=lambda t: t[1], reverse=True)
        for idx, inc in candidates[:_SPIKE_TOP_N]:
            spike_indices.add(idx)
            markers.append(
                {
                    "idx": idx,
                    "uuid": uuids[idx],
                    "kind": "spike",
                    "inc_usd": inc,
                    "cum_usd": cum[idx],
                }
            )

    if n >= _SLOPE_MIN_N:
        deltas: list[tuple[int, float]] = []
        positive_deltas: list[float] = []
        for i in range(_SLOPE_WINDOW, n):
            delta = cum[i] - cum[i - _SLOPE_WINDOW]
            deltas.append((i, delta))
            if delta > 0:
                positive_deltas.append(delta)
        # Require ≥ 2 positive deltas AND non-zero stdev so that a degenerate
        # distribution (all cheap, or one identical positive value) doesn't
        # collapse the threshold to 0 and mark every non-decreasing window.
        if len(positive_deltas) >= _SLOPE_MIN_POSITIVE_DELTAS:
            mean_d = sum(positive_deltas) / len(positive_deltas)
            var = sum((d - mean_d) ** 2 for d in positive_deltas) / len(positive_deltas)
            sigma = math.sqrt(var)
            if sigma > 0:
                # Same absolute floor the spike branch uses so noise near
                # the cent level can't fire a marker even if sigma is tiny
                # but nonzero.
                threshold = max(_SLOPE_SIGMA_MULTIPLIER * sigma, _SPIKE_ABS_FLOOR)
                # Filter + de-dupe against spikes
                filtered = [
                    (i, d)
                    for i, d in deltas
                    if d >= threshold
                    and not any(
                        abs(i - s) <= _SLOPE_DEDUPE_RADIUS for s in spike_indices
                    )
                ]
                filtered.sort(key=lambda t: t[1], reverse=True)
                for idx, _d in filtered[:_SLOPE_TOP_N]:
                    markers.append(
                        {
                            "idx": idx,
                            "uuid": uuids[idx],
                            "kind": "slope",
                            "inc_usd": inc_usd[idx],
                            "cum_usd": cum[idx],
                        }
                    )

    markers.sort(key=lambda m: m["idx"])
    return markers


def _bucket_series(
    increments_by_series: dict[str, list[float]],
    total_messages: int,
) -> tuple[dict[str, list[float]], int]:
    """Bucket per-series *increments* into ≤120 buckets, then cumsum.

    Bucketing increments (not cumulatives) keeps stacked series aligned to
    within float rounding error — well below SVG pixel resolution — whereas
    bucketing cumulatives independently could accumulate visible drift.

    Returns (cumulative_by_series, bucket_size).
    """
    bucket_size = 1
    if total_messages > _CUMULATIVE_RAW_THRESHOLD:
        bucket_size = max(1, math.ceil(total_messages / _CUMULATIVE_MAX_BUCKETS))

    result: dict[str, list[float]] = {}
    for name, incs in increments_by_series.items():
        # Bucket the increments: sum each chunk.
        bucketed_incs: list[float] = []
        for i in range(0, total_messages, bucket_size):
            chunk = incs[i : i + bucket_size]
            bucketed_incs.append(sum(chunk))
        # Cumsum.
        running = 0.0
        cum: list[float] = []
        for inc in bucketed_incs:
            running += inc
            cum.append(running)
        result[name] = cum
    return result, bucket_size


_template_activated: list[bool] = [False]


def _ensure_template() -> None:
    """Register nolegend's "tufte" template the first time a chart is built.

    Mirrors the cost-overview lazy activator so importing the handler from
    a CLI / test that doesn't render a chart leaves Plotly's default
    template alone.
    """
    if not _template_activated[0]:
        nolegend.activate()
        _template_activated[0] = True


_SPIKE_MARKER_COLOR = "#c62828"
_SLOPE_MARKER_COLOR = "#c0a000"
_AGENT_SUB_COLOR = "#8a5ad0"
_CATEGORY_READ_COLOR = "#c86b1a"
_CATEGORY_CREATED_COLOR = "#2e8b57"
_CATEGORY_CONVERSATION_COLOR = "#5a6e9a"


def _rank_invocations(
    uuids: list[str],
    inc_usd: list[float],
    invocation_ids: list[int | None],
) -> tuple[list[dict], dict[int, str], dict[int, int], dict[str, list[float]]]:
    """Rank subagent invocations by cost and build their per-series increments.

    Returns ``(ranked, first_uuid_by_inv, top_inv_index, per_inv_inc)``:

    * ``ranked``  — all invocations sorted by ``cost_usd`` descending. Each
      row is ``{"inv": inv_id, "cost_usd": float, "messages": int}``.
    * ``first_uuid_by_inv`` — inv_id → uuid of that invocation's earliest
      sidechain message (for deep-links).
    * ``top_inv_index`` — inv_id → 0-based rank for invocations that fit in
      the palette; others are absent and will roll up into ``inv_other``.
    * ``per_inv_inc`` — series-key → per-message increment list, keyed by
      ``inv_<rank>`` for the top-N and ``inv_other`` for the remainder.
    """
    n = len(uuids)
    per_inv_totals: dict[int, dict] = {}
    first_uuid_by_inv: dict[int, str] = {}
    for i in range(n):
        inv = invocation_ids[i]
        if inv is None:
            continue
        bucket = per_inv_totals.setdefault(
            inv, {"inv": inv, "cost_usd": 0.0, "messages": 0}
        )
        bucket["cost_usd"] += inc_usd[i]
        bucket["messages"] += 1
        if inv not in first_uuid_by_inv:
            first_uuid_by_inv[inv] = uuids[i]
    ranked = sorted(per_inv_totals.values(), key=lambda d: d["cost_usd"], reverse=True)
    top_invs = ranked[:_INVOCATION_TOP_N]
    top_inv_index = {entry["inv"]: rank for rank, entry in enumerate(top_invs)}
    has_other = len(ranked) > _INVOCATION_TOP_N
    per_inv_inc: dict[str, list[float]] = {
        f"inv_{rank}": [0.0] * n for rank in range(len(top_invs))
    }
    if has_other:
        per_inv_inc["inv_other"] = [0.0] * n
    for i in range(n):
        inv = invocation_ids[i]
        if inv is None:
            continue
        rank = top_inv_index.get(inv)
        key = f"inv_{rank}" if rank is not None else "inv_other"
        per_inv_inc[key][i] = inc_usd[i]
    return ranked, first_uuid_by_inv, top_inv_index, per_inv_inc


def _build_invocation_views(
    *,
    ranked: list[dict],
    invocation_types: list[str],
    per_inv_first_uuid: dict[int, str],
    main_cost: float,
    main_messages: int,
) -> tuple[list[dict], list[dict]]:
    """Build the chart-legend series + summary-table rows for invocations."""
    top_invs = ranked[:_INVOCATION_TOP_N]
    other_invs = ranked[_INVOCATION_TOP_N:]

    def _label(entry: dict, rank: int) -> str:
        stype = invocation_types[entry["inv"]] or "(unknown)"
        return f"#{rank + 1} {stype}"

    series: list[dict] = [
        {
            "key": "main",
            "name": "Main",
            "color": _MAIN_AGENT_COLOR,
            "cost": format_cost(main_cost),
            "messages": main_messages,
            "first_uuid": None,
        },
    ]
    for rank, entry in enumerate(top_invs):
        series.append(
            {
                "key": f"inv_{rank}",
                "name": _label(entry, rank),
                "color": _INVOCATION_COLOR_PALETTE[rank],
                "cost": format_cost(entry["cost_usd"]),
                "messages": entry["messages"],
                "first_uuid": per_inv_first_uuid.get(entry["inv"]),
            }
        )
    if other_invs:
        other_cost = sum(e["cost_usd"] for e in other_invs)
        other_msgs = sum(e["messages"] for e in other_invs)
        series.append(
            {
                "key": "inv_other",
                "name": f"Other ({len(other_invs)} invocations)",
                "color": _INVOCATION_OTHER_COLOR,
                "cost": format_cost(other_cost),
                "messages": other_msgs,
                "first_uuid": None,
            }
        )

    summary = [
        {
            "rank": rank + 1,
            "subagent_type": invocation_types[entry["inv"]] or "(unknown)",
            "cost": format_cost(entry["cost_usd"]),
            "cost_usd": entry["cost_usd"],
            "messages": entry["messages"],
            "first_uuid": per_inv_first_uuid.get(entry["inv"]),
            "color": _INVOCATION_COLOR_PALETTE[rank]
            if rank < _INVOCATION_TOP_N
            else _INVOCATION_OTHER_COLOR,
        }
        for rank, entry in enumerate(ranked)
    ]
    return series, summary


# Smaller deltas than this are treated as "no offset needed" — saves a
# stray leader arrow on labels whose spread position rounds back to the
# actual y within float noise.
_LABEL_OFFSET_EPSILON = 1e-9


def _spread_y(actual_ys: list[float], min_gap: float) -> list[float]:
    """Spread label y-positions so adjacent labels are at least ``min_gap`` apart.

    Sort descending, push each label down until it's ``min_gap`` below
    its predecessor, then shift the whole stack to keep its midpoint at
    the original midpoint (so labels stay roughly centred on their
    actual positions). Returns positions in the original order.
    """
    n = len(actual_ys)
    if n <= 1 or min_gap <= 0:
        return list(actual_ys)
    order = sorted(range(n), key=lambda i: -actual_ys[i])
    sorted_ys = [actual_ys[i] for i in order]
    spread = [sorted_ys[0]]
    for i in range(1, n):
        spread.append(min(sorted_ys[i], spread[-1] - min_gap))
    actual_mid = (max(sorted_ys) + min(sorted_ys)) / 2
    spread_mid = (spread[0] + spread[-1]) / 2
    shift = actual_mid - spread_mid
    spread = [y + shift for y in spread]
    result = [0.0] * n
    for k, orig_i in enumerate(order):
        result[orig_i] = spread[k]
    return result


def _build_label_annotations(
    view_label_specs: dict[str, list[tuple[str, str, float]]],
    *,
    x_max: int,
    default_view: str,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Build Plotly annotations for direct line-end labels, per view.

    Returns ``(annotations, annotation_view_map)``. The map keys are
    view names ("total", "agent", "category", "invocations") and the
    values are the indices in ``annotations`` belonging to that view —
    the JS toggle flips each annotation's ``visible`` based on this
    lookup since Plotly's Annotation type doesn't permit a custom
    ``meta`` field.

    Within a view, label y-positions are spread to avoid overlap; when
    a label is offset from its line's actual endpoint, a thin leader
    arrow connects them.
    """
    annotations: list[dict] = []
    view_map: dict[str, list[int]] = {view: [] for view in view_label_specs}
    for view, specs in view_label_specs.items():
        if not specs:
            continue
        actual_ys = [y for _name, _color, y in specs]
        y_range = max(actual_ys) - min(actual_ys) if len(actual_ys) > 1 else 0.0
        min_gap = max(y_range * 0.14, max(actual_ys) * 0.06, 0.0001)
        spread = _spread_y(actual_ys, min_gap)
        for (name, color, actual_y), label_y in zip(specs, spread, strict=True):
            offset = abs(actual_y - label_y) > _LABEL_OFFSET_EPSILON
            view_map[view].append(len(annotations))
            annotations.append(
                {
                    "x": x_max,
                    "y": actual_y,
                    "xref": "x",
                    "yref": "y",
                    "text": name,
                    "ax": x_max,
                    "ay": label_y,
                    "axref": "x",
                    "ayref": "y",
                    "xanchor": "left",
                    "yanchor": "middle",
                    "xshift": 8,
                    "showarrow": offset,
                    "arrowhead": 0,
                    "arrowwidth": 0.8,
                    "arrowcolor": color,
                    "font": {"color": color, "size": 12},
                    "visible": view == default_view,
                }
            )
    return annotations, view_map


def _render_multi_chart(  # noqa: PLR0913, PLR0915
    uuids: list[str],
    inc_usd: list[float],
    is_sidechain_list: list[bool],
    categories: list[str],
    invocation_ids: list[int | None],
    invocation_types: list[str],
) -> dict:
    """Build the per-series Plotly figure + marker overlay + view map.

    ``invocation_ids[i]`` is the 0-based Task/Agent invocation index the
    i-th message belongs to (``None`` for main-agent messages).
    ``invocation_types`` lists each invocation's ``subagent_type`` in
    called_at order — so ``invocation_types[invocation_ids[i]]`` names the
    agent type of message i.

    Returns a dict consumed by ``_session_cost.html``:

    * ``figure_json``        — serialised ``go.Figure``: a Scatter trace
      per series (total / main / sub / read / created / conversation /
      per-invocation) plus one marker overlay trace.
    * ``view_map``           — mapping ``{view_name: [trace_indices]}``
      so the client toggle can flip ``visible`` via ``Plotly.restyle``.
    * ``marker_trace``       — index of the always-on marker overlay.
    * ``messages``, ``points``, ``bucket_size`` — chart metadata for the
      caption / box-select-precision warning.
    * ``has_subagents``      — gate for showing the "By invocation" view.
    * ``invocation_series``  — legend rows used in the per-invocation
      summary table below the chart.
    * ``invocation_summary`` — outlier table shown beneath the chart.

    The "by invocation" view ranks invocations by total cost descending;
    top-N get distinct palette colours, the rest fold into "Other".
    """
    n_messages = len(uuids)
    empty_view_map = {"total": [], "agent": [], "category": [], "invocations": []}
    if n_messages == 0:
        return {
            "messages": 0,
            "points": 0,
            "bucket_size": 1,
            "figure_json": "",
            "view_map": empty_view_map,
            "annotation_view_map": dict(empty_view_map),
            "marker_trace": -1,
            "invocation_series": [],
            "invocation_summary": [],
            "has_subagents": False,
        }

    main_inc = [0.0 if is_sidechain_list[i] else inc_usd[i] for i in range(n_messages)]
    sub_inc = [inc_usd[i] if is_sidechain_list[i] else 0.0 for i in range(n_messages)]
    read_inc = [
        inc_usd[i] if categories[i] == "Read" else 0.0 for i in range(n_messages)
    ]
    created_inc = [
        inc_usd[i] if categories[i] == "Created" else 0.0 for i in range(n_messages)
    ]
    convo_inc = [
        inc_usd[i] if categories[i] == "Conversation" else 0.0
        for i in range(n_messages)
    ]

    ranked, per_inv_first_uuid, _top_inv_index, per_inv_inc = _rank_invocations(
        uuids, inc_usd, invocation_ids
    )

    raw_cum: list[float] = []
    running = 0.0
    for inc in inc_usd:
        running += inc
        raw_cum.append(running)
    raw_markers = _detect_inflection_points(uuids, inc_usd, raw_cum)

    cumulatives, bucket_size = _bucket_series(
        {
            "total": inc_usd,
            "main": main_inc,
            "sub": sub_inc,
            "read": read_inc,
            "created": created_inc,
            "conversation": convo_inc,
            **per_inv_inc,
        },
        n_messages,
    )
    points = len(cumulatives["total"])

    invocation_series, invocation_summary = _build_invocation_views(
        ranked=ranked,
        invocation_types=invocation_types,
        per_inv_first_uuid=per_inv_first_uuid,
        main_cost=sum(main_inc),
        main_messages=sum(1 for i in range(n_messages) if not is_sidechain_list[i]),
    )

    # Per-bucket customdata: [first_uuid, last_uuid, msg_count]. Box-select
    # reads first-of-leftmost and last-of-rightmost so the resolved range
    # always covers every raw message under the dragged region.
    bucket_customdata: list[list[object]] = []
    for bidx in range(points):
        raw_first = bidx * bucket_size
        raw_last = min((bidx + 1) * bucket_size, n_messages) - 1
        bucket_customdata.append(
            [uuids[raw_first], uuids[raw_last], raw_last - raw_first + 1]
        )

    _ensure_template()
    fig = go.Figure()
    view_map: dict[str, list[int]] = {
        "total": [],
        "agent": [],
        "category": [],
        "invocations": [],
    }
    # Per-view label spec: (trace_name, color, last_y). Used after all
    # traces are added to compute spread label positions and emit one
    # annotation per visible series, tagged with `meta=view` so the JS
    # toggle can flip annotation visibility alongside trace visibility.
    view_label_specs: dict[str, list[tuple[str, str, float]]] = {
        "total": [],
        "agent": [],
        "category": [],
        "invocations": [],
    }
    x_axis = list(range(points))

    def _add_series(*, name: str, color: str, ys: list[float], view: str) -> None:
        view_map[view].append(len(fig.data))
        if ys:
            view_label_specs[view].append((name, color, ys[-1]))
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=ys,
                mode="lines",
                name=name,
                line={"color": color, "width": 1.5},
                customdata=bucket_customdata,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "msg %{customdata[2]} in bucket<br>"
                    "cum $%{y:.4f}<extra></extra>"
                ),
            )
        )

    _add_series(
        name="Total", color=_MAIN_AGENT_COLOR, ys=cumulatives["total"], view="total"
    )
    _add_series(
        name="Main", color=_MAIN_AGENT_COLOR, ys=cumulatives["main"], view="agent"
    )
    _add_series(
        name="Subagent", color=_AGENT_SUB_COLOR, ys=cumulatives["sub"], view="agent"
    )
    _add_series(
        name="Read", color=_CATEGORY_READ_COLOR, ys=cumulatives["read"], view="category"
    )
    _add_series(
        name="Created",
        color=_CATEGORY_CREATED_COLOR,
        ys=cumulatives["created"],
        view="category",
    )
    _add_series(
        name="Conversation",
        color=_CATEGORY_CONVERSATION_COLOR,
        ys=cumulatives["conversation"],
        view="category",
    )
    for s in invocation_series:
        ys = cumulatives.get(s["key"], [])
        if not ys:
            continue
        _add_series(name=s["name"], color=s["color"], ys=ys, view="invocations")

    total_cum = cumulatives["total"]
    marker_x: list[int] = []
    marker_y: list[float] = []
    marker_color: list[str] = []
    marker_customdata: list[list[object]] = []
    for m in raw_markers:
        raw_idx = m["idx"]
        bucket_idx = min(raw_idx // bucket_size, points - 1)
        marker_x.append(bucket_idx)
        marker_y.append(total_cum[bucket_idx])
        marker_color.append(
            _SPIKE_MARKER_COLOR if m["kind"] == "spike" else _SLOPE_MARKER_COLOR
        )
        marker_customdata.append(
            [
                m["uuid"],
                m["kind"],
                format_cost(m["inc_usd"]),
                format_cost(m["cum_usd"]),
            ]
        )
    marker_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            name="Markers",
            marker={
                "color": marker_color,
                "size": 14,
                "symbol": "line-ns",
                "line": {"color": marker_color, "width": 3},
            },
            customdata=marker_customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "inc %{customdata[2]}<br>"
                "cum %{customdata[3]}<extra></extra>"
            ),
        )
    )

    # Plotly's `visible` accepts ``True | False | "legendonly"`` — typed
    # as the union so ty doesn't widen the literal-string list.
    visible: list[bool | str] = ["legendonly"] * len(fig.data)
    for tidx in view_map["total"]:
        visible[tidx] = True
    visible[marker_trace_idx] = True
    for tidx, v in enumerate(visible):
        fig.data[tidx].visible = v

    # Per-view label annotations with vertical spreading + leader lines.
    # The JS view-toggle flips visibility per annotation index using the
    # returned ``annotation_view_map`` since Plotly's Annotation type
    # doesn't accept custom meta tags.
    label_annotations, annotation_view_map = _build_label_annotations(
        view_label_specs, x_max=points - 1, default_view="total"
    )

    fig.update_layout(
        template="tufte",
        showlegend=False,
        hovermode="closest",
        dragmode="select",
        selectdirection="h",
        xaxis_title="Message #"
        if bucket_size == 1
        else f"Bucket # ({bucket_size}/bucket)",
        yaxis_title="Cumulative cost (USD)",
        # Right margin reserves room for direct labels on the line ends.
        margin={"l": 60, "r": 130, "t": 20, "b": 60},
        annotations=label_annotations,
    )

    return {
        "messages": n_messages,
        "points": points,
        "bucket_size": bucket_size,
        "figure_json": fig.to_json(),
        "view_map": view_map,
        "annotation_view_map": annotation_view_map,
        "marker_trace": marker_trace_idx,
        "invocation_series": invocation_series,
        "invocation_summary": invocation_summary,
        "has_subagents": bool(ranked),
    }


def _build_chart_from_attrib(
    attrib_rows: list[tuple],
    subagent_type_timeline: list[tuple],
) -> dict:
    """Extract per-message arrays from the attribution query and render chart.

    ``subagent_type_timeline`` is a list of ``(called_at, subagent_type)``
    pairs in ascending time order — Task/Agent tool_use calls from the main
    thread. Each entry defines one invocation. Sidechain messages inherit
    the invocation id (= index into this list) of the most recent preceding
    call. Sequential subagents are attributed exactly; fully parallel
    subagents from one turn get lumped under whichever call came first —
    the visual breakdown stays meaningful even so.
    """
    invocation_types = [s for _t, s in subagent_type_timeline]
    uuids: list[str] = []
    inc_usd: list[float] = []
    is_sidechain_list: list[bool] = []
    categories: list[str] = []
    invocation_ids: list[int | None] = []
    timeline_idx = 0
    current_invocation: int | None = None
    for row in attrib_rows:
        (
            _session_id,
            uuid,
            timestamp,
            is_side,
            model,
            in_tok,
            out_tok,
            cr_tok,
            cc_total,
            cc_5m,
            cc_1h,
            user_block_type,
            tool_name,
            tool_input_raw,
        ) = row
        # Advance timeline pointer: every Task/Agent call whose called_at is
        # <= this message's timestamp is "in effect" for any sidechain rows
        # that follow. The pointer index itself is the invocation id.
        while (
            timeline_idx < len(subagent_type_timeline)
            and subagent_type_timeline[timeline_idx][0] <= timestamp
        ):
            current_invocation = timeline_idx
            timeline_idx += 1
        in_tok_i = int(in_tok or 0)
        out_tok_i = int(out_tok or 0)
        cr_tok_i = int(cr_tok or 0)
        cc_total_i = int(cc_total or 0)
        cc_5m_i = int(cc_5m or 0)
        cc_1h_i = int(cc_1h or 0)
        eff_5m = (
            cc_total_i
            if (cc_5m_i == 0 and cc_1h_i == 0 and cc_total_i > 0)
            else cc_5m_i
        )
        cost = compute_cost_usd(
            model=model,
            input_tokens=in_tok_i,
            output_tokens=out_tok_i,
            cache_read_tokens=cr_tok_i,
            cache_creation_5m=eff_5m,
            cache_creation_1h=cc_1h_i,
        )
        _bucket, category = _classify_bucket(
            tool_name=tool_name,
            tool_input_raw=tool_input_raw,
            user_block_type=user_block_type,
        )
        uuids.append(uuid)
        inc_usd.append(cost)
        is_sidechain_list.append(bool(is_side))
        categories.append(category)
        invocation_ids.append(current_invocation if is_side else None)

    return _render_multi_chart(
        uuids,
        inc_usd,
        is_sidechain_list,
        categories,
        invocation_ids,
        invocation_types,
    )


# --- Tokenscape -------------------------------------------------------
#
# "Where the cost went": each piece of content (system prompt, a user
# message, a tool_result file read, an assistant text block) becomes
# a *band* — a horizontal slab in a stacked-area chart that exists from
# the turn the content arrived until the cache that holds it gets
# discarded. A single early big read shows up as one wide rectangle
# spanning many turns: the visual area equals the cost it caused.
#
# We have exact API token totals per turn (input + cache_read =
# context the model saw). Within-turn allocation between bands uses
# char counts, normalised so each turn's column sums to the API total.
# The band heights for any given content are constant across the
# turns where it persists.
#
# Two events truncate or rebuild the cache:
# * 5-minute cache TTL break — cache_read drops to ~0, but the same
#   total context is re-sent and recached. Bands persist; we draw a
#   thin marker because the cache rebuild is itself expensive.
# * /compact — total context drops sharply, history collapses into a
#   summary. Bands reset.

_TOKENSCAPE_TOP_READS = 5
# Colour family for tool_result bands. The most expensive read gets
# the brightest coral; lesser reads fade through into a muted grey.
_TOKENSCAPE_READ_COLORS = (
    "#e3a08c",
    "#e9b3a3",
    "#eec5b9",
    "#f1d3ca",
    "#e6d7d0",
)
_TOKENSCAPE_OTHER_READ_COLOR = "#d8c8c0"
_TOKENSCAPE_CATEGORY_COLORS = {
    "system": "#b8b3a3",
    "user": "#7fbef0",
    "assistant_text": "#bda6e6",
}
# Drop ratios for distinguishing /compact (truncates) from a 5-min
# cache TTL break (rebuilds same context) — see module-doc above.
_TOKENSCAPE_COMPACT_DROP_RATIO = 0.5
_TOKENSCAPE_COMPACT_MIN_DROP = 5_000
# Anthropic's prompt cache TTL is 5 minutes for the default tier;
# longer idle gaps force the cache to be rebuilt next turn (cache_read
# drops to ~0, cache_creation jumps to roughly the previous context).
# Use a slight buffer below 5m to absorb timestamp jitter.
_TOKENSCAPE_TTL_BREAK_MIN_GAP_SECONDS = 4 * 60 + 30
_TOKENSCAPE_TTL_BREAK_CACHE_DROP_RATIO = 0.5
_TOKENSCAPE_TTL_BREAK_MIN_CACHE = 5_000


def _tokenscape_classify(kind: str | None) -> str | None:
    """Map ``session_messages_enriched.kind`` to a tokenscape category."""
    if kind in ("human_prompt", "slash_command", "subagent_prompt"):
        return "user"
    if kind in ("agent_text", "agent_thinking", "agent_tool_call"):
        return "assistant_text"
    if kind == "tool_result":
        return "tool_result"
    return None


def _tokenscape_label(tool_name: str, tool_input_raw: str) -> str:
    """One-line label for a tool_result band, e.g. ``Read settings.py``."""
    if not tool_name:
        return "tool result"
    if tool_name == "Read":
        d = _safe_json(tool_input_raw)
        return f"Read {_basename(d.get('file_path'))}"
    if tool_name == "Bash":
        d = _safe_json(tool_input_raw)
        cmd = (d.get("command") or "").strip().split()
        head = " ".join(cmd[:2]) if cmd else "(empty)"
        return f"Bash · {head}"
    if tool_name in ("WebFetch", "WebSearch"):
        return tool_name
    if tool_name.startswith("mcp__"):
        return f"mcp · {tool_name[len('mcp__') :]}"
    return tool_name


def _build_tokenscape_context(db, session_id: str) -> dict:
    """Build the streamgraph context for the Tokenscape tab.

    Each piece of content becomes a band that lives from its arrival
    turn until /compact wipes it. Bands stack to API context totals;
    top reads are individually labelled.
    """
    rows = db.execute(
        """
        WITH result_meta AS (
            -- tool_result rows don't expose ``id`` in the view
            -- (the JSON field is ``tool_use_id``), so look it up
            -- from raw_messages and extract the content size in
            -- the same pass.
            SELECT
                rm.uuid AS user_uuid,
                e.block_idx AS block_idx,
                json_extract_string(
                    rm.message, '$.content[' || e.block_idx || '].tool_use_id'
                ) AS result_tool_use_id,
                LENGTH(COALESCE(
                    json_extract_string(
                        rm.message, '$.content[' || e.block_idx || '].content'
                    ),
                    ''
                )) AS result_char_count
            FROM session_messages_enriched e
            JOIN raw_messages rm ON rm.uuid = e.uuid
            WHERE e.session_id = ? AND e.kind = 'tool_result'
        )
        SELECT
            e.uuid,
            e.timestamp,
            e.kind,
            COALESCE(e.tool_name, tc.tool_name) AS tool_name,
            COALESCE(e.tool_input, tc.tool_input) AS tool_input,
            CASE
                WHEN e.kind = 'tool_result' THEN COALESCE(rmeta.result_char_count, 0)
                WHEN e.kind = 'agent_tool_call' THEN
                    LENGTH(COALESCE(e.tool_input, ''))
                WHEN e.kind = 'agent_thinking' THEN
                    LENGTH(COALESCE(e.thinking_text, ''))
                ELSE
                    LENGTH(COALESCE(e.text, ''))
            END AS char_count,
            amc.input_tokens,
            amc.cache_read_tokens,
            amc.cache_creation_tokens
        FROM session_messages_enriched e
        LEFT JOIN result_meta rmeta
          ON rmeta.user_uuid = e.uuid AND rmeta.block_idx = e.block_idx
        LEFT JOIN tool_calls tc ON tc.tool_use_id = rmeta.result_tool_use_id
        LEFT JOIN assistant_message_costs amc ON amc.uuid = e.uuid
        WHERE e.session_id = ? AND NOT e.is_sidechain
        ORDER BY e.timestamp ASC, e.block_idx ASC
        """,
        [session_id, session_id],
    ).fetchall()
    if not rows:
        return {"has_data": False}

    blocks, turns, events = _tokenscape_walk(rows)
    if not turns:
        return {"has_data": False}

    figure_json, top_reads = _render_tokenscape_streamgraph(blocks, turns, events)
    return {
        "has_data": True,
        "figure_json": figure_json,
        "turn_count": len(turns),
        "compact_count": sum(1 for e in events if e["kind"] == "compact"),
        "ttl_break_count": sum(1 for e in events if e["kind"] == "ttl_break"),
        "top_reads": top_reads,
        "total_context_tokens": int(sum(t["api_context"] for t in turns)),
    }


def _tokenscape_walk(  # noqa: PLR0912
    rows: list[tuple],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Single pass over enriched message rows.

    Produces ``(blocks, turns, events)``:

    * ``blocks`` — every content block we can attribute to a band, with
      ``arrival_turn`` set; ``end_turn`` is filled in below as we
      discover /compact events.
    * ``turns`` — one row per assistant message: API totals (input,
      cache_read, cache_creation) plus a flag for any event that
      occurred entering this turn.
    * ``events`` — list of ``{turn, kind}`` for the marker overlay,
      ``kind`` ∈ ``{"compact", "ttl_break"}``.
    """
    blocks: list[dict] = []
    turns: list[dict] = []
    events: list[dict] = []
    seen_assistant_uuid: set[str] = set()
    prev_turn_ts = None

    for (
        uuid,
        ts,
        kind,
        tool_name,
        tool_input,
        char_count,
        input_tokens,
        cache_read_tokens,
        cache_creation_tokens,
    ) in rows:
        category = _tokenscape_classify(kind)
        chars = int(char_count or 0)
        if category is not None and chars > 0:
            block = {
                "category": category,
                "char_count": chars,
                "arrival_turn": len(turns) + 1,
                "end_turn": None,  # filled in after /compact detection
                "label": "",
            }
            if category == "tool_result":
                block["label"] = _tokenscape_label(tool_name or "", tool_input or "")
            blocks.append(block)

        if input_tokens is not None and uuid and uuid not in seen_assistant_uuid:
            seen_assistant_uuid.add(uuid)
            turn_idx = len(turns) + 1
            api_in = int(input_tokens or 0)
            api_cache = int(cache_read_tokens or 0)
            api_creation = int(cache_creation_tokens or 0)
            # Context the model actually processed = every token sent in
            # this request, regardless of how it's billed. cache_creation
            # is content that's being written to the cache *and* sent to
            # the model on this turn — so a 5m TTL break preserves
            # api_context (everything re-sent as cache_creation) even
            # though cache_read drops to 0.
            api_context = api_in + api_cache + api_creation

            event_kind: str | None = None
            if turns:
                prev = turns[-1]
                drop = prev["api_context"] - api_context
                # /compact: total context plummets — history truncated.
                # Take precedence over TTL detection because /compact is
                # the more impactful event and can co-occur with an idle
                # gap (running /compact after stepping away).
                if (
                    prev["api_context"] > 0
                    and drop > _TOKENSCAPE_COMPACT_MIN_DROP
                    and api_context
                    < prev["api_context"] * _TOKENSCAPE_COMPACT_DROP_RATIO
                ):
                    event_kind = "compact"
                # 5m TTL: idle gap >= 5min AND the cache was rebuilt
                # (cache_read collapsed). Bands persist; only the
                # cache write is "wasted" cost.
                elif (
                    prev_turn_ts is not None
                    and ts is not None
                    and (ts - prev_turn_ts).total_seconds()
                    >= _TOKENSCAPE_TTL_BREAK_MIN_GAP_SECONDS
                    and prev["api_cache_read"] > _TOKENSCAPE_TTL_BREAK_MIN_CACHE
                    and api_cache
                    < prev["api_cache_read"] * _TOKENSCAPE_TTL_BREAK_CACHE_DROP_RATIO
                ):
                    event_kind = "ttl_break"

            if event_kind == "compact":
                # Close out every still-open band on the previous turn.
                for b in blocks:
                    if b["end_turn"] is None:
                        b["end_turn"] = turn_idx - 1
                events.append({"turn": turn_idx, "kind": "compact"})
            elif event_kind == "ttl_break":
                # Bands persist — just record the cache rebuild cost.
                events.append(
                    {
                        "turn": turn_idx,
                        "kind": "ttl_break",
                        "rebuild_tokens": api_creation,
                    }
                )

            turns.append(
                {
                    "turn": turn_idx,
                    "api_input": api_in,
                    "api_cache_read": api_cache,
                    "api_cache_creation": api_creation,
                    "api_context": api_context,
                    "event": event_kind,
                }
            )
            prev_turn_ts = ts

    # Fill end_turn for any band still open at the end of the session.
    last_turn = turns[-1]["turn"] if turns else 0
    for b in blocks:
        if b["end_turn"] is None:
            b["end_turn"] = last_turn

    return blocks, turns, events


def _render_tokenscape_streamgraph(  # noqa: PLR0912, PLR0915
    blocks: list[dict],
    turns: list[dict],
    events: list[dict],
) -> tuple[str, list[dict]]:
    """Render the stacked-area streamgraph + return top-reads metadata."""
    n_turns = len(turns)
    if n_turns == 0:
        return "", []

    # Solve system baseline at turn 1 from the gap between the API
    # context total and content blocks active at turn 1.
    turn1_blocks = [b for b in blocks if b["arrival_turn"] == 1]
    turn1_chars = sum(b["char_count"] for b in turn1_blocks) or 1
    api_context_t1 = turns[0]["api_context"]
    # We don't know exact chars/token, but ~4 chars/token is a
    # reasonable proxy for English+code. Whatever's left over after
    # accounting for message content is the system prompt + tool
    # definitions baseline. Negative gap (tiny turn 1) clamps to 0.
    estimated_content_tokens = turn1_chars / 4.0
    system_tokens = max(0, api_context_t1 - estimated_content_tokens)

    # Rank tool_result blocks by char_count x persisted_turns to pick
    # the top N for individual bands. Smaller reads aggregate.
    read_blocks = [b for b in blocks if b["category"] == "tool_result"]
    for b in read_blocks:
        persisted = max(1, (b["end_turn"] or 1) - b["arrival_turn"] + 1)
        b["weight"] = b["char_count"] * persisted
        b["persisted"] = persisted
    read_blocks.sort(key=lambda b: b["weight"], reverse=True)
    named = read_blocks[:_TOKENSCAPE_TOP_READS]
    other_reads = read_blocks[_TOKENSCAPE_TOP_READS:]

    # --- Per-turn band heights ---
    # For each turn, sum char_count of every active block per band-id.
    # band-ids: 'system', 'user', 'assistant_text', 'other_reads', or
    # the index of a named read block.
    def _band_id(b: dict) -> str | int:
        if b["category"] == "user":
            return "user"
        if b["category"] == "assistant_text":
            return "assistant_text"
        if b in named:
            return f"read_{named.index(b)}"
        return "other_reads"

    band_order = (
        ["system", "user", "assistant_text"]
        + [f"read_{i}" for i in range(len(named))]
        + (["other_reads"] if other_reads else [])
    )
    band_chars: dict[str | int, list[float]] = {
        bid: [0.0] * n_turns for bid in band_order
    }

    for b in blocks:
        bid = _band_id(b)
        if bid == "system":
            continue
        end = b["end_turn"] or n_turns
        for t in range(b["arrival_turn"], end + 1):
            if 1 <= t <= n_turns:
                band_chars[bid][t - 1] += b["char_count"]

    # Convert per-turn char counts to tokens, scaling to the API
    # context total minus the (constant) system baseline.
    band_tokens: dict[str | int, list[float]] = {
        bid: [0.0] * n_turns for bid in band_order
    }
    for i, t in enumerate(turns):
        active_chars = sum(band_chars[bid][i] for bid in band_order if bid != "system")
        target = max(0.0, t["api_context"] - system_tokens)
        if active_chars > 0 and target > 0:
            scale = target / active_chars
            for bid in band_order:
                if bid == "system":
                    band_tokens[bid][i] = system_tokens
                else:
                    band_tokens[bid][i] = band_chars[bid][i] * scale
        else:
            band_tokens["system"][i] = float(t["api_context"])

    # --- Build figure ---
    _ensure_template()
    fig = go.Figure()
    xs = [t["turn"] for t in turns]

    band_specs: list[tuple[str | int, str, str]] = [
        ("system", "system + tool defs", _TOKENSCAPE_CATEGORY_COLORS["system"]),
        ("user", "user", _TOKENSCAPE_CATEGORY_COLORS["user"]),
        (
            "assistant_text",
            "assistant text",
            _TOKENSCAPE_CATEGORY_COLORS["assistant_text"],
        ),
    ]
    for i, b in enumerate(named):
        color = _TOKENSCAPE_READ_COLORS[min(i, len(_TOKENSCAPE_READ_COLORS) - 1)]
        band_specs.append((f"read_{i}", b["label"], color))
    if other_reads:
        band_specs.append(
            ("other_reads", "other tool results", _TOKENSCAPE_OTHER_READ_COLOR)
        )

    for bid, label, color in band_specs:
        ys = band_tokens[bid]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="none",
                name=label,
                stackgroup="one",
                fillcolor=color,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "turn %{x}<br>"
                    "≈ %{y:,.0f} tokens<extra></extra>"
                ),
            )
        )

    # --- Annotations: label each named-read band at its centroid ---
    annotations: list[dict] = []
    # Cumulative below-band y for each turn — needed to anchor labels
    # at the centre of the band slab. Keyed parallel to band_order.
    cumulative_below = [0.0] * n_turns
    for bid, label, _color in band_specs:
        if not str(bid).startswith("read_"):
            for i in range(n_turns):
                cumulative_below[i] += band_tokens[bid][i]
            continue
        idx = int(str(bid).split("_", 1)[1])
        block = named[idx]
        # Place label at the persistence midpoint (in turn coords).
        mid_turn = (
            block["arrival_turn"] + (block["end_turn"] or block["arrival_turn"])
        ) // 2
        i = max(0, min(n_turns - 1, mid_turn - 1))
        band_height = band_tokens[bid][i]
        if band_height <= 0:
            for j in range(n_turns):
                cumulative_below[j] += band_tokens[bid][j]
            continue
        y_center = cumulative_below[i] + band_height / 2
        approx_tokens = max(band_tokens[bid])
        annotations.append(
            {
                "x": mid_turn,
                "y": y_center,
                "xref": "x",
                "yref": "y",
                "text": (
                    f"<b>{label}</b><br>"
                    f"≈ {approx_tokens:,.0f} tokens · {block['persisted']} turns"
                ),
                "showarrow": False,
                "font": {"size": 11, "color": "#5a3a25"},
                "align": "center",
                "bgcolor": "rgba(255,255,255,0.7)",
                "borderpad": 3,
            }
        )
        for j in range(n_turns):
            cumulative_below[j] += band_tokens[bid][j]

    # --- Event markers: /compact (dashed) and 5m TTL break (dotted) ---
    for ev in events:
        x = ev["turn"] - 0.5
        if ev["kind"] == "compact":
            fig.add_vline(x=x, line_dash="dash", line_color="#888", line_width=1)
            annotations.append(
                {
                    "x": x,
                    "y": 1.02,
                    "xref": "x",
                    "yref": "paper",
                    "text": "/compact",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#666"},
                    "xanchor": "left",
                }
            )
        else:  # ttl_break
            fig.add_vline(x=x, line_dash="dot", line_color="#c0a000", line_width=1)
            annotations.append(
                {
                    "x": x,
                    "y": 1.02,
                    "xref": "x",
                    "yref": "paper",
                    "text": (
                        f"5m gap · cache rebuilt (+{ev.get('rebuild_tokens', 0):,} tok)"
                    ),
                    "showarrow": False,
                    "font": {"size": 10, "color": "#a08000"},
                    "xanchor": "left",
                }
            )

    fig.update_layout(
        template="tufte",
        showlegend=False,
        hovermode="x unified",
        xaxis_title="turn",
        yaxis_title="input tokens",
        margin={"l": 70, "r": 30, "t": 40, "b": 60},
        annotations=annotations,
    )

    # Serialise the top-reads list for the table beneath the chart.
    total_context = sum(t["api_context"] for t in turns) or 1
    top_reads = [
        {
            "label": b["label"],
            "tokens": int(max(band_tokens[f"read_{i}"])),
            "arrival_turn": b["arrival_turn"],
            "persisted": b["persisted"],
            "share_pct": (
                100.0 * max(band_tokens[f"read_{i}"]) * b["persisted"] / total_context
            ),
        }
        for i, b in enumerate(named)
    ]
    return fig.to_json(), top_reads


_VALID_TABS = {"messages", "cost", "tokenscape"}


async def session_detail(
    request: Request, session_id: str, tab: str = "messages"
) -> HTMLResponse:
    """Full session detail: Messages tab (default) or Cost tab (?tab=cost)."""
    db = conn(request)

    active_tab = tab if tab in _VALID_TABS else "messages"

    session_info = db.execute(
        """
        SELECT session_id, started_at, ended_at, duration,
               user_messages, assistant_messages, model, cwd, git_branch
        FROM logical_sessions
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    token_usage = fetch_token_usage(db, session_id=session_id)

    # Tool call summary
    tool_summary = db.execute(
        """
        SELECT
            COUNT(*) AS total_calls,
            COUNT(*) FILTER (WHERE is_error = 'true') AS failed_calls,
            MODE(tool_name) AS most_used_tool
        FROM tool_calls
        WHERE session_id = ?
    """,
        [session_id],
    ).fetchone()

    # File metrics (from file_reads / file_writes views)
    session_cwd = session_info[7] if session_info else ""
    file_metrics = db.execute(
        """
        WITH sr AS (
            SELECT DISTINCT file_path FROM file_reads
            WHERE session_id = ?
        ), sw AS (
            SELECT DISTINCT file_path FROM file_writes
            WHERE session_id = ?
        )
        SELECT
            (SELECT COUNT(*) FROM sr),
            (SELECT COUNT(*) FROM sw),
            (SELECT COUNT(*) FROM sr
             WHERE file_path NOT IN (SELECT file_path FROM sw)),
            (SELECT COUNT(*) FROM sr
             WHERE NOT starts_with(file_path, ?))
    """,
        [session_id, session_id, session_cwd or ""],
    ).fetchone()

    parsed_messages: list[dict] = []
    cost_ctx: dict = {}
    tokenscape_ctx: dict = {}
    if active_tab == "messages":
        parsed_messages = _build_messages_context(db, session_id)
    elif active_tab == "tokenscape":
        try:
            tokenscape_ctx = _build_tokenscape_context(db, session_id)
        except Exception as exc:
            logger.exception("Failed to build tokenscape for %s", session_id)
            tokenscape_ctx = {
                "has_data": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:  # "cost"
        cost_ctx = _build_cost_context(db, session_id)

    return templates.TemplateResponse(
        request,
        "session_detail.html",
        {
            "parent": parent(request),
            "session": session_info,
            "session_id": session_id,
            "messages": parsed_messages,
            "token_usage": token_usage,
            "tool_summary": tool_summary,
            "active_tab": active_tab,
            "cost_ctx": cost_ctx,
            "tokenscape_ctx": tokenscape_ctx,
            "file_metrics": {
                "files_read": file_metrics[0] or 0,
                "files_edited": file_metrics[1] or 0,
                "files_read_only": file_metrics[2] or 0,
                "files_outside": file_metrics[3] or 0,
            }
            if file_metrics
            else None,
        },
    )


async def cost_bloat_panel(
    request: Request,
    session_id: str,
    from_uuid: str | None = None,
    to_uuid: str | None = None,
) -> HTMLResponse:
    """Re-render the bloat-rollup + top-contributors block for a uuid range.

    HTMX swaps the result into ``#session-cost-bloat-panel``. When both
    ``from_uuid`` and ``to_uuid`` are supplied, the tables scope to that
    inclusive range; with neither, they cover the whole session.
    """
    db = conn(request)
    range_filter: tuple[str, str] | None = None
    if from_uuid and to_uuid:
        range_filter = (from_uuid, to_uuid)
    cost_ctx = _build_cost_context(db, session_id, range_filter=range_filter)
    return templates.TemplateResponse(
        request,
        "_session_cost_bloat.html",
        {
            "cost_ctx": cost_ctx,
            "session_id": session_id,
            "parent": parent(request),
        },
    )
