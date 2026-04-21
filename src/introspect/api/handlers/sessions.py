"""Session-related route handlers."""

import json
import math
import re
from pathlib import PurePosixPath

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.pricing import compute_cost_usd
from introspect.search import ensure_search_corpus, fts_available

from ._helpers import (
    OBVIOUS_COMMANDS_SQL,
    SESSION_INFO_JOINS,
    SESSION_INFO_SELECT,
    SESSIONS_PAGE_SIZES,
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_COLS,
    SESSIONS_SORT_DEFAULT,
    conn,
    fetch_token_usage,
    format_cost,
    parent,
    session_row_to_dict,
    templates,
)

_MESSAGE_HARD_CAP = 5000
_THINKING_PREVIEW_MAX = 200
_TOOL_HINT_MAX = 120
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
        where_clauses.append("ls.model = ?")
        params.append(model.strip())
    if project.strip():
        where_clauses.append("ls.project = ?")
        params.append(project.strip())
    if branch.strip():
        where_clauses.append("ls.git_branch = ?")
        params.append(branch.strip())
    if command.strip():
        where_clauses.append(
            "EXISTS (SELECT 1 FROM message_commands mc"
            " WHERE mc.session_id = ls.session_id AND mc.command = ?)"
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
                "WHERE sc.session_id = ls.session_id AND sc.score IS NOT NULL)"
            )
            params.append(search_query)
        else:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM search_corpus sc"
                " WHERE sc.session_id = ls.session_id"
                " AND sc.content_text ILIKE ?)"
            )
            params.append(f"%{search_query}%")

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Count with filters
    total = db.execute(
        f"SELECT COUNT(*) FROM logical_sessions ls {where}",  # noqa: S608
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
        FROM logical_sessions ls
        {SESSION_INFO_JOINS}
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
            tc.execution_time
        FROM session_messages_enriched e
        LEFT JOIN tool_calls tc ON tc.tool_use_id = e.tool_use_id
        WHERE e.session_id = ?
          AND e.kind <> 'tool_result'
        ORDER BY e.timestamp ASC, e.block_idx ASC
        """,
        [session_id],
    )
    col_names = [d[0] for d in cur.description]
    parsed_messages: list[dict] = []
    for row in cur.fetchall():
        rec = dict(zip(col_names, row, strict=True))
        kind = rec["kind"]
        exec_secs = (
            rec["execution_time"].total_seconds()
            if rec["execution_time"] is not None
            else None
        )
        thinking_preview, thinking_has_more = _thinking_preview(rec["thinking_text"])
        parsed_messages.append(
            {
                "timestamp": str(rec["timestamp"])[:19] if rec["timestamp"] else "",
                "kind": kind,
                "is_sidechain": bool(rec["is_sidechain"]),
                "text": _cap(rec["text"]),
                "thinking_text": _cap(rec["thinking_text"]),
                "thinking_preview": thinking_preview,
                "thinking_has_more": thinking_has_more,
                "command_label": (
                    _slash_command_label(rec["text"]) if kind == "slash_command" else ""
                ),
                "tool_name": rec["tool_name"] or "",
                "tool_hint": _tool_hint(rec["tool_name"] or "", rec["tool_input"]),
                "tool_input": _pretty_tool_input(rec["tool_input"]),
                "tool_result": _cap(rec["tool_use_result"]),
                "is_error": _coerce_bool(rec["is_error"]),
                "exec_time": (
                    _format_exec_time(exec_secs) if exec_secs is not None else ""
                ),
                "tool_use_id": rec["tool_use_id"] or "",
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


def _aggregate_per_model(rows: list[tuple]) -> tuple[list[dict], list[tuple], float]:
    """Group rows by model, returning (per_model, cumulative_pts, total_cost)."""
    by_model: dict[str, dict] = {}
    cumulative_pts: list[tuple] = []
    running = 0.0
    for cost_row in rows:
        ts, _is_side, model = cost_row[0], cost_row[1], cost_row[2]
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
        cumulative_pts.append((str(ts), running))
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
    return per_model, cumulative_pts, running


def _aggregate_bloat(rows: list[tuple]) -> tuple[dict, dict]:
    """Aggregate bloat rows into bucket_totals and category_totals dicts.

    Both dicts are keyed by (label, agent) so the main vs. subagent split is
    preserved as an orthogonal dimension, not collapsed into one category.
    """
    bucket_totals: dict[tuple[str, str], dict] = {}
    category_totals: dict[tuple[str, str], dict] = {
        (c, a): {"category": c, "agent": a, "tokens": 0, "cost_usd": 0.0}
        for c in _BLOAT_CATEGORIES
        for a in _BLOAT_AGENTS
    }
    for bloat_row in rows:
        (
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
            },
        )
        bucket_entry["tokens"] += cc_total
        bucket_entry["cost_usd"] += cost
    return bucket_totals, category_totals


def _build_cost_context(db, session_id: str) -> dict:
    """Build the data structures the Cost tab template needs."""
    rows = db.execute(
        """
        SELECT
            timestamp,
            is_sidechain,
            model,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            cache_creation_5m,
            cache_creation_1h,
            parent_uuid
        FROM assistant_message_costs
        WHERE session_id = ?
        ORDER BY timestamp
        """,
        [session_id],
    ).fetchall()

    per_model, cumulative_pts, total_cost = _aggregate_per_model(rows)
    sparkline = _build_sparkline(cumulative_pts, total_cost)

    # Note on the parent-user-message join: `u.message.content` is an array
    # of blocks (parallel-tool calls produce one tool_result per block).  We
    # unnest blocks first, then prefer a tool_result block when one exists —
    # otherwise the outer LEFT JOIN's ON clause would silently drop the
    # tool_use_id whenever the result lived at index >= 1, mis-attributing
    # legitimate file reads as "human input".
    bloat_rows = db.execute(
        """
        WITH amc AS (
            SELECT * FROM assistant_message_costs WHERE session_id = ?
        ),
        parent_blocks AS (
            SELECT
                amc.uuid AS amc_uuid,
                json_extract_string(
                    u.message, '$.content[' || i.idx || '].type'
                ) AS block_type,
                json_extract_string(
                    u.message, '$.content[' || i.idx || '].tool_use_id'
                ) AS block_tool_use_id,
                i.idx AS block_idx
            FROM amc
            JOIN raw_messages u
              ON u.uuid = amc.parent_uuid AND u.type = 'user'
              AND json_array_length(json_extract(u.message, '$.content')) > 0,
              generate_series(
                  0,
                  CAST(json_array_length(
                      json_extract(u.message, '$.content')
                  ) - 1 AS BIGINT)
              ) AS i(idx)
        ),
        chosen_block AS (
            -- Prefer a tool_result block (any index) over a text block.
            SELECT amc_uuid,
                   FIRST(block_type ORDER BY
                         CASE WHEN block_type = 'tool_result' THEN 0 ELSE 1 END,
                         block_idx) AS user_block_type,
                   FIRST(block_tool_use_id ORDER BY
                         CASE WHEN block_type = 'tool_result' THEN 0 ELSE 1 END,
                         block_idx) AS tool_use_id
            FROM parent_blocks
            GROUP BY amc_uuid
        )
        SELECT
            amc.is_sidechain,
            amc.model,
            amc.cache_creation_tokens AS cc_total,
            amc.cache_creation_5m,
            amc.cache_creation_1h,
            cb.user_block_type,
            tc.tool_name,
            tc.tool_input
        FROM amc
        LEFT JOIN chosen_block cb ON cb.amc_uuid = amc.uuid
        LEFT JOIN tool_calls tc ON tc.tool_use_id = cb.tool_use_id
        ORDER BY amc.timestamp
        """,
        [session_id],
    ).fetchall()

    bucket_totals, category_totals = _aggregate_bloat(bloat_rows)
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
        }
        for bucket in sorted_buckets[:_BLOAT_TOP_N]
    ]

    return {
        "per_model": per_model,
        "total_cost_usd": total_cost,
        "total_cost": format_cost(total_cost),
        "sparkline": sparkline,
        "bloat_rollup": rollup,
        "bloat_rollup_totals": rollup_totals,
        "bloat_top_buckets": top_buckets_view,
        "bloat_extra_count": extra_count,
        "bloat_total_tokens": raw_tokens,
        "bloat_total_cost": format_cost(total_bloat_cost),
        "has_data": bool(rows),
    }


def _build_sparkline(points: list[tuple[str, float]], total_cost: float) -> dict:
    """Bucket the running-cost series + render an inline-SVG polyline string.

    Returns a dict with the SVG path data and metadata so the template can
    render the chart without any JS lib.  ``messages`` is the raw API-call
    count; ``points`` is the (possibly bucketed) sample count plotted.
    """
    if not points:
        return {
            "polyline": "",
            "width": 0,
            "height": 0,
            "max": 0.0,
            "messages": 0,
            "points": 0,
        }

    total_messages = len(points)
    if total_messages > _CUMULATIVE_RAW_THRESHOLD:
        bucketed: list[tuple[str, float]] = []
        # math.ceil keeps us within ≤_CUMULATIVE_MAX_BUCKETS buckets even
        # for sizes that round down (e.g. 12_000 // 120 = 100 buckets).
        bucket_size = max(1, math.ceil(total_messages / _CUMULATIVE_MAX_BUCKETS))
        for i in range(0, total_messages, bucket_size):
            chunk = points[i : i + bucket_size]
            # The chunk endpoint is the running total at the end of the chunk.
            bucketed.append((chunk[-1][0], chunk[-1][1]))
        series = bucketed
    else:
        series = points

    width, height = 600, 80
    max_cost = total_cost if total_cost > 0 else 1.0
    n = max(1, len(series) - 1)
    coords = []
    for i, (_, cost) in enumerate(series):
        x = (i / n) * width if n else 0
        y = height - (cost / max_cost) * height
        coords.append(f"{x:.1f},{y:.1f}")
    return {
        "polyline": " ".join(coords),
        "width": width,
        "height": height,
        "max": max_cost,
        "messages": total_messages,
        "points": len(series),
    }


_VALID_TABS = {"messages", "cost"}


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
    if active_tab == "messages":
        parsed_messages = _build_messages_context(db, session_id)
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
