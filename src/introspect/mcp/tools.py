"""MCP tool definitions for introspect."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import cast, get_args

import duckdb

from introspect.cache_ttl import compare_ttl
from introspect.db import (
    DEFAULT_CODEX_GLOB,
    DEFAULT_DB_PATH,
    configured_memory_limit,
    connect_read_hardened,
    get_read_connection,
)
from introspect.mcp import refresh_bridge
from introspect.query_templates import (
    QUERY_TEMPLATES,
    Param,
    QueryTemplate,
    TemplateKind,
    get_template,
    templates_by_kind,
)
from introspect.refresh import RefreshOutcome, wait_for_refresh
from introspect.search import ensure_search_corpus, fts_search
from introspect.sql_fragments import (
    CACHE_READ_COST_SQL,
    CACHE_WRITE_COST_SQL,
    COST_EXPR_SQL,
    OUTPUT_COST_SQL,
    SESSION_COST_SUBQUERY,
    session_cost_subquery_filtered,
)
from introspect.sql_query import (
    CELL_TRUNCATION_MARKER,
    MCP_BUDGET,
    MCP_SQL_CELL_CAP,
    MCP_SQL_ROW_CAP,
    SqlTimeoutError,
    clamp_row_limit,
    execute_bounded,
    validate_read_only_sql,
)

_VALID_ROLES = {"user", "assistant"}
# Derived from the QueryTemplate.kind Literal so the two can't drift.
_VALID_TEMPLATE_KINDS: frozenset[str] = frozenset(get_args(TemplateKind))

# Max characters per rendered cell in run_sql output. Deliberately one marker
# wider than ``MCP_SQL_CELL_CAP``: ``execute_bounded`` normalizes and clips
# every cell to ``cell_cap`` + CELL_TRUNCATION_MARKER, and a ceiling equal to
# ``cell_cap`` would re-clip all of them and eat the marker. Nothing should
# exceed this by the time rows get here — it is a backstop, not the bound.
_SQL_CELL_MAX = MCP_SQL_CELL_CAP + len(CELL_TRUNCATION_MARKER)
# Hard cap on run_sql rows regardless of caller's `limit` argument.
# Re-exported alias for tests/back-compat — canonical value lives in sql_query.
_SQL_ROW_CAP = MCP_SQL_ROW_CAP
# Hard cap on expensive_sessions rows displayed (clamped from caller's limit).
_EXPENSIVE_SESSIONS_ROW_CAP = 50

# Pareto cumulative-cost cutoff — mirrors cost_overview.PARETO_CUTOFF.
# Deliberate duplication: do NOT refactor cost_overview in this change.
_PARETO_CUTOFF = 0.80

# Generous compared to the HTTP refresh handler — MCP callers tolerate
# longer waits than a browser HTMX swap. Module-level so tests can shrink it
# without waiting half a minute on the STILL_RUNNING branch.
REFRESH_TIMEOUT = 30.0


def _get_read_connection() -> duckdb.DuckDBPyConnection:
    """``get_read_connection`` with the Codex default glob applied.

    Mirrors ``cli._db``'s ``DEFAULT_CODEX_GLOB`` passthrough so the
    lazy-view fallback surfaces Codex sessions too, not just Claude ones.
    """
    return get_read_connection(codex_glob=DEFAULT_CODEX_GLOB)


def _validate_since(since: str) -> str | None:
    """Return a friendly error message if `since` isn't empty or ISO-parseable.

    Shared by every tool using the empty-string-disables `since` convention
    (`search_conversations`, `expensive_sessions`, `tool_failure_rate`) so the
    validation and error wording can't drift between them.
    """
    if not since:
        return None
    try:
        datetime.fromisoformat(since)
    except ValueError as exc:
        return f"Error: invalid 'since' (expected ISO date/timestamp): {exc}"
    return None


def search_conversations(  # noqa: PLR0913
    query: str,
    limit: int = 10,
    offset: int = 0,
    cwd_prefix: str = "",
    role: str = "",
    since: str = "",
    session_id: str = "",
    require_all: bool = False,
) -> str:
    """Full-text search across conversation logs (BM25 when available).

    Returns session summaries with context-windowed snippets.

    Optional filters (empty string disables each filter — FastMCP exposes
    these as plain strings rather than Optional for tool-schema simplicity):
      - ``cwd_prefix``: match sessions whose working directory starts with this
        string (e.g. ``/home/matous/git/logogame`` or just ``/home/matous/git``).
      - ``role``: ``'user'`` or ``'assistant'``.
      - ``since``: ISO date/timestamp (e.g. ``'2026-04-01'``); matches messages
        at or after this point.
      - ``session_id``: restrict to a single session.
      - ``require_all``: multi-word queries must match ALL terms (AND mode).
      - ``offset``: skip N results — use with ``limit`` for pagination.
    """
    if role and role not in _VALID_ROLES:
        return f"Error: role must be one of {sorted(_VALID_ROLES)} (got {role!r})."
    since_error = _validate_since(since)
    if since_error:
        return since_error

    conn = _get_read_connection()
    try:
        ensure_search_corpus(conn)

        results = fts_search(
            conn,
            query,
            limit=limit,
            offset=offset,
            cwd_prefix=cwd_prefix or None,
            role=role or None,
            since=since or None,
            session_id=session_id or None,
            require_all=require_all,
        )
        if not results:
            return "No results found."

        lines: list[str] = []
        for sid, timestamp, msg_role, cwd, snippet, score in results:
            ts = str(timestamp)[:19] if timestamp else "?"
            score_str = f"{score:.4f}" if score is not None else "?"
            lines.append(
                f"[{ts}] session={sid} role={msg_role} "
                f"cwd={cwd or '?'} score={score_str}\n  {snippet}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


def get_session(session_id: str) -> str:
    """Get full session content by session ID.

    Returns all messages as structured data.
    """
    conn = _get_read_connection()
    try:
        # Session metadata
        meta = conn.execute(
            """
            SELECT
                session_id, started_at, ended_at, duration,
                user_messages, assistant_messages, model, cwd, git_branch
            FROM logical_sessions
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if not meta:
            return f"Session '{session_id}' not found."

        lines: list[str] = [
            f"Session: {meta[0]}",
            f"Started: {meta[1]}",
            f"Ended: {meta[2]}",
            f"Duration: {meta[3]}",
            f"User messages: {meta[4]}",
            f"Assistant messages: {meta[5]}",
            f"Model: {meta[6]}",
            f"CWD: {meta[7]}",
            f"Branch: {meta[8]}",
            "",
            "--- Messages ---",
        ]

        turns = conn.execute(
            """
            SELECT turn_order, type, content_text
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY turn_order
            """,
            [session_id],
        ).fetchall()

        for turn_order, msg_type, content in turns:
            label = "User" if msg_type == "user" else "Assistant"
            text = (content or "")[:500]
            lines.append(f"\n[{turn_order}] {label}:\n{text}")

        return "\n".join(lines)
    finally:
        conn.close()


def recent_sessions(n: int = 10) -> str:
    """List the most recent N sessions with metadata."""
    conn = _get_read_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                session_id, started_at, ended_at, duration,
                user_messages, assistant_messages, model, cwd, git_branch
            FROM logical_sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [n],
        ).fetchall()

        if not rows:
            return "No sessions found."

        lines: list[str] = []
        for row in rows:
            started = str(row[1])[:19] if row[1] else "?"
            duration = str(row[3]) if row[3] else "?"
            lines.append(
                f"session={row[0]}\n"
                f"  started={started} duration={duration}\n"
                f"  user_msgs={row[4]} asst_msgs={row[5]}\n"
                f"  model={row[6]} cwd={row[7]} branch={row[8]}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


def _format_rows(
    columns: list[str], rows: list[tuple], footnote: str | None = None
) -> str:
    """Format a result set as an aligned text table with truncated cells.

    Cells are already normalized and clipped by :func:`execute_bounded` —
    lists, structs, maps and blobs are stringified there, under the cell cap,
    so this clip is a backstop rather than the thing that bounds the output.
    """
    if not rows:
        return f"(0 rows)\ncolumns: {', '.join(columns)}"

    def cell(value: object) -> str:
        text = "NULL" if value is None else str(value)
        text = text.replace("\n", " ").replace("\r", " ")
        if len(text) > _SQL_CELL_MAX:
            text = text[: _SQL_CELL_MAX - 1] + "…"
        return text

    str_rows = [[cell(v) for v in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    header = render(columns)
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(render(r) for r in str_rows)
    tail = f"\n{footnote}" if footnote else ""
    return f"{header}\n{sep}\n{body}\n({len(rows)} rows){tail}"


def _run_sql_error(exc: Exception) -> str:
    """Render a `run_sql` execution failure as tool output.

    Out-of-memory gets its own wording naming the configured budget: the
    engine raises it cleanly (the process survives), so the useful thing to
    say is what the query was allowed to use.
    """
    if isinstance(exc, SqlTimeoutError):
        return f"Error: {exc}"
    if isinstance(exc, duckdb.OutOfMemoryException):
        return (
            f"Error: query exceeded the {configured_memory_limit()} memory limit: {exc}"
        )
    return f"SQL error ({type(exc).__name__}): {exc}"


def run_sql(sql: str, limit: int = 100) -> str:
    """Execute a read-only SELECT query against the introspect DB.

    The statement is parsed by DuckDB and must be exactly one SELECT (``WITH``
    and DuckDB's FROM-first form count as SELECT). Writes, ATTACH, PRAGMA,
    SET, INSTALL, LOAD, COPY and multi-statement scripts are rejected, as are
    functions that read outside the database (``read_csv``, ``read_text``,
    ``glob``, ``sqlite_scan``, …). The connection itself has filesystem,
    network and extension access disabled, so those are refused by the engine
    too. Use `describe_schema` to discover available views and columns.

    Bounded on five axes: `limit` rows (max 500, pushed into the planner as an
    outer LIMIT), 64 KB of total output, 200 characters per cell, 8 KB of SQL
    text, and a 20 s wall clock. The last line names every cap that fired. The
    cell cap only shortens values — seeing it there means the rows are all
    here but some were too wide, so re-running with a smaller `limit` will not
    help; select narrower columns or `substr()` instead. Returns an aligned
    text table.
    """
    error = validate_read_only_sql(sql, max_bytes=MCP_BUDGET.max_sql_bytes)
    if error:
        return f"Error: {error}"

    # Fresh hardened read-only connection — do NOT route through
    # get_read_connection(), which silently falls back to a writable
    # connection over lazy JSONL views when the materialized DB is missing.
    if not DEFAULT_DB_PATH.exists():
        return (
            f"Error: materialized DB not found at {DEFAULT_DB_PATH}. "
            "Start `introspect serve` once to materialize views."
        )
    try:
        conn = connect_read_hardened(DEFAULT_DB_PATH)
    except duckdb.Error as exc:
        return f"Error opening DB ({type(exc).__name__}): {exc}"

    try:
        result = execute_bounded(
            conn,
            sql,
            replace(MCP_BUDGET, row_cap=clamp_row_limit(limit, _SQL_ROW_CAP)),
        )
    except (SqlTimeoutError, duckdb.Error) as exc:
        return _run_sql_error(exc)
    finally:
        conn.close()

    footnote = (
        f"truncated: hit the {result.truncation_reason}" if result.truncated else None
    )
    return _format_rows(result.columns, result.rows, footnote)


def describe_schema() -> str:
    """List views/tables available to `run_sql` with their columns.

    Returns a compact listing grouped by table, pulled from the attached
    DuckDB's information_schema. Use this before writing a `run_sql` query
    to discover column names and types.
    """
    conn = _get_read_connection()
    try:
        rows = conn.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No tables or views found."

    by_table: dict[str, list[str]] = {}
    for table_name, column_name, data_type in rows:
        by_table.setdefault(table_name, []).append(f"{column_name} {data_type}")

    # Surface the views a caller most often wants first; everything else
    # follows alphabetically.
    priority = [
        "logical_sessions",
        "tool_calls",
        "conversation_turns",
        "session_titles",
        "search_corpus",
    ]
    ordered: list[str] = [name for name in priority if name in by_table]
    ordered.extend(name for name in sorted(by_table) if name not in ordered)

    lines: list[str] = []
    for table_name in ordered:
        lines.append(f"{table_name}:")
        lines.extend(f"  {col}" for col in by_table[table_name])
        lines.append("")
    schema_text = "\n".join(lines).rstrip()
    return (
        f"{schema_text}\n\n"
        f"{len(QUERY_TEMPLATES)} query templates available via "
        "list_query_templates()."
    )


def _format_param(p: Param) -> str:
    """Render one Param as a cookbook bullet line."""
    requiredness = "required" if p.required else f"default={p.default!r}"
    return f"  - {p.name} ({p.type}, {requiredness}): {p.description}"


def list_query_templates(kind: str = "") -> str:
    """Render the curated SQL investigation-template cookbook.

    These are STARTING POINTS to adapt, not canned answers — read the SQL,
    tweak it for the question actually being asked, and run it through
    `run_sql`. Each entry shows the question it answers, its parameters
    (`$named` DuckDB placeholders), the SQL itself, and a `note` carrying the
    non-obvious schema knowledge (e.g. dedup rules, what a flag column really
    means) needed to adapt it correctly.

    A `kind="deterministic"` template ("one fixed query answers this") may
    also be available as a dedicated MCP tool — check the tool list first.
    A `kind="exploratory"` template ("the value is in adapting and following
    threads") is meant to be reshaped per investigation.

    Parameters
    ----------
    kind:
        Optional filter: 'deterministic' or 'exploratory'. Empty string
        (default) returns all — empty string rather than Optional for
        tool-schema simplicity, matching the rest of this module.
    """
    if kind and kind not in _VALID_TEMPLATE_KINDS:
        return (
            f"Error: kind must be one of {sorted(_VALID_TEMPLATE_KINDS)} "
            f"(got {kind!r})."
        )

    templates = (
        templates_by_kind(cast("TemplateKind", kind)) if kind else QUERY_TEMPLATES
    )
    if not templates:
        return f"No templates of kind {kind!r}."

    blocks: list[str] = []
    for template in templates:
        param_lines = [_format_param(p) for p in template.params]
        params_text = "\n".join(param_lines) if param_lines else "  (none)"
        blocks.append(
            f"### {template.name}  [{template.kind}]\n"
            f"Q: {template.question}\n"
            f"Params:\n{params_text}\n"
            f"SQL:\n{template.sql}\n"
            f"Note: {template.note}"
        )

    header = (
        "Query template cookbook — starting points to adapt, not canned "
        "answers. Adjust the SQL to the actual question, then run it "
        "through `run_sql`.\n"
    )
    return header + "\n\n".join(blocks)


async def refresh_data() -> str:
    """Trigger an immediate rebuild of the materialized DB from JSONL files.

    The server normally rescans every 10 minutes by default
    (``INTROSPECT_REFRESH_INTERVAL_SECONDS``, in seconds). Call this when
    you need to observe events from the very recent past — e.g. a session
    that just ended. The rebuild only runs when JSONL files actually
    changed, so calling this on an unchanged filesystem is a fast no-op.

    Returns a status string describing what happened.
    """
    state = refresh_bridge.get_state()
    if state is None:
        return (
            "Manual refresh unavailable: introspect is not running as a server. "
            "Start `introspect serve` to enable refresh."
        )

    result = await wait_for_refresh(state, finish_timeout=REFRESH_TIMEOUT)

    match result.outcome:
        case RefreshOutcome.DISABLED:
            return (
                "Auto-refresh is disabled "
                "(INTROSPECT_REFRESH_INTERVAL_SECONDS=0); "
                "manual refresh unavailable."
            )
        case RefreshOutcome.COMPLETED:
            # COMPLETED only fires when last_refreshed_at advanced to a non-None
            # value (see wait_for_refresh); the ``or "?"`` is defensive against
            # future contract drift, not a real branch under correct inputs.
            ts = (
                result.last_refreshed_at.isoformat()
                if result.last_refreshed_at
                else "?"
            )
            return f"Refresh complete. Last refreshed at {ts}."
        case RefreshOutcome.STILL_RUNNING:
            return (
                f"Refresh started but did not complete within "
                f"{int(REFRESH_TIMEOUT)} seconds; still running."
            )
        case RefreshOutcome.UNCHANGED:
            return "No refresh needed: JSONL files unchanged since last refresh."
        case _:
            # Defensive: future variants of RefreshOutcome must update this
            # match. Raising rather than returning a vague string surfaces the
            # gap loudly at runtime if static checks miss it.
            msg = f"unhandled refresh outcome: {result.outcome}"
            raise RuntimeError(msg)


def run_query_template(
    template: QueryTemplate,
    conn: duckdb.DuckDBPyConnection,
    params: dict[str, object],
) -> list[tuple]:
    """Bind `params` and execute `template.sql` on `conn`.

    The single execution path for a registry entry's SQL — deterministic
    tools (e.g. `tool_failure_rate`) call this, so the SQL the tool actually
    runs can never drift from the SQL the cookbook advertises. The
    registry/tool parity test (`test_tool_failure_rate_parity_with_registry_sql`)
    exercises this path indirectly: it calls `tool_failure_rate`, which calls
    `run_query_template`, and compares the result to `template.sql` executed
    directly. `params` is bound as a DuckDB `$named` dict, never
    string-interpolated.
    """
    return conn.execute(template.sql, params).fetchall()


def tool_failure_rate(limit: int = 20, since: str = "", min_calls: int = 5) -> str:
    """Rank tools by failure rate — which tools fail most, by rate and count?

    Groups `tool_calls` by `tool_name` and reports call count, failure count,
    and failure rate, sorted worst-first. Executes the `tool_failure_rate`
    entry from the query-template registry (`list_query_templates`) so the
    tool and the cookbook never drift apart.

    Parameters
    ----------
    limit:
        Max rows to return (default 20).
    since:
        Optional ISO date/timestamp (e.g. ``'2026-06-01'``); only calls at or
        after this point are counted. Empty string (default) means all time
        — matching `expensive_sessions`' empty-string-disables convention.
    min_calls:
        Suppress low-N noise: only tools with at least this many calls are
        included (default 5).
    """
    since_error = _validate_since(since)
    if since_error:
        return since_error

    template = get_template("tool_failure_rate")
    if template is None:  # pragma: no cover - defensive, registry can't drop this
        return "Error: 'tool_failure_rate' template not found in registry."

    conn = _get_read_connection()
    try:
        rows = run_query_template(
            template,
            conn,
            {"limit": limit, "since": since or None, "min_calls": min_calls},
        )
    finally:
        conn.close()

    if not rows:
        return "No tools met the min_calls threshold."

    lines: list[str] = []
    for tool_name, calls, failures, failure_rate in rows:
        lines.append(
            f"{tool_name}: {failures}/{calls} failed ({float(failure_rate):.1%})"
        )
    return "\n".join(lines)


# Offsets into the `cache_ttl_choice` template's SELECT list, past the
# leading `project` column. Named so a reordered SELECT fails loudly at one
# place rather than silently mislabelling every number.
_TTL_N_REQUESTS = 0
_TTL_N_RECOVERABLE = 1
_TTL_N_BREAKS = 2
_TTL_N_STRUCTURAL = 3
_TTL_OBSERVED = 4
_TTL_COST_5M = 5
_TTL_COST_1H = 6


def cache_ttl_choice(limit: int = 20, since: str = "", sidechain: bool = False) -> str:
    """Which prompt-cache TTL is cheaper per project — 5m or 1h?

    Replays every API request under both policies and reports the margin.
    Executes the `cache_ttl_choice` entry from the query-template registry
    (`list_query_templates`) so the tool and the cookbook cannot drift.

    A negative delta means 1h is cheaper. Read the margin before acting:
    under ~2% is inside the simulation's modelling error and is not a
    decision. Gaps longer than an hour are counted separately as breaks —
    no TTL setting recovers them, so they are not evidence for switching.

    Parameters
    ----------
    limit:
        Max projects to return (default 20).
    since:
        Optional ISO date/timestamp; only requests at or after this point.
        Empty string (default) means all time.
    sidechain:
        False (default) scores the main conversation, governed by
        `promptCacheTtl`. True scores subagents, which have their own
        `subagentPromptCacheTtl` — the two are never merged.
    """
    since_error = _validate_since(since)
    if since_error:
        return since_error

    template = get_template("cache_ttl_choice")
    if template is None:  # pragma: no cover - defensive, registry can't drop this
        return "Error: 'cache_ttl_choice' template not found in registry."

    conn = _get_read_connection()
    try:
        rows = run_query_template(
            template,
            conn,
            {"limit": limit, "since": since or None, "sidechain": sidechain},
        )
    finally:
        conn.close()

    if not rows:
        return "No cache data in range."

    lines: list[str] = []
    for project, *rollup in rows:
        # Route through ``compare_ttl`` rather than re-deriving the verdict
        # from the SQL's own margin column: the "under N% is not a decision"
        # rule and the sign convention live in one place.
        verdict = compare_ttl(
            cost_5m=float(rollup[_TTL_COST_5M] or 0.0),
            cost_1h=float(rollup[_TTL_COST_1H] or 0.0),
            n_requests=int(rollup[_TTL_N_REQUESTS] or 0),
            n_gaps_recoverable=int(rollup[_TTL_N_RECOVERABLE] or 0),
            n_gaps_unrecoverable=int(rollup[_TTL_N_BREAKS] or 0),
            n_structural=int(rollup[_TTL_N_STRUCTURAL] or 0),
            ttl_observed_dominant=rollup[_TTL_OBSERVED],
        )
        if verdict.decisive:
            call = (
                f"{verdict.recommendation} saves ${verdict.savings:.2f} "
                f"({verdict.margin_pct:.1f}%)"
            )
        else:
            call = f"either (only {verdict.margin_pct:.1f}% apart — within noise)"
        lines.append(
            f"{project}: {call} "
            f"[5m ${verdict.cost_5m:.2f} vs 1h ${verdict.cost_1h:.2f}; "
            f"{verdict.n_requests} requests, "
            f"{verdict.n_gaps_recoverable} recoverable gap(s), "
            f"{verdict.n_gaps_unrecoverable} break(s) >1h, "
            f"{verdict.n_structural} structural; "
            f"currently billed at {verdict.ttl_observed_dominant or '?'}]"
        )
    return "\n".join(lines)


def tool_failures(command_prefix: str = "", limit: int = 20) -> str:
    """List failed tool calls, optionally filtered by tool name prefix."""
    conn = _get_read_connection()
    try:
        if command_prefix:
            rows = conn.execute(
                """
                SELECT
                    session_id, called_at, tool_name,
                    LEFT(tool_input, 200) AS input_preview,
                    LEFT(tool_use_result::VARCHAR, 200) AS result_preview,
                    execution_time
                FROM tool_calls
                WHERE is_error = 'true'
                  AND tool_name LIKE ? || '%'
                ORDER BY called_at DESC
                LIMIT ?
                """,
                [command_prefix, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    session_id, called_at, tool_name,
                    LEFT(tool_input, 200) AS input_preview,
                    LEFT(tool_use_result::VARCHAR, 200) AS result_preview,
                    execution_time
                FROM tool_calls
                WHERE is_error = 'true'
                ORDER BY called_at DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        if not rows:
            return "No failed tool calls found."

        lines: list[str] = []
        for row in rows:
            called = str(row[1])[:19] if row[1] else "?"
            lines.append(
                f"[{called}] session={row[0]} tool={row[2]} exec_time={row[5]}\n"
                f"  input: {row[3]}\n"
                f"  result: {row[4]}"
            )
        return "\n\n".join(lines)
    finally:
        conn.close()


def _parse_ts_epoch(ts_val: object) -> float | None:
    """Return epoch seconds from a DB timestamp value.

    Handles: datetime objects, strings with/without trailing Z, with/without
    fractional seconds. Unlike cost_overview._parse_timestamp, this function
    does not apply `.replace(tzinfo=UTC)` — it returns local-naive epochs
    suitable only for computing intra-session time deltas (where the constant
    offset cancels out). Do not compare the returned values against wall-clock
    UTC timestamps.
    """
    ts = str(ts_val).rstrip("Z").replace("T", " ")
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        try:
            return datetime.fromisoformat(ts.split(".")[0]).timestamp()
        except ValueError:
            return None


def expensive_sessions(limit: int = 15, since: str = "") -> str:  # noqa: PLR0912, PLR0915
    """Return the most expensive sessions ranked by cost, with Pareto analysis.

    Lists sessions in descending cost order. A Pareto marker ``[pareto]``
    flags the sessions that together account for 80% of total spend — the
    last marked row is the one that tips the cumulative share over 80%.

    Each block includes: session ID (pass to ``get_session`` for the full
    conversation), project, start time, model, message counts, tool count,
    file activity, cost split (cache read / cache write / output), spend
    shape (total duration + front-load %), subagent flag, and slash commands
    used.

    Results match the web Cost Overview page's Pareto table.

    Parameters
    ----------
    limit:
        Number of sessions to display (1-50, default 15). Header totals
        always cover *all* sessions regardless of limit.
    since:
        Optional ISO date or timestamp (e.g. ``'2026-06-01'``). When set,
        only costs from messages at or after this point are counted. Empty
        string means all time.
    """
    since_error = _validate_since(since)
    if since_error:
        return since_error

    # Clamp limit to [1, _EXPENSIVE_SESSIONS_ROW_CAP].
    display_limit = max(1, min(limit, _EXPENSIVE_SESSIONS_ROW_CAP))

    # Choose the cost subquery: filtered or unfiltered.
    # Trust contract: `since` was round-tripped through fromisoformat above;
    # splicing it into SQL is safe — mirrors cost_overview._cost_subquery.
    if since:
        cost_subquery = session_cost_subquery_filtered(f"timestamp >= '{since}'")
    else:
        cost_subquery = SESSION_COST_SUBQUERY

    conn = _get_read_connection()
    try:
        # --- Query 1: Pareto ranking + metadata ---
        # Mirrors cost_overview._build_pareto — keep cutoff semantics in sync.
        # Fetches ALL rows (needed for pareto_session_count and totals);
        # display is capped to `display_limit` in Python.
        pareto_rows = conn.execute(
            f"""
            WITH session_costs AS (
                SELECT session_id, cost_usd FROM {cost_subquery}
            ),
            ranked AS (
                SELECT
                    session_id,
                    cost_usd,
                    SUM(cost_usd) OVER () AS grand_total,
                    SUM(cost_usd) OVER (
                        ORDER BY cost_usd DESC, session_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cumulative
                FROM session_costs
                WHERE cost_usd IS NOT NULL AND cost_usd > 0
            )
            SELECT
                r.session_id,
                r.cost_usd,
                r.cumulative,
                r.grand_total,
                r.cumulative / NULLIF(r.grand_total, 0) AS cum_frac,
                ss.started_at,
                ss.duration,
                ss.project,
                ss.model,
                ss.git_branch,
                ss.user_messages,
                ss.assistant_messages,
                ss.tool_count,
                ss.files_read,
                ss.files_edited,
                ss.commands,
                ss.first_prompt
            FROM ranked r
            LEFT JOIN session_stats ss ON ss.session_id = r.session_id
            ORDER BY r.cost_usd DESC, r.session_id
            """  # noqa: S608
        ).fetchall()

        if not pareto_rows:
            return "No sessions with cost found."

        # --- Pareto arithmetic (Python) ---
        # Same loop as _build_pareto: row is in-pareto while prev_cum_frac < 0.80;
        # the tipping row is included.
        grand_total = float(pareto_rows[0][3] or 0.0)
        total_sessions = len(pareto_rows)

        prev_cum_frac = 0.0
        cutoff_seen = False
        pareto_count = 0
        pareto_cost_usd = 0.0
        enriched: list[dict] = []
        for row in pareto_rows:
            (
                session_id,
                cost_usd_raw,
                cumulative_raw,
                _grand_total,
                cum_frac_raw,
                started_at,
                duration,
                project,
                model,
                git_branch,
                user_messages,
                assistant_messages,
                tool_count,
                files_read,
                files_edited,
                commands,
                first_prompt,
            ) = row
            cost_usd = float(cost_usd_raw or 0.0)
            cumulative = float(cumulative_raw or 0.0)
            cum_frac = float(cum_frac_raw or 0.0)

            if cutoff_seen:
                in_pareto = False
            elif prev_cum_frac >= _PARETO_CUTOFF:
                in_pareto = False
                cutoff_seen = True
            else:
                in_pareto = True

            is_cutoff = in_pareto and cum_frac >= _PARETO_CUTOFF
            if in_pareto:
                pareto_count += 1
                pareto_cost_usd = cumulative

            enriched.append(
                {
                    "session_id": session_id,
                    "cost_usd": cost_usd,
                    "cum_frac": cum_frac,
                    "in_pareto": in_pareto,
                    "is_cutoff": is_cutoff,
                    "started_at": str(started_at)[:16] if started_at else "?",
                    "duration": duration,
                    "project": project or "",
                    "model": model or "",
                    "git_branch": git_branch or "",
                    "user_messages": user_messages or 0,
                    "assistant_messages": assistant_messages or 0,
                    "tool_count": tool_count or 0,
                    "files_read": files_read or 0,
                    "files_edited": files_edited or 0,
                    "commands": commands or "",
                    "first_prompt": first_prompt or "",
                }
            )
            prev_cum_frac = cum_frac

        # Rows to display
        display_rows = enriched[:display_limit]
        display_ids = [r["session_id"] for r in display_rows]

        # --- Query 2: spend shape/split for displayed sessions only ---
        if display_ids:
            placeholders = ", ".join("?" * len(display_ids))
            shape_params: list = list(display_ids)
            since_filter = ""
            if since:
                since_filter = " AND timestamp >= ?"
                shape_params.append(since)
            shape_rows = conn.execute(
                f"""
                SELECT
                    session_id,
                    timestamp,
                    ({COST_EXPR_SQL}) / 1e6      AS cost_usd,
                    ({CACHE_READ_COST_SQL}) / 1e6  AS read_usd,
                    ({CACHE_WRITE_COST_SQL}) / 1e6 AS write_usd,
                    ({OUTPUT_COST_SQL}) / 1e6      AS output_usd
                FROM assistant_message_costs
                WHERE session_id IN ({placeholders}){since_filter}
                ORDER BY session_id, timestamp
                """,  # noqa: S608
                shape_params,
            ).fetchall()

            # Aggregate per session
            per_read: dict[str, float] = {}
            per_write: dict[str, float] = {}
            per_output: dict[str, float] = {}
            per_msgs: dict[str, list[tuple[float, float]]] = {}
            for sr in shape_rows:
                sid = sr[0]
                cost = float(sr[2] or 0.0)
                read = float(sr[3] or 0.0)
                write = float(sr[4] or 0.0)
                output = float(sr[5] or 0.0)
                per_read[sid] = per_read.get(sid, 0.0) + read
                per_write[sid] = per_write.get(sid, 0.0) + write
                per_output[sid] = per_output.get(sid, 0.0) + output
                epoch = _parse_ts_epoch(sr[1]) if sr[1] is not None else None
                if epoch is not None:
                    per_msgs.setdefault(sid, []).append((epoch, cost))
        else:
            per_read = per_write = per_output = {}  # type: ignore[assignment]
            per_msgs = {}

        # --- Query 3: subagent flags for displayed sessions ---
        # The `since` predicate is applied to both UNION branches so the flag
        # reflects only activity within the cost window — consistent with Query 1
        # and Query 2. Without it, a session could show subagents=yes due to a
        # sidechain message or Task/Agent call that occurred before `since`.
        # Note: assistant_message_costs uses `timestamp`; tool_calls uses `called_at`.
        subagent_ids: set[str] = set()
        if display_ids:
            placeholders = ", ".join("?" * len(display_ids))
            amc_since_filter = " AND timestamp >= ?" if since else ""
            tc_since_filter = " AND called_at >= ?" if since else ""
            since_params = [since] if since else []
            flag_rows = conn.execute(
                f"""
                SELECT DISTINCT session_id FROM (
                    SELECT session_id FROM assistant_message_costs
                    WHERE is_sidechain = TRUE
                      AND session_id IN ({placeholders}){amc_since_filter}
                    UNION ALL
                    SELECT session_id FROM tool_calls
                    WHERE tool_name IN ('Task', 'Agent')
                      AND session_id IN ({placeholders}){tc_since_filter}
                ) sub
                """,  # noqa: S608
                display_ids + since_params + display_ids + since_params,
            ).fetchall()
            subagent_ids = {r[0] for r in flag_rows}

    finally:
        conn.close()

    # --- Build output ---
    # Lazy import: clean_title lives in the web layer (FastAPI/Jinja2 top-level
    # imports); keeping it lazy prevents those imports at stdio MCP startup.
    from introspect.api.handlers._helpers import clean_title  # noqa: PLC0415

    since_clause = f" (since {since})" if since else ""
    header_lines = [
        f"Total: ${grand_total:.2f} across {total_sessions} sessions{since_clause}.",
        f"Pareto: {pareto_count} sessions account for 80% (${pareto_cost_usd:.2f}).",
        f"Top {display_limit} by cost:",
    ]

    blocks: list[str] = []
    for i, row in enumerate(display_rows, start=1):
        sid = row["session_id"]
        cum_pct = round(row["cum_frac"] * 100)
        in_pareto = row["in_pareto"]
        is_cutoff = row["is_cutoff"]

        if not in_pareto:
            pareto_marker = ""
        elif is_cutoff:
            pareto_marker = "  [pareto, crosses 80%]"
        else:
            pareto_marker = "  [pareto]"

        # Line 1: cost + cumulative % + pareto marker
        line1 = f"{i}. ${row['cost_usd']:.2f}  cum {cum_pct}%{pareto_marker}"

        # Line 2: session metadata
        line2 = (
            f"   session={sid} project={row['project'] or '—'}"
            f" started={row['started_at']}"
        )

        # Line 3: duration / model / branch
        line3 = (
            f"   duration={row['duration'] or '?'}"
            f" model={row['model'] or '?'}"
            f" branch={row['git_branch'] or '?'}"
        )

        # Line 4: message counts / tools / files / subagents / commands
        has_subagents = sid in subagent_ids
        subagent_str = "yes" if has_subagents else "none"
        commands_str = row["commands"] or "none"
        line4 = (
            f"   msgs={row['user_messages']}u/{row['assistant_messages']}a"
            f" tools={row['tool_count']}"
            f" files={row['files_read']}r/{row['files_edited']}w"
            f" subagents={subagent_str}"
            f" commands={commands_str}"
        )

        # Line 5: split
        read_usd = per_read.get(sid, 0.0)
        write_usd = per_write.get(sid, 0.0)
        output_usd = per_output.get(sid, 0.0)
        split_total = read_usd + write_usd + output_usd
        if split_total > 0:
            read_pct = round(100 * read_usd / split_total)
            write_pct = round(100 * write_usd / split_total)
            output_pct = 100 - read_pct - write_pct
            line5 = (
                f"   split: cache read {read_pct}% (${read_usd:.2f})"
                f" · cache write {write_pct}% (${write_usd:.2f})"
                f" · output {output_pct}% (${output_usd:.2f})"
            )
        else:
            line5 = "   split: —"

        # Line 6: shape
        msgs = per_msgs.get(sid, [])
        if len(msgs) <= 1:
            line6 = "   shape: single message"
        else:
            t0 = msgs[0][0]
            t1 = msgs[-1][0]
            dur_secs = t1 - t0
            dur_min = round(dur_secs / 60)
            total_cost_msgs = sum(c for _, c in msgs)
            mid = t0 + dur_secs / 2
            front = sum(c for ts, c in msgs if ts <= mid)
            front_pct = (
                round(100 * front / total_cost_msgs) if total_cost_msgs > 0 else 0
            )
            line6 = f"   shape: {dur_min} min · {front_pct}% of spend in first half"

        # Line 7: title
        title = clean_title(row["first_prompt"])[:120] if row["first_prompt"] else "—"
        line7 = f"   title: {title}"

        blocks.append("\n".join([line1, line2, line3, line4, line5, line6, line7]))

    return "\n".join(header_lines) + "\n\n" + "\n\n".join(blocks)
