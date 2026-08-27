"""CLI interface for introspect."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import subprocess

    import duckdb

import typer
from rich.console import Console
from rich.table import Table

from introspect.cache_ttl import (
    MAX_RECOVERABLE_GAP_SECONDS,
    TTL_5M_SECONDS,
    TtlComparison,
    gap_histogram,
    global_ttl_comparison,
    parity_residuals,
    project_ttl_comparisons,
    split_coverage,
)
from introspect.db import (
    DEFAULT_CODEX_GLOB,
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_GLOB,
    DatabaseLockedError,
    connect_writable,
    ensure_materialized,
    get_read_connection,
    materialize_views,
)
from introspect.search import build_search_corpus, ensure_search_corpus, fts_search
from introspect.sql_query import is_loopback_host
from introspect.version_check import maybe_notify_update

SID_TRUNCATE = 12

# Rows shown per section in ``stats`` before it collapses to a count.
_STATS_TOP_N = 10

app = typer.Typer(help="Explore Claude Code conversation logs.")
console = Console()


@app.callback()
def main(ctx: typer.Context) -> None:
    """Explore Claude Code conversation logs."""
    # Shared entry for every command. The update nag gates itself (mcp / non-TTY
    # / CI / opt-out all short-circuit) and defers its single stderr line to
    # command close, so it prints after the real output rather than mid-render.
    maybe_notify_update(ctx.invoked_subcommand, ctx.call_on_close)


def _truncate_sid(val) -> str:
    s = str(val) if val else ""
    return s[:SID_TRUNCATE] + "..." if len(s) > SID_TRUNCATE else s


_JUST_NOW_SECONDS = 30
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400


def _format_relative(dt: datetime | None) -> str:
    """Render ``dt`` as a coarse relative time (``"5m ago"``, ``"just now"``, ...)."""
    if dt is None:
        return "never"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = (datetime.now(UTC) - dt).total_seconds()
    if delta < _JUST_NOW_SECONDS:
        return "just now"
    if delta < _SECONDS_PER_HOUR:
        return f"{int(delta // _SECONDS_PER_MINUTE)}m ago"
    if delta < _SECONDS_PER_DAY:
        return f"{int(delta // _SECONDS_PER_HOUR)}h ago"
    return f"{int(delta // _SECONDS_PER_DAY)}d ago"


def _db(
    db_path: Path | None = None,
    jsonl_glob: str | None = None,
):
    """Return a read connection, materializing the DB on first use.

    The CLI shares its DB with ``introspect serve``; if the server has already
    built the on-disk tables we reuse them. Otherwise we build them now so
    every command sees the same fast path. A header line tells the user when
    the data was last materialized so stale results are obvious.

    Defaults are resolved at call time (rather than via parameter defaults)
    so that ``monkeypatch.setattr("introspect.cli.DEFAULT_DB_PATH", ...)`` in
    tests redirects this code path the same way it redirects ``materialize``.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if jsonl_glob is None:
        jsonl_glob = DEFAULT_JSONL_GLOB
    try:
        materialized_at = ensure_materialized(
            db_path, jsonl_glob, codex_glob=DEFAULT_CODEX_GLOB
        )
    except DatabaseLockedError as e:
        _print_lock_error(e.db_path)
        raise typer.Exit(code=1) from None
    _print_materialized_banner(materialized_at)
    return get_read_connection(db_path, jsonl_glob, DEFAULT_CODEX_GLOB)


def _print_materialized_banner(materialized_at: datetime | None) -> None:
    """Print the 'last materialized' header above command output."""
    if materialized_at is None:
        console.print("[dim]Last materialized: unknown[/dim]")
        return
    iso = materialized_at.strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[dim]Last materialized: {iso} ({_format_relative(materialized_at)})[/dim]"
    )


def _print_lock_error(db_path: Path) -> None:
    """Print the user-facing 'another process has the DB' message."""
    console.print(
        "[red]Error:[/red] Another Introspect process is already using the "
        f"database at [cyan]{db_path}[/cyan].\n"
        "It looks like a server (or other writer) is running elsewhere — "
        "stop that instance before starting a new one."
    )


@app.command()
def sessions(
    limit: int = typer.Option(20, help="Number of sessions to show"),
):
    """List recent sessions with timestamps."""
    conn = _db()
    rows = conn.execute(
        """
        SELECT
            session_id,
            started_at,
            ended_at,
            duration,
            user_messages,
            assistant_messages,
            model,
            cwd,
        FROM logical_sessions
        ORDER BY started_at DESC
        LIMIT ?
    """,
        [limit],
    ).fetchall()

    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan", max_width=12)
    table.add_column("Started", style="green")
    table.add_column("Duration")
    table.add_column("User Msgs", justify="right")
    table.add_column("Asst Msgs", justify="right")
    table.add_column("Model")
    table.add_column("CWD", max_width=30)

    for row in rows:
        sid = _truncate_sid(row[0])
        started = str(row[1])[:19] if row[1] else ""
        duration = str(row[3]) if row[3] else ""
        table.add_row(
            sid,
            started,
            duration,
            str(row[4] or 0),
            str(row[5] or 0),
            row[6] or "",
            row[7] or "",
        )

    console.print(table)
    conn.close()


@app.command()
def tools(
    failed: bool = typer.Option(False, "--failed", help="Show only failed tool calls"),
    tool_name: str | None = typer.Option(None, "--name", help="Filter by tool name"),
    limit: int = typer.Option(30, help="Number of results"),
):
    """Show tool call history."""
    conn = _db()

    where_clauses = []
    params = []
    if failed:
        where_clauses.append("is_error = 'true'")
    if tool_name:
        where_clauses.append("tool_name = ?")
        params.append(tool_name)

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT
            session_id,
            called_at,
            tool_name,
            is_error,
            LEFT(tool_input, 100) AS input_preview,
            execution_time,
        FROM tool_calls
        {where}
        ORDER BY called_at DESC
        LIMIT ?
    """,  # noqa: S608
        params,
    ).fetchall()

    table = Table(title="Tool Calls" + (" (failed)" if failed else ""))
    table.add_column("Session", style="cyan", max_width=12)
    table.add_column("Called At", style="green")
    table.add_column("Tool", style="yellow")
    table.add_column("Error", justify="center")
    table.add_column("Input Preview", max_width=50)
    table.add_column("Exec Time")

    for row in rows:
        sid = _truncate_sid(row[0])
        called = str(row[1])[:19] if row[1] else ""
        error_str = "Yes" if row[3] == "true" else ""
        style = "red" if row[3] == "true" else None
        table.add_row(
            sid,
            called,
            row[2] or "",
            error_str,
            row[4] or "",
            str(row[5] or ""),
            style=style,
        )

    console.print(table)
    conn.close()


@app.command()
def tables():
    """List available SQL views and tables for use with the query command."""
    conn = _db()
    try:
        rows = conn.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_type, table_name
        """).fetchall()

        table = Table(title="Available Tables & Views")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Columns")

        for name, ttype in rows:
            cols = conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ? AND table_schema = 'main'
                ORDER BY ordinal_position
                """,
                [name],
            ).fetchall()
            col_names = ", ".join(c[0] for c in cols)
            table.add_row(name, ttype.lower(), col_names)

        console.print(table)
    finally:
        conn.close()


@app.command()
def query(
    sql: str = typer.Argument(help="SQL query to execute"),
):
    """Run an ad-hoc SQL query against the views."""
    conn = _db()
    try:
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        table = Table()
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(v) if v is not None else "" for v in row])

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    finally:
        conn.close()


@app.command()
def raw(
    limit: int = typer.Option(5, help="Number of records to show"),
    session: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
):
    """Show raw unfiltered JSONL records — all fields, no transformation."""
    conn = _db()
    try:
        where = ""
        params: list[str] = []
        if session:
            where = "WHERE CAST(sessionId AS VARCHAR) LIKE ?"
            params.append(f"{session}%")
        result = conn.execute(
            f"SELECT * FROM raw_data {where} LIMIT {limit}",  # noqa: S608
            params,
        )
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        if not rows:
            console.print("[yellow]No records found.[/yellow]")
            raise typer.Exit()

        console.print(f"[dim]{len(columns)} columns: {', '.join(columns)}[/dim]\n")

        for i, row in enumerate(rows):
            console.print(f"[bold cyan]--- Record {i + 1} ---[/bold cyan]")
            for col, val in zip(columns, row, strict=True):
                if val is None:
                    continue
                val_str = str(val)
                max_display = 200
                if len(val_str) > max_display:
                    val_str = val_str[:max_display] + "..."
                console.print(f"  [yellow]{col}:[/yellow] {val_str}")
            console.print()
    finally:
        conn.close()


@app.command()
def stats():
    """Show summary statistics."""
    conn = _db()

    session_count = conn.execute("SELECT COUNT(*) FROM logical_sessions").fetchone()[0]
    tool_count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    failed_count = conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE is_error = 'true'"
    ).fetchone()[0]

    tool_breakdown = conn.execute("""
        SELECT tool_name, COUNT(*) AS cnt
        FROM tool_calls
        GROUP BY tool_name
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    console.print(f"\n[bold]Sessions:[/bold] {session_count}")
    console.print(f"[bold]Tool calls:[/bold] {tool_count}")
    console.print(f"[bold]Failed tool calls:[/bold] {failed_count}")

    if tool_breakdown:
        console.print("\n[bold]Top tools:[/bold]")
        table = Table()
        table.add_column("Tool")
        table.add_column("Count", justify="right")
        for name, cnt in tool_breakdown:
            table.add_row(name or "?", str(cnt))
        console.print(table)

    ttl_rows = project_ttl_comparisons(conn)
    if ttl_rows:
        console.print(
            "\n[bold]Prompt-cache TTL[/bold] "
            "[dim](introspy cache-ttl for the detail)[/dim]"
        )
        # Capped like the tool breakdown above — this is a summary, and a
        # machine with fifty projects shouldn't turn it into a wall.
        for project, verdict in ttl_rows[:_STATS_TOP_N]:
            console.print(f"  {project}: {_format_verdict(verdict)}")
        if len(ttl_rows) > _STATS_TOP_N:
            console.print(f"  [dim]… {len(ttl_rows) - _STATS_TOP_N} more[/dim]")

    conn.close()


def _format_verdict(verdict: TtlComparison) -> str:
    """One-line recommendation with its margin, or an honest shrug."""
    if verdict.n_requests == 0:
        return "[dim]no requests[/dim]"
    if not verdict.decisive:
        return (
            f"[dim]either ({verdict.margin_pct:.1f}% apart — inside "
            f"modelling error)[/dim]"
        )
    colour = "green" if verdict.recommendation == "1h" else "cyan"
    return (
        f"[{colour}]{verdict.recommendation}[/{colour}] saves "
        f"${verdict.savings:.2f} ({verdict.margin_pct:.1f}%)"
    )


def _ttl_table(rows: list[tuple[str, TtlComparison]]) -> Table:
    """Per-project recommendation table."""
    table = Table()
    table.add_column("Project")
    table.add_column("Reqs", justify="right")
    table.add_column("Recoverable", justify="right")
    table.add_column("Breaks", justify="right")
    table.add_column("5m", justify="right")
    table.add_column("1h", justify="right")
    table.add_column("Observed TTL")
    table.add_column("Recommendation")
    for project, verdict in rows:
        table.add_row(
            project,
            str(verdict.n_requests),
            str(verdict.n_gaps_recoverable),
            str(verdict.n_gaps_unrecoverable),
            f"${verdict.cost_5m:.2f}",
            f"${verdict.cost_1h:.2f}",
            verdict.ttl_observed_dominant or "?",
            _format_verdict(verdict),
        )
    return table


@app.command(name="cache-ttl")
def cache_ttl(
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Show the simulation's parity residuals and 5m/1h split coverage "
        "instead of the recommendation.",
    ),
    subagents: bool = typer.Option(
        False,
        "--subagents",
        help="Score sidechain traffic (subagentPromptCacheTtl) instead of the "
        "main conversation. Never merged with the main-chain verdict.",
    ),
):
    """Would a 1h or 5m prompt-cache TTL have been cheaper?

    Replays every API request under both policies. The prefix a request
    re-sends is the same either way; only the read/write split moves, so the
    comparison turns on one thing — which gaps each TTL would have kept warm.
    """
    conn = _db()
    try:
        if verify:
            _print_ttl_verification(conn)
            return

        overall = global_ttl_comparison(conn, sidechain=subagents)
        scope = "Subagents" if subagents else "Main conversation"
        console.print(f"\n[bold]{scope}[/bold] — {overall.n_requests} requests")
        if overall.n_requests == 0:
            console.print("[dim]No cache data.[/dim]")
            return

        console.print(f"  5m: [bold]${overall.cost_5m:.2f}[/bold]")
        console.print(f"  1h: [bold]${overall.cost_1h:.2f}[/bold]")
        console.print(f"  → {_format_verdict(overall)}")
        console.print(
            f"  [dim]{overall.n_gaps_recoverable} recoverable gap(s), "
            f"{overall.n_gaps_unrecoverable} break(s) over "
            f"{MAX_RECOVERABLE_GAP_SECONDS // 60} min (no TTL recovers those), "
            f"{overall.n_structural} structural invalidation(s).[/dim]"
        )

        rows = project_ttl_comparisons(conn, sidechain=subagents)
        if rows:
            console.print("\n[bold]By project[/bold] (the setting is per project):")
            console.print(_ttl_table(rows))

        console.print("\n[bold]Gaps between requests:[/bold]")
        hist = Table()
        hist.add_column("Gap")
        hist.add_column("Requests", justify="right")
        hist.add_column("Prefix tokens at stake", justify="right")
        hist.add_column("Recoverable")
        for bucket in gap_histogram(conn, sidechain=subagents):
            hist.add_row(
                bucket["bucket"],
                str(bucket["count"]),
                f"{bucket['prefix_tokens']:,}",
                "yes" if bucket["recoverable"] else "no",
            )
        console.print(hist)
        console.print(
            "\n[dim]Costs are list API prices; subscription dollars are "
            "treated as API-equivalent.[/dim]"
        )
    finally:
        conn.close()


def _print_ttl_verification(conn: "duckdb.DuckDBPyConnection") -> None:
    """Parity residuals + split coverage — the gate on the simulation.

    Simulating the TTL a session was *actually* billed at has to reproduce
    its bill. A non-zero residual means the gap definition misclassified a
    request's warmth, and nothing built on the counterfactual is trustworthy
    until it is explained.
    """
    coverage = split_coverage(conn)
    console.print("\n[bold]cache_creation 5m/1h split coverage by month[/bold]")
    table = Table()
    table.add_column("Month")
    table.add_column("Requests", justify="right")
    table.add_column("With writes", justify="right")
    table.add_column("Missing split", justify="right")
    table.add_column("Sum mismatch", justify="right")
    for row in coverage:
        table.add_row(
            row["month"],
            str(row["n_requests"]),
            str(row["n_with_writes"]),
            f"{row['n_missing_split']} ({row['pct_missing_split']:.1f}%)",
            str(row["n_split_mismatch"]),
        )
    console.print(table)
    if any(row["n_split_mismatch"] for row in coverage):
        console.print(
            "[red]Split does not sum to cache_creation_tokens on some rows — "
            "prefix_total is unreliable there.[/red]"
        )

    residuals = parity_residuals(conn)
    console.print(
        f"\n[bold]Parity: simulated vs observed[/bold] "
        f"({len(residuals)} uniform-TTL session(s))"
    )
    if not residuals:
        console.print(
            "[dim]No session has a single observed TTL throughout — nothing "
            "to reproduce.[/dim]"
        )
        return
    worst = residuals[:10]
    table = Table()
    table.add_column("Session")
    table.add_column("TTL")
    table.add_column("Reqs", justify="right")
    table.add_column("Observed", justify="right")
    table.add_column("Simulated", justify="right")
    table.add_column("Residual", justify="right")
    for row in worst:
        table.add_row(
            _truncate_sid(row["session_id"]),
            row["ttl_observed"],
            str(row["n_requests"]),
            f"${row['observed_usd']:.4f}",
            f"${row['simulated_usd']:.4f}",
            f"{row['residual_pct']:+.2f}%",
        )
    console.print(table)
    max_residual = max(abs(row["residual_pct"]) for row in residuals)
    console.print(
        f"Worst residual: [bold]{max_residual:.3f}%[/bold] "
        f"(gap threshold {TTL_5M_SECONDS}s / "
        f"{MAX_RECOVERABLE_GAP_SECONDS}s)"
    )


@app.command()
def search(
    query_text: str = typer.Argument(help="Text to search for"),
    limit: int = typer.Option(20, help="Number of results"),
):
    """Full-text search across conversation logs."""
    conn = _db()
    try:
        ensure_search_corpus(conn)

        results = fts_search(conn, query_text, limit)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            raise typer.Exit()

        table = Table(title=f"Search results for: {query_text}")
        table.add_column("Session ID", style="cyan", max_width=12)
        table.add_column("Timestamp", style="green")
        table.add_column("Role")
        table.add_column("CWD", style="dim", max_width=30)
        table.add_column("Snippet", max_width=80)
        table.add_column("Score", justify="right")

        for session_id, timestamp, role, cwd, snippet, score in results:
            table.add_row(
                _truncate_sid(session_id),
                str(timestamp)[:19] if timestamp else "",
                role or "",
                cwd or "",
                snippet or "",
                f"{score:.4f}" if score is not None else "",
            )

        console.print(table)
    finally:
        conn.close()


@app.command()
def materialize(
    days: int = typer.Option(
        10, "-d", "--days", help="Days of history to load (0 = no limit)"
    ),
    no_resolve_projects: bool = typer.Option(
        False,
        "--no-resolve-projects",
        help="Skip git worktree resolution for project names",
    ),
):
    """Materialize data into DuckDB for fast CLI and MCP queries."""
    db_path = DEFAULT_DB_PATH
    jsonl_glob = DEFAULT_JSONL_GLOB
    codex_glob = DEFAULT_CODEX_GLOB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = connect_writable(db_path)
    except DatabaseLockedError as e:
        _print_lock_error(e.db_path)
        raise typer.Exit(code=1) from None
    try:
        if days > 0:
            console.print(f"[dim]Materializing last {days} days of data...[/dim]")
        else:
            console.print("[dim]Materializing all data (no day limit)...[/dim]")
        materialize_views(
            conn,
            jsonl_glob,
            days,
            resolve_projects=not no_resolve_projects,
            codex_glob=codex_glob,
        )
        build_search_corpus(conn)
        row = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
        count = row[0] if row else 0
        console.print(f"[green]Materialized {count} messages into {db_path}[/green]")
    finally:
        conn.close()


# Deliberately off the beaten path: 8000/8080 collide with every other dev
# server, and the MCP registration in Claude Code must match the live port.
DEFAULT_PORT = 8347

PORT_PROBE_ATTEMPTS = 20


def _find_available_port(host: str, start_port: int, attempts: int) -> int | None:
    """Return the first free port at or after `start_port`, or None if none found.

    Only skips ports that are in use. Permission errors (e.g. privileged ports
    below 1024 without capability) propagate so the user sees a real error
    instead of silently ending up on a different port.
    """
    import errno  # noqa: PLC0415
    import socket  # noqa: PLC0415

    for candidate in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    continue
                raise
            else:
                return candidate
    return None


def _run_web_ui(
    host: str,
    port: int,
    days: int,
    no_resolve_projects: bool,
    reload: bool,
) -> None:
    """Shared uvicorn launcher for the `serve` and `devserve` commands."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    # Preflight: detect "server already running" before handing off to uvicorn,
    # since a DatabaseLockedError raised from inside the lifespan is surfaced as
    # an ugly uvicorn startup traceback. Small TOCTOU window here is harmless —
    # if the first server exits between probe and lifespan, the new one starts.
    db_path = Path(os.environ.get("INTROSPECT_DB_PATH", str(DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connect_writable(db_path).close()
    except DatabaseLockedError as e:
        _print_lock_error(e.db_path)
        raise typer.Exit(code=1) from None

    available = _find_available_port(host, port, PORT_PROBE_ATTEMPTS)
    if available is None:
        console.print(
            f"[red]Error:[/red] Tried {PORT_PROBE_ATTEMPTS} ports starting at "
            f"{port} on [cyan]{host}[/cyan]; none were free.\n"
            "Stop whatever is holding those ports, or pass [cyan]--port[/cyan] "
            "with a different starting value."
        )
        raise typer.Exit(code=1)
    if available != port:
        console.print(
            f"[yellow]Port {port} is in use; using port {available} instead.[/yellow]"
        )
        port = available

    os.environ["INTROSPECT_DAYS"] = str(days)
    # The app reads INTROSPECT_HOST in its lifespan to decide whether to expose
    # the local-only SQL API (loopback bind only). Keep it in sync with `host`.
    os.environ["INTROSPECT_HOST"] = host
    if no_resolve_projects:
        os.environ["INTROSPECT_RESOLVE_PROJECTS"] = "0"

    banner = "dev server" if reload else "web UI"
    console.print(f"[bold]Starting Introspect {banner} on http://{host}:{port}[/bold]")
    console.print(f"[dim]MCP endpoint: http://{host}:{port}/mcp[/dim]")
    if is_loopback_host(host):
        console.print(
            f"[dim]SQL API: POST http://{host}:{port}/api/query (local only)[/dim]"
        )
    if days > 0:
        console.print(f"[dim]Loading last {days} days of data...[/dim]")
    else:
        console.print("[dim]Loading all data (no day limit)...[/dim]")

    kwargs: dict[str, object] = {
        "host": host,
        "port": port,
        "log_level": "info",
    }
    if reload:
        reload_dir = str(Path(__file__).resolve().parent)
        console.print(f"[dim]Auto-reload watching {reload_dir}[/dim]")
        kwargs["reload"] = True
        kwargs["reload_dirs"] = [reload_dir]
    uvicorn.run("introspect.api.main:app", **kwargs)  # ty: ignore[invalid-argument-type]


# Dev-server databases are namespaced per git branch so that switching branches
# (which may change the schema) never reuses a stale DB built on another branch.
_BRANCH_DB_PREFIX = "introspect-"


def _current_git_branch() -> str | None:
    """Return the current git branch, or None on detached HEAD / not a repo."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip() or None


def _sanitize_branch(branch: str) -> str:
    """Make a branch name safe for use in a filename (``feat/x`` -> ``feat-x``)."""
    import re  # noqa: PLC0415

    return re.sub(r"[^A-Za-z0-9._-]", "-", branch)


def _branch_db_path(branch: str) -> Path:
    """Per-branch DuckDB path alongside the default DB."""
    return (
        DEFAULT_DB_PATH.parent / f"{_BRANCH_DB_PREFIX}{_sanitize_branch(branch)}.duckdb"
    )


def _remove_db(db_path: Path) -> None:
    """Delete a DuckDB file and its DuckDB/refresh sidecars, if present."""
    for f in (
        db_path,
        db_path.with_name(db_path.name + ".wal"),
        db_path.with_name(db_path.name + ".next"),
    ):
        f.unlink(missing_ok=True)


def _git_toplevel(cwd: str | Path) -> Path | None:
    """Return the git worktree root for ``cwd``, or None if not a repo."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return Path(result.stdout.strip()).resolve()


def _prune_stale_branch_dbs(keep: Path) -> list[Path]:
    """Delete branch DBs whose branch no longer exists locally.

    Branch DBs are keyed by bare branch name and shared across all repos in
    ``~/.introspect/``, so pruning is only safe when ``devserve`` is launched
    from the introspect repo itself (its own dev scenario) — otherwise another
    project's branch list would nuke introspect's dev DBs. Bail out unless the
    cwd's git worktree is the same repo this code is running from.

    Never touches ``keep`` (the current branch's DB) or the shared default DB.
    Returns the removed primary DB paths.
    """
    import subprocess  # noqa: PLC0415

    parent = DEFAULT_DB_PATH.parent
    if not parent.exists():
        return []
    cwd_repo = _git_toplevel(".")
    src_repo = _git_toplevel(Path(__file__).parent)
    if cwd_repo is None or cwd_repo != src_repo:
        return []  # not developing introspect here — don't touch its DBs
    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []  # not a git repo — don't delete anything
    live = {
        f"{_BRANCH_DB_PREFIX}{_sanitize_branch(b)}.duckdb"
        for b in result.stdout.split()
        if b
    }
    removed: list[Path] = []
    for db in parent.glob(f"{_BRANCH_DB_PREFIX}*.duckdb"):
        if db == keep or db.name in live:
            continue
        _remove_db(db)
        removed.append(db)
    return removed


@app.command()
def serve(
    port: int = typer.Option(DEFAULT_PORT, help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    days: int = typer.Option(
        10, "-d", "--days", help="Days of history to load (0 = no limit)"
    ),
    no_resolve_projects: bool = typer.Option(
        False,
        "--no-resolve-projects",
        help="Skip git worktree resolution for project names",
    ),
):
    """Launch the web UI."""
    _run_web_ui(host, port, days, no_resolve_projects, reload=False)


@app.command()
def devserve(
    port: int = typer.Option(DEFAULT_PORT, help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    days: int = typer.Option(
        10, "-d", "--days", help="Days of history to load (0 = no limit)"
    ),
    no_resolve_projects: bool = typer.Option(
        False,
        "--no-resolve-projects",
        help="Skip git worktree resolution for project names",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Rebuild this branch's dev DB from scratch instead of reusing it",
    ),
):
    """Launch the web UI with auto-reload on source changes.

    Uses a per-branch DuckDB (`introspect-<branch>.duckdb`) so dev servers on
    different branches never share a schema-mismatched database. Respects an
    explicit `INTROSPECT_DB_PATH` if set. Also prunes dev DBs for branches
    that no longer exist.
    """
    import os  # noqa: PLC0415

    if "INTROSPECT_DB_PATH" not in os.environ:
        branch = _current_git_branch()
        if branch is not None:
            db_path = _branch_db_path(branch)
            os.environ["INTROSPECT_DB_PATH"] = str(db_path)
            console.print(f"[dim]Dev DB (branch '{branch}'): {db_path}[/dim]")
            if clean:
                _remove_db(db_path)
                console.print("[dim]--clean: removed existing dev DB[/dim]")
            pruned = _prune_stale_branch_dbs(keep=db_path)
            if pruned:
                console.print(
                    f"[dim]Pruned {len(pruned)} dev DB(s) for deleted branches[/dim]"
                )

    _run_web_ui(host, port, days, no_resolve_projects, reload=True)


@app.command()
def mcp():
    """Run the MCP server (stdio transport) for Claude Code integration."""
    from introspect.mcp.server import create_mcp_server  # noqa: PLC0415

    create_mcp_server().run(transport="stdio")


# How long to wait for a freshly-spawned server to become connectable.
# First start materialises the DuckDB, which can take tens of seconds on a
# large history — be generous.
SERVER_START_TIMEOUT_SECONDS = 120.0
SERVER_POLL_INTERVAL_SECONDS = 0.5
# Grace period for a spawned server to shut down after SIGTERM before we
# escalate to SIGKILL.
SERVER_STOP_TIMEOUT_SECONDS = 10.0


def _serve_log_path() -> Path:
    """Return the path to the background server log file.

    Defined as a function (rather than a constant) so tests can monkeypatch it
    to a temporary directory and avoid writing into the real ``~/.introspect``.
    """
    return DEFAULT_DB_PATH.parent / "serve.log"


def _ensure_server_running(host: str, port: int) -> "subprocess.Popen[bytes] | None":
    """Ensure an introspect server is listening on *host*:*port*.

    If the port is already connectable, return ``None`` immediately.
    Otherwise spawn ``python -m introspect.cli serve`` as a detached
    background process, stream its output to ``_serve_log_path()``, poll
    until the port becomes connectable, and return the child process so the
    caller can stop it when the session ends.

    Raises ``typer.Exit(code=1)`` on child death or timeout without touching
    the child process (it may still be materialising the DB).
    """
    import socket  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    # Fast path: server is already up.
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError:
        pass
    else:
        return None

    log_path = _serve_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[dim]Starting introspect server in the background "
        f"([cyan]{host}:{port}[/cyan])[/dim]\n"
        f"[dim]Log: {log_path}[/dim]\n"
        "[dim]First start may take a while while the DB is materialised...[/dim]"
    )

    # TOCTOU caveat: if another process grabs the requested port between our
    # probe above and the child's bind, _run_web_ui will shift to port+1 and
    # our readiness poll on the original port will time out.  Acceptable edge
    # case; do not engineer around it.
    with log_path.open("ab") as log_fh:
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "introspect.cli",
                "serve",
                "--port",
                str(port),
                "--host",
                host,
            ],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(SERVER_POLL_INTERVAL_SECONDS)

        if proc.poll() is not None:
            console.print(
                f"[red]Error:[/red] introspect server exited with code "
                f"{proc.returncode} before becoming ready.\n"
                f"Check the log for details: [cyan]{log_path}[/cyan]"
            )
            raise typer.Exit(code=1)

        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except OSError:
            pass
        else:
            console.print(f"[green]Server ready on http://{host}:{port}[/green]")
            return proc

    console.print(
        f"[red]Error:[/red] Server did not become ready within "
        f"{SERVER_START_TIMEOUT_SECONDS:.0f} s.\n"
        f"It was left running (pid {proc.pid}) — check the log for the actual "
        "bound port and any startup errors (a port-shift or slow DB "
        "materialisation can both cause this).\n"
        f"Log: [cyan]{log_path}[/cyan]"
    )
    raise typer.Exit(code=1)


def _stop_server(proc: "subprocess.Popen[bytes]") -> None:
    """Terminate the server we spawned, escalating to kill if it hangs."""
    import subprocess  # noqa: PLC0415

    if proc.poll() is not None:
        return  # already gone
    console.print("[dim]Stopping introspect server...[/dim]")
    proc.terminate()
    try:
        proc.wait(timeout=SERVER_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _finish_connected_session(
    server_proc: "subprocess.Popen[bytes] | None",
    *,
    host: str,
    port: int,
    keep_server: bool,
) -> None:
    """Clean up an Introspect server spawned for an interactive client."""
    if server_proc is None:
        return
    if keep_server:
        console.print(
            f"[dim]Server left running on http://{host}:{port} "
            f"(pid {server_proc.pid})[/dim]"
        )
        return
    _stop_server(server_proc)


# Added to the dedicated Claude Code / Codex session prompt. The session is
# dedicated to log analysis, so steer it toward the MCP tools instead of
# spelunking the raw rollout logs with Bash.
SESSION_INSTRUCTIONS = (
    "This session is dedicated to analyzing coding-agent conversation logs "
    "via the `introspect` MCP server. Prefer the mcp__introspect__* tools "
    "(run_sql, describe_schema, search_conversations, get_session, "
    "recent_sessions, tool_failures, tool_failure_rate, refresh_data, "
    "expensive_sessions, cache_ttl_choice, list_query_templates) over reading "
    "~/.claude/projects JSONL files directly — the views already handle "
    "session stitching, cost attribution, and project resolution. For "
    "ranked expensive sessions with cost split and Pareto analysis, call "
    "expensive_sessions. For the prompt-cache TTL question — would 1h or 5m "
    "have been cheaper — call cache_ttl_choice rather than deriving it "
    "from cache-miss waste, which cannot answer it. Call describe_schema "
    "before writing SQL, and "
    "list_query_templates for curated starting-point queries to adapt."
)

# Kept as a named alias because Claude Code calls this an appended system
# prompt, while Codex calls it developer instructions.
CLAUDE_SYSTEM_PROMPT_SUFFIX = SESSION_INSTRUCTIONS
CODEX_DEVELOPER_INSTRUCTIONS = SESSION_INSTRUCTIONS

# Permission rule covering every tool on the introspect MCP server, so the
# dedicated session doesn't permission-prompt on each query.
MCP_PERMISSION_RULE = "mcp__introspect"


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def claude(
    ctx: typer.Context,
    port: int = typer.Option(
        DEFAULT_PORT, help="Port the introspect server is listening on"
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Host the introspect server is bound to"
    ),
    keep_server: bool = typer.Option(
        False,
        "--keep-server",
        help="Leave the auto-started server running after Claude Code exits",
    ),
):
    """Launch Claude Code connected to the introspect MCP server.

    Starts `introspy serve` automatically in the background when nothing is
    listening on the target port (log at `~/.introspect/serve.log`) and stops
    it again when Claude Code exits (pass `--keep-server` to leave it up).
    A server that was already running is never touched.  The MCP config is
    passed inline, so nothing is written to your Claude Code settings — the
    server is only registered for this session.

    Any extra arguments are forwarded to the `claude` CLI as-is, e.g.
    `introspy claude -- --model opus --resume` or
    `introspy claude -- -p "recent sessions"`. Use `--` to separate
    introspect's own options from the ones meant for Claude Code.
    """
    import json  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        console.print(
            "[red]Error:[/red] `claude` CLI not found on PATH. "
            "Install Claude Code first: https://claude.com/claude-code"
        )
        raise typer.Exit(code=1)

    server_proc = _ensure_server_running(host, port)

    config = json.dumps(
        {
            "mcpServers": {
                "introspect": {"type": "http", "url": f"http://{host}:{port}/mcp"}
            }
        }
    )
    try:
        exit_code = subprocess.call(  # noqa: S603
            [
                claude_bin,
                "--mcp-config",
                config,
                "--append-system-prompt",
                CLAUDE_SYSTEM_PROMPT_SUFFIX,
                "--allowedTools",
                MCP_PERMISSION_RULE,
                *ctx.args,
            ]
        )
    finally:
        _finish_connected_session(
            server_proc, host=host, port=port, keep_server=keep_server
        )
    raise typer.Exit(code=exit_code)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def codex(
    ctx: typer.Context,
    port: int = typer.Option(
        DEFAULT_PORT, help="Port the introspect server is listening on"
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Host the introspect server is bound to"
    ),
    keep_server: bool = typer.Option(
        False,
        "--keep-server",
        help="Leave the auto-started server running after Codex exits",
    ),
):
    """Launch Codex connected to the introspect MCP server.

    Starts `introspy serve` automatically in the background when nothing is
    listening on the target port (log at `~/.introspect/serve.log`) and stops
    it again when Codex exits (pass `--keep-server` to leave it up). A server
    that was already running is never touched. The MCP configuration and
    developer instructions are passed as command-line overrides, so nothing is
    written to the user's Codex configuration.

    Any extra arguments are forwarded to the `codex` CLI as-is, e.g.
    `introspy codex -- --model gpt-5.4` or
    `introspy codex -- "what are the most expensive sessions"`. Use `--`
    to separate introspect's own options from the ones meant for Codex.
    """
    import json  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    codex_bin = shutil.which("codex")
    if codex_bin is None:
        console.print(
            "[red]Error:[/red] `codex` CLI not found on PATH. "
            "Install Codex first: https://developers.openai.com/codex"
        )
        raise typer.Exit(code=1)

    server_proc = _ensure_server_running(host, port)
    mcp_url = f"http://{host}:{port}/mcp"
    try:
        exit_code = subprocess.call(  # noqa: S603
            [
                codex_bin,
                "--config",
                f"mcp_servers.introspect.url={json.dumps(mcp_url)}",
                "--config",
                f"developer_instructions={json.dumps(CODEX_DEVELOPER_INSTRUCTIONS)}",
                *ctx.args,
            ]
        )
    finally:
        _finish_connected_session(
            server_proc, host=host, port=port, keep_server=keep_server
        )
    raise typer.Exit(code=exit_code)


@app.command()
def refresh():
    """Rebuild the search corpus table and FTS index."""
    db_path = DEFAULT_DB_PATH
    jsonl_glob = DEFAULT_JSONL_GLOB
    codex_glob = DEFAULT_CODEX_GLOB
    try:
        ensure_materialized(db_path, jsonl_glob, codex_glob=codex_glob)
        conn = connect_writable(db_path)
    except DatabaseLockedError as e:
        _print_lock_error(e.db_path)
        raise typer.Exit(code=1) from None
    try:
        console.print("[dim]Rebuilding search index...[/dim]")
        build_search_corpus(conn)
        row = conn.execute("SELECT COUNT(*) FROM search_corpus").fetchone()
        count = row[0] if row else 0
        console.print(f"[green]Search index rebuilt with {count} entries.[/green]")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
