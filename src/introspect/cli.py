"""CLI interface for introspect."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from introspect.db import (
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_GLOB,
    get_read_connection,
    materialize_views,
)
from introspect.search import build_search_corpus, ensure_search_corpus, fts_search

SID_TRUNCATE = 12

app = typer.Typer(help="Explore Claude Code conversation logs.")
console = Console()


def _truncate_sid(val) -> str:
    s = str(val) if val else ""
    return s[:SID_TRUNCATE] + "..." if len(s) > SID_TRUNCATE else s


def _db(db_path: Path = DEFAULT_DB_PATH, jsonl_glob: str = DEFAULT_JSONL_GLOB):
    return get_read_connection(db_path, jsonl_glob)


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
    """,  # nosec B608
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
            f"SELECT * FROM raw_data {where} LIMIT {limit}",  # nosec B608
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

    conn.close()


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
        table.add_column("Snippet", max_width=80)
        table.add_column("Score", justify="right")

        for row in results:
            sid = _truncate_sid(row[0])
            ts = str(row[1])[:19] if row[1] else ""
            table.add_row(
                sid,
                ts,
                row[2] or "",
                row[3] or "",
                f"{row[4]:.4f}" if row[4] is not None else "",
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
    import duckdb  # noqa: PLC0415

    db_path = DEFAULT_DB_PATH
    jsonl_glob = DEFAULT_JSONL_GLOB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        if days > 0:
            console.print(f"[dim]Materializing last {days} days of data...[/dim]")
        else:
            console.print("[dim]Materializing all data (no day limit)...[/dim]")
        materialize_views(
            conn, jsonl_glob, days, resolve_projects=not no_resolve_projects
        )
        build_search_corpus(conn)
        row = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
        count = row[0] if row else 0
        console.print(f"[green]Materialized {count} messages into {db_path}[/green]")
    finally:
        conn.close()


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to listen on"),
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
    import os  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    os.environ["INTROSPECT_DAYS"] = str(days)
    if no_resolve_projects:
        os.environ["INTROSPECT_RESOLVE_PROJECTS"] = "0"
    console.print(f"[bold]Starting Introspect web UI on http://{host}:{port}[/bold]")
    console.print(f"[dim]MCP endpoint: http://{host}:{port}/mcp[/dim]")
    if days > 0:
        console.print(f"[dim]Loading last {days} days of data...[/dim]")
    else:
        console.print("[dim]Loading all data (no day limit)...[/dim]")
    uvicorn.run("introspect.api.main:app", host=host, port=port, log_level="info")


@app.command()
def mcp():
    """Run the MCP server (stdio transport) for Claude Code integration."""
    from introspect.mcp.server import create_mcp_server  # noqa: PLC0415

    create_mcp_server().run(transport="stdio")


@app.command()
def refresh():
    """Rebuild the search corpus table and FTS index."""
    conn = _db()
    try:
        console.print("[dim]Rebuilding search index...[/dim]")
        build_search_corpus(conn)
        count = conn.execute("SELECT COUNT(*) FROM search_corpus").fetchone()[0]
        console.print(f"[green]Search index rebuilt with {count} entries.[/green]")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
