"""CLI interface for introspect."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from introspect.db import DEFAULT_DB_PATH, DEFAULT_JSONL_GLOB, get_connection

SID_TRUNCATE = 12

app = typer.Typer(help="Explore Claude Code conversation logs.")
console = Console()


def _truncate_sid(val) -> str:
    s = str(val) if val else ""
    return s[:SID_TRUNCATE] + "..." if len(s) > SID_TRUNCATE else s


def _db(db_path: Path = DEFAULT_DB_PATH, jsonl_glob: str = DEFAULT_JSONL_GLOB):
    return get_connection(db_path, jsonl_glob)


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
    """,
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


if __name__ == "__main__":
    app()
