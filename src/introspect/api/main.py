"""FastAPI application for introspect web UI."""

import asyncio
import contextlib
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from introspect.api.routes import router
from introspect.db import (
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_GLOB,
    connect_writable,
    materialize_views,
)
from introspect.mcp.refresh_bridge import set_state as set_mcp_refresh_state
from introspect.mcp.server import create_mcp_server
from introspect.refresh import refresh_loop
from introspect.search import build_search_corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Materialize views on startup, then start MCP session manager."""
    db_path = Path(os.environ.get("INTROSPECT_DB_PATH", str(DEFAULT_DB_PATH)))
    jsonl_glob = os.environ.get("INTROSPECT_JSONL_GLOB", DEFAULT_JSONL_GLOB)
    days = int(os.environ.get("INTROSPECT_DAYS", "10"))
    interval = float(os.environ.get("INTROSPECT_REFRESH_INTERVAL_SECONDS", "600"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    resolve_projects = os.environ.get("INTROSPECT_RESOLVE_PROJECTS", "1") != "0"
    try:
        materialize_views(conn, jsonl_glob, days, resolve_projects=resolve_projects)
        build_search_corpus(conn)
    finally:
        conn.close()

    # Open a shared read-only connection for request handling
    app.state.read_conn = duckdb.connect(str(db_path), read_only=True)
    app.state.last_refreshed_at = datetime.now(UTC)
    app.state.refresh_in_progress = False
    # Always set the attribute (None when disabled) so callers can check
    # ``state.refresh_trigger is None`` instead of falling back to ``getattr``.
    app.state.refresh_trigger = None

    refresh_task: asyncio.Task[None] | None = None
    if interval > 0:
        app.state.refresh_trigger = asyncio.Event()
        refresh_task = asyncio.create_task(
            refresh_loop(
                app,
                db_path,
                jsonl_glob,
                days,
                resolve_projects,
                interval,
                trigger=app.state.refresh_trigger,
            )
        )

    # Create a fresh MCP server and replace the placeholder mount
    mcp_server = create_mcp_server()
    mcp_app = mcp_server.streamable_http_app()
    for route in app.routes:
        if getattr(route, "path", None) == "/mcp":
            route.app = mcp_app  # ty: ignore[unresolved-attribute]
            break
    # Rebuild middleware stack to pick up the new mount
    app.middleware_stack = app.build_middleware_stack()
    set_mcp_refresh_state(app.state)
    async with mcp_server.session_manager.run():
        try:
            yield
        finally:
            set_mcp_refresh_state(None)
            if refresh_task is not None:
                refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresh_task
            app.state.read_conn.close()


app = FastAPI(title="Introspect", lifespan=lifespan)


@app.middleware("http")
async def db_middleware(request: Request, call_next):
    """Attach a DuckDB cursor to each request from the shared connection."""
    read_conn = getattr(request.app.state, "read_conn", None)
    if read_conn is not None:
        request.state.conn = read_conn.cursor()
    else:
        # Fallback for tests or when lifespan hasn't run
        db_path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
        request.state.conn = duckdb.connect(str(db_path))
    try:
        response = await call_next(request)
    finally:
        request.state.conn.close()
    return response


app.include_router(router)
# Placeholder mount — replaced with a fresh MCP app in lifespan
app.mount("/mcp", FastAPI())


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    """Chrome DevTools automatic workspace discovery."""
    workspace_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    workspace_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, workspace_root))
    return JSONResponse({"workspace": {"root": workspace_root, "uuid": workspace_uuid}})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an SVG favicon."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<text y=".9em" font-size="90">&#128269;</text>'
        "</svg>"
    )
    return HTMLResponse(content=svg, media_type="image/svg+xml")
