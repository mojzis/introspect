"""FastAPI application for introspect web UI."""

import asyncio
import contextlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from introspect.api.routes import router
from introspect.db import (
    DEFAULT_CODEX_GLOB,
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_GLOB,
    connect_writable,
    materialize_views,
)
from introspect.mcp.refresh_bridge import set_state as set_mcp_refresh_state
from introspect.mcp.server import create_mcp_server
from introspect.refresh import (
    DEFAULT_WINDOW,
    VALID_WINDOWS,
    refresh_loop,
    window_to_days,
)
from introspect.search import build_search_corpus
from introspect.sql_query import is_loopback_host

log = logging.getLogger(__name__)


def _configure_sql_api(app: FastAPI) -> None:
    """Decide whether to expose the local-only SQL API and record it on state.

    Exposed only when the server is bound to a loopback address (the CLI
    passes the bind host through ``INTROSPECT_HOST``); an explicit
    ``INTROSPECT_SQL_API=off`` disables it even on loopback.

    Fails closed: an unset ``INTROSPECT_HOST`` (e.g. launching the app under
    bare ``uvicorn`` instead of via ``introspy serve``) counts as "not known
    to be loopback" and leaves the API disabled — the bind host must be
    explicitly loopback to expose it.
    """
    bind_host = os.environ.get("INTROSPECT_HOST", "")
    api_toggle = os.environ.get("INTROSPECT_SQL_API", "").strip().lower()
    app.state.bind_host = bind_host
    app.state.sql_api_enabled = api_toggle != "off" and is_loopback_host(bind_host)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Materialize views on startup, then start MCP session manager."""
    db_path = Path(os.environ.get("INTROSPECT_DB_PATH", str(DEFAULT_DB_PATH)))
    jsonl_glob = os.environ.get("INTROSPECT_JSONL_GLOB", DEFAULT_JSONL_GLOB)
    codex_glob = os.environ.get("INTROSPECT_CODEX_GLOB", DEFAULT_CODEX_GLOB)
    interval = float(os.environ.get("INTROSPECT_REFRESH_INTERVAL_SECONDS", "600"))
    refresh_window = os.environ.get("INTROSPECT_REFRESH_WINDOW", DEFAULT_WINDOW)
    if refresh_window not in VALID_WINDOWS:
        log.warning(
            "Invalid INTROSPECT_REFRESH_WINDOW=%r; falling back to %s",
            refresh_window,
            DEFAULT_WINDOW,
        )
        refresh_window = DEFAULT_WINDOW
    # ``INTROSPECT_DAYS`` is the explicit override (used heavily in tests with
    # ``"0"`` for "no limit"). When it isn't set, we resolve from the picker
    # window so the initial materialize matches what the UI advertises and
    # the refresh loop's first tick doesn't rebuild a second time.
    days_env = os.environ.get("INTROSPECT_DAYS")
    days = int(days_env) if days_env is not None else window_to_days(refresh_window)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_writable(db_path)
    resolve_projects = os.environ.get("INTROSPECT_RESOLVE_PROJECTS", "1") != "0"
    try:
        materialize_views(
            conn,
            jsonl_glob,
            days,
            resolve_projects=resolve_projects,
            codex_glob=codex_glob,
        )
        build_search_corpus(conn)
    finally:
        conn.close()

    # Persist config on app.state so middleware can open per-request connections
    # (avoids the swap-during-query 500 caused by a shared read connection).
    app.state.db_path = db_path
    app.state.days = days
    app.state.refresh_window = refresh_window
    app.state.last_built_days = days
    app.state.last_refreshed_at = datetime.now(UTC)
    app.state.refresh_in_progress = False
    app.state.refresh_started_at = None
    # Always set the attribute (None when disabled) so callers can check
    # ``state.refresh_trigger is None`` instead of falling back to ``getattr``.
    app.state.refresh_trigger = None

    _configure_sql_api(app)

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
                codex_glob=codex_glob,
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


app = FastAPI(title="Introspect", lifespan=lifespan)


@app.middleware("http")
async def db_middleware(request: Request, call_next):
    """Open a fresh read-only DuckDB connection per request.

    Per-request connections decouple in-flight queries from the background
    refresh: ``_swap_in`` is now just ``os.replace`` and never closes a
    connection out from under a live cursor. ``contextlib.closing`` makes
    the ownership structural — if anything below the ``with`` raises, the
    connection still closes deterministically.
    """
    db_path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
    with contextlib.closing(duckdb.connect(str(db_path), read_only=True)) as conn:
        request.state.conn = conn
        return await call_next(request)


app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/manifest.webmanifest", include_in_schema=False)
async def manifest() -> FileResponse:
    """Serve the web app manifest at the site root.

    Served from a dedicated route (rather than under ``/static/``) so the
    ``application/manifest+json`` media type is explicit — ``.webmanifest`` is
    absent from Python's default ``mimetypes`` table — and so ``scope`` /
    ``start_url`` resolve against the root.
    """
    return FileResponse(
        STATIC_DIR / "manifest.webmanifest",
        media_type="application/manifest+json",
    )


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
