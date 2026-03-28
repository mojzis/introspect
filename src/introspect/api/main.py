"""FastAPI application for introspect web UI."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from introspect.api.routes import router
from introspect.db import DEFAULT_DB_PATH, DEFAULT_JSONL_GLOB, materialize_views
from introspect.mcp.server import create_mcp_server
from introspect.search import build_search_corpus

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Materialize views on startup, then start MCP session manager."""
    db_path = Path(os.environ.get("INTROSPECT_DB_PATH", str(DEFAULT_DB_PATH)))
    jsonl_glob = os.environ.get("INTROSPECT_JSONL_GLOB", DEFAULT_JSONL_GLOB)
    days = int(os.environ.get("INTROSPECT_DAYS", "10"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        materialize_views(conn, jsonl_glob, days)
        build_search_corpus(conn)
    finally:
        conn.close()
    app.state.db_path = db_path

    # Create a fresh MCP server and replace the placeholder mount
    mcp_server = create_mcp_server()
    mcp_app = mcp_server.streamable_http_app()
    for route in app.routes:
        if getattr(route, "path", None) == "/mcp":
            route.app = mcp_app  # ty: ignore[unresolved-attribute]
            break
    # Rebuild middleware stack to pick up the new mount
    app.middleware_stack = app.build_middleware_stack()
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="Introspect", lifespan=lifespan)


@app.middleware("http")
async def db_middleware(request: Request, call_next):
    """Attach a DuckDB connection to each request, close after."""
    db_path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
    conn = duckdb.connect(str(db_path))
    request.state.conn = conn
    try:
        response = await call_next(request)
    finally:
        conn.close()
    return response


app.include_router(router)
# Placeholder mount — replaced with a fresh MCP app in lifespan
app.mount("/mcp", FastAPI())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty response for favicon requests."""
    return HTMLResponse(content="", status_code=204)
