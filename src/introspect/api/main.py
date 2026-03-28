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
from introspect.search import build_search_corpus

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Materialize views on startup for fast querying."""
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an SVG favicon."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<text y=".9em" font-size="90">&#128269;</text>'
        "</svg>"
    )
    return HTMLResponse(content=svg, media_type="image/svg+xml")
