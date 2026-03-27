"""FastAPI application for introspect web UI."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from introspect.api.routes import router
from introspect.db import get_connection

app = FastAPI(title="Introspect")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.middleware("http")
async def db_middleware(request: Request, call_next):
    """Attach a DuckDB connection to each request, close after."""
    conn = get_connection()
    request.state.conn = conn
    try:
        response = await call_next(request)
    finally:
        conn.close()
    return response


app.include_router(router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty response for favicon requests."""
    return HTMLResponse(content="", status_code=204)
