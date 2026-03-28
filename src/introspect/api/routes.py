"""Route definitions for introspect web UI.

Each route delegates to a handler function in the handlers/ package.
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from introspect.api.handlers._helpers import (
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_DEFAULT,
)
from introspect.api.handlers.dashboard import dashboard as _dashboard
from introspect.api.handlers.mcps import mcps as _mcps
from introspect.api.handlers.raw import raw_data as _raw_data
from introspect.api.handlers.search import search as _search
from introspect.api.handlers.sessions import session_detail as _session_detail
from introspect.api.handlers.sessions import sessions as _sessions
from introspect.api.handlers.stats import stats as _stats
from introspect.api.handlers.tools import tools as _tools

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return await _dashboard(request)


@router.get("/sessions", response_class=HTMLResponse)
async def sessions(  # noqa: PLR0913
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(SESSIONS_PER_PAGE_DEFAULT, ge=1, le=500),
    sort: str = Query(SESSIONS_SORT_DEFAULT),
    order: str = Query("desc"),
    model: str = Query("", alias="model"),
    project: str = Query("", alias="project"),
    branch: str = Query("", alias="branch"),
):
    return await _sessions(
        request, page, page_size, sort, order, model, project, branch
    )


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    return await _session_detail(request, session_id)


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Query("", alias="q")):
    return await _search(request, q)


@router.get("/tools", response_class=HTMLResponse)
async def tools(
    request: Request,
    failed: bool = Query(False),
    name: str = Query("", alias="name"),
):
    return await _tools(request, failed, name)


@router.get("/raw", response_class=HTMLResponse)
async def raw_data(
    request: Request,
    page: int = Query(1, ge=1),
    session: str = Query("", alias="session"),
    record_type: str = Query("", alias="type"),
):
    return await _raw_data(request, page, session, record_type)


@router.get("/mcps", response_class=HTMLResponse)
async def mcps(
    request: Request,
    server: str = Query("", alias="server"),
    command: str = Query("", alias="command"),
    failed: bool = Query(False),
):
    return await _mcps(request, server, command, failed)


@router.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    return await _stats(request)
