"""Route definitions for introspect web UI.

Each route delegates to a handler function in the handlers/ package.
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from introspect.api.handlers._helpers import (
    SESSIONS_PER_PAGE_DEFAULT,
    SESSIONS_SORT_DEFAULT,
)
from introspect.api.handlers.bash import bash as _bash
from introspect.api.handlers.cost_breakdown import (
    DEFAULT_BREAKDOWN as _COST_DEFAULT_BREAKDOWN,
)
from introspect.api.handlers.cost_breakdown import daily_panel as _cost_daily_panel
from introspect.api.handlers.cost_breakdown import hourly_panel as _cost_hourly_panel
from introspect.api.handlers.cost_overview import (
    cost_overview as _cost_overview,
)
from introspect.api.handlers.cost_overview import (
    cost_portfolio_panel as _cost_portfolio_panel,
)
from introspect.api.handlers.dashboard import dashboard as _dashboard
from introspect.api.handlers.mcps import mcps as _mcps
from introspect.api.handlers.raw import raw_data as _raw_data
from introspect.api.handlers.refresh import refresh_now as _refresh_now
from introspect.api.handlers.refresh import refresh_status as _refresh_status
from introspect.api.handlers.search import search as _search
from introspect.api.handlers.sessions import cost_bloat_panel as _cost_bloat_panel
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
    page: str = Query(""),
    page_size: str = Query(""),
    sort: str = Query(SESSIONS_SORT_DEFAULT),
    order: str = Query("desc"),
    model: str = Query("", alias="model"),
    project: str = Query("", alias="project"),
    branch: str = Query("", alias="branch"),
    command: str = Query("", alias="command"),
    q: str = Query("", alias="q"),
):
    return await _sessions(
        request,
        max(1, int(page)) if page.strip().isdigit() else 1,
        int(page_size) if page_size.strip().isdigit() else SESSIONS_PER_PAGE_DEFAULT,
        sort,
        order,
        model,
        project,
        branch,
        command,
        q,
    )


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(
    request: Request,
    session_id: str,
    tab: str = Query("messages"),
):
    return await _session_detail(request, session_id, tab)


@router.get("/sessions/{session_id}/cost/bloat", response_class=HTMLResponse)
async def session_cost_bloat(
    request: Request,
    session_id: str,
    from_uuid: str | None = Query(default=None),
    to_uuid: str | None = Query(default=None),
):
    return await _cost_bloat_panel(request, session_id, from_uuid, to_uuid)


@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query("", alias="q"),
    page: int = Query(1, ge=1),
):
    return await _search(request, q, page)


@router.get("/tools", response_class=HTMLResponse)
async def tools(  # noqa: PLR0913
    request: Request,
    failed: bool = Query(False),
    name: str = Query("", alias="name"),
    session: str = Query("", alias="session"),
    project: str = Query("", alias="project"),
    q: str = Query("", alias="q"),
    page: int = Query(1, ge=1),
):
    return await _tools(request, failed, name, session, project, q, page)


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
    page: int = Query(1, ge=1),
):
    return await _mcps(request, server, command, failed, page)


@router.get("/bash", response_class=HTMLResponse)
async def bash(  # noqa: PLR0913
    request: Request,
    prefix: str = Query("", alias="prefix"),
    session: str = Query("", alias="session"),
    project: str = Query("", alias="project"),
    q: str = Query("", alias="q"),
    failed: bool = Query(False),
    page: int = Query(1, ge=1),
):
    return await _bash(request, prefix, session, project, q, failed, page)


@router.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    return await _stats(request)


@router.get("/cost-overview", response_class=HTMLResponse)
async def cost_overview(request: Request):
    return await _cost_overview(request)


@router.get("/cost-overview/breakdown", response_class=HTMLResponse)
async def cost_daily_panel(
    request: Request,
    breakdown: str = Query(_COST_DEFAULT_BREAKDOWN),
):
    return await _cost_daily_panel(request, breakdown)


@router.get("/cost-overview/breakdown/{day}", response_class=HTMLResponse)
async def cost_hourly_panel(
    request: Request,
    day: str,
    breakdown: str = Query(_COST_DEFAULT_BREAKDOWN),
):
    return await _cost_hourly_panel(request, day, breakdown)


@router.get("/cost-overview/portfolio", response_class=HTMLResponse)
async def cost_portfolio_panel(
    request: Request,
    day: str | None = Query(default=None),
    hour: str | None = Query(default=None),
):
    return await _cost_portfolio_panel(request, day, hour)


@router.post("/refresh", response_class=HTMLResponse)
async def refresh(request: Request):
    return await _refresh_now(request)


@router.get("/refresh-status", response_class=HTMLResponse)
async def refresh_status(request: Request):
    return await _refresh_status(request)
