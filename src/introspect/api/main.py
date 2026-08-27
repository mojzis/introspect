"""FastAPI application for introspect web UI."""

import asyncio
import contextlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles

from introspect.api.errors import register_exception_handlers
from introspect.api.routes import router
from introspect.db import (
    DEFAULT_CODEX_GLOB,
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_GLOB,
    connect_read_hardened,
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

# ``Host`` values a loopback-bound server answers to, port already stripped.
# Enforced by ``host_guard`` — but only when the bind is loopback; see
# ``host_allowlist_applies``.
ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")


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
    mcp_server = create_mcp_server(app.state.bind_host)
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

# One place decides what a failed request looks like — an HTML fragment for
# HTMX, a full page for a browser, JSON for the machine endpoints — always
# with the real status code. See introspect.api.errors.
#
# ``host_guard`` and ``local_api_guard`` below keep their own representation
# because they *return* a response rather than raising, so nothing routes them
# through these handlers. Note the layering if that ever changes: the
# ``Exception`` handler lives in Starlette's outermost ``ServerErrorMiddleware``
# (a crash inside those middlewares does reach it), while the
# ``HTTPException`` handler lives in ``ExceptionMiddleware``, inside them.
register_exception_handlers(app)


@app.middleware("http")
async def db_middleware(request: Request, call_next):
    """Open a fresh hardened read-only DuckDB connection per request.

    Per-request connections decouple in-flight queries from the background
    refresh: ``_swap_in`` is now just ``os.replace`` and never closes a
    connection out from under a live cursor. ``contextlib.closing`` makes
    the ownership structural — if anything below the ``with`` raises, the
    connection still closes deterministically.

    ``connect_read_hardened`` is not optional here: DuckDB refuses a second
    connection to a file whose instance was opened with a different config,
    so a bare ``duckdb.connect`` in this middleware would break the MCP
    ``run_sql`` tool running in the same process (and vice versa).
    """
    db_path = getattr(request.app.state, "db_path", DEFAULT_DB_PATH)
    with contextlib.closing(connect_read_hardened(db_path)) as conn:
        request.state.conn = conn
        return await call_next(request)


# Paths that a hostile web page could usefully reach: they read the user's
# conversation logs, or (for /mcp) drive a tool-calling session.
_ORIGIN_GUARDED_PREFIXES = ("/api/query", "/api/schema", "/mcp")

# Custom header required on POST /api/query. Its only job is to force a CORS
# preflight — a cross-origin `fetch` cannot send an unrecognized header
# without one, and with no CORS middleware registered the preflight has no
# Access-Control-Allow-* response to satisfy it, so the request never
# happens. Documented in docs/usage/sql-api.md; notebooks just set it.
CLIENT_HEADER = "X-Introspect-Client"


@app.middleware("http")
async def local_api_guard(request: Request, call_next):
    """Reject cross-origin and drive-by requests to the data endpoints.

    Binding to loopback stops *remote* clients; it does nothing about a page
    the user has open in a browser, which can reach ``127.0.0.1`` directly or
    via DNS rebinding. Two cheap checks close that:

    * an ``Origin`` header whose host is not loopback is refused outright.
      Requests without ``Origin`` (curl, httpx, notebooks) are unaffected —
      browsers always send it on cross-origin requests.
    * ``POST /api/query`` additionally requires ``X-Introspect-Client``,
      which a cross-origin ``fetch`` cannot set without a successful CORS
      preflight.

    ``Host`` is handled separately by :func:`host_guard`.
    """
    path = request.url.path
    if path.startswith(_ORIGIN_GUARDED_PREFIXES):
        origin = request.headers.get("origin")
        if origin and not _is_loopback_origin(origin):
            return JSONResponse(
                {"error": "Cross-origin requests are not allowed."},
                status_code=403,
            )
        if (
            request.method == "POST"
            and path == "/api/query"
            and CLIENT_HEADER not in request.headers
        ):
            return JSONResponse(
                {"error": f"Missing required {CLIENT_HEADER} header."},
                status_code=403,
            )
    return await call_next(request)


def _is_loopback_origin(origin: str) -> bool:
    """True when an ``Origin`` header names a loopback host.

    ``Origin: null`` (sandboxed iframes, some file:// contexts) is not
    loopback and is refused along with everything else.
    """
    host = urlsplit(origin).hostname
    return bool(host) and is_loopback_host(host)


def host_allowlist_applies(bind_host: str) -> bool:
    """Should the loopback ``Host`` allowlist be enforced for this bind?

    Yes for a loopback bind — that is the DNS-rebinding case, and no
    legitimate request can carry any other ``Host``.

    No for a deliberate non-loopback bind. ``serve --host 0.0.0.0`` is a
    supported way to reach the UI from another machine, and we cannot know
    which hostnames or LAN addresses that user will type. Enforcing a
    loopback allowlist there would 400 every page, not merely narrow the SQL
    API — which is already disabled on such a bind, along with everything
    else that reads logs (see ``_configure_sql_api``).

    An unset ``INTROSPECT_HOST`` (bare ``uvicorn``, whose own default is
    ``127.0.0.1``) fails closed *into* enforcement, matching how the SQL API
    gate fails closed in the other direction. Set ``INTROSPECT_HOST`` to the
    real bind address if you launch the app yourself on a routable one.
    """
    return not bind_host or is_loopback_host(bind_host)


@app.middleware("http")
async def host_guard(request: Request, call_next):
    """Refuse a request whose ``Host`` is not one this server answers to.

    A DNS-rebinding attack works by pointing a hostname the browser already
    trusts at 127.0.0.1; the request then arrives here carrying that
    attacker-controlled hostname. Matching ``Host`` against a loopback
    allowlist rejects it before it reaches a route — or a database
    connection.

    Written as a middleware rather than Starlette's ``TrustedHostMiddleware``
    because the allowlist has to depend on the bind host, which is only known
    once the lifespan has read the environment; ``TrustedHostMiddleware``
    fixes its ``allowed_hosts`` at construction. The comparison is otherwise
    the same: the port is stripped before matching.
    """
    if (
        host_allowlist_applies(getattr(request.app.state, "bind_host", ""))
        and _host_without_port(request.headers.get("host", "")) not in ALLOWED_HOSTS
    ):
        return PlainTextResponse("Invalid host header", status_code=400)
    return await call_next(request)


def _host_without_port(header: str) -> str:
    """Lowercase the ``Host`` header and drop its port.

    IPv6 literals are bracketed (``[::1]:8347``) and full of colons, so a
    plain ``split(":")`` yields ``"["``. The brackets are kept — that is how
    the host appears in a URL authority, and how ``ALLOWED_HOSTS`` spells it.
    """
    host = header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":")[0]


# NOTE: CORSMiddleware must never be added to this app — certainly not with
# ``allow_origins=["*"]``. The SQL API has no authentication; permissive CORS
# would let any web page the user visits read their entire conversation
# history out of localhost. ``tests/e2e/test_sql_hardening.py`` asserts it is
# absent.

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
