"""One project-wide policy for surfacing request failures.

HTMX's documented default is to *not* swap a 4xx/5xx response and to fire an
event nobody listens to, so a failed fragment request looks to the user like
a click that did nothing. This module is the server half of the fix; the
client half lives in ``templates/base.html`` (the ``htmx-config`` meta, the
``#errors`` container, and the ``htmx:responseError`` / ``htmx:sendError``
listeners).

The policy, in three rules:

* **The status code stays honest.** A failure is never dressed up as a 200 —
  tests, logs and monitoring all read the status line.
* **An HTMX request gets an HTML fragment**, retargeted at ``#errors`` and
  appended there, so a failed refresh never wipes the data the user was
  looking at. A normal request gets the same information as a full page.
* **Machine endpoints keep their machine representation.** ``/api/*`` and
  ``/mcp`` answer notebooks and MCP clients, not browsers; they fall through
  to FastAPI's JSON defaults.

Tracebacks reach the browser only when ``INTROSPECT_DEBUG`` is set. They are
always written to the server log regardless.
"""

from __future__ import annotations

import logging
import os
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from introspect.api.handlers._helpers import is_htmx, templates

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request
    from fastapi.responses import Response

log = logging.getLogger(__name__)

# The fixed container in ``base.html`` that every error fragment lands in.
# Kept here (not spelled out per element) so the policy has exactly one
# definition; ``tests/routes/test_error_handling.py`` asserts the template
# and these constants agree.
ERROR_CONTAINER_ID = "errors"
ERROR_TARGET = f"#{ERROR_CONTAINER_ID}"

# Append rather than replace: two failures in a row should both be visible,
# and neither should destroy the element the request was originally aimed at.
ERROR_SWAP = "beforeend"

# Set on responses whose body already *is* rendered error markup. The
# client-side ``htmx:responseError`` listener is only a backstop for
# responses we did not format (a proxy 502, an HTML-less body); this header
# is how it tells the two apart instead of double-reporting.
ERROR_RENDERED_HEADER = "X-Introspect-Error-Rendered"

# Endpoints whose callers are programs, not browsers. ``/api/*`` is the local
# SQL API (documented as returning ``{"error": ...}``), ``/mcp`` is the MCP
# transport, ``/static`` and ``/.well-known`` serve assets. Handing any of
# them a toast fragment would be a regression, so they keep the framework
# defaults.
#
# The MCP transport is mounted at exactly ``/mcp``, so it is matched as a whole
# path and not as a prefix: ``/mcps`` is the MCP *statistics page*, a normal
# HTML route, and a bare ``startswith("/mcp")`` would silently exempt it from
# the entire policy.
MACHINE_PATHS = frozenset({"/mcp"})
MACHINE_PATH_PREFIXES = ("/api/", "/mcp/", "/static/", "/.well-known/")

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# Status codes at or above this are the server's fault, however they were
# raised, and are logged accordingly.
HTTP_SERVER_ERROR = 500
HTTP_UNPROCESSABLE_ENTITY = 422

_GENERIC_SERVER_MESSAGE = (
    "Something went wrong while building this view. The server log has the details."
)


def debug_enabled(request: Request) -> bool:
    """Whether tracebacks may be rendered into the response body.

    Two conditions, both required. ``INTROSPECT_DEBUG`` must be explicitly
    truthy — read per call, not cached at import, so a test (or ``devserve``,
    which sets it) can flip it without reloading the module. And the server
    must be bound to loopback: ``serve --host 0.0.0.0`` is supported, and a
    variable a user exported once for ``devserve`` must not start serving file
    paths and SQL to the LAN. Same gate, same reasoning as the SQL API.
    """
    if os.environ.get("INTROSPECT_DEBUG", "").strip().lower() not in _TRUE_VALUES:
        return False
    # Local import: ``api.main`` imports this module to register the handlers.
    from introspect.api.main import host_allowlist_applies  # noqa: PLC0415

    return host_allowlist_applies(getattr(request.app.state, "bind_host", ""))


def _wants_html_error(request: Request) -> bool:
    """Whether this request should get an HTML error rather than JSON."""
    path = request.url.path
    return path not in MACHINE_PATHS and not path.startswith(MACHINE_PATH_PREFIXES)


def _reason(status_code: int) -> str:
    """Human phrase for a status code (``"Not Found"``), or a bare fallback."""
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def _new_request_id() -> str:
    """Short correlation id, shared by the log line and the rendered error.

    We have no request-id middleware and don't need one: an id only earns its
    keep on a failure, which is exactly where this mints it. It lets a user
    quote "request 3f2a1c9d" and the maintainer grep the log for it.
    """
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class ErrorView:
    """One failure, in the terms both framings render it in.

    Bundled rather than passed loose so the fragment and the full page can
    never be handed different facts about the same error.
    """

    status_code: int
    message: str
    request_id: str
    #: Rendered only under :func:`debug_enabled`; ``None`` everywhere else.
    traceback_text: str | None = None

    def context(self) -> dict[str, object]:
        """Template context for ``error.html`` / ``_error.html``."""
        return {
            "status_code": self.status_code,
            "reason": _reason(self.status_code),
            "message": self.message,
            "request_id": self.request_id,
            "traceback": self.traceback_text,
        }


def _render_error(
    request: Request,
    view: ErrorView,
    headers: Mapping[str, str] | None = None,
) -> Response:
    """Render one failure, as a fragment for HTMX and a full page otherwise.

    The HTMX branch reuses :func:`introspect.api.handlers._helpers.is_htmx` —
    the same header test ``parent()`` uses to choose ``base.html`` vs
    ``partial.html`` — so the error path can't drift from the success path.

    The fragment carries ``HX-Retarget`` / ``HX-Reswap`` so it lands in the
    error container instead of the element the request came from, and
    ``HX-Push-Url: false`` so a failed navigation doesn't leave the address
    bar claiming a page that never rendered.
    """
    context = view.context()
    if not is_htmx(request):
        return templates.TemplateResponse(
            request,
            "error.html",
            context,
            status_code=view.status_code,
            headers=headers,
        )
    response = templates.TemplateResponse(
        request,
        "_error.html",
        context,
        status_code=view.status_code,
        headers=headers,
    )
    response.headers["HX-Retarget"] = ERROR_TARGET
    response.headers["HX-Reswap"] = ERROR_SWAP
    response.headers["HX-Push-Url"] = "false"
    response.headers[ERROR_RENDERED_HEADER] = "1"
    return response


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    """Deliberate 4xx/5xx raised by a handler (``raise HTTPException(...)``).

    A deliberate 4xx is an expected outcome — a malformed ``?day=``, a uuid
    that isn't in the session — so it is logged at info, and its ``detail`` is
    shown as-is: handlers write it for a human. A deliberate *5xx* is still a
    server fault and is logged as one.
    """
    if not _wants_html_error(request):
        return await default_http_exception_handler(request, exc)
    request_id = _new_request_id()
    log.log(
        logging.WARNING if exc.status_code >= HTTP_SERVER_ERROR else logging.INFO,
        "HTTP %s on %s %s [%s]: %s",
        exc.status_code,
        request.method,
        request.url.path,
        request_id,
        exc.detail,
    )
    detail = exc.detail if isinstance(exc.detail, str) else _reason(exc.status_code)
    return _render_error(
        request,
        ErrorView(exc.status_code, detail, request_id),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Request-shape errors from FastAPI's own parsing (422).

    The raw pydantic error list is noise in a toast, so the browser gets a
    one-liner and the log gets the structure.
    """
    if not _wants_html_error(request):
        return await default_validation_exception_handler(request, exc)
    request_id = _new_request_id()
    log.info(
        "Validation error on %s %s [%s]: %s",
        request.method,
        request.url.path,
        request_id,
        exc.errors(),
    )
    return _render_error(
        request,
        ErrorView(
            HTTP_UNPROCESSABLE_ENTITY,
            "The request was not in a form this endpoint accepts.",
            request_id,
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Anything that escaped a handler: log the traceback, then show a 500.

    The log write happens first and unconditionally — the rendered fragment
    is a courtesy to the user, the traceback in the log is the thing that
    gets the bug fixed. The browser only ever sees it under
    ``INTROSPECT_DEBUG``.

    Note this runs in Starlette's ``ServerErrorMiddleware``, *outside*
    ``db_middleware``, so ``request.state.conn`` is already closed. Nothing
    here may touch the database.
    """
    request_id = _new_request_id()
    log.error(
        "Unhandled exception on %s %s [%s]",
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
    )
    if not _wants_html_error(request):
        # ``{"error": ...}`` is the shape docs/usage/sql-api.md promises and
        # notebook clients parse; a plain-text 500 would break it.
        return JSONResponse({"error": "Internal Server Error"}, status_code=500)
    traceback_text = (
        "".join(traceback.format_exception(exc)) if debug_enabled(request) else None
    )
    return _render_error(
        request,
        ErrorView(
            HTTP_SERVER_ERROR,
            _GENERIC_SERVER_MESSAGE,
            request_id,
            traceback_text=traceback_text,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the three handlers that implement the policy.

    ``StarletteHTTPException`` (not ``fastapi.HTTPException``) is the class
    Starlette dispatches on, and the one ``StaticFiles`` and routing raise
    too, so registering it covers 404s that never reach a handler.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # ty: ignore[invalid-argument-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # ty: ignore[invalid-argument-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
