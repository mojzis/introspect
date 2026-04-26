"""Manual refresh endpoint — pokes the background loop's ``asyncio.Event``.

The handler returns the ``_refresh_indicator.html`` fragment (HTMX swaps it
into ``#refresh-state``). If auto-refresh is disabled (no trigger on
``app.state``), the fragment renders a muted "auto-refresh off" label. The
rebuild still only happens when mtimes changed (or the window changed) —
the trigger just wakes the loop early.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.api.handlers._helpers import templates
from introspect.refresh import (
    DEFAULT_WINDOW,
    VALID_WINDOWS,
    wait_for_refresh,
    window_to_days,  # noqa: F401  # re-exported for back-compat / external use
)

# Poll budget for observing the background loop after setting the trigger.
# Kept short so the HTTP response stays snappy; the indicator may show
# "refreshing…" if we give up early and the template polls /refresh-status.
_WAIT_FOR_FINISH_SECONDS = 3.0

# Relative-time thresholds for :func:`format_relative`.
_JUST_NOW_SECONDS = 30
_MINUTE_SECONDS = 60
_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400


def format_relative(dt: datetime | None) -> str:
    """Format ``dt`` relative to ``now()`` for the top-bar indicator.

    Always a duration — the label never includes a date (negative ``delta``
    from clock skew collapses into the "just now" branch).

    * ``None``  -> ``"never"``
    * < 30 s    -> ``"just now"``
    * < 1 h     -> ``"Nm ago"``
    * < 24 h    -> ``"Nh ago"``
    * otherwise -> ``"Nd ago"``
    """
    if dt is None:
        return "never"
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = (now - dt).total_seconds()
    if delta < _JUST_NOW_SECONDS:
        return "just now"
    if delta < _HOUR_SECONDS:
        return f"{int(delta // _MINUTE_SECONDS)}m ago"
    if delta < _DAY_SECONDS:
        return f"{int(delta // _HOUR_SECONDS)}h ago"
    return f"{int(delta // _DAY_SECONDS)}d ago"


# Expose to Jinja so base.html's {% include %} path renders the same relative
# label as the HTMX fragment (previously it fell back to strftime).
templates.env.globals["format_relative"] = format_relative  # ty: ignore[invalid-assignment]


def _render(
    request: Request,
    *,
    disabled: bool,
    in_progress: bool,
    last_refreshed_at: datetime | None,
    window: str,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "_refresh_indicator.html",
        {
            "disabled": disabled,
            "in_progress": in_progress,
            "last_refreshed_at": last_refreshed_at,
            "relative": format_relative(last_refreshed_at),
            "window": window,
        },
    )


def _current_window(request: Request) -> str:
    """Read the current window token from ``app.state``, defaulting to ``"30"``."""
    value = getattr(request.app.state, "refresh_window", DEFAULT_WINDOW)
    return value or DEFAULT_WINDOW


def _current_indicator(request: Request) -> HTMLResponse:
    """Render the indicator from current ``app.state`` — no side effects.

    ``lifespan`` always sets ``refresh_trigger`` (to ``None`` when disabled),
    so direct attribute access is safe here — no ``getattr`` fallback needed.
    """
    state = request.app.state
    trigger: asyncio.Event | None = state.refresh_trigger
    window = _current_window(request)
    if trigger is None:
        return _render(
            request,
            disabled=True,
            in_progress=False,
            last_refreshed_at=None,
            window=window,
        )
    return _render(
        request,
        disabled=False,
        in_progress=bool(state.refresh_in_progress),
        last_refreshed_at=state.last_refreshed_at,
        window=window,
    )


async def refresh_status(request: Request) -> HTMLResponse:
    """GET /refresh-status — the fragment polls this while ``in_progress``.

    Does NOT set the trigger; just a read of current state so the "refreshing…"
    label can flip to "Last refreshed …" once the background rebuild finishes.
    """
    return _current_indicator(request)


async def refresh_now(request: Request) -> HTMLResponse:
    """POST /refresh — wake the background loop and re-render the indicator.

    Accepts an optional ``window`` form field. Valid values
    (``"1" / "7" / "30" / "month"``) update ``app.state.refresh_window``;
    invalid or missing input keeps the existing sticky choice.
    """
    form = await request.form()
    submitted = form.get("window")
    if isinstance(submitted, str) and submitted in VALID_WINDOWS:
        request.app.state.refresh_window = submitted
    # The handler ignores the helper's outcome; it just re-renders the
    # indicator from current state. If the rebuild didn't finish inside the
    # HTTP budget, the template polls ``/refresh-status`` until it flips.
    await wait_for_refresh(request.app.state, finish_timeout=_WAIT_FOR_FINISH_SECONDS)
    return _current_indicator(request)
