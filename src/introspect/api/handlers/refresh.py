"""Manual refresh endpoint — pokes the background loop's ``asyncio.Event``.

The handler returns the ``_refresh_indicator.html`` fragment (HTMX swaps it
into ``#refresh-state``). If auto-refresh is disabled (no trigger on
``app.state``), the fragment renders a muted "auto-refresh off" label. The
rebuild still only happens when mtimes changed (or the window changed) —
the trigger just wakes the loop early.

The POST handler returns immediately after waking the loop — the user
keeps browsing on the (atomically swapped) old DB, and the polling fragment
flips the label once the rebuild lands.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.api.handlers._helpers import templates
from introspect.refresh import DEFAULT_WINDOW, VALID_WINDOWS

# Relative-time thresholds for :func:`format_relative`.
_JUST_NOW_SECONDS = 30
_MINUTE_SECONDS = 60
_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400

# Adaptive poll cadence for the in-progress fragment. As the rebuild runs
# longer, the chance it just finished grows — so we tighten the interval
# instead of polling at a single fixed rate. Tuples are
# ``(elapsed_seconds_threshold, delay_milliseconds)``, evaluated in order.
_POLL_SCHEDULE: tuple[tuple[float, int], ...] = (
    (2.0, 3000),
    (4.0, 2000),
    (6.0, 1000),
    (float("inf"), 500),
)
# How recently a completed refresh counts as "just finished" — we render the
# subtle ✓ flash for that window so the user notices completion.
_FLASH_WINDOW_SECONDS = 4.0


def _poll_delay_ms(started_at: datetime | None) -> int:
    """Pick the next ``hx-trigger`` delay based on how long the rebuild has run."""
    if started_at is None:
        return _POLL_SCHEDULE[0][1]
    elapsed = (datetime.now(UTC) - started_at).total_seconds()
    for threshold, delay in _POLL_SCHEDULE:
        if elapsed < threshold:
            return delay
    return _POLL_SCHEDULE[-1][1]


def _just_completed(in_progress: bool, last_refreshed_at: datetime | None) -> bool:
    """True when the indicator should show the brief ✓ flash."""
    if in_progress or last_refreshed_at is None:
        return False
    if last_refreshed_at.tzinfo is None:
        last_refreshed_at = last_refreshed_at.replace(tzinfo=UTC)
    elapsed = (datetime.now(UTC) - last_refreshed_at).total_seconds()
    return 0 <= elapsed < _FLASH_WINDOW_SECONDS


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


def _render(  # noqa: PLR0913
    request: Request,
    *,
    disabled: bool,
    in_progress: bool,
    last_refreshed_at: datetime | None,
    window: str,
    poll_delay_ms: int,
    notify: bool,
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
            "poll_delay_ms": poll_delay_ms,
            "notify": notify,
        },
    )


def _current_window(request: Request) -> str:
    """Read the current window token from ``app.state``, defaulting to ``"30"``."""
    value = getattr(request.app.state, "refresh_window", DEFAULT_WINDOW)
    return value or DEFAULT_WINDOW


def _current_indicator(request: Request, *, notify: bool = False) -> HTMLResponse:
    """Render the indicator from current ``app.state`` — no side effects.

    ``lifespan`` always sets ``refresh_trigger`` (to ``None`` when disabled),
    so direct attribute access is safe here — no ``getattr`` fallback needed.
    ``notify`` lets the caller opt into the brief ✓ flash when the indicator
    is about to render the just-finished state.
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
            poll_delay_ms=_POLL_SCHEDULE[0][1],
            notify=False,
        )
    in_progress = bool(state.refresh_in_progress)
    last_refreshed_at = state.last_refreshed_at
    started_at = getattr(state, "refresh_started_at", None)
    show_flash = notify and _just_completed(in_progress, last_refreshed_at)
    return _render(
        request,
        disabled=False,
        in_progress=in_progress,
        last_refreshed_at=last_refreshed_at,
        window=window,
        poll_delay_ms=_poll_delay_ms(started_at),
        notify=show_flash,
    )


async def refresh_status(request: Request) -> HTMLResponse:
    """GET /refresh-status — the fragment polls this while ``in_progress``.

    Does NOT set the trigger; just a read of current state so the "refreshing…"
    label can flip to "Last refreshed …" once the background rebuild finishes.
    Opts into the ✓ flash since this is the path that catches completion.
    """
    return _current_indicator(request, notify=True)


async def refresh_now(request: Request) -> HTMLResponse:
    """POST /refresh — wake the background loop and re-render the indicator.

    Accepts an optional ``window`` form field. Valid values
    (``"1" / "7" / "30" / "month"``) update ``app.state.refresh_window``;
    invalid or missing input keeps the existing sticky choice.

    Returns immediately after setting the trigger — the user keeps browsing
    on the existing DB while the rebuild runs in the background; the
    fragment self-polls ``/refresh-status`` until completion.

    Optimistically marks ``refresh_in_progress = True`` and seeds
    ``refresh_started_at`` before rendering so the fragment is guaranteed to
    emit polling attributes even though ``trigger.set()`` doesn't yield to
    the background loop. The loop overwrites both fields shortly after; if
    the rebuild short-circuits (mtimes unchanged AND window unchanged), it
    clears them in its ``finally``.
    """
    form = await request.form()
    submitted = form.get("window")
    if isinstance(submitted, str) and submitted in VALID_WINDOWS:
        request.app.state.refresh_window = submitted
    state = request.app.state
    trigger = state.refresh_trigger
    if trigger is not None:
        state.refresh_in_progress = True
        state.refresh_started_at = datetime.now(UTC)
        trigger.set()
    return _current_indicator(request)
