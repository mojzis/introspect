"""Module-level handle so MCP tools can reach the FastAPI app state.

MCP tools are stateless functions invoked by FastMCP — they don't have
direct access to ``request.app.state``. The FastAPI lifespan registers
its state object here on startup so the ``refresh_data`` tool can read
``refresh_trigger`` / ``refresh_in_progress`` / ``last_refreshed_at``.
``set_state(None)`` is called on shutdown to release the reference.

Single-app assumption: this module assumes one FastAPI app per process.
``set_state`` raises if a second non-``None`` registration arrives without
an intervening clear, surfacing accidental multi-app setups.
"""

from __future__ import annotations

from introspect.refresh import RefreshState

_DOUBLE_REGISTRATION_MSG = (
    "refresh_bridge already has a registered state; "
    "call set_state(None) on the previous app's shutdown first."
)


class _BridgeHolder:
    """Mutable holder for the registered state.

    Wrapping the state in a class attribute avoids ``global`` (PLW0603) while
    keeping the module's ``set_state`` / ``get_state`` API unchanged.
    """

    state: RefreshState | None = None


def set_state(state: RefreshState | None) -> None:
    """Register (or clear, with ``None``) the app state used by ``refresh_data``.

    Raises ``RuntimeError`` on double registration to prevent silent
    overwrite of an existing app's state by a second app in the same process.
    """
    if state is not None and _BridgeHolder.state is not None:
        raise RuntimeError(_DOUBLE_REGISTRATION_MSG)
    _BridgeHolder.state = state


def get_state() -> RefreshState | None:
    """Return the currently registered app state, or ``None`` if unset."""
    return _BridgeHolder.state
