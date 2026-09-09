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


class DataNotReadyError(RuntimeError):
    """Raised when a standalone MCP read precedes its preview publication."""

    def __init__(self, state: RefreshState):
        loading = getattr(state, "loading_state", None)
        phase = getattr(getattr(loading, "phase", None), "value", "unknown")
        target = getattr(loading, "target", None) or getattr(
            state, "refresh_target", None
        )
        window = getattr(target, "window", "unknown")
        days = getattr(target, "days", "unknown")
        candidate_count = getattr(loading, "candidate_count", 0)
        completed = getattr(loading, "completed_candidates", 0)
        progress = (
            f"; candidates={completed}/{candidate_count}" if candidate_count else ""
        )
        error = getattr(loading, "error", None)
        if phase == "failed":
            detail = f" Error: {error}" if error else ""
            super().__init__(
                "Data unavailable: startup data loading failed."
                f"{detail} No database snapshot is available; restart the "
                "standalone MCP server after correcting the configuration. "
                f"(phase={phase}; target={window} ({days} days){progress})"
            )
            return
        super().__init__(
            "Data loading: no preview is ready yet "
            f"(phase={phase}; target={window} ({days} days){progress}). "
            "Retry this tool shortly; results will be available as soon as "
            "the preview or warm snapshot is published."
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
