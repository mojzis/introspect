"""Background refresh loop for keeping the materialized DB in sync with JSONL files."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import duckdb

from introspect.db import materialize_views
from introspect.search import build_search_corpus

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


# Allowed tokens for the refresh-window picker. Owned by this module because
# both the background loop and the HTTP handler need them; placing them here
# keeps the dependency direction (handler -> refresh) one-way.
VALID_WINDOWS = frozenset({"1", "7", "30", "month"})
DEFAULT_WINDOW = "30"
# Days returned for fixed-length tokens. ``"month"`` is computed at call time.
_FIXED_WINDOW_DAYS: dict[str, int] = {"1": 1, "7": 7, "30": 30}

# Cold-start discovery is deliberately bounded.  A mtime guard band keeps a
# recently written Claude transcript whose timestamps lag its file mtime in
# the preview; Codex has stable YYYY/MM/DD partitions, so it does not need to
# scan the entire tree for the common case.
MAX_COLD_START_CANDIDATES = 10_000
CLAUDE_MTIME_GUARD_DAYS = 1


class LoadingPhase(StrEnum):
    """Lifecycle phases visible to the web UI and startup diagnostics."""

    DISCOVERING = "discovering"
    PREVIEWING = "previewing"
    PREVIEW_READY = "preview_ready"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class LoadingStage(StrEnum):
    """Concrete work currently being performed for a loading phase."""

    PROVIDER = "provider"
    DERIVED = "derived"
    SEARCH = "search"
    SWAP = "swap"


@dataclass(frozen=True)
class RefreshTarget:
    """The sole value used by a refresh build to identify its target window."""

    window: str
    days: int
    generation: int = 0


@dataclass(frozen=True)
class LoadingState:
    """Typed, truthful startup/refresh state.

    Candidate counts are counts of discovered files, not guessed percentages
    or ETAs.  ``target`` is captured with the state so a build can be checked
    before it promotes its sidecar.
    """

    phase: LoadingPhase
    target: RefreshTarget
    candidate_count: int = 0
    completed_candidates: int = 0
    error: str | None = None
    stage: LoadingStage | None = None


@dataclass(frozen=True)
class CandidateFiles:
    """Explicit files selected for a bounded cold-start preview."""

    claude: tuple[str, ...]
    codex: tuple[str, ...]
    truncated: bool = False

    @property
    def total(self) -> int:
        return len(self.claude) + len(self.codex)


def window_to_days(window: str) -> int:
    """Convert a window token to a positive ``days`` value for ``materialize_views``.

    * ``"1"`` / ``"7"`` / ``"30"`` -> the literal int.
    * ``"month"`` -> days since the first of the current UTC calendar month
      (inclusive of today). On the 1st returns ``1``. UTC matches the
      timezone used by ``materialize_views``' day filter.
    * Anything else -> the ``DEFAULT_WINDOW`` days value (defensive; the
      handler and lifespan both pre-validate input).
    """
    if window in _FIXED_WINDOW_DAYS:
        return _FIXED_WINDOW_DAYS[window]
    if window == "month":
        today = datetime.now(UTC).date()
        return (today - today.replace(day=1)).days + 1
    return _FIXED_WINDOW_DAYS[DEFAULT_WINDOW]


def target_for_window(
    window: str,
    *,
    days: int | None = None,
    generation: int = 0,
) -> RefreshTarget:
    """Build a target, keeping an explicit numeric override displayable."""
    if days is not None and days > 0 and days != window_to_days(window):
        display_window = str(days)
    else:
        display_window = window if window in VALID_WINDOWS else DEFAULT_WINDOW
    return RefreshTarget(
        window=display_window,
        days=window_to_days(window) if days is None else max(0, days),
        generation=generation,
    )


def set_refresh_target(
    state: RefreshState,
    window: str,
    *,
    days: int | None = None,
) -> RefreshTarget:
    """Atomically publish a new target and advance its generation.

    The compatibility ``refresh_window`` attribute remains mirrored for
    existing templates/callers, but refresh decisions use ``refresh_target``.
    """
    current = getattr(state, "refresh_target", None)
    generation = current.generation + 1 if isinstance(current, RefreshTarget) else 1
    target = target_for_window(window, days=days, generation=generation)
    state.refresh_target = target
    state.refresh_window = target.window
    state.refresh_pending = True
    return target


def _partition_date(path: str) -> date | None:
    """Extract a Codex YYYY/MM/DD partition from any path component sequence."""
    parts = Path(path).parts
    for index in range(len(parts) - 2):
        try:
            year, month, day = (int(value) for value in parts[index : index + 3])
            return date(year, month, day)
        except (TypeError, ValueError):
            continue
    return None


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)  # noqa: PTH204
    except (FileNotFoundError, OSError):
        return 0.0


def _bounded(paths: Iterable[str], limit: int) -> tuple[tuple[str, ...], bool]:
    """Return newest paths first without retaining more than ``limit`` paths."""
    import heapq  # noqa: PLC0415

    iterator: Iterator[str] = iter(paths)
    if limit < 1:
        return (), next(iterator, None) is not None

    selected: list[tuple[float, str]] = []
    truncated = False
    for path in iterator:
        candidate = (_safe_mtime(path), path)
        if len(selected) < limit:
            heapq.heappush(selected, candidate)
        elif candidate > selected[0]:
            heapq.heapreplace(selected, candidate)
            truncated = True
        else:
            truncated = True
    selected.sort(reverse=True)
    return tuple(path for _mtime, path in selected), truncated


def _codex_candidate_paths(
    codex_glob: str, *, moment: datetime, span_days: int
) -> Iterator[str]:
    """Yield Codex files from only the recent YYYY/MM/DD partitions."""
    if "**" not in codex_glob:
        return glob.iglob(codex_glob, recursive=True)  # noqa: PTH207

    def paths():
        seen: set[str] = set()
        for offset in range(span_days):
            partition = (moment - timedelta(days=offset)).date()
            partition_glob = codex_glob.replace(
                "**", f"{partition:%Y}/{partition:%m}/{partition:%d}", 1
            )
            for path in glob.iglob(partition_glob, recursive=True):  # noqa: PTH207
                if path not in seen:
                    seen.add(path)
                    yield path

    return paths()


def discover_cold_start_candidates(
    jsonl_glob: str,
    codex_glob: str | None,
    *,
    days: int = 1,
    now: datetime | None = None,
    max_candidates: int = MAX_COLD_START_CANDIDATES,
) -> CandidateFiles:
    """Discover a bounded, conservative file set for the initial preview.

    Claude selection uses a one-day mtime guard band around the requested
    window.  Codex selection prefers its date partitions and falls back to the
    same mtime guard when a test/custom path has no date partition.  The SQL
    timestamp filter remains authoritative after this preselection.
    """
    if max_candidates < 1:
        return CandidateFiles(
            (),
            (),
            next(glob.iglob(jsonl_glob, recursive=True), None) is not None,  # noqa: PTH207
        )
    moment = now or datetime.now(UTC)
    span_days = max(1, days)
    claude_cutoff = (
        moment - timedelta(days=span_days + CLAUDE_MTIME_GUARD_DAYS)
    ).timestamp()
    claude = (
        path
        for path in glob.iglob(jsonl_glob, recursive=True)  # noqa: PTH207
        if _safe_mtime(path) >= claude_cutoff
    )

    codex: Iterable[str] = ()
    if codex_glob is not None:
        fallback_cutoff = (
            moment - timedelta(days=span_days + CLAUDE_MTIME_GUARD_DAYS)
        ).timestamp()
        codex = (
            path
            for path in _codex_candidate_paths(
                codex_glob, moment=moment, span_days=span_days
            )
            if (
                _partition_date(path) is not None
                or _safe_mtime(path) >= fallback_cutoff
            )
        )
        # Date-partitioned Codex files are selected by partition, while
        # unpartitioned custom paths retain the conservative mtime fallback.
        codex = (
            path
            for path in codex
            if (
                (partition := _partition_date(path)) is None
                or partition >= (moment - timedelta(days=span_days - 1)).date()
            )
        )

    claude_selected, claude_truncated = _bounded(claude, max_candidates)
    remaining = max(0, max_candidates - len(claude_selected))
    codex_selected, codex_truncated = _bounded(codex, remaining)
    return CandidateFiles(
        claude_selected,
        codex_selected,
        claude_truncated or codex_truncated,
    )


class RefreshState(Protocol):
    """Contract between :func:`refresh_loop` (writer) and :func:`wait_for_refresh`
    (reader). FastAPI's ``app.state`` satisfies this after :mod:`api.main` sets
    the attributes during startup.
    """

    refresh_trigger: asyncio.Event | None
    refresh_in_progress: bool
    refresh_started_at: datetime | None
    last_refreshed_at: datetime | None
    refresh_window: str
    last_built_days: int
    refresh_target: RefreshTarget
    refresh_pending: bool
    loading_state: LoadingState


class RefreshOutcome(Enum):
    """Result classes for :func:`wait_for_refresh`."""

    DISABLED = "disabled"  # No trigger configured (auto-refresh off).
    UNCHANGED = "unchanged"  # Loop woke but JSONL files were unchanged.
    COMPLETED = "completed"  # Rebuild finished within the wait budget.
    STILL_RUNNING = "still_running"  # Started but did not finish in time.


@dataclass(frozen=True)
class RefreshResult:
    outcome: RefreshOutcome
    last_refreshed_at: datetime | None


# Internal poll cadence — kept private because callers tune *budgets*, not
# step granularity. Two phases: a brief one to detect that the loop picked up
# the trigger, then a longer one to wait for completion.
_START_TIMEOUT = 0.5
_START_STEP = 0.05
_FINISH_STEP = 0.1


async def wait_for_refresh(
    state: RefreshState,
    *,
    finish_timeout: float = 3.0,
) -> RefreshResult:
    """Set the refresh trigger and wait until the background loop finishes.

    Returns one of four outcomes based on what the loop did:

    * ``DISABLED`` — no trigger on ``state`` (auto-refresh is off).
    * ``UNCHANGED`` — loop woke but JSONL mtimes were unchanged; no rebuild ran.
    * ``COMPLETED`` — rebuild finished within ``finish_timeout``.
    * ``STILL_RUNNING`` — rebuild started but did not finish in time.
    """
    if state.refresh_trigger is None:
        return RefreshResult(RefreshOutcome.DISABLED, state.last_refreshed_at)

    last_before = state.last_refreshed_at
    state.refresh_trigger.set()

    waited = 0.0
    while waited < _START_TIMEOUT:
        if state.refresh_in_progress:
            break
        await asyncio.sleep(_START_STEP)
        waited += _START_STEP

    if state.refresh_in_progress:
        waited = 0.0
        while waited < finish_timeout:
            if not state.refresh_in_progress:
                break
            await asyncio.sleep(_FINISH_STEP)
            waited += _FINISH_STEP

    last_after = state.last_refreshed_at
    if last_after != last_before and last_after is not None:
        return RefreshResult(RefreshOutcome.COMPLETED, last_after)
    if state.refresh_in_progress:
        return RefreshResult(RefreshOutcome.STILL_RUNNING, last_after)
    return RefreshResult(RefreshOutcome.UNCHANGED, last_after)


def newest_mtime(jsonl_glob: str, codex_glob: str | None = None) -> float:
    """Return the newest mtime among files matching ``jsonl_glob`` (and,
    when given, ``codex_glob``), so the caller can watch both trees.

    Returns ``0.0`` if nothing matches. Defensively skips files that disappear
    between ``glob`` and ``os.path.getmtime``.
    """
    paths = glob.glob(jsonl_glob, recursive=True)  # noqa: PTH207
    if codex_glob is not None:
        paths += glob.glob(codex_glob, recursive=True)  # noqa: PTH207
    latest = 0.0
    for p in paths:
        try:
            mtime = os.path.getmtime(p)  # noqa: PTH204
        except FileNotFoundError:
            continue
        latest = max(latest, mtime)
    return latest


def _rebuild_sidecar(  # noqa: PLR0913
    sidecar: Path,
    jsonl_glob: str,
    days: int,
    resolve_projects: bool,
    codex_glob: str | None = None,
    *,
    jsonl_candidates: list[str] | None = None,
    codex_candidates: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    stage: Callable[[LoadingStage], None] | None = None,
) -> None:
    """Rebuild the materialized DB into a fresh sidecar file."""
    with contextlib.suppress(FileNotFoundError):
        sidecar.unlink()
    conn = duckdb.connect(str(sidecar))
    try:
        materialize_views(
            conn,
            jsonl_glob,
            days,
            resolve_projects=resolve_projects,
            codex_glob=codex_glob,
            jsonl_candidates=jsonl_candidates,
            codex_candidates=codex_candidates,
            progress=progress,
            phase=(
                (lambda name: stage(LoadingStage(name))) if stage is not None else None
            ),
        )
        if stage is not None:
            stage(LoadingStage.SEARCH)
        build_search_corpus(conn)
    finally:
        conn.close()


def _swap_in(db_path: Path, sidecar: Path) -> None:
    """Atomically rename ``sidecar`` over ``db_path``.

    Per-request connections are opened directly from ``db_path`` in the
    middleware, so this is now just an atomic file swap. In-flight cursors
    keep reading from the old inode (which lingers until they close), and
    new connections after the swap see the fresh data.
    """
    os.replace(str(sidecar), str(db_path))  # noqa: PTH105


def _compute_days(state: RefreshState, default: int) -> int:
    """Resolve the days-window value from ``state.refresh_window``.

    Falls back to ``default`` only when the state attribute is missing
    entirely. An invalid token is delegated to :func:`window_to_days`, which
    returns the ``DEFAULT_WINDOW`` days value — matching the lifespan's
    invalid-env fallback so the two code paths agree.
    """
    target = getattr(state, "refresh_target", None)
    if isinstance(target, RefreshTarget):
        return target.days
    window = getattr(state, "refresh_window", None)
    if not isinstance(window, str):
        return default
    return window_to_days(window)


def _window_changed(state: RefreshState, current_days: int) -> bool:
    """Has the window changed since the last successful rebuild?

    Used by :func:`refresh_loop` to force a rebuild when the user picks a new
    window even though JSONL mtimes are unchanged.
    """
    last = getattr(state, "last_built_days", None)
    return last != current_days


def _current_target(state: RefreshState, default_days: int) -> RefreshTarget:
    target = getattr(state, "refresh_target", None)
    if isinstance(target, RefreshTarget):
        return target
    window = getattr(state, "refresh_window", DEFAULT_WINDOW)
    return target_for_window(window, days=_compute_days(state, default_days))


def _set_loading(  # noqa: PLR0913
    state: RefreshState,
    phase: LoadingPhase,
    target: RefreshTarget,
    *,
    stage: LoadingStage | None = None,
    candidate_count: int = 0,
    completed_candidates: int = 0,
    error: str | None = None,
) -> None:
    state.loading_state = LoadingState(
        phase,
        target,
        stage=stage,
        candidate_count=candidate_count,
        completed_candidates=completed_candidates,
        error=error,
    )


async def refresh_loop(  # noqa: PLR0913, PLR0915
    app: FastAPI,
    db_path: Path,
    jsonl_glob: str,
    days: int,
    resolve_projects: bool,
    interval_seconds: float,
    trigger: asyncio.Event,
    codex_glob: str | None = None,
    *,
    initial: bool = False,
    one_shot: bool = False,
) -> None:
    """Poll JSONL mtime and rebuild the materialized DB when files change.

    The loop sleeps up to ``interval_seconds`` between ticks, but wakes early
    when ``trigger`` is set (e.g. a manual "Refresh now" click). The mtime
    short-circuit still gates the rebuild — a manual wake on an unchanged
    filesystem is a fast no-op unless the user picked a new window.

    The ``days`` parameter is the initial default; each rebuild re-reads
    ``app.state.refresh_window`` so the picker's choice is honoured by both
    manual refreshes and idle ticks.

    ``codex_glob``, when given, is watched alongside ``jsonl_glob`` so new
    Codex sessions trigger a rebuild too.
    """
    sidecar = db_path.with_name(db_path.name + ".next")
    last_mtime = newest_mtime(jsonl_glob, codex_glob)
    first_run = initial
    try:
        while True:
            try:
                if first_run:
                    first_run = False
                else:
                    if one_shot:
                        return
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(trigger.wait(), timeout=interval_seconds)
                    trigger.clear()
                current = newest_mtime(jsonl_glob, codex_glob)
                target = _current_target(app.state, days)
                current_days = target.days
                # Skip when nothing changed AND the window matches the last build.
                # A manual wake with a new window forces a rebuild even on an
                # idle filesystem. The handler may have optimistically flipped
                # ``refresh_in_progress`` true on POST so the polling fragment
                # always starts; clear it here so a no-op tick doesn't leave the
                # indicator polling forever.
                pending = bool(getattr(app.state, "refresh_pending", False))
                if (
                    current <= last_mtime
                    and not _window_changed(app.state, current_days)
                    and not pending
                ):
                    # The handler may have optimistically flipped these true on
                    # POST so the polling fragment always starts. Clear them on
                    # a no-op tick so the indicator doesn't poll forever.
                    # last_refreshed_at intentionally stays put — nothing was
                    # refreshed, so the UI honestly reverts to the prior value.
                    app.state.refresh_in_progress = False
                    app.state.refresh_started_at = None
                    continue
                log.info("JSONL changed; rebuilding materialized DB")
                app.state.refresh_started_at = datetime.now(UTC)
                app.state.refresh_in_progress = True
                _set_loading(
                    app.state,
                    LoadingPhase.LOADING,
                    target,
                    stage=LoadingStage.PROVIDER,
                )

                def report_progress(
                    completed: int, total: int, *, build_target: RefreshTarget = target
                ) -> None:
                    # The callback runs in the worker thread. The state fields
                    # are replaced atomically, and stale workers must not
                    # overwrite a newer target's progress.
                    if _current_target(app.state, days) == build_target:
                        _set_loading(
                            app.state,
                            LoadingPhase.LOADING,
                            build_target,
                            stage=LoadingStage.PROVIDER,
                            candidate_count=total,
                            completed_candidates=completed,
                        )

                def report_stage(
                    current_stage: LoadingStage,
                    *,
                    build_target: RefreshTarget = target,
                ) -> None:
                    if _current_target(app.state, days) == build_target:
                        previous = getattr(app.state, "loading_state", None)
                        _set_loading(
                            app.state,
                            LoadingPhase.LOADING,
                            build_target,
                            stage=current_stage,
                            candidate_count=(
                                previous.candidate_count
                                if isinstance(previous, LoadingState)
                                else 0
                            ),
                            completed_candidates=(
                                previous.completed_candidates
                                if isinstance(previous, LoadingState)
                                else 0
                            ),
                        )

                build_task = asyncio.create_task(
                    asyncio.to_thread(
                        _rebuild_sidecar,
                        sidecar,
                        jsonl_glob,
                        current_days,
                        resolve_projects,
                        codex_glob,
                        progress=report_progress,
                        stage=report_stage,
                    )
                )
                try:
                    await asyncio.shield(build_task)
                except asyncio.CancelledError:
                    # Cancellation must not leave a worker writing the
                    # sidecar after shutdown has removed it. Wait for the
                    # non-cancellable thread, then let cancellation proceed.
                    with contextlib.suppress(BaseException):
                        await build_task
                    raise
                # A picker change while the sidecar was being built makes it
                # stale.  Never publish a result for an obsolete generation.
                if _current_target(app.state, days) != target:
                    with contextlib.suppress(FileNotFoundError):
                        sidecar.unlink()
                    _set_loading(
                        app.state,
                        LoadingPhase.DISCOVERING,
                        _current_target(app.state, days),
                    )
                    app.state.refresh_in_progress = False
                    app.state.refresh_started_at = None
                    first_run = one_shot
                    continue
                report_stage(LoadingStage.SWAP)
                await asyncio.to_thread(_swap_in, db_path, sidecar)
                app.state.database_snapshot = False
                app.state.database_label = "authoritative"
                # Record the post-swap window first so a freak exception on
                # the timestamp assignment can't leave state thinking the DB
                # still holds the previous window's data and force a needless
                # rebuild on the next tick.
                app.state.last_built_days = current_days
                app.state.last_refreshed_at = datetime.now(UTC)
                app.state.refresh_pending = False
                _set_loading(app.state, LoadingPhase.READY, target)
                app.state.refresh_in_progress = False
                app.state.refresh_started_at = None
                if one_shot:
                    return
                last_mtime = current
                log.info("refresh complete")
            except asyncio.CancelledError:
                raise
            except Exception:
                target = _current_target(app.state, days)
                app.state.refresh_in_progress = False
                app.state.refresh_started_at = None
                _set_loading(
                    app.state,
                    LoadingPhase.FAILED,
                    target,
                    error="refresh failed; retrying",
                )
                app.state.refresh_pending = True
                log.warning(
                    "refresh failed; will retry next tick",
                    exc_info=True,
                )
                if one_shot:
                    return
                continue
    finally:
        app.state.refresh_in_progress = False
        app.state.refresh_started_at = None
        with contextlib.suppress(FileNotFoundError):
            sidecar.unlink()
