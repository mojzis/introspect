"""Tests for the refresh endpoint and indicator UI."""

import asyncio
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from introspect.api.main import app
from introspect.refresh import LoadingPhase, LoadingState, RefreshTarget

from .conftest import _patched_client


def test_post_refresh_sets_trigger_and_renders_fragment():
    """POST /refresh should set the event and return the indicator fragment."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        trigger = asyncio.Event()
        app.state.refresh_trigger = trigger
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        try:
            response = client.post("/refresh")
            assert response.status_code == 200
            assert 'id="refresh-state"' in response.text
            assert "Refresh now" in response.text
            assert trigger.is_set()
        finally:
            # Clean up to avoid bleed into other tests that share ``app``.
            for attr in ("refresh_trigger", "refresh_in_progress", "last_refreshed_at"):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_post_refresh_disabled_when_interval_zero():
    """With trigger=None on app.state, POST /refresh renders the disabled label."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # Lifespan always sets the attribute; ``None`` models interval==0.
        app.state.refresh_trigger = None
        response = client.post("/refresh")
        assert response.status_code == 200
        assert "auto-refresh off" in response.text


def test_base_html_shows_refresh_indicator():
    """The nav-embedded indicator should appear on a normal GET."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="refresh-state"' in response.text


def test_refresh_status_renders_without_setting_trigger():
    """GET /refresh-status returns the fragment without poking the loop."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        trigger = asyncio.Event()
        app.state.refresh_trigger = trigger
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert 'id="refresh-state"' in response.text
            # Status endpoint must NOT poke the trigger — polling should be idle.
            assert not trigger.is_set()
        finally:
            for attr in ("refresh_trigger", "refresh_in_progress", "last_refreshed_at"):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_polls_while_in_progress():
    """While in_progress, the fragment includes hx-get polling attributes."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = True
        app.state.last_refreshed_at = datetime.now(UTC)
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert 'hx-get="/refresh-status"' in response.text
            assert "hx-trigger=" in response.text
            assert "refreshing" in response.text
        finally:
            for attr in ("refresh_trigger", "refresh_in_progress", "last_refreshed_at"):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_is_an_accessible_live_region_with_progress():
    """The shared lifecycle state is visible to assistive technology."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = True
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.loading_state = LoadingState(
            LoadingPhase.LOADING,
            RefreshTarget("14", 14),
            candidate_count=4,
            completed_candidates=2,
        )
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert 'role="status"' in response.text
            assert 'aria-live="polite"' in response.text
            assert 'aria-busy="true"' in response.text
            assert "2/4 files" in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "loading_state",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_indicator_label_has_no_date():
    """The rendered label never contains a YYYY-MM-DD date — only relative time."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        # 3 days ago — previously rendered via strftime("%Y-%m-%d %H:%M").
        app.state.last_refreshed_at = datetime.now(UTC) - timedelta(days=3)
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert "3d ago" in response.text
            # No ISO-like date substring.
            assert re.search(r"\d{4}-\d{2}-\d{2}", response.text) is None
        finally:
            for attr in ("refresh_trigger", "refresh_in_progress", "last_refreshed_at"):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_post_refresh_with_window_updates_app_state():
    """POST /refresh with window=7 should make 7 the selected option."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        trigger = asyncio.Event()
        app.state.refresh_trigger = trigger
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_window = "30"
        try:
            response = client.post("/refresh", data={"window": "7"})
            assert response.status_code == 200
            assert app.state.refresh_window == "7"
            assert '<option value="7" selected>7 days</option>' in response.text
            # The other tokens must NOT be marked selected.
            assert '<option value="30" selected' not in response.text
            assert '<option value="1" selected' not in response.text
            assert '<option value="month" selected' not in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_window",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_post_refresh_accepts_custom_and_all_data_targets():
    """The web control can continue a CLI target or select all data."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_window = "30"
        try:
            response = client.post("/refresh", data={"window": "14"})
            assert response.status_code == 200
            assert app.state.refresh_target.days == 14
            assert '<option value="14" selected>14 days (CLI)</option>' in response.text

            response = client.post("/refresh", data={"window": "0"})
            assert response.status_code == 200
            assert app.state.refresh_target.days == 0
            assert '<option value="0" selected>All data</option>' in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_window",
                "refresh_target",
                "refresh_pending",
                "refresh_started_at",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_post_refresh_invalid_window_keeps_current():
    """An invalid window token is ignored — sticky choice survives."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_window = "7"
        try:
            response = client.post("/refresh", data={"window": "bogus"})
            assert response.status_code == 200
            assert app.state.refresh_window == "7"
            assert '<option value="7" selected>7 days</option>' in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_window",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_includes_picker():
    """GET /refresh-status must render the window picker with all four options."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_window = "month"
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            text = response.text
            assert '<select name="window"' in text
            assert ">Today<" in text
            assert ">7 days<" in text
            assert ">30 days<" in text
            assert ">This month<" in text
            assert '<option value="month" selected>This month</option>' in text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_window",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_post_refresh_returns_immediately_without_waiting():
    """POST /refresh must not block waiting for the rebuild to finish.

    Sets the trigger and returns the current indicator state in one shot —
    the polling fragment is responsible for catching completion. Asserts the
    response arrives well under the old 3-second wait budget even with the
    in-progress flag stuck on.
    """
    import time  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        trigger = asyncio.Event()
        app.state.refresh_trigger = trigger
        app.state.refresh_in_progress = True  # Loop "stuck" — handler must not wait.
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_started_at = datetime.now(UTC)
        try:
            start = time.monotonic()
            response = client.post("/refresh")
            elapsed = time.monotonic() - start
            assert response.status_code == 200
            assert elapsed < 0.5, f"POST /refresh blocked for {elapsed:.2f}s"
            assert trigger.is_set()
            assert "refreshing" in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_started_at",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_poll_delay_tightens_with_elapsed_time():
    """The hx-trigger delay shrinks (3s → 500ms) as the rebuild runs longer."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = True
        app.state.last_refreshed_at = datetime.now(UTC)
        try:
            for elapsed_seconds, expected_ms in (
                (0.0, 3000),
                (3.0, 2000),
                (5.0, 1000),
                (10.0, 500),
            ):
                app.state.refresh_started_at = datetime.now(UTC) - timedelta(
                    seconds=elapsed_seconds,
                )
                response = client.get("/refresh-status")
                assert response.status_code == 200
                assert f"load delay:{expected_ms}ms" in response.text, (
                    f"expected {expected_ms}ms delay at elapsed={elapsed_seconds}s, "
                    f"got: {response.text}"
                )
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_started_at",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_reloads_page_after_completion():
    """A completed polled refresh tells HTMX to reload the current page."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.refresh_started_at = datetime.now(UTC) - timedelta(seconds=2)
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert "refresh-flash" in response.text
            assert "refreshed" in response.text
            assert response.headers["HX-Refresh"] == "true"
            # Polling stops once the flash is shown — no load-delay trigger remains.
            # (The window picker still has hx-trigger="change", which is unrelated.)
            assert "load delay:" not in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "refresh_started_at",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_labels_preview_database():
    """The indicator identifies data served from the startup preview."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.database_snapshot = False
        app.state.database_label = "preview"
        app.state.loading_state = LoadingState(
            LoadingPhase.PREVIEW_READY,
            RefreshTarget("30", 30),
        )
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert "preview" in response.text
            assert "refreshed" in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "database_snapshot",
                "database_label",
                "loading_state",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_labels_warm_snapshot():
    """The indicator identifies a compatible prior database snapshot."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.database_snapshot = True
        app.state.database_label = "warm snapshot"
        app.state.loading_state = LoadingState(
            LoadingPhase.PREVIEW_READY,
            RefreshTarget("30", 30),
        )
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert "warm snapshot" in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "database_snapshot",
                "database_label",
                "loading_state",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_refresh_status_reports_failed_authoritative_load():
    """A failed build is visible while the prior snapshot remains usable."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        app.state.refresh_trigger = asyncio.Event()
        app.state.refresh_in_progress = False
        app.state.last_refreshed_at = datetime.now(UTC)
        app.state.database_snapshot = True
        app.state.database_label = "warm snapshot"
        app.state.loading_state = LoadingState(
            LoadingPhase.FAILED,
            RefreshTarget("30", 30),
            error="refresh failed; retrying",
        )
        try:
            response = client.get("/refresh-status")
            assert response.status_code == 200
            assert "Refresh failed" in response.text
            assert "warm snapshot" in response.text
        finally:
            for attr in (
                "refresh_trigger",
                "refresh_in_progress",
                "last_refreshed_at",
                "database_snapshot",
                "database_label",
                "loading_state",
            ):
                if hasattr(app.state, attr):
                    delattr(app.state, attr)


def test_base_html_include_does_not_render_flash():
    """A normal page render (not a polling response) must not trigger the flash."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # Even with a brand-new last_refreshed_at, the base.html include path
        # has notify undefined and must default to no flash. The CSS class
        # name itself appears in the <style> block — check for the rendered
        # span and the literal "refreshed" label instead.
        response = client.get("/")
        assert response.status_code == 200
        assert 'class="refresh-flash"' not in response.text
        assert "&check; refreshed" not in response.text


def test_window_to_days_month(monkeypatch):
    """``window_to_days('month')`` matches days-since-start-of-current-month.

    Pinning ``datetime.now`` removes the microsecond-window flake that would
    otherwise occur if the test crosses a UTC midnight between the two calls.
    """
    from introspect import refresh as refresh_mod  # noqa: PLC0415

    fixed = datetime(2026, 4, 26, tzinfo=UTC)

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(refresh_mod, "datetime", _FakeDT)
    assert refresh_mod.window_to_days("month") == 26


def test_window_to_days_simple_tokens():
    """``window_to_days`` returns the literal int for fixed tokens."""
    from introspect.refresh import window_to_days  # noqa: PLC0415

    assert window_to_days("1") == 1
    assert window_to_days("7") == 7
    assert window_to_days("30") == 30
