"""The HTMX error policy, end to end.

Three things have to hold together or a failure goes back to being invisible:
the status code stays honest, an HTMX request gets a fragment retargeted at
the error container, and the base template carries the client-side half
(``htmx-config`` + ``#errors``) that makes the fragment land somewhere.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from introspect.api.errors import (
    ERROR_CONTAINER_ID,
    ERROR_RENDERED_HEADER,
    ERROR_SWAP,
    ERROR_TARGET,
)

from .conftest import SID, _patched_client

HTMX = {"HX-Request": "true"}

# A route that fails for a real reason, rather than a test-only endpoint:
# ``?day=garbage`` fails ``parse_day`` before any DB work and raises
# HTTPException(400) from the handler.
BAD_REQUEST_URL = "/cost-overview/portfolio?day=garbage"
# ``from_uuid``/``to_uuid`` that aren't in the session raise HTTPException(404).
NOT_FOUND_URL = f"/sessions/{SID}/cost/bloat?from_uuid=nope&to_uuid=nope"
# ``page`` is declared ``Query(1, ge=1)``, so FastAPI's own validation rejects
# ``page=0`` before the handler runs — the RequestValidationError branch.
VALIDATION_URL = "/search?q=x&page=0"
MCPS_VALIDATION_URL = "/mcps?page=0"


def _is_full_document(body: str) -> bool:
    """True when the body is a whole page rather than a swappable fragment."""
    lowered = body.lower()
    return "<!doctype html" in lowered or "<html" in lowered


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as c:
        yield c


CRASH_MESSAGE = "deliberate failure from test_error_handling"


async def _boom(_request):
    raise RuntimeError(CRASH_MESSAGE)


@contextmanager
def _crashing_client(extra_env: dict[str, str] | None = None):
    """Client whose dashboard route raises, with the 500 response observable.

    Patching ``routes._dashboard`` (not the handler module) is what actually
    takes effect: ``routes.py`` binds the function at import time.
    """
    with (
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(
            Path(tmp), extra_env=extra_env, raise_server_exceptions=False
        ) as client,
        patch("introspect.api.routes._dashboard", _boom),
    ):
        yield client


@pytest.fixture
def crashing_client():
    with _crashing_client() as client:
        yield client


# --- HTMX requests ------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected_status"),
    [
        (BAD_REQUEST_URL, 400),
        (NOT_FOUND_URL, 404),
        (VALIDATION_URL, 422),
        # ``/mcps`` starts with ``/mcp``, the MCP transport mount. A prefix
        # test that doesn't distinguish them exempts a real HTML page from the
        # whole policy — and under the new swap rule htmx would then paste a
        # pydantic error blob over the page content.
        (MCPS_VALIDATION_URL, 422),
    ],
)
def test_htmx_error_is_a_retargeted_fragment(client, url, expected_status):
    """The real status code, an HTML fragment, and headers aiming it at #errors."""
    response = client.get(url, headers=HTMX)

    assert response.status_code == expected_status
    assert not _is_full_document(response.text)
    assert response.headers["HX-Retarget"] == ERROR_TARGET
    assert response.headers["HX-Reswap"] == ERROR_SWAP
    # A failed request must not leave the address bar claiming a page that
    # never rendered.
    assert response.headers["HX-Push-Url"] == "false"
    assert response.headers[ERROR_RENDERED_HEADER] == "1"

    toast = BeautifulSoup(response.text, "html.parser").select_one(".error-toast")
    assert toast is not None
    assert str(expected_status) in toast.get_text()
    assert toast.select_one("[data-error-dismiss]") is not None


def test_htmx_error_fragment_does_not_target_the_original_element(client):
    """``HX-Retarget`` is the only targeting the fragment does.

    Nothing in the markup may carry its own ``hx-target``/``hx-swap-oob``: a
    failed refresh must not be able to touch the data the user was reading.
    """
    response = client.get(BAD_REQUEST_URL, headers=HTMX)
    soup = BeautifulSoup(response.text, "html.parser")
    assert soup.select("[hx-target]") == []
    assert soup.select("[hx-swap-oob]") == []


# --- Non-HTMX requests --------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected_status"),
    [(BAD_REQUEST_URL, 400), (NOT_FOUND_URL, 404), (VALIDATION_URL, 422)],
)
def test_plain_request_error_is_a_full_page(client, url, expected_status):
    """Typed URL or reload: same status code, framed as a whole page."""
    response = client.get(url)

    assert response.status_code == expected_status
    assert _is_full_document(response.text)
    soup = BeautifulSoup(response.text, "html.parser")
    # The full page keeps the app chrome, so the user can navigate away.
    assert soup.select_one("nav") is not None
    heading = soup.select_one("h1")
    assert heading is not None
    assert str(expected_status) in heading.get_text()
    # Retargeting is meaningless without HTMX, so it isn't sent.
    assert "HX-Retarget" not in response.headers


def test_validation_error_fragment_is_a_one_liner(client):
    """The pydantic error list is noise in a toast; it belongs in the log."""
    response = client.get(VALIDATION_URL, headers=HTMX)

    assert response.status_code == 422
    assert "int_parsing" not in response.text
    assert "greater_than_equal" not in response.text
    assert "form this endpoint accepts" in response.text


# --- Unhandled exceptions -----------------------------------------------------


def test_unhandled_exception_is_a_500_fragment_for_htmx(crashing_client):
    """A crash is a 500 with feedback, never a 200 with error markup."""
    response = crashing_client.get("/", headers=HTMX)

    assert response.status_code == 500
    assert not _is_full_document(response.text)
    assert response.headers["HX-Retarget"] == ERROR_TARGET
    assert response.headers["HX-Reswap"] == ERROR_SWAP
    assert "500" in response.text


def test_unhandled_exception_is_a_500_page_without_htmx(crashing_client):
    """The full-page branch covers crashes too."""
    response = crashing_client.get("/")

    assert response.status_code == 500
    assert _is_full_document(response.text)
    assert "500" in response.text


def test_unhandled_exception_is_logged_with_its_traceback(crashing_client, caplog):
    """The fragment is a courtesy; the logged traceback is what gets it fixed."""
    with caplog.at_level("ERROR", logger="introspect.api.errors"):
        crashing_client.get("/", headers=HTMX)

    record = next(r for r in caplog.records if r.name == "introspect.api.errors")
    assert record.exc_info is not None
    assert CRASH_MESSAGE in caplog.text


# --- Debug mode ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "traceback_visible", "why"),
    [
        ({"INTROSPECT_DEBUG": ""}, False, "off by default"),
        ({"INTROSPECT_DEBUG": "1"}, True, "explicitly on, loopback bind"),
        # A variable exported once for devserve must not start serving file
        # paths and SQL to the LAN — same gate as the SQL API.
        (
            {"INTROSPECT_DEBUG": "1", "INTROSPECT_HOST": "0.0.0.0"},
            False,
            "debug is inert on a non-loopback bind",
        ),
    ],
)
def test_traceback_is_shown_only_in_debug_mode(env, traceback_visible, why):
    """Tracebacks are always logged; reaching the browser needs both gates."""
    with _crashing_client(extra_env=env) as client:
        response = client.get("/", headers=HTMX)

    assert response.status_code == 500
    pre = BeautifulSoup(response.text, "html.parser").select_one(
        ".error-toast-trace pre"
    )
    if traceback_visible:
        assert pre is not None, why
        assert "RuntimeError" in pre.get_text()
        assert CRASH_MESSAGE in response.text
    else:
        assert pre is None, why
        assert CRASH_MESSAGE not in response.text


def test_unrouted_url_gets_the_policy_too(client):
    """A 404 that never reaches a handler is still a policy error.

    Routing (and ``StaticFiles``) raise *Starlette's* ``HTTPException``, and
    ``fastapi.HTTPException`` is a distinct subclass — so a handler registered
    on the FastAPI class would miss this entirely. That is why ``errors.py``
    imports ``starlette.exceptions`` directly, and why ``starlette`` is a
    declared dependency rather than a transitive one.
    """
    response = client.get("/no-such-page", headers=HTMX)

    assert response.status_code == 404
    assert response.headers["HX-Retarget"] == ERROR_TARGET
    assert response.headers[ERROR_RENDERED_HEADER] == "1"


# --- Machine endpoints keep their machine representation ----------------------


def test_api_errors_stay_json():
    """``/api/*`` answers notebooks; a toast fragment there would be a bug."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _patched_client(Path(tmp), extra_env={"INTROSPECT_HOST": "0.0.0.0"}) as client,
    ):
        response = client.get("/api/schema", headers=HTMX)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "HX-Retarget" not in response.headers
    assert response.json() == {"detail": "Not found."}


def test_base_template_refuses_to_swap_an_unformatted_error(client):
    """A response we did not format must never overwrite the original target.

    HTMX swaps 4xx/5xx under our config, which is what lands the server's
    fragment. A proxy 502 or a middleware's plain-text 400 carries no
    ``HX-Retarget``, so without this guard htmx would swap that foreign body
    into the element the request came from.
    """
    body = client.get("/").text

    assert "htmx:beforeSwap" in body
    assert "shouldSwap = false" in body


# --- The client-side half -----------------------------------------------------


def test_base_template_declares_the_response_handling_rules(client):
    """4xx/5xx must be configured to swap *and* flag an error.

    Without this HTMX's default silently drops the server's error fragment,
    which is the whole failure mode this policy exists to close.
    """
    soup = BeautifulSoup(client.get("/").text, "html.parser")
    meta = soup.select_one('meta[name="htmx-config"]')
    assert meta is not None

    rules = json.loads(str(meta["content"]))["responseHandling"]
    by_code = {rule["code"]: rule for rule in rules}

    assert by_code["[45].."] == {"code": "[45]..", "swap": True, "error": True}
    # HTMX's own 204 / 2xx-3xx behaviour is preserved.
    assert by_code["204"]["swap"] is False
    assert by_code["[23].."]["swap"] is True


def test_base_template_has_the_error_container(client):
    """The fixed container the fragments are retargeted into."""
    soup = BeautifulSoup(client.get("/").text, "html.parser")
    container = soup.select_one(f"#{ERROR_CONTAINER_ID}")

    assert container is not None
    # Empty on a good page — it fills up only when something fails.
    assert container.get_text(strip=True) == ""


def test_base_template_listens_for_the_two_failure_events(client):
    """The backstop for responses the server never got to format."""
    body = client.get("/").text

    assert "htmx:responseError" in body
    assert "htmx:sendError" in body
    # The backstop must stay quiet when the server already rendered a toast.
    assert ERROR_RENDERED_HEADER in body
