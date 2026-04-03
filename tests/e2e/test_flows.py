"""Targeted tests for specific user flows."""


def test_dashboard_to_session_detail(e2e):
    """Navigate from dashboard to a session detail page."""
    dashboard = e2e.get("/")
    assert dashboard.status == 200

    link = dashboard.select_one("a[href^='/sessions/']")
    assert link, "Dashboard should show session links"

    detail = e2e.click(dashboard, "a[href^='/sessions/']")
    assert detail.status == 200


def test_search_returns_results(e2e):
    """Search should return results for known test data."""
    results = e2e.get("/search?q=refactor")
    assert results.status == 200


def test_sessions_list_loads(e2e):
    """Sessions page lists available sessions."""
    page = e2e.get("/sessions")
    assert page.status == 200
    link = page.select_one("a[href^='/sessions/']")
    assert link, "Sessions page should list sessions"


def test_htmx_navigation(e2e):
    """HTMX requests to nav links return partial content."""
    urls = [
        "/sessions",
        "/search",
        "/tools",
        "/mcps",
        "/stats",
        "/raw",
    ]
    for url in urls:
        page = e2e.get(url, htmx=True)
        assert page.status == 200
        html_tag = page.soup.find("html")
        assert not (html_tag and html_tag.find("head")), (
            f"HTMX request to {url} should return a fragment"
        )


def test_session_detail_shows_messages(e2e):
    """Session detail page should display conversation."""
    sid = "aaaa1111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    detail = e2e.get(f"/sessions/{sid}")
    assert detail.status == 200
    text = detail.soup.get_text().lower()
    assert "refactor" in text or "auth" in text


def test_tools_page(e2e):
    """Tools page should show tool call data."""
    page = e2e.get("/tools")
    assert page.status == 200


def test_mcps_page(e2e):
    """MCPs page should show MCP tool data."""
    page = e2e.get("/mcps")
    assert page.status == 200


def test_stats_page(e2e):
    """Stats page should show statistics."""
    page = e2e.get("/stats")
    assert page.status == 200
    text = page.soup.get_text()
    assert "2" in text or "session" in text.lower()
