"""Automatic crawl test — exercises all discoverable endpoints."""


def test_crawl_all_pages(e2e):
    """Crawl the entire app from / and verify nothing errors."""
    result = e2e.crawl("/", max_pages=100)

    assert result.all_ok, f"Crawl found errors:\n{result.summary()}"
    assert len(result.visited) > 5, (
        f"Expected multiple pages, found {len(result.visited)}"
    )


def test_htmx_fragments_are_fragments(e2e):
    """HTMX requests should return fragments, not full pages."""
    result = e2e.crawl("/", follow_links=False, follow_htmx=True)

    for url, page in result.visited.items():
        if url == "/":
            continue  # Start page is full
        html_tag = page.soup.find("html")
        assert not (html_tag and html_tag.find("head")), (
            f"HTMX endpoint {url} returned full page"
        )


def test_all_pages_have_title(e2e):
    """Full page loads should have a <title>."""
    result = e2e.crawl("/", follow_htmx=False, follow_links=True)

    for url, page in result.visited.items():
        title = page.soup.find("title")
        assert title and title.string, f"Page {url} missing <title>"
