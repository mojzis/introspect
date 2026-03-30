"""E2E test infrastructure for HTMX-based crawling."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin, urlparse

import pytest
from bs4 import BeautifulSoup, Tag
from fastapi.testclient import TestClient
from httpx import Response


@dataclass
class HTMXResponse:
    """Wrapper for HTML responses with convenient selectors."""

    response: Response
    soup: BeautifulSoup
    url: str

    @property
    def status(self) -> int:
        return self.response.status_code

    def select_one(self, selector: str) -> Tag | None:
        return self.soup.select_one(selector)

    def select(self, selector: str) -> list[Tag]:
        return self.soup.select(selector)

    def text(self, selector: str) -> str:
        el = self.select_one(selector)
        return el.get_text(strip=True) if el else ""

    def htmx_attr(self, selector: str, attr: str = "hx-get") -> str | None:
        el = self.select_one(selector)
        return str(el.get(attr)) if el else None

    def find_all_endpoints(self) -> list[dict]:
        """Extract all HTMX and link endpoints from the page."""
        endpoints: list[dict] = []

        for el in self.soup.select("[hx-get]"):
            endpoints.append(
                {
                    "url": str(el["hx-get"]),
                    "method": "GET",
                    "htmx": True,
                    "source": f"hx-get on <{el.name}>",
                }
            )

        for el in self.soup.select("[hx-post]"):
            endpoints.append(
                {
                    "url": str(el["hx-post"]),
                    "method": "POST",
                    "htmx": True,
                    "source": f"hx-post on <{el.name}>",
                }
            )

        for el in self.soup.select("a[href]"):
            href = str(el["href"])
            parsed = urlparse(href)
            if href.startswith(("/", ".")) or not parsed.netloc:
                endpoints.append(
                    {
                        "url": href,
                        "method": "GET",
                        "htmx": el.has_attr("hx-boost"),
                        "source": "href on <a>",
                    }
                )

        for el in self.soup.select("form[action]"):
            endpoints.append(
                {
                    "url": str(el.get("action", "")),
                    "method": str(el.get("method", "GET")).upper(),
                    "htmx": (el.has_attr("hx-post") or el.has_attr("hx-get")),
                    "source": "form action",
                }
            )

        return endpoints


@dataclass
class CrawlResult:
    """Result of automatic crawling."""

    visited: dict[str, HTMXResponse] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"Crawled {len(self.visited)} URLs"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  {err['url']}: {err['status']} ({err['source']})")
        return "\n".join(lines)


class HTMXTestClient:
    """Test client that understands HTMX patterns."""

    def __init__(self, client: TestClient):
        self._client = client

    def get(self, url: str, *, htmx: bool = False, **kwargs) -> HTMXResponse:
        headers = kwargs.pop("headers", {})
        if htmx:
            headers["HX-Request"] = "true"
        resp = self._client.get(url, headers=headers, **kwargs)
        soup = BeautifulSoup(resp.text, "html.parser")
        return HTMXResponse(resp, soup, url)

    def post(self, url: str, *, htmx: bool = False, **kwargs) -> HTMXResponse:
        headers = kwargs.pop("headers", {})
        if htmx:
            headers["HX-Request"] = "true"
        resp = self._client.post(url, headers=headers, **kwargs)
        soup = BeautifulSoup(resp.text, "html.parser")
        return HTMXResponse(resp, soup, url)

    def click(self, page: HTMXResponse, selector: str) -> HTMXResponse:
        """Simulate clicking an HTMX-enabled element."""
        el = page.select_one(selector)
        if not el:
            msg = f"No element matching: {selector}"
            raise ValueError(msg)

        if url := el.get("hx-get"):
            return self.get(str(url), htmx=True)
        if url := el.get("hx-post"):
            return self.post(str(url), htmx=True)
        if url := el.get("href"):
            return self.get(str(url), htmx=el.has_attr("hx-boost"))
        msg = f"Element has no hx-get/hx-post/href: {selector}"
        raise ValueError(msg)

    def _visit(self, endpoint: dict) -> HTMXResponse | None:
        """Make a request for a single crawl endpoint."""
        method = endpoint["method"]
        if method == "GET":
            return self.get(endpoint["url"], htmx=endpoint["htmx"])
        if method == "POST":
            return self.post(endpoint["url"], htmx=endpoint["htmx"])
        return None

    def crawl(
        self,
        start_url: str = "/",
        *,
        max_pages: int = 50,
        follow_htmx: bool = True,
        follow_links: bool = True,
        allowed_prefixes: tuple[str, ...] = ("/",),
    ) -> CrawlResult:
        """Crawl all discoverable endpoints from start_url."""
        result = CrawlResult()
        queue: deque[dict] = deque(
            [
                {
                    "url": start_url,
                    "method": "GET",
                    "htmx": False,
                    "source": "start",
                }
            ]
        )
        seen: set[tuple[str, str, bool]] = set()

        while queue and len(result.visited) < max_pages:
            ep = queue.popleft()
            key = (ep["url"], ep["method"], ep["htmx"])
            if key in seen:
                continue
            seen.add(key)

            if not any(ep["url"].startswith(p) for p in allowed_prefixes):
                continue

            try:
                page = self._visit(ep)
            except Exception as e:
                result.errors.append(
                    {
                        "url": ep["url"],
                        "method": ep["method"],
                        "status": f"exception: {e}",
                        "source": ep["source"],
                    }
                )
                continue

            if page is None:
                continue

            if page.status >= 400:
                result.errors.append(
                    {
                        "url": ep["url"],
                        "method": ep["method"],
                        "status": page.status,
                        "source": ep["source"],
                    }
                )
                continue

            result.visited[ep["url"]] = page
            self._enqueue_endpoints(
                page,
                ep["url"],
                queue,
                follow_htmx=follow_htmx,
                follow_links=follow_links,
            )

        return result

    @staticmethod
    def _enqueue_endpoints(
        page: HTMXResponse,
        base_url: str,
        queue: deque[dict],
        *,
        follow_htmx: bool,
        follow_links: bool,
    ) -> None:
        """Discover and enqueue new endpoints from a page."""
        for ep in page.find_all_endpoints():
            ep_url = ep["url"]
            if not ep_url.startswith("/"):
                ep_url = urljoin(base_url, ep_url)

            should_follow = (follow_htmx and ep["htmx"]) or (
                follow_links and not ep["htmx"] and ep["method"] == "GET"
            )
            if should_follow:
                queue.append({**ep, "url": ep_url})


# === Fixtures ===

E2E_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def e2e(tmp_path):
    """E2E test client with test data from tests/e2e/data/."""
    from introspect.search import _fts_cache  # noqa: PLC0415

    glob_pattern = str(E2E_DATA_DIR / "projects" / "**" / "*.jsonl")
    db_path = tmp_path / "test.duckdb"
    _fts_cache.clear()

    with patch.dict(
        os.environ,
        {
            "INTROSPECT_JSONL_GLOB": glob_pattern,
            "INTROSPECT_DB_PATH": str(db_path),
            "INTROSPECT_DAYS": "0",
        },
    ):
        from introspect.api.main import app  # noqa: PLC0415

        with TestClient(app) as client:
            yield HTMXTestClient(client)
