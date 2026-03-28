"""Search route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.search import ensure_search_corpus, fts_search

from ._helpers import conn, parent, templates

SEARCH_PAGE_SIZE = 50


async def search(request: Request, q: str, page: int = 1) -> HTMLResponse:
    """Search results with snippets."""
    db = conn(request)
    results = []
    has_next = False

    if q.strip():
        ensure_search_corpus(db)
        offset = (page - 1) * SEARCH_PAGE_SIZE
        # Fetch one extra to detect next page
        results = fts_search(db, q, SEARCH_PAGE_SIZE + 1, offset)
        if len(results) > SEARCH_PAGE_SIZE:
            has_next = True
            results = results[:SEARCH_PAGE_SIZE]

    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "parent": parent(request),
            "query": q,
            "results": results,
            "page": page,
            "has_next": has_next,
        },
    )
