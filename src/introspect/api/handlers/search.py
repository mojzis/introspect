"""Search route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.search import ensure_search_corpus, fts_search

from ._helpers import conn, parent, templates


async def search(request: Request, q: str) -> HTMLResponse:
    """Search results with snippets."""
    db = conn(request)
    results = []

    if q.strip():
        ensure_search_corpus(db)
        results = fts_search(db, q, 50)

    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "parent": parent(request),
            "query": q,
            "results": results,
        },
    )
