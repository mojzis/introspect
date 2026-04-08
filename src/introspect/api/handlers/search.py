"""Search route handler."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.search import ensure_search_corpus, fts_available, fts_search

from ._helpers import (
    _EMPTY_SESSION_INFO,
    DEFAULT_PAGE_SIZE,
    SESSION_INFO_JOINS,
    SESSION_INFO_SELECT,
    conn,
    parent,
    session_row_to_dict,
    templates,
)


async def search(request: Request, q: str, page: int = 1) -> HTMLResponse:
    """Search results with snippets, enriched with session info."""
    db = conn(request)
    enriched: list[dict] = []
    has_next = False

    if q.strip():
        ensure_search_corpus(db)
        offset = (page - 1) * DEFAULT_PAGE_SIZE
        results = fts_search(db, q, DEFAULT_PAGE_SIZE + 1, offset)
        if len(results) > DEFAULT_PAGE_SIZE:
            has_next = True
            results = results[:DEFAULT_PAGE_SIZE]

        if results:
            session_ids = list({r[0] for r in results})
            placeholders = ", ".join("?" for _ in session_ids)
            rows = db.execute(
                f"""
                SELECT {SESSION_INFO_SELECT}
                FROM logical_sessions ls
                {SESSION_INFO_JOINS}
                WHERE ls.session_id IN ({placeholders})
            """,  # noqa: S608
                session_ids,
            ).fetchall()

            info_map = {row[0]: session_row_to_dict(row) for row in rows}

            for session_id, timestamp, role, snippet, score in results:
                info = info_map.get(session_id, _EMPTY_SESSION_INFO)
                enriched.append(
                    {
                        "session_id": session_id,
                        "timestamp": str(timestamp)[:19] if timestamp else "",
                        "role": role or "",
                        "snippet": (snippet or "")[:200],
                        "score": score,
                        **info,
                    }
                )

    fts_loaded = fts_available(db)

    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "parent": parent(request),
            "query": q,
            "results": enriched,
            "page": page,
            "has_next": has_next,
            "fts_loaded": fts_loaded,
        },
    )
