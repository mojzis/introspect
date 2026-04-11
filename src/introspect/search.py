"""Full-text search over conversation logs using DuckDB FTS."""

from __future__ import annotations

from datetime import datetime

import duckdb

_fts_cache: dict[str, bool] = {}

# Snippet window size (characters) for results — wide enough to see context
# around the match but small enough to keep LLM responses readable.
_SNIPPET_WINDOW = 240


def _windowed_snippet(text: str | None, terms: list[str]) -> str:
    """Return a window of ``text`` centered on the first term hit.

    ``terms`` is a pre-lowercased list so callers can hoist the split out
    of tight loops. Falls back to the head of the text when no term matches
    (can happen with stemmed FTS matches or ILIKE fallback quirks). Collapses
    newlines and adds ellipses when the window has more text on either side.
    """
    if not text:
        return ""
    flat = text.replace("\n", " ").replace("\r", " ")
    lower = flat.lower()
    best = len(flat) + 1
    for term in terms:
        idx = lower.find(term)
        if idx != -1 and idx < best:
            best = idx
    if best > len(flat):
        return flat[:_SNIPPET_WINDOW]
    half = _SNIPPET_WINDOW // 2
    start = max(0, best - half)
    end = min(len(flat), start + _SNIPPET_WINDOW)
    start = max(0, end - _SNIPPET_WINDOW)
    core = flat[start:end]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return prefix + core + suffix


def fts_available(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if the FTS extension can be installed and loaded.

    Attempts INSTALL (downloads if needed) then LOAD.
    Result is cached for the process lifetime. Clear ``_fts_cache`` to reset.
    """
    if "available" in _fts_cache:
        return _fts_cache["available"]
    try:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        _fts_cache["available"] = True
    except (duckdb.IOException, duckdb.CatalogException, duckdb.HTTPException):
        _fts_cache["available"] = False
    return _fts_cache["available"]


def ensure_search_corpus(conn: duckdb.DuckDBPyConnection) -> None:
    """Build the search corpus table if it doesn't already exist."""
    tables = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name = 'search_corpus' AND table_type = 'BASE TABLE'
    """).fetchall()
    if not tables:
        build_search_corpus(conn)


def build_search_corpus(conn: duckdb.DuckDBPyConnection) -> None:
    """Create or refresh the search_corpus table and optionally FTS index.

    Materializes text content from raw_messages into a searchable table.
    If the FTS extension is available, builds a BM25 index; otherwise
    falls back to ILIKE-based search.
    """
    conn.execute("DROP TABLE IF EXISTS search_corpus")

    conn.execute("""
        CREATE TABLE search_corpus AS
        SELECT
            row_number() OVER () AS rowid,
            session_id,
            timestamp,
            type AS role,
            COALESCE(
                -- User text messages: content is a plain string
                CASE
                    WHEN type = 'user'
                        AND json_extract_string(
                            message, '$.content[0].type'
                        ) IS NULL
                    THEN json_extract_string(message, '$.content')
                END,
                -- Assistant text blocks
                CASE
                    WHEN type = 'assistant'
                        AND json_extract_string(
                            message, '$.content[0].type'
                        ) = 'text'
                    THEN json_extract_string(
                        message, '$.content[0].text'
                    )
                END,
                -- Tool use inputs (assistant)
                CASE
                    WHEN type = 'assistant'
                        AND json_extract_string(
                            message, '$.content[0].type'
                        ) = 'tool_use'
                    THEN json_extract_string(
                        message, '$.content[0].name'
                    )
                        || ': '
                        || COALESCE(
                            json_extract_string(
                                message, '$.content[0].input'
                            ),
                            ''
                        )
                END,
                -- Tool results (user with tool_result content)
                CASE
                    WHEN type = 'user'
                        AND json_extract_string(
                            message, '$.content[0].type'
                        ) = 'tool_result'
                    THEN COALESCE(
                        json_extract_string(
                            message, '$.content[0].content'
                        ),
                        ''
                    )
                END,
                ''
            ) AS content_text,
        FROM raw_messages
    """)

    # Remove rows with empty content
    conn.execute("DELETE FROM search_corpus WHERE content_text = ''")

    # Try to build FTS index if extension is available
    if fts_available(conn):
        conn.execute(
            "PRAGMA create_fts_index("
            "'search_corpus', 'rowid', 'content_text',"
            " overwrite=1)"
        )


def _collect_filters(
    cwd_prefix: str | None,
    role: str | None,
    since: str | datetime | None,
    session_id: str | None,
) -> tuple[list[str], list[object]]:
    """Translate structured filter kwargs into WHERE clauses + params.

    Returns (clauses, params) — clauses reference the joined ``search_corpus``
    as ``s`` and ``logical_sessions`` as ``ls``.
    """
    clauses: list[str] = []
    params: list[object] = []
    if cwd_prefix:
        clauses.append("ls.cwd LIKE ? || '%'")
        params.append(cwd_prefix)
    if role:
        clauses.append("s.role = ?")
        params.append(role)
    if since:
        clauses.append("s.timestamp >= ?")
        params.append(since)
    if session_id:
        clauses.append("s.session_id = ?")
        params.append(session_id)
    return clauses, params


def fts_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    limit: int = 20,
    offset: int = 0,
    *,
    cwd_prefix: str | None = None,
    role: str | None = None,
    since: str | datetime | None = None,
    session_id: str | None = None,
    require_all: bool = False,
) -> list[tuple]:
    """Search the corpus. Uses FTS/BM25 if available, else ILIKE.

    Returns rows of ``(session_id, timestamp, role, cwd, snippet, score)``.
    Snippets are windowed around the first query-term hit for context.

    Optional filters:
      - ``cwd_prefix``: only sessions whose working directory starts with this.
      - ``role``: restrict to ``'user'`` or ``'assistant'`` messages.
      - ``since``: ISO timestamp or ``datetime``; only messages at or after.
      - ``session_id``: restrict to a single session.
      - ``require_all``: multi-word queries must match all terms (FTS AND mode).
    """
    filter_clauses, filter_params = _collect_filters(
        cwd_prefix, role, since, session_id
    )
    filter_sql = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

    if fts_available(conn):
        # ``conjunctive`` is inlined, not parameterized — DuckDB doesn't accept
        # ``?`` placeholders for FTS named arguments, and the value comes from
        # a typed bool we control, not user input.
        conjunctive = 1 if require_all else 0
        sql = f"""
            SELECT
                s.session_id,
                s.timestamp,
                s.role,
                ls.cwd,
                s.content_text,
                s.score,
            FROM (
                SELECT *,
                    fts_main_search_corpus.match_bm25(
                        rowid, ?, conjunctive := {conjunctive}
                    ) AS score
                FROM search_corpus
            ) s
            JOIN logical_sessions ls USING (session_id)
            WHERE s.score IS NOT NULL{filter_sql}
            ORDER BY s.score DESC
            LIMIT ?
            OFFSET ?
        """  # noqa: S608
        params: list[object] = [query, *filter_params, limit, offset]
        rows = conn.execute(sql, params).fetchall()
    else:
        # Fallback: ILIKE search with word-count scoring.
        terms = query.strip().split()
        if not terms:
            return []

        match_op = " AND " if require_all else " OR "
        where_clauses = match_op.join("s.content_text ILIKE ?" for _ in terms)
        score_expr = " + ".join(
            "CASE WHEN s.content_text ILIKE ? THEN 1 ELSE 0 END" for _ in terms
        )
        like_terms = [f"%{t}%" for t in terms]
        params = [*like_terms, *like_terms, *filter_params, limit, offset]

        sql = f"""
            SELECT
                s.session_id,
                s.timestamp,
                s.role,
                ls.cwd,
                s.content_text,
                ({score_expr}) AS score,
            FROM search_corpus s
            JOIN logical_sessions ls USING (session_id)
            WHERE ({where_clauses}){filter_sql}
            ORDER BY score DESC, s.timestamp DESC
            LIMIT ?
            OFFSET ?
        """  # noqa: S608
        rows = conn.execute(sql, params).fetchall()

    snippet_terms = [t for t in query.lower().split() if t]
    return [
        (
            session_id,
            timestamp,
            role,
            cwd,
            _windowed_snippet(content_text, snippet_terms),
            score,
        )
        for session_id, timestamp, role, cwd, content_text, score in rows
    ]
