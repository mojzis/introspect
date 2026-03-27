"""Full-text search over conversation logs using DuckDB FTS."""

from __future__ import annotations

import duckdb

_fts_checked: bool | None = None


def _fts_available(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if the FTS extension is already installed and loadable.

    Does NOT attempt INSTALL — avoids hanging when there's no network.
    """
    global _fts_checked  # noqa: PLW0603
    if _fts_checked is not None:
        return _fts_checked
    try:
        conn.execute("LOAD fts")
        _fts_checked = True
    except (duckdb.IOException, duckdb.CatalogException):
        _fts_checked = False
    return _fts_checked


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
    if _fts_available(conn):
        conn.execute(
            "PRAGMA create_fts_index("
            "'search_corpus', 'rowid', 'content_text',"
            " overwrite=1)"
        )


def fts_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    limit: int = 20,
) -> list[tuple]:
    """Search the corpus. Uses FTS/BM25 if available, else ILIKE.

    Returns rows of (session_id, timestamp, role, snippet, score).
    """
    if _fts_available(conn):
        return conn.execute(
            """
            SELECT
                s.session_id,
                s.timestamp,
                s.role,
                LEFT(s.content_text, 200) AS snippet,
                score,
            FROM (
                SELECT *,
                    fts_main_search_corpus.match_bm25(
                        rowid, ?
                    ) AS score
                FROM search_corpus
            ) s
            WHERE score IS NOT NULL
            ORDER BY score
            LIMIT ?
            """,
            [query, limit],
        ).fetchall()

    # Fallback: ILIKE search with word-count scoring
    terms = query.strip().split()
    if not terms:
        return []

    # WHERE: at least one term must match
    where_clauses = " OR ".join("content_text ILIKE ?" for _ in terms)
    # Score: count how many terms match (higher = better)
    score_expr = " + ".join(
        "CASE WHEN content_text ILIKE ? THEN 1 ELSE 0 END" for _ in terms
    )
    # Parameters: score terms first, then where terms, then limit
    params: list[str | int] = [f"%{t}%" for t in terms]
    params.extend(f"%{t}%" for t in terms)
    params.append(limit)

    return conn.execute(
        f"""
        SELECT
            session_id,
            timestamp,
            role,
            LEFT(content_text, 200) AS snippet,
            ({score_expr}) AS score,
        FROM search_corpus
        WHERE {where_clauses}
        ORDER BY score DESC, timestamp DESC
        LIMIT ?
        """,  # nosec B608
        params,
    ).fetchall()
