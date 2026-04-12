"""Shared SQL validation for read-only query enforcement.

Used by both the MCP tools and the web UI SQL page.
"""

import re

SQL_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
SQL_COMMENT_LINE = re.compile(r"--[^\n]*")
# Only single-quoted strings are SQL literals; double-quoted tokens are
# identifiers and must not be rewritten by the validator.
SQL_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
SQL_ALLOWED_FIRST_KEYWORDS = {"select", "with"}


def validate_read_only_sql(sql: str) -> str | None:
    """Return an error message if `sql` is not a safe read-only query.

    This is the PRIMARY guard -- do not weaken it assuming the connection is
    read-only. ``run_sql`` opens a fresh ``read_only=True`` connection as a
    defense-in-depth backstop, but even that permits some side-effecting
    statements (e.g. ``COPY ... TO '/file'`` can write outside the DB).
    The "first keyword must be SELECT or WITH" check blocks ATTACH, INSTALL,
    LOAD, PRAGMA, COPY, INSERT, UPDATE, DELETE, DROP, CREATE, CALL, etc.
    """
    stripped = SQL_COMMENT_BLOCK.sub(" ", sql)
    stripped = SQL_COMMENT_LINE.sub(" ", stripped)
    # Replace string-literal contents before scanning so a `;` or keyword
    # inside a literal doesn't trip the multi-statement / first-keyword
    # checks. Double-quoted identifiers are intentionally preserved.
    scan = SQL_STRING_LITERAL.sub("''", stripped)
    scan = scan.strip().rstrip(";").strip()
    if not scan:
        return "SQL is empty."
    if ";" in scan:
        return "Multiple statements are not allowed."
    first_word = scan.split(None, 1)[0].lower()
    if first_word not in SQL_ALLOWED_FIRST_KEYWORDS:
        return f"Only SELECT / WITH queries are allowed (got: {first_word!r})."
    return None
