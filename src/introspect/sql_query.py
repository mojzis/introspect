"""Shared read-only SQL guard used by the MCP `run_sql` tool and the HTTP API.

The validator here is the PRIMARY safety boundary for both callers — do not
weaken it on the assumption that the connection is read-only. A DuckDB
``read_only=True`` connection still permits some side-effecting statements
(e.g. ``COPY ... TO '/file'`` can write outside the DB), so the "first keyword
must be SELECT or WITH" check is what actually blocks ATTACH, INSTALL, LOAD,
PRAGMA, COPY, INSERT, UPDATE, DELETE, DROP, CREATE, CALL, etc.
"""

from __future__ import annotations

import ipaddress
import re

# Hostnames that resolve to loopback but aren't parseable as IPs.
_LOOPBACK_HOSTNAMES = {"localhost"}


def is_loopback_host(host: str) -> bool:
    """True if binding to `host` only accepts local (loopback) connections.

    ``127.0.0.1`` / ``::1`` (and the ``127.0.0.0/8`` range) and ``localhost``
    are loopback; ``0.0.0.0`` / ``::`` / any routable address are not. This is
    the gate for exposing the HTTP SQL API — bound to loopback, the OS itself
    refuses non-local TCP, so no per-request client check is needed.
    """
    candidate = host.strip().lower()
    if candidate in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # Unknown hostname — fail closed rather than resolving DNS here.
        return False


# Hard cap on rows returned by the MCP `run_sql` tool — kept small because an
# LLM context is the consumer.
MCP_SQL_ROW_CAP = 500
# Hard cap for the local HTTP SQL API — a notebook doing analysis legitimately
# wants more rows than fit in an LLM context, but it's still bounded so a
# runaway query can't stream unbounded results.
API_SQL_ROW_CAP = 10_000

_SQL_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_COMMENT_LINE = re.compile(r"--[^\n]*")
# Only single-quoted strings are SQL literals; double-quoted tokens are
# identifiers and must not be rewritten by the validator.
_SQL_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_SQL_ALLOWED_FIRST_KEYWORDS = {"select", "with"}


def validate_read_only_sql(sql: str) -> str | None:
    """Return an error message if `sql` is not a safe read-only query, else None.

    See the module docstring for why the first-keyword check — not the
    read-only connection — is the real guard.
    """
    stripped = _SQL_COMMENT_BLOCK.sub(" ", sql)
    stripped = _SQL_COMMENT_LINE.sub(" ", stripped)
    # Replace string-literal contents before scanning so a `;` or keyword
    # inside a literal doesn't trip the multi-statement / first-keyword
    # checks. Double-quoted identifiers are intentionally preserved.
    scan = _SQL_STRING_LITERAL.sub("''", stripped)
    scan = scan.strip().rstrip(";").strip()
    if not scan:
        return "SQL is empty."
    if ";" in scan:
        return "Multiple statements are not allowed."
    first_word = scan.split(None, 1)[0].lower()
    if first_word not in _SQL_ALLOWED_FIRST_KEYWORDS:
        return f"Only SELECT / WITH queries are allowed (got: {first_word!r})."
    return None


def clamp_row_limit(limit: int, cap: int) -> int:
    """Clamp a caller-supplied row limit to [1, cap]."""
    return max(1, min(limit, cap))


def wrap_with_row_cap(sql: str, capped_limit: int) -> str:
    """Wrap `sql` so the row cap is applied by the planner, not just by fetch.

    Safe to string-interpolate `capped_limit` because callers pass it through
    :func:`clamp_row_limit` (a clamped int) and `sql` has already passed
    :func:`validate_read_only_sql`.
    """
    inner = sql.strip().rstrip(";").strip()
    return f"SELECT * FROM ({inner}) AS _introspect_q LIMIT {capped_limit}"  # noqa: S608
