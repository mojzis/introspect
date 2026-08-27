"""End-to-end guards for the ad-hoc SQL surface.

Every attack in ``ATTACK_CORPUS`` is checked twice, deliberately:

* against :func:`validate_read_only_sql`, the readable-error layer, and
* against a real hardened connection with the validator bypassed, which is
  what actually holds if the validator is ever weakened or side-stepped.

See ``docs/security.md`` for the threat model these correspond to.
"""

from __future__ import annotations

import ast
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import duckdb
import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from introspect.api.main import (
    ALLOWED_HOSTS,
    CLIENT_HEADER,
    app,
    host_allowlist_applies,
)
from introspect.db import (
    _read_config,
    connect_read_hardened,
    materialize_views,
)
from introspect.search import build_search_corpus
from introspect.sql_query import (
    API_BUDGET,
    MCP_SQL_CELL_CAP,
    SqlBudget,
    SqlTimeoutError,
    execute_bounded,
    validate_read_only_sql,
    wrap_with_row_cap,
)

from ..conftest import (
    glob_pattern,
    local_client,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def budget(**overrides) -> SqlBudget:
    """An ``API_BUDGET`` with the limits under test narrowed.

    Starting from the real budget rather than hand-building one keeps these
    tests honest about the defaults; only the limit a test is exercising is
    overridden. ``timeout_s`` defaults low so a bug cannot stall the suite
    for 30 s.
    """
    return replace(API_BUDGET, **{"timeout_s": 10.0, **overrides})


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def hardened_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A materialized DB with the FTS index built, ready to open hardened.

    Module-scoped: materializing is the slow part and every test here treats
    the DB as read-only. Sharing it also matches production, where one DuckDB
    instance is shared by every connection in the process.
    """
    tmp_path = tmp_path_factory.mktemp("hardened")
    write_jsonl(
        tmp_path,
        SID,
        [
            make_user_message(
                SID, "u1", None, "2026-03-27T10:00:00.000Z", "index this haystack"
            ),
            make_assistant_message(
                SID,
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "a needle in the haystack"}],
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
            make_assistant_message(
                SID,
                "a2",
                "a1",
                "2026-03-27T10:00:02.000Z",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_hard1",
                        "name": "Bash",
                        "input": {"command": "echo hi", "description": "probe"},
                    }
                ],
            ),
            # A tool_result carrying toolUseResult: materialize_views binds
            # that column, so a fixture without one fails to load at all.
            make_user_message(
                SID,
                "u2",
                "a2",
                "2026-03-27T10:00:03.000Z",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_hard1",
                        "content": "hi\n",
                        "is_error": False,
                    }
                ],
                tool_use_result={"stdout": "hi\n", "stderr": ""},
                source_tool_uuid="a2",
            ),
        ],
    )
    db_path = tmp_path / "hardened.duckdb"
    conn = duckdb.connect(str(db_path))
    materialize_views(conn, glob_pattern(tmp_path), 0, resolve_projects=False)
    build_search_corpus(conn)
    conn.close()
    return db_path


@pytest.fixture
def hardened_conn(hardened_db: Path):
    """A hardened read-only connection to ``hardened_db``."""
    conn = connect_read_hardened(hardened_db)
    try:
        yield conn
    finally:
        conn.close()


# --- the attack corpus --------------------------------------------------------


class Attack(NamedTuple):
    """One hostile query and which layers are expected to stop it."""

    label: str
    sql: str
    #: The readable-error layer rejects it before execution.
    validator_blocks: bool
    #: The hardened engine refuses it even with the validator bypassed.
    engine_blocks: bool
    #: Why the engine lets it through, when it does.
    engine_note: str = ""


# Everything that reaches outside the database, plus every non-SELECT
# statement type. Two entries are engine_blocks=False on purpose: they are
# *harmless* on a hardened instance rather than permitted escapes, and saying
# so here is more useful than pretending the engine covers them.
ATTACK_CORPUS: list[Attack] = [
    Attack(
        "read_csv arbitrary file",
        "SELECT * FROM read_csv('/etc/passwd', delim=':', header=false)",
        True,
        True,
    ),
    Attack(
        "glob filesystem enumeration",
        "SELECT count(*) FROM glob('/home/**')",
        True,
        True,
    ),
    Attack("read_text", "SELECT * FROM read_text('/etc/hostname')", True, True),
    Attack("read_blob", "SELECT * FROM read_blob('/etc/hostname')", True, True),
    Attack("sniff_csv", "SELECT * FROM sniff_csv('/etc/passwd')", True, True),
    Attack(
        "parquet_metadata",
        "SELECT * FROM parquet_metadata('/tmp/x.parquet')",
        True,
        True,
    ),
    Attack(
        "read_json over http",
        "SELECT * FROM read_json('https://example.com/a.json')",
        True,
        True,
    ),
    Attack("read_ndjson", "SELECT * FROM read_ndjson('/etc/passwd')", True, True),
    Attack("read_parquet", "SELECT * FROM read_parquet('/tmp/x.parquet')", True, True),
    Attack(
        "sqlite_scan extension pull",
        "SELECT * FROM sqlite_scan('/tmp/x.db', 't')",
        True,
        True,
    ),
    Attack(
        "postgres_scan", "SELECT * FROM postgres_scan('', 'public', 't')", True, True
    ),
    Attack("duckdb_extensions", "SELECT * FROM duckdb_extensions()", True, True),
    Attack("SET memory_limit", "SET memory_limit='16GB'", True, True),
    Attack("PRAGMA memory_limit", "PRAGMA memory_limit='16GB'", True, True),
    Attack("INSTALL", "INSTALL httpfs", True, True),
    Attack("LOAD httpfs", "LOAD httpfs", True, True),
    Attack("ATTACH", "ATTACH '/tmp/evil.db' AS evil", True, True),
    Attack("COPY TO", "COPY (SELECT 1) TO '/tmp/out.csv'", True, True),
    Attack("DELETE", "DELETE FROM logical_sessions", True, True),
    Attack("CREATE TABLE", "CREATE TABLE evil AS SELECT 1", True, True),
    Attack(
        "comment-hidden second statement",
        "SELECT 1 /* x */; DROP TABLE raw_data",
        True,
        True,
    ),
    Attack(
        "semicolon in string then DROP", "SELECT 'a;b'; DROP TABLE raw_data", True, True
    ),
    # Legal single SELECTs that only the engine configuration stops — the
    # cases proving the validator is not the boundary.
    Attack("bare csv path in FROM", "SELECT * FROM 'file.csv'", False, True),
    Attack(
        "query() smuggling INSTALL",
        "SELECT * FROM query('INSTALL httpfs')",
        False,
        True,
    ),
    # Rejected by statement type / denylist only. Verified on DuckDB 1.5.3:
    # the engine runs both and neither discloses anything — duckdb_secrets()
    # is empty because allow_persistent_secrets=false means none can exist,
    # and pragma_version() returns the DuckDB build string.
    Attack(
        "duckdb_secrets",
        "SELECT * FROM duckdb_secrets()",
        True,
        False,
        "returns an empty set; allow_persistent_secrets=false leaves nothing to list",
    ),
    Attack(
        "CALL",
        "CALL pragma_version()",
        True,
        False,
        "returns the DuckDB version string, which the /about page already shows",
    ),
]

# Every function name the denylist is expected to name in its error message,
# paired with a SELECT that uses it. ``engine_blocks`` mirrors ``Attack``.
DENYLISTED_CALLS: list[Attack] = [
    Attack("read_csv", "SELECT * FROM read_csv('/etc/passwd')", True, True),
    Attack("read_text", "SELECT * FROM read_text('/etc/passwd')", True, True),
    Attack("read_blob", "SELECT * FROM read_blob('/etc/passwd')", True, True),
    Attack("read_json", "SELECT * FROM read_json('/etc/passwd')", True, True),
    Attack("read_json_auto", "SELECT * FROM read_json_auto('/etc/passwd')", True, True),
    Attack("read_ndjson", "SELECT * FROM read_ndjson('/etc/passwd')", True, True),
    Attack("read_parquet", "SELECT * FROM read_parquet('/tmp/x.parquet')", True, True),
    Attack("sniff_csv", "SELECT * FROM sniff_csv('/etc/passwd')", True, True),
    Attack("glob", "SELECT * FROM glob('/home/**')", True, True),
    Attack(
        "parquet_metadata",
        "SELECT * FROM parquet_metadata('/tmp/x.parquet')",
        True,
        True,
    ),
    Attack("sqlite_scan", "SELECT * FROM sqlite_scan('/tmp/x.db', 't')", True, True),
    Attack(
        "postgres_scan", "SELECT * FROM postgres_scan('', 'public', 't')", True, True
    ),
    Attack("duckdb_extensions", "SELECT * FROM duckdb_extensions()", True, True),
    Attack(
        "duckdb_secrets",
        "SELECT * FROM duckdb_secrets()",
        True,
        False,
        "returns an empty set; allow_persistent_secrets=false leaves nothing to list",
    ),
]

ENGINE_BLOCKED = [a for a in ATTACK_CORPUS if a.engine_blocks]


@pytest.mark.parametrize("attack", ATTACK_CORPUS, ids=[a.label for a in ATTACK_CORPUS])
def test_validator_rejects_attack(attack: Attack):
    """The readable-error layer catches everything it claims to."""
    error = validate_read_only_sql(attack.sql)
    if attack.validator_blocks:
        assert error is not None, f"{attack.label}: validator let it through"
    else:
        assert error is None, (
            f"{attack.label}: expected the engine, not the validator, to block this"
        )


@pytest.mark.parametrize(
    "attack", ENGINE_BLOCKED, ids=[a.label for a in ENGINE_BLOCKED]
)
def test_engine_rejects_attack(hardened_conn, attack: Attack):
    """The same corpus run raw against the engine, validator bypassed.

    This is the layer that matters: if the validator were removed tomorrow,
    every one of these would still fail.
    """
    with pytest.raises(duckdb.Error):
        hardened_conn.execute(attack.sql).fetchall()


ENGINE_PERMITTED = [a for a in ATTACK_CORPUS if not a.engine_blocks]


@pytest.mark.parametrize(
    "attack", ENGINE_PERMITTED, ids=[a.label for a in ENGINE_PERMITTED]
)
def test_engine_permitted_attacks_are_harmless(hardened_conn, attack: Attack):
    """The two the engine runs anyway disclose nothing.

    Pinning this keeps the docs honest: if a DuckDB upgrade ever makes
    ``duckdb_secrets()`` return rows on a hardened instance, this fails.
    """
    assert attack.engine_note, "an engine-permitted attack must say why it is safe"
    rows = hardened_conn.execute(attack.sql).fetchall()
    if attack.label == "duckdb_secrets":
        # allow_persistent_secrets=false means there is nothing to list.
        assert rows == []
    else:
        # pragma_version() — one row of DuckDB build metadata, no user data.
        assert len(rows) == 1
        assert rows[0][0].startswith("v")


@pytest.mark.parametrize(
    "attack", DENYLISTED_CALLS, ids=[a.label for a in DENYLISTED_CALLS]
)
def test_denylisted_function_is_named_in_the_error(attack: Attack):
    """The denylist explains itself instead of leaking a DuckDB exception."""
    error = validate_read_only_sql(attack.sql)
    assert error is not None
    assert attack.label in error
    assert "not allowed" in error


DENYLISTED_ENGINE_BLOCKED = [a for a in DENYLISTED_CALLS if a.engine_blocks]


@pytest.mark.parametrize(
    "attack",
    DENYLISTED_ENGINE_BLOCKED,
    ids=[a.label for a in DENYLISTED_ENGINE_BLOCKED],
)
def test_denylisted_function_is_independently_blocked_by_the_engine(
    hardened_conn, attack: Attack
):
    """Each denylist entry is refused by the engine with the validator bypassed."""
    with pytest.raises(duckdb.Error):
        hardened_conn.execute(attack.sql).fetchall()


def test_denylist_does_not_reject_the_name_inside_a_string_literal(hardened_conn):
    """Searching the logs for the text 'read_csv' is a legitimate query."""
    sql = "SELECT 'read_csv(...)' AS mention WHERE 'a' ILIKE '%read_csv%'"
    assert validate_read_only_sql(sql) is None
    assert hardened_conn.execute(sql).fetchall() == []


# --- happy path ---------------------------------------------------------------

HAPPY_PATH_SQL = [
    ("plain select", "SELECT 1 AS n"),
    ("from-first", "SELECT n FROM (SELECT 1 AS n) t"),
    ("duckdb from-first syntax", "FROM logical_sessions SELECT session_id"),
    ("with", "WITH c AS (SELECT 1 AS n) SELECT n FROM c"),
    ("trailing line comment", "SELECT 1 AS n -- a trailing note"),
    ("block comment", "SELECT 1 /* inline */ AS n"),
    ("semicolon inside literal", "SELECT ';' AS punctuation"),
    ("trailing semicolon", "SELECT 1 AS n;"),
    ("describe", "DESCRIBE logical_sessions"),
]


@pytest.mark.parametrize(
    ("label", "sql"), HAPPY_PATH_SQL, ids=[case[0] for case in HAPPY_PATH_SQL]
)
def test_happy_path_queries_validate_and_run(hardened_conn, label: str, sql: str):
    """Legitimate read-only SQL survives both the validator and the engine."""
    del label
    assert validate_read_only_sql(sql) is None
    result = execute_bounded(hardened_conn, sql, budget(row_cap=10))
    assert result.columns


def test_wrapper_survives_a_trailing_line_comment():
    """Regression: the closing paren must not land inside a ``--`` comment."""
    wrapped = wrap_with_row_cap("SELECT 1 AS n -- note", 5)
    assert duckdb.connect(":memory:").execute(wrapped).fetchall() == [(1,)]


# --- resource bounds ----------------------------------------------------------


def test_runaway_recursive_cte_times_out_instead_of_hanging(hardened_conn):
    """``WITH RECURSIVE`` with no base case must be interrupted, not survived."""
    sql = (
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) "
        "SELECT count(*) FROM r"
    )
    assert validate_read_only_sql(sql) is None
    started = time.monotonic()
    with pytest.raises(SqlTimeoutError):
        execute_bounded(hardened_conn, sql, budget(timeout_s=2))
    assert time.monotonic() - started < 10


def test_connection_is_reusable_after_a_timeout(hardened_conn):
    """An interrupt must not poison the connection for the next query."""
    with pytest.raises(SqlTimeoutError):
        execute_bounded(
            hardened_conn,
            "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) "
            "SELECT count(*) FROM r",
            budget(timeout_s=1),
        )
    assert hardened_conn.execute("SELECT 42").fetchone() == (42,)


def test_byte_cap_stops_a_single_enormous_row(hardened_conn):
    """``LIMIT`` cannot bound output size; the byte cap has to."""
    result = execute_bounded(
        hardened_conn,
        "SELECT repeat('x', 100000) AS big",
        budget(byte_cap=1024, cell_cap=500),
    )
    assert len(result.rows[0][0]) < 1000


def test_row_cap_marks_the_result_truncated(hardened_conn):
    """Hitting the row cap sets ``truncated`` and names the cap."""
    result = execute_bounded(
        hardened_conn, "SELECT * FROM range(0, 100) AS t(n)", budget(row_cap=5)
    )
    assert len(result.rows) == 5
    assert result.truncated
    assert "row cap" in (result.truncation_reason or "")


def test_cell_cap_clips_wide_values(hardened_conn):
    """A single wide cell is clipped with a visible marker."""
    result = execute_bounded(
        hardened_conn,
        f"SELECT repeat('y', {MCP_SQL_CELL_CAP * 3}) AS wide",
        budget(cell_cap=MCP_SQL_CELL_CAP),
    )
    value = result.rows[0][0]
    assert value.startswith("y" * MCP_SQL_CELL_CAP)
    assert "truncated" in value


def test_out_of_memory_raises_without_killing_the_process():
    """``memory_limit`` must produce an exception, not a dead interpreter.

    Run against a throwaway in-memory instance: the point is DuckDB's
    behaviour at the limit, and a separate instance keeps this from fighting
    the shared-config rule the rest of the file relies on.

    ``max_temp_directory_size`` is part of the shape being tested, not test
    tuning. Without it DuckDB spills to disk indefinitely and this query
    grinds for minutes instead of failing — which is exactly why the real
    factory caps the temp directory alongside memory.
    """
    conn = duckdb.connect(
        ":memory:",
        config={
            "memory_limit": "256MB",
            "threads": 2,
            "max_temp_directory_size": "64MB",
        },
    )
    try:
        with pytest.raises(duckdb.OutOfMemoryException):
            conn.execute(
                "SELECT count(*) FROM ("
                "  SELECT range AS a, repeat('x', 4000) AS b FROM range(50000000)"
                ") GROUP BY b, a"
            ).fetchall()
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()


def test_sql_length_cap_rejects_oversized_input():
    """A megabyte of generated SQL is refused before it reaches the parser."""
    error = validate_read_only_sql("SELECT " + "1," * 20_000 + "1", max_bytes=1024)
    assert error is not None
    assert "too long" in error


# --- connection factory -------------------------------------------------------


def test_hardened_instance_config_is_shared_not_conflicting(hardened_db: Path):
    """An API-style and an MCP-style connection coexist in one process.

    Regression for ``ConnectionException: Can't open a connection to same
    database file with a different configuration than existing connections``,
    which is what a second ``duckdb.connect`` without the shared config would
    raise while the first is open.
    """
    api_conn = connect_read_hardened(hardened_db)
    try:
        mcp_conn = connect_read_hardened(hardened_db)
        try:
            assert mcp_conn.execute("SELECT 1").fetchone() == (1,)
            assert api_conn.execute("SELECT 1").fetchone() == (1,)
        finally:
            mcp_conn.close()
    finally:
        api_conn.close()


def test_reconnecting_to_a_locked_instance_is_idempotent(hardened_db: Path):
    """Re-running the factory on a locked instance must not raise.

    Re-issuing the SETs would fail with ``InvalidInputException: Cannot change
    configuration option "..." - the configuration has been locked``.
    """
    first = connect_read_hardened(hardened_db)
    try:
        for _ in range(3):
            conn = connect_read_hardened(hardened_db)
            try:
                assert conn.execute(
                    "SELECT current_setting('lock_configuration')"
                ).fetchone() == (True,)
            finally:
                conn.close()
    finally:
        first.close()


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        ("INTROSPECT_DB_THREADS", "abc"),
        ("INTROSPECT_DB_THREADS", "0"),
        ("INTROSPECT_DB_MEMORY_LIMIT", "not-a-size"),
    ],
)
def test_a_malformed_resource_override_falls_back_instead_of_500ing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
):
    """A typo in these must not turn every page into a 500.

    They are applied in the request middleware, so an unhandled
    ``InvalidInputException`` here would break the whole UI with nothing in
    the log naming the variable.
    """
    monkeypatch.setitem(os.environ, env_var, bad_value)
    _read_config.cache_clear()
    db = tmp_path / "override.duckdb"
    duckdb.connect(str(db)).close()
    conn = connect_read_hardened(db)
    try:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()
        _read_config.cache_clear()


def test_hardened_connection_is_locked_and_external_access_is_off(hardened_conn):
    """The two settings the whole design rests on are actually in force."""
    assert hardened_conn.execute(
        "SELECT current_setting('lock_configuration')"
    ).fetchone() == (True,)
    assert hardened_conn.execute(
        "SELECT current_setting('enable_external_access')"
    ).fetchone() == (False,)


def test_fts_still_works_on_a_hardened_connection(hardened_conn):
    """LOAD-before-lock keeps BM25 search alive with external access off.

    Skips on the index, not on ``fts_available``. Those are different facts:
    the session-scoped ``_prewarm_fts_cache`` fixture may have decided FTS was
    unavailable (so ``build_search_corpus`` never created the index) on a
    machine where the extension nonetheless loads. Asking ``fts_available``
    here would then say "yes" and the query would die on a missing index —
    which is exactly how this failed in CI. Reading the cache is also not
    free: clearing it re-enables the ~80s offline INSTALL probe the fixture
    exists to avoid, for this test and every one after it in the process.
    """
    index_exists = hardened_conn.execute(
        "SELECT count(*) FROM information_schema.schemata "
        "WHERE schema_name = 'fts_main_search_corpus'"
    ).fetchone()
    if not (index_exists and index_exists[0]):
        pytest.skip("no BM25 index on search_corpus (FTS unavailable at build time)")
    rows = hardened_conn.execute(
        "SELECT fts_main_search_corpus.match_bm25(rowid, 'needle') AS score "
        "FROM search_corpus"
    ).fetchall()
    assert any(row[0] is not None for row in rows)


def test_fts_install_is_attempted_at_most_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing extension must not cost a network INSTALL on every request.

    Connections are per-request and DuckDB drops an instance when its last
    connection closes, so ``_load_fts`` runs again on every request. Without
    the once-per-process latch, an offline machine with no FTS extension
    would pay the DNS timeout on each page load.
    """
    from introspect import db as db_module  # noqa: PLC0415

    installs: list[str] = []

    def no_extension_on_disk(conn):
        return False

    real_execute = duckdb.DuckDBPyConnection.execute
    offline = duckdb.IOException("simulated: no network")

    def counting_execute(self, sql, *args, **kwargs):
        if isinstance(sql, str) and sql.strip().upper().startswith("INSTALL"):
            installs.append(sql)
            raise offline
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(db_module, "_try_load_fts", no_extension_on_disk)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", counting_execute)
    monkeypatch.setattr(db_module, "_fts_install_failed", [False])

    db = tmp_path / "no_fts.duckdb"
    duckdb.connect(str(db)).close()
    for _ in range(3):
        connect_read_hardened(db).close()

    assert len(installs) == 1, f"INSTALL ran {len(installs)} times, want 1"


def _read_only_connect_sites(path: Path) -> list[int]:
    """Line numbers of ``duckdb.connect(..., read_only=...)`` calls in `path`.

    Parsed rather than grepped so a module that merely *discusses*
    ``read_only=True`` in a docstring isn't reported as an offender.
    """
    tree = ast.parse(path.read_text())
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and any(kw.arg == "read_only" for kw in node.keywords)
    ]


def test_only_the_factory_opens_the_main_db_read_only():
    """Grep guard: no module may hand-roll a read-only connect.

    A second ``duckdb.connect(..., read_only=True)`` on the main DB would
    either skip the hardening or collide with the shared instance config
    (``ConnectionException: Can't open a connection to same database file
    with a different configuration than existing connections``).
    """
    src = Path(__file__).resolve().parents[2] / "src" / "introspect"
    offenders = {
        str(path.relative_to(src)): sites
        for path in sorted(src.rglob("*.py"))
        if path.name != "db.py" and (sites := _read_only_connect_sites(path))
    }
    assert not offenders, (
        "These modules open a read-only DuckDB connection directly instead of "
        f"calling db.connect_read_hardened(): {offenders}"
    )


def test_db_module_only_connects_read_only_inside_the_factory():
    """``db.py``'s own single read-only connect lives in the factory."""
    db_path = Path(__file__).resolve().parents[2] / "src" / "introspect" / "db.py"
    sites = _read_only_connect_sites(db_path)
    assert len(sites) == 1, f"expected exactly one read-only connect, found {sites}"

    tree = ast.parse(db_path.read_text())
    factory = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "connect_read_hardened"
    )
    assert factory.lineno < sites[0] < (factory.end_lineno or 0)


# --- HTTP boundary ------------------------------------------------------------


def _patch_app_env(monkeypatch: pytest.MonkeyPatch, hardened_db: Path) -> None:
    """Point the app at ``hardened_db`` and nothing on the host machine."""
    monkeypatch.setitem(os.environ, "INTROSPECT_DB_PATH", str(hardened_db))
    monkeypatch.setitem(
        os.environ, "INTROSPECT_JSONL_GLOB", glob_pattern(hardened_db.parent)
    )
    monkeypatch.setitem(
        os.environ,
        "INTROSPECT_CODEX_GLOB",
        str(hardened_db.parent / "codex" / "**" / "*.jsonl"),
    )
    monkeypatch.setitem(os.environ, "INTROSPECT_DAYS", "0")
    # Keep the responsiveness probe below from waiting out the 30 s default.
    monkeypatch.setitem(os.environ, "INTROSPECT_SQL_TIMEOUT_SECONDS", "3")


@pytest.fixture
def api_client(hardened_db: Path, monkeypatch: pytest.MonkeyPatch):
    """A TestClient with the SQL API enabled against ``hardened_db``."""
    _patch_app_env(monkeypatch, hardened_db)
    monkeypatch.setitem(os.environ, "INTROSPECT_HOST", "127.0.0.1")
    with local_client(app) as client:
        yield client


def test_query_happy_path_through_http(api_client):
    """The guarded endpoint still answers a normal request."""
    resp = api_client.post(
        "/api/query", json={"sql": "SELECT 1 AS n"}, headers={CLIENT_HEADER: "1"}
    )
    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1]]


def test_rebinding_host_is_rejected(api_client):
    """A Host header naming an attacker domain never reaches a route."""
    resp = api_client.post(
        "/api/query",
        json={"sql": "SELECT 1 AS n"},
        headers={CLIENT_HEADER: "1", "Host": "evil.example"},
    )
    assert resp.status_code in (400, 403)


def test_rebinding_host_is_rejected_on_the_mcp_mount(api_client):
    """The host guard covers the /mcp sub-app too."""
    resp = api_client.post("/mcp", json={}, headers={"Host": "evil.example"})
    assert resp.status_code in (400, 403)


def test_cross_origin_request_is_rejected(api_client):
    """A page on another origin is refused even though it can reach loopback."""
    resp = api_client.post(
        "/api/query",
        json={"sql": "SELECT 1 AS n"},
        headers={CLIENT_HEADER: "1", "Origin": "https://evil.example"},
    )
    assert resp.status_code == 403


def test_cross_origin_request_is_rejected_on_schema(api_client):
    """The Origin guard covers /api/schema, not just /api/query."""
    resp = api_client.get("/api/schema", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 403


def test_loopback_origin_is_accepted(api_client):
    """The web UI's own fetches carry a loopback Origin and must pass."""
    resp = api_client.post(
        "/api/query",
        json={"sql": "SELECT 1 AS n"},
        headers={CLIENT_HEADER: "1", "Origin": "http://127.0.0.1:8347"},
    )
    assert resp.status_code == 200


def test_missing_client_header_is_rejected(api_client):
    """Without the custom header a drive-by POST never gets a preflight."""
    resp = api_client.post("/api/query", json={"sql": "SELECT 1 AS n"})
    assert resp.status_code == 403
    assert CLIENT_HEADER in resp.json()["error"]


def test_no_cors_middleware_is_registered():
    """Permissive CORS would hand the whole log corpus to any web page."""
    registered = [m.cls for m in app.user_middleware]
    assert CORSMiddleware not in registered


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.0.0.1:8347",
        "localhost",
        "localhost:8347",
        "[::1]",
        "[::1]:8347",
    ],
)
def test_every_loopback_host_spelling_is_accepted(api_client, host: str):
    """Including bracketed IPv6, where a naive port split yields ``"["``."""
    assert api_client.get("/", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8347", "", "127.0.0.1.evil.example"]
)
def test_non_loopback_host_spellings_are_rejected(api_client, host: str):
    """Anything not on the allowlist is refused before it reaches a route."""
    assert api_client.get("/", headers={"Host": host}).status_code == 400


def test_allowed_hosts_are_loopback_only():
    """The Host allowlist must never grow a routable name."""
    assert set(ALLOWED_HOSTS) == {"localhost", "127.0.0.1", "[::1]"}


@pytest.mark.parametrize("bind_host", ["", "127.0.0.1", "localhost", "::1"])
def test_host_allowlist_applies_to_a_loopback_bind(bind_host: str):
    """Loopback — and an unknown bind, which fails closed — are enforced."""
    assert host_allowlist_applies(bind_host)


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "::", "192.168.1.50"])
def test_host_allowlist_is_skipped_for_a_deliberate_lan_bind(bind_host: str):
    """``serve --host 0.0.0.0`` is the user opting into remote access."""
    assert not host_allowlist_applies(bind_host)


@pytest.fixture
def lan_client(hardened_db: Path, monkeypatch: pytest.MonkeyPatch):
    """A TestClient for a server the user deliberately bound to 0.0.0.0."""
    _patch_app_env(monkeypatch, hardened_db)
    monkeypatch.setitem(os.environ, "INTROSPECT_HOST", "0.0.0.0")
    with TestClient(app, base_url="http://192.168.1.50:8347") as client:
        yield client


def test_lan_bound_server_serves_a_lan_host(lan_client):
    """Regression: the Host allowlist must not 400 the whole UI off-loopback.

    ``serve --host 0.0.0.0`` is supported, and ``_configure_sql_api``'s
    non-loopback branch exists precisely so the UI keeps working there with
    only the SQL API disabled.
    """
    assert lan_client.get("/").status_code == 200
    assert lan_client.get("/sessions").status_code == 200


def test_lan_bound_server_still_disables_the_sql_api(lan_client):
    """Skipping the Host check does not open the endpoint that reads logs."""
    resp = lan_client.post(
        "/api/query", json={"sql": "SELECT 1 AS n"}, headers={CLIENT_HEADER: "1"}
    )
    assert resp.status_code == 404


def test_event_loop_stays_responsive_during_a_slow_query(api_client):
    """A blocking query must not freeze the UI, MCP and refresh with it.

    ``run_query`` hands DuckDB to a worker thread precisely so this holds;
    before that, ``db.execute`` on the event loop stalled every other request.
    """
    slow_sql = (
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) "
        "SELECT count(*) FROM r"
    )
    latencies: list[float] = []

    def hammer() -> None:
        # Give the slow query a moment to actually start executing.
        time.sleep(0.5)
        started = time.monotonic()
        api_client.get("/")
        latencies.append(time.monotonic() - started)

    prober = threading.Thread(target=hammer)
    prober.start()
    try:
        api_client.post(
            "/api/query",
            json={"sql": slow_sql},
            headers={CLIENT_HEADER: "1"},
        )
    finally:
        prober.join()

    assert latencies and latencies[0] < 3.0, (
        f"GET / took {latencies[0]:.2f}s while a query ran — the event loop is blocked."
    )
