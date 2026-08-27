# Security

Introspect reads your entire Claude Code and Codex history into a local
database and then hands two ad-hoc SQL surfaces over it — `POST /api/query`
and the MCP `run_sql` tool. This page describes what those surfaces are
guarded against, how, and what is left over.

## Threat model

In priority order — the ordering matters, because it is what decided where the
effort went.

### 1. Prompt injection through the logs themselves

This is the one that motivated the rest. Session JSONL files are full of
untrusted text: fetched web pages, command output, code from third-party
repositories, MCP tool results. `run_sql` is called by an LLM that reads those
logs. A single line planted in any of them can ask the model to run
`SELECT * FROM read_csv('~/.ssh/id_ed25519')`, or `read_text('.env')`, or
enumerate the disk with `glob('/home/**')` — and the result lands in the
model's context, one `WebFetch` away from leaving the machine.

Before this hardening, all of those worked. `read_only=True` does not stop a
`SELECT` from reading a file, and a first-keyword regex sees nothing wrong with
a query that starts with `SELECT`.

### 2. DNS rebinding from a web page

The server binds to loopback, which stops *remote* clients. It does nothing
about a page the user has open in a browser: that page runs on the user's own
machine, and can either address `127.0.0.1` directly or resolve an
attacker-controlled hostname to it (DNS rebinding). Either way the request
reaches the loopback port from a context the user never authorised.

### 3. Denial of service, deliberate or accidental

`WITH RECURSIVE` with no base case runs until something kills it.
`string_agg` over the whole corpus is one row of hundreds of megabytes.
`LIMIT` bounds neither. This is as often a typo as an attack, and the
consequence is the same.

## The layers

Four layers, listed from the one that actually holds to the one that only
produces a nicer message.

### Engine configuration — the real boundary

`introspect.db.connect_read_hardened` is the single place the codebase opens a
read-only connection to the main database. It opens with resource caps, loads
FTS, disables everything else, and then locks the configuration:

| Setting | Effect |
|---|---|
| `enable_external_access = false` | No `read_csv` / `read_text` / `read_blob` / `read_json` / `read_parquet` / `sniff_csv` / `glob` / `COPY … TO` / `ATTACH`; no HTTP. |
| `allow_community_extensions = false` | No third-party extension code. |
| `allow_unsigned_extensions = false` | No unsigned extension code. |
| `autoinstall_known_extensions = false` | A query cannot trigger an extension download (this is what `sqlite_scan` used to do from inside a `SELECT`). |
| `autoload_known_extensions = false` | A query cannot pull in an extension that happens to be on disk. |
| `allow_persistent_secrets = false` | No stored credentials to read back. |
| `lock_configuration = true` | None of the above can be undone by `SET` or `PRAGMA` for the lifetime of the instance. |

Resource caps (`memory_limit`, `threads`, `max_temp_directory_size`,
`temp_directory`) are applied at connect time rather than as `SET`s, and the
same lock makes them equally irreversible. `max_temp_directory_size` is a
security setting as much as a performance one: without it DuckDB answers a
memory-limit breach by spilling to disk indefinitely instead of failing.

Order is load bearing in two places, both verified against DuckDB 1.5.3:

* **`LOAD fts` must come before `enable_external_access = false`.** Afterwards
  it fails with `PermissionException: Loading external extensions is disabled
  through configuration`, and `INSTALL fts` fails with `PermissionException:
  Cannot access directory ".../.duckdb/extensions/..."`. Re-`LOAD`ing an
  extension the instance has *already* loaded is a no-op that still succeeds
  after the lock, which is what keeps full-text search working.
* **`lock_configuration = true` comes last,** and the factory must be
  idempotent. DuckDB settings are instance-global and concurrent callers join
  the cached instance, so re-issuing the SETs on an already-locked instance
  raises `InvalidInputException: Cannot change configuration option "…" - the
  configuration has been locked`. The factory checks
  `current_setting('lock_configuration')` first and returns immediately when
  it is already true.

This is also why there is exactly one factory rather than one connect call per
caller: DuckDB refuses a second connection to a file whose instance was opened
with a different configuration (`ConnectionException: Can't open a connection
to same database file with a different configuration than existing
connections`). A hardened API request and an unhardened MCP `run_sql` in the
same process would break each other. `tests/e2e/test_sql_hardening.py` parses
`src/` and fails if any module outside `db.py` opens a read-only connection
itself.

The background refresh writes to a separate sidecar file and swaps it in with
`os.replace`, so it is unaffected by any of this.

### HTTP boundary

* `Host` must be `localhost`, `127.0.0.1` or `[::1]` (port stripped before
  matching). A rebinding request arrives carrying the attacker's hostname and
  is refused before it reaches a route — or a database connection. This
  covers the whole app, the `/mcp` mount included.

    This check applies **only when the server is bound to loopback**, which is
    where the rebinding threat lives. `serve --host 0.0.0.0` is a supported way
    to reach the UI from another machine, and no allowlist we could write would
    know which hostnames that user will type — so the check steps aside there,
    as does the MCP transport's equivalent. Nothing is opened up by that:
    a non-loopback bind already disables the SQL API and everything else that
    serves logs over the API. An unset `INTROSPECT_HOST` fails closed *into*
    enforcement, since bare `uvicorn` defaults to `127.0.0.1`.
* Requests to `/api/query`, `/api/schema` and `/mcp` carrying an `Origin`
  whose host is not loopback get a 403. Requests with no `Origin` — curl,
  httpx, notebooks — are unaffected, because browsers always send it
  cross-origin.
* `POST /api/query` additionally requires an `X-Introspect-Client` header.
  Its only job is to force a CORS preflight: a cross-origin `fetch` cannot
  send an unrecognised header without one, and with no CORS middleware
  registered there is no `Access-Control-Allow-*` response to satisfy the
  preflight. That kills the whole drive-by class for the cost of one header.
* **`CORSMiddleware` must never be added to this app**, least of all with
  `allow_origins=["*"]`. The SQL API has no authentication; permissive CORS
  would hand any page the user visits their entire conversation history. A
  test asserts it is absent.
* The MCP transport runs the SDK's own DNS-rebinding protection
  (`TransportSecuritySettings`, `mcp >= 1.10`) with the same loopback host and
  origin allowlists.
* The existing gates still apply: the SQL API is exposed only on a loopback
  bind, and `INTROSPECT_SQL_API=off` disables it outright. See
  [Configuration](configuration.md#sql-api-gating).
* **Error responses never leak internals by default.** A failed request gets
  its real status code and a short message — see
  [Error handling](usage/web-ui.md#error-handling). The traceback, which
  carries file paths, SQL text and DuckDB internals, is written to the server
  log on every unhandled exception but reaches the browser only when
  `INTROSPECT_DEBUG` is truthy **and** the bind is loopback. `devserve` sets
  the variable; `serve` never does, and on `serve --host 0.0.0.0` the variable
  is inert — the same gate, for the same reason, as the SQL API.

### Resource bounds

`introspect.sql_query.execute_bounded` runs every ad-hoc query for both
callers, in a worker thread. That last part is not a detail: `db.execute` is
blocking, and on the event loop one slow query froze the web UI, the MCP
endpoint and the background refresh together.

| Bound | `POST /api/query` | MCP `run_sql` |
|---|---|---|
| Wall clock | 30 s | 20 s |
| Rows | 10 000 | 500 |
| Total output | 8 MB | 64 KB |
| Per cell | 4 000 chars | 200 chars |
| SQL text | 32 KB | 8 KB |

All five limits for one caller live in a single `SqlBudget`
(`MCP_BUDGET` / `API_BUDGET`), so a call site cannot pick up four of them and
forget the fifth.

The timeout is a `threading.Timer` calling `interrupt()` on the connection
that is executing; DuckDB raises `InterruptException`, which surfaces as a
distinct "query timed out" error rather than a generic failure. It is the only
thing that stops a base-case-less `WITH RECURSIVE`, and it fires within about a
second of the deadline. Both timeouts are env-overridable but cannot be
disabled — a non-positive value falls back to the default.

Row limits are pushed into the planner as an outer `LIMIT` rather than applied
at fetch time, and results are read with `fetchmany` so the byte cap can stop a
single enormous row. `LIMIT` alone does not bound output size, and `fetchall`
builds Python objects outside DuckDB's memory accounting.

Memory is bounded by the connection's `memory_limit` (see
[Configuration](configuration.md#resource-limits)). Exceeding it raises
`OutOfMemoryException` cleanly — the process survives — and both callers turn
that into a 400 naming the configured budget.

### Statement validator — defense in depth

`introspect.sql_query.validate_read_only_sql` parses with
`duckdb.extract_statements` and requires exactly one statement of type
`SELECT`. Using DuckDB's own parser rather than a keyword regex fixes four
things at once: comments and string literals stop being the validator's
problem (`SELECT ';' AS x` is one statement; `SELECT 'a;b'; DROP TABLE t` is
two), DuckDB's FROM-first syntax (`FROM t SELECT a`) is correctly a `SELECT`,
`SET`/`PRAGMA`/`INSTALL`/`LOAD`/`ATTACH`/`COPY`/`CALL` each get their own
statement type and cannot hide behind a leading comment, and a syntax error is
reported as one before execution.

A small denylist of file- and network-reading functions (`read_csv`,
`read_text`, `read_blob`, `read_json*`, `read_ndjson*`, `read_parquet`,
`sniff_csv`, `glob`, `parquet_*`, `sqlite_scan`, `postgres_scan`,
`duckdb_secrets`, `duckdb_extensions`) rides along. The engine already refuses
all of them; the denylist exists so the caller gets "`read_csv` is not allowed
— it reads outside the conversation database" instead of a raw
`PermissionException`, and so a future loosening of the engine config fails
loudly. It matches only function *calls* outside string literals, so querying
the logs for the text `read_csv` still works.

Every item in the attack corpus is tested against both layers independently:
once through the validator, and once raw against a real hardened connection
with the validator bypassed.

## What this does not cover

* **Paths are still visible.** `duckdb_settings()` and `current_setting()`
  disclose the database path, the temp directory, and the rest of the engine
  configuration. Nothing is readable through them; they are metadata.
* **`query()` and `query_table()` remain callable.** They inherit the same
  restrictions — `query('INSTALL httpfs')` fails at parse time because the
  argument must be a single `SELECT`, and any `SELECT` it runs hits the same
  locked configuration.
* **`CALL pragma_version()` and `duckdb_secrets()` run.** Both are rejected by
  the validator, and both are harmless if it is bypassed: the first returns the
  DuckDB build string, and the second returns an empty set because
  `allow_persistent_secrets = false` means no secret can exist to list. Tests
  pin that emptiness, so a DuckDB upgrade that changes it will fail CI.
* **A compromised process is still running as you.** Everything here bounds
  what *SQL* can do. It does nothing about arbitrary code execution in the
  Python process, which has your file permissions by construction. OS-level
  sandboxing (seccomp, a container, a separate uid) is the only answer beyond
  this point, and it is out of scope for a tool you run on your own laptop
  against your own logs.
* **There is no authentication.** Anything that can already reach loopback on
  that port *and* is not a browser can read your conversation logs. That is the
  design: the guards above are aimed at browsers and at injected SQL, not at
  other processes running as you.

## Upgrade policy

The engine configuration is only as good as the DuckDB release enforcing it,
so treat DuckDB security advisories as upgrade-now, not upgrade-eventually.

One past advisory is worth naming because it shows the failure mode: in older
versions `sniff_csv` bypassed `enable_external_access` and read files anyway.
It is fixed in 1.5.3, and `sniff_csv` stays in both the attack corpus and the
denylist regardless — a regression there would otherwise be silent.

Run `uv run pytest tests/e2e/test_sql_hardening.py` after any DuckDB upgrade.
It is the check that says whether the boundary still holds.
