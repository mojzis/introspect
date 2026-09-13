# Functional QA: standalone MCP startup

Run the isolated synthetic consumer from the repository root:

```bash
uv run python scripts/qa_mcp_startup.py
```

The script spawns real `introspy mcp` stdio processes for cold startup, warm
startup, terminal startup and directory-preparation failures, unlimited history (`INTROSPECT_DAYS=0`),
and a larger synthetic transcript containing 10,000 messages. It performs
`initialize`, `tools/list`, `recent_sessions`, and `refresh_data` calls, then
prints measured timings and observed results as JSON. Handshake and discovery
must complete within five seconds in each successful scenario; the failure
scenarios verify discovery but do not measure its latency. The route has a
two-minute overall timeout.

Cold and warm runs must expose partial data before publishing complete,
unmarked results. Unlimited cold startup must publish complete, ready results
directly. Every successful run disables periodic refresh with interval `0`;
a later request to change the window must report manual refresh unavailable
and retain the startup target. Both failure cases require `Data unavailable`
and the startup-failure detail in data and refresh responses; this route does
not assert the specific underlying filesystem error text.

All inputs and databases are disposable synthetic files, removed afterward.
No pytest, conversation logs, shared `~/.introspect` database, or real
home-directory glob is used as evidence.

# Functional QA: Pycoati accepted findings

Run these commands from the repository root. They use the locked environment
and write only to the explicitly named disposable files under `/tmp`:

```bash
uv run --locked pycoati --version
uv run --locked pycoati . --no-accept --output /tmp/introspect-pycoati-raw.json
uv run --locked pycoati . --output /tmp/introspect-pycoati-actionable.json
uv run --locked pycoati . --include-accepted --output /tmp/introspect-pycoati-full.json
uv run --locked python scripts/qa_pycoati_acceptance.py
```

The version command must report `pycoati 0.2.9`. The three repository scans
must report `tool.ran_pytest: true` and `tool.ran_coverage: true`; compare
`suite.test_count`, `suite.line_coverage_pct`, all raw test/file records, and
scores across them. Natural runtime and `slowest_tests` ordering can vary.
`--no-accept` and `--include-accepted` retain the accepted test in the
shortlist; the default scan filters it while retaining the same raw counts and
scores. Inspect stderr for pytest/coverage and stale-acceptance warnings.

The synthetic consumer must print one JSON object with `status: "pass"`,
Pycoati 0.2.9, pytest and coverage enabled, checked subprocess recognition
counts of 1 for `check_call`, `check_output`, `run(check=True)`, and
`check_returncode`, and 0 for unchecked `run`, `run(check=False)`, a replaced
subprocess name, and checked calls swallowed by `try`/`except` or
`contextlib.suppress`. It also proves a failing checked child reaches pytest,
keeps one distinct active mock signal actionable, and observes exactly one each
of `unknown_test`, `signal_not_active`, and `content_changed` stale states. Its temporary project
and all child processes self-clean; it does not read conversation logs,
personal databases, or shared services.

If the scan output or consumer fails, preserve stderr and the JSON for
diagnosis. Clean only the named disposable scan files afterward:

```bash
rm -f /tmp/introspect-pycoati-raw.json \
  /tmp/introspect-pycoati-actionable.json \
  /tmp/introspect-pycoati-full.json
```
