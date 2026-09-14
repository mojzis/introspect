# Pycoati 0.2.9 audit

Audit date: 2026-09-14. Base: `559573f324fbb308a1d7eabd7c81e84e81da938b`.
Tested implementation revision: `7a048ac4ccb95006d7d6666fdb4df89ccdca2123`.

## Reproducibility

The dependency floor moved from Pycoati 0.2.7 to 0.2.9. The locked
installation and lock validation passed:

```text
uv sync --locked
uv lock --check
uv run --locked pycoati --version  # pycoati 0.2.9
```

The final locked audits were run from the repository root:

```text
uv run --locked pycoati --version
uv run --locked pycoati . --no-accept --output /tmp/introspect-pycoati-raw.json
uv run --locked pycoati . --output /tmp/introspect-pycoati-actionable.json
uv run --locked pycoati . --include-accepted --output /tmp/introspect-pycoati-full.json
uv run --locked python scripts/qa_pycoati_acceptance.py
```

Each audit reported `tool.ran_pytest: true`, `tool.ran_coverage: true`, 1,132
collected tests, and 92.167% line coverage. The raw, default, and
`--include-accepted` records retained the same measured counts, files, test
records, scores, and coverage. Acceptance metadata and the expected shortlist
membership were the only policy differences; runtime and slow-test ordering
naturally varied (100.91s raw, 86.89s default, 77.91s include-accepted).

| Mode | Accepted | Stale | Shortlist | Result |
|---|---:|---:|---:|---|
| `--no-accept` | 0 | 0 | 20 raw findings | authoritative raw inventory |
| default | 7 | 0 | 20 actionable findings | accepted tests filtered |
| `--include-accepted` | 7 | 0 | 20 full findings | accepted tests restored |

The synthetic consumer printed `status: "pass"`. It verified checked
subprocess recognition (count 1 for `check_call`, `check_output`, `run` with
`check=True`, and `check_returncode`; count 0 for unchecked and
`check=False` runs, a replaced subprocess name, and checked calls swallowed by
`try`/`except` or `contextlib.suppress`), propagated a failing child into
pytest, and exercised one each of `unknown_test`, `signal_not_active`, and
`content_changed`. A distinct mock-only signal remained actionable. The
consumer uses only temporary synthetic projects and self-cleans them.

## Reviewed baseline

The seven accepted findings are all `zero_asserts`, with exact nodeids,
current fingerprints, review date `2026-09-13`, and project-specific reasons
in `.pycoati-accept.toml`:

| Test | Rationale |
|---|---|
| `test_every_cli_command_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_env_var_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_relation_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_mcp_tool_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_mcp_prompt_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_query_template_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |
| `test_every_route_is_documented` | Missing documentation reaches `pytest.fail` through the shared helper. |

No tests were changed: the reviewed signals are intentional verification
patterns, not confirmed gaps. The integrated `main` revision added an explicit
assertion to `test_finish_connected_session_leaves_existing_server_alone`, so
that test no longer needs an acceptance entry. The raw actionable inventory still includes
`tests/e2e/test_sql_hardening.py::test_fts_install_is_attempted_at_most_once_per_process`
(`mock_overuse`, 3 stubs, 1 assertion, setup ratio 37.0) and the remaining
ranked candidates. It remains visible and actionable; it was not accepted to
shorten the report.

## Repository checks and smoke

The implementation and QA-evidence commits passed the repository hook checks:
ruff, ty, biston, zorilla, and gerenuk. Repository tests passed with 1,131
passed and 1 skipped, with 92% source coverage. The prepared
application smoke passed:

```text
uv run poe test
uv run poe check
uv run python scripts/qa_mcp_startup.py
uv run mkdocs build --strict
```

Pycoati is a periodic audit only; it is not installed in a hook or CI gate.
