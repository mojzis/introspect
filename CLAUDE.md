# Python Template

## Commands

- `uv run poe check` — run lint, typecheck, security, vulns, then tests
- `uv run poe fix` — auto-format and fix lint issues
- `uv run poe test` — run tests only
- `uv run poe check-all` — run all checks including dead-code and unused-deps

## Stack

uv, ruff (lint/format), ty (type check), pytest, poethepoet (task runner)

## Notes

- ty is in beta — may produce false positives. Prefer `# ty: ignore[rule]` over blanket suppression.
- Pre-commit hook auto-fixes and restages files. Only blocks on unfixable issues.
