#!/usr/bin/env bash
set -euo pipefail

# Get staged Python files
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')

if [ -z "$STAGED_PY_FILES" ]; then
    exit 0
fi

echo "Pre-commit: checking staged Python files..."

# Phase 1: Auto-fix (format + lint fix)
echo "$STAGED_PY_FILES" | xargs uv run ruff format --quiet
echo "$STAGED_PY_FILES" | xargs uv run ruff check --fix --quiet 2>/dev/null || true

# Re-stage any auto-fixed files
echo "$STAGED_PY_FILES" | xargs git add

# Phase 2: Report-only checks
FAILED=0

echo "$STAGED_PY_FILES" | xargs uv run ruff check --no-fix || FAILED=1

uv run ty check || FAILED=1

# Run bandit only on src files (skip tests)
STAGED_SRC_FILES=$(echo "$STAGED_PY_FILES" | grep -v '^tests/' || true)
if [ -n "$STAGED_SRC_FILES" ]; then
    echo "$STAGED_SRC_FILES" | xargs uv run bandit -q || FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Pre-commit checks failed. Fix the issues above or bypass with:"
    echo "  git commit --no-verify"
    exit 1
fi
