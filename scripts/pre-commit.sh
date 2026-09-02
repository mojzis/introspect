#!/usr/bin/env bash
#
# The single commit gate for this repo. Install it with `uv run poe setup`.
#
# Stages, in order — each one prints how long it took:
#   1. ruff format + ruff check --fix on staged Python, re-staged
#   2. ruff check --no-fix + ty check          (correctness)
#   3. biston  — structural clone detection, scoped to the staged files
#   4. gerenuk -> tyf -> pytest                (only the impacted tests)
#
# Bypass with `git commit --no-verify`.
set -euo pipefail

step_start=0
start_step() {
    printf '\n[pre-commit] %s\n' "$1"
    step_start=$SECONDS
}
end_step() {
    printf '[pre-commit] ...%ss\n' "$((SECONDS - step_start))"
}

STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')

if [ -z "$STAGED_PY_FILES" ]; then
    echo "[pre-commit] no staged Python files, nothing to do"
    exit 0
fi

TOTAL_START=$SECONDS
FAILED=0

# --- 1. auto-fix -------------------------------------------------------------
start_step "ruff format + autofix"
echo "$STAGED_PY_FILES" | xargs uv run ruff format --quiet
echo "$STAGED_PY_FILES" | xargs uv run ruff check --fix --quiet 2>/dev/null || true
echo "$STAGED_PY_FILES" | xargs git add
end_step

# --- 2. lint + types ---------------------------------------------------------
start_step "ruff check + ty check"
echo "$STAGED_PY_FILES" | xargs uv run ruff check --no-fix || FAILED=1
uv run ty check || FAILED=1
end_step

# --- 3. clone detection ------------------------------------------------------
# --files-from keeps the whole tree in the comparison (so a staged file cloning
# an untouched one is still caught) while only reporting pairs that involve a
# staged file. Piping the list handles "no staged files" as "no pairs", where
# --files would silently expand to a full-tree scan.
start_step "biston (clones in staged files)"
echo "$STAGED_PY_FILES" | uv run biston scan --files-from - . || FAILED=1
end_step

# --- 4. impacted tests -------------------------------------------------------
# The selector exits non-zero when it cannot map the diff to a trustworthy set;
# that always means "run the whole suite", never "skip the tests".
start_step "gerenuk + tyf -> impacted tests"
if IMPACTED=$(uv run python scripts/impacted_tests.py); then
    if [ -z "$IMPACTED" ]; then
        echo "[pre-commit] no test is impacted by this diff"
    else
        echo "$IMPACTED" | sed 's/^/[pre-commit]   /'
        # shellcheck disable=SC2086
        uv run pytest -o addopts= -x -qq --tb=short --no-header -n auto $IMPACTED || FAILED=1
    fi
else
    echo "[pre-commit] selection inconclusive — running the full suite"
    uv run poe test-quick || FAILED=1
fi
end_step

printf '\n[pre-commit] total %ss\n' "$((SECONDS - TOTAL_START))"

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Pre-commit checks failed. Fix the issues above or bypass with:"
    echo "  git commit --no-verify"
    exit 1
fi
