#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <branch-name> [base-ref]" >&2
  echo "  Creates ~/worktrees/introspect-<branch-name> from a fresh base-ref (default: origin/main)." >&2
  exit 1
fi

BRANCH="$1"
BASE="${2:-origin/main}"
WORKTREES_DIR="${WORKTREES_DIR:-$HOME/worktrees}"
TARGET="$WORKTREES_DIR/introspect-$BRANCH"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ -e "$TARGET" ]; then
  echo "error: $TARGET already exists" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "error: branch '$BRANCH' already exists locally" >&2
  exit 1
fi

echo "==> Fetching $BASE"
REMOTE="${BASE%%/*}"
if [ "$REMOTE" != "$BASE" ]; then
  git fetch "$REMOTE" "${BASE#*/}"
fi

mkdir -p "$WORKTREES_DIR"

echo "==> Creating worktree at $TARGET (branch '$BRANCH' from $BASE)"
git worktree add -b "$BRANCH" "$TARGET" "$BASE"

if [ -f "$REPO_ROOT/.claude/settings.local.json" ]; then
  echo "==> Copying .claude/settings.local.json"
  mkdir -p "$TARGET/.claude"
  cp "$REPO_ROOT/.claude/settings.local.json" "$TARGET/.claude/settings.local.json"
fi

echo "==> Running uv sync"
(cd "$TARGET" && uv sync --quiet)

echo
echo "Done. cd $TARGET"
