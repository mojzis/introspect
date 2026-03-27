#!/usr/bin/env bash
set -euo pipefail

HOOK_PATH=".git/hooks/pre-commit"
SCRIPT_PATH="scripts/pre-commit.sh"

cp "$SCRIPT_PATH" "$HOOK_PATH"
chmod +x "$HOOK_PATH"
echo "Pre-commit hook installed."
