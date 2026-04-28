#!/bin/bash
set -euo pipefail

CLAUDE_PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$CLAUDE_PROJECT_DIR"

# Report venv status for the user's outer shell.
# Affects both user-typed `! cmd` and Claude's Bash subprocesses (which inherit the parent shell's env).
# When unactivated, prefer `uv run <tool>` for venv binaries.
project_venv="$CLAUDE_PROJECT_DIR/.venv"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "venv not activated in the user's outer shell — \`! cmd\` will not see .venv/bin. Suggest activating: fish \`source .venv/bin/activate.fish\`, bash/zsh \`source .venv/bin/activate\`."
elif [ "$VIRTUAL_ENV" != "$project_venv" ]; then
  echo "VIRTUAL_ENV is '$VIRTUAL_ENV' but project venv is '$project_venv'."
fi

# Remote-only setup (Claude Code on the web)
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  uv sync --quiet
  bash scripts/install-hooks.sh
fi
