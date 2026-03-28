"""Git worktree-aware project resolution."""

import subprocess  # nosec B404
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def get_canonical_project(cwd: str) -> str:
    """Get canonical project root from a working directory.

    Uses ``git rev-parse --git-common-dir`` so that both the main repo
    checkout and any worktrees resolve to the same path.
    """
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "-C", cwd, "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = Path(result.stdout.strip())
        # git may return a relative path (e.g. ".git"); resolve it
        # relative to the target cwd, NOT the process's cwd.
        git_common = (Path(cwd) / raw).resolve() if not raw.is_absolute() else raw

        if git_common.name == ".git":
            return str(git_common.parent)
        if "worktrees" in git_common.parts:
            idx = git_common.parts.index("worktrees")
            return str(Path(*git_common.parts[:idx]).parent)
        return str(git_common.parent)
    except (subprocess.CalledProcessError, OSError):
        return cwd  # fallback if not a git repo or path missing


def resolve_project_map(cwds: list[str]) -> dict[str, str]:
    """Resolve a list of cwds to their canonical project roots in parallel."""
    if not cwds:
        return {}

    with ThreadPoolExecutor(max_workers=min(8, len(cwds))) as pool:
        results = pool.map(get_canonical_project, cwds)

    return dict(zip(cwds, results, strict=True))
