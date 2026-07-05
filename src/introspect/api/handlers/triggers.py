"""First-prompt triggers route handler.

Shows, per session, how the *opening prompt* set the session up: which
file/dir paths it named, whether the harness auto-loaded context (nested
CLAUDE.md / .claude/rules, @-file expansions, the skill menu), and which
skills/commands ran. See ``docs/first-prompt-triggers-plan.md``.
"""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from introspect.query_templates import FIRST_PROMPT_PATH_REGEX
from introspect.sql_fragments import (
    CONTEXT_LOADS_ROLLUP_SQL,
    SKILLS_INVOKED_ROLLUP_SQL,
)

from ._helpers import DEFAULT_PAGE_SIZE, clean_title, conn, parent, templates

# One row per session with the trigger rollup. Shares the same path regex and
# the same skills / session_context_loads rollup fragments as the
# ``first_prompt_triggers`` query template, so classification can't drift; this
# adds pagination and project scoping for the web view. All interpolated pieces
# are hardcoded module constants, never user input.
_BASE_CTE = f"""
    WITH base AS (
        SELECT
            ss.session_id,
            ss.project,
            ss.started_at,
            ss.first_prompt,
            ss.commands,
            len(regexp_extract_all(ss.first_prompt, '{FIRST_PROMPT_PATH_REGEX}'))
                AS n_referenced_paths,
            COALESCE(sk.skills_invoked, 0) AS skills_invoked,
            COALESCE(l.auto_loaded_claude_md, FALSE) AS auto_loaded_claude_md,
            COALESCE(l.n_auto_loaded_files, 0) AS n_auto_loaded_files,
            COALESCE(l.skill_menu_loaded, FALSE) AS skill_menu_loaded
        FROM session_stats ss
        LEFT JOIN ({SKILLS_INVOKED_ROLLUP_SQL}) sk USING (session_id)
        LEFT JOIN ({CONTEXT_LOADS_ROLLUP_SQL}) l USING (session_id)
        WHERE ss.first_prompt IS NOT NULL
    )
"""  # noqa: S608 — interpolated pieces are hardcoded constants, not input


async def triggers(
    request: Request,
    project: str,
    page: int = 1,
) -> HTMLResponse:
    """Per-session view of what each opening prompt referenced and triggered."""
    db = conn(request)
    project_name = project.strip()

    where = ""
    params: list[str | int] = []
    if project_name:
        where = "WHERE project = ?"
        params.append(project_name)

    # Aggregate: share of "vague openers" — prompts that named no path AND
    # triggered no skill (candidates for a sharper opening prompt).
    stats = db.execute(
        f"""
        {_BASE_CTE}
        SELECT
            count(*) AS total,
            count(*) FILTER (
                WHERE n_referenced_paths = 0 AND skills_invoked = 0
            ) AS vague_openers,
            count(*) FILTER (WHERE auto_loaded_claude_md) AS with_claude_md
        FROM base
        {where}
    """,  # noqa: S608
        params,
    ).fetchone()

    # Project filter chips. Counts the same population as ``base`` (sessions
    # with a first prompt) but skips its regex + rollup joins, none of which
    # feed a plain per-project count.
    projects = db.execute(
        """
        SELECT project, count(*) AS cnt
        FROM session_stats
        WHERE first_prompt IS NOT NULL AND project IS NOT NULL
        GROUP BY project
        ORDER BY cnt DESC
        LIMIT 25
    """
    ).fetchall()

    total = stats[0] if stats else 0
    total_pages = max(1, math.ceil(total / DEFAULT_PAGE_SIZE))
    offset = (page - 1) * DEFAULT_PAGE_SIZE

    rows = db.execute(
        f"""
        {_BASE_CTE}
        SELECT
            session_id,
            project,
            started_at,
            n_referenced_paths,
            auto_loaded_claude_md,
            n_auto_loaded_files,
            skills_invoked,
            skill_menu_loaded,
            commands,
            first_prompt
        FROM base
        {where}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """,  # noqa: S608
        [*params, DEFAULT_PAGE_SIZE, offset],
    ).fetchall()

    vague_pct = round(100 * stats[1] / stats[0], 1) if stats and stats[0] else 0.0

    return templates.TemplateResponse(
        request,
        "triggers.html",
        {
            "parent": parent(request),
            "rows": rows,
            "stats": stats,
            "vague_pct": vague_pct,
            "projects": projects,
            "filter_project": project,
            "page": page,
            "total_pages": total_pages,
            "clean_title": clean_title,
        },
    )
