"""Subagents tab: per-run cost and file-activity breakdown for a session.

One row per agent — the "Main agent" (the non-sidechain part of the session)
and one row per Task/Agent subagent invocation.  Rows are sorted cost-desc so
the most expensive agent is at the top.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from introspect.pricing import rates_for

from ._helpers import format_cost, format_duration, safe_json
from .cost_overview import _build_cost_split_shape
from .tokenscape import (
    _fetch_chain_rows,
    _fetch_subagent_prompts,
    _group_sidechain_rows,
    _label_and_merge_subagent_groups,
)

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

# Tool names whose tool_input carries a ``file_path`` or ``notebook_path``
# that the agent *wrote* (created or modified).
_EDIT_TOOL_NAMES = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

# Minimum timestamps needed to compute a duration.
_MIN_TIMESTAMPS_FOR_DURATION = 2


def _to_posix_timestamp(t: object) -> float:
    """Convert a timestamp value (datetime, str, int, float) to POSIX seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=UTC).timestamp()
        return t.timestamp()
    s = str(t)
    # "Z" suffix is valid ISO-8601 but Python < 3.11 doesn't support it
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except ValueError:
        # Strip sub-second precision and retry
        return (
            datetime.fromisoformat(s.split(".", maxsplit=1)[0])
            .replace(tzinfo=UTC)
            .timestamp()
        )


def _compute_duration(timestamps: list[object]) -> str:
    """Format duration from a list of timestamps, returning M:SS or '—'."""
    if len(timestamps) < _MIN_TIMESTAMPS_FOR_DURATION:
        return "—"
    try:
        posix = sorted(_to_posix_timestamp(t) for t in timestamps)
        return format_duration(posix[-1] - posix[0])
    except (ValueError, OSError, TypeError):
        return "—"


def _accumulate_cost(  # noqa: PLR0913
    row: tuple,
    total_cost_usd: float,
    total_read_usd: float,
    total_write_usd: float,
    total_output_usd: float,
    asst_uuids: set[str],
) -> tuple[float, float, float, float]:
    """Accumulate cost components for one (deduped) assistant message row.

    Returns ``(total_cost_usd, total_read_usd, total_write_usd, total_output_usd)``
    with the row's contribution added.
    """
    uuid = row[0]
    model = row[13]

    if row[7] is not None:
        asst_uuids.add(uuid)

    input_tokens = int(row[7] or 0)
    cache_read_tokens = int(row[8] or 0)
    cc_total = int(row[9] or 0)
    cc_5m = int(row[10] or 0)
    cc_1h = int(row[11] or 0)
    output_tokens = int(row[12] or 0)

    # Legacy 5m fallback: if cc_5m == 0 and cc_1h == 0 but cc_total > 0
    eff_5m = cc_5m
    if cc_5m == 0 and cc_1h == 0 and cc_total > 0:
        eff_5m = cc_total

    # Compute all sub-costs from a single rate lookup so the split bar and
    # row total are derived from the same rates and can never drift apart.
    r = rates_for(model)
    input_usd = input_tokens * r.input / 1_000_000
    read_usd = cache_read_tokens * r.cache_read / 1_000_000
    write_usd = (eff_5m * r.cache_write_5m + cc_1h * r.cache_write_1h) / 1_000_000
    output_usd = output_tokens * r.output / 1_000_000
    row_cost_usd = input_usd + read_usd + write_usd + output_usd

    return (
        total_cost_usd + row_cost_usd,
        total_read_usd + read_usd,
        total_write_usd + write_usd,
        total_output_usd + output_usd,
    )


def _process_tool_call(
    row: tuple,
    read_paths: set[str],
    edited_paths: set[str],
    skills: set[str],
) -> None:
    """Extract file-path and skill information from a tool-call row (in-place)."""
    tool_name = row[3]
    inp = safe_json(row[4])

    if tool_name == "Read":
        fp = inp.get("file_path") or inp.get("notebook_path")
        if fp:
            read_paths.add(fp)
    elif tool_name in _EDIT_TOOL_NAMES:
        fp = inp.get("file_path") or inp.get("notebook_path")
        if fp:
            edited_paths.add(fp)
            read_paths.add(fp)  # edits also count as reads
    elif tool_name == "Skill":
        skill = inp.get("skill")
        if skill:
            skills.add(skill)


def _aggregate_rows(rows: list[tuple], session_cwd: str) -> dict:
    """Aggregate a flat list of chain rows into per-run metrics.

    Columns of each row (15-column tuples from _fetch_chain_rows):
      0 uuid, 1 timestamp, 2 kind, 3 tool_name, 4 tool_input,
      5 char_count, 6 head, 7 input_tokens, 8 cache_read_tokens,
      9 cache_creation_tokens (cc_total), 10 cache_creation_5m,
      11 cache_creation_1h, 12 output_tokens, 13 model, 14 parent_uuid

    One row per *content block*, so a multi-block assistant message repeats
    its amc usage on every block.  Dedupe usage by uuid before summing.
    """
    seen_uuids: set[str] = set()
    models: set[str] = set()
    user_uuids: set[str] = set()
    asst_uuids: set[str] = set()
    total_cost_usd = 0.0
    total_read_usd = 0.0
    total_write_usd = 0.0
    total_output_usd = 0.0
    tool_count = 0
    read_paths: set[str] = set()
    edited_paths: set[str] = set()
    skills: set[str] = set()
    timestamps: list[object] = []

    for row in rows:
        uuid = row[0]
        kind = row[2]

        if row[1] is not None:
            timestamps.append(row[1])

        if kind in ("human_prompt", "slash_command", "subagent_prompt"):
            if uuid is not None:
                user_uuids.add(uuid)
            continue

        # Dedupe assistant usage by uuid (one row per block, usage repeated)
        if uuid is not None and uuid not in seen_uuids:
            seen_uuids.add(uuid)
            if row[13]:
                models.add(row[13])
            total_cost_usd, total_read_usd, total_write_usd, total_output_usd = (
                _accumulate_cost(
                    row,
                    total_cost_usd,
                    total_read_usd,
                    total_write_usd,
                    total_output_usd,
                    asst_uuids,
                )
            )

        if kind == "agent_tool_call" and row[3]:
            tool_count += 1
            _process_tool_call(row, read_paths, edited_paths, skills)

    read_only_paths = read_paths - edited_paths
    outside_paths = (
        {p for p in read_paths if not p.startswith(session_cwd)}
        if session_cwd
        else set()
    )
    first_uuid = next((r[0] for r in rows if r[0] is not None), None)

    return {
        "first_uuid": first_uuid,
        "model": ", ".join(sorted(models)) if models else "—",
        "duration": _compute_duration(timestamps),
        "user_msgs": len(user_uuids),
        "asst_msgs": len(asst_uuids),
        "tool_count": tool_count,
        "files_read": len(read_paths),
        "files_edited": len(edited_paths),
        "files_read_only": len(read_only_paths),
        "files_outside": len(outside_paths),
        "skills": sorted(skills),
        "cost_usd": total_cost_usd,
        "cost": format_cost(total_cost_usd),
        "split": _build_cost_split_shape(
            total_read_usd, total_write_usd, total_output_usd
        ),
    }


def build_subagent_breakdown_context(
    db: duckdb.DuckDBPyConnection,
    session_id: str,
    session_cwd: str | None,
) -> dict:
    """Build template context for the Subagents tab.

    Returns a dict with keys:
        has_data        bool — any rows at all
        rows            list[dict] — one per agent, sorted cost desc
        total_cost      str — formatted session total
        total_cost_usd  float
        subagent_count  int — number of subagent runs (excludes main row)
    """
    cwd = session_cwd or ""

    # Fetch and partition rows using the same pipeline as tokenscape
    side_rows = _fetch_chain_rows(db, session_id, sidechain=True)
    groups = _group_sidechain_rows(side_rows)
    prompts = _fetch_subagent_prompts(db, session_id)
    runs = _label_and_merge_subagent_groups(db, session_id, groups, prompts)
    main_rows = _fetch_chain_rows(db, session_id, sidechain=False)

    result_rows: list[dict] = []

    if main_rows:
        metrics = _aggregate_rows(main_rows, cwd)
        metrics["label"] = "Main agent"
        metrics["is_main"] = True
        result_rows.append(metrics)

    for run_rows, label in runs:
        if not run_rows:
            continue
        metrics = _aggregate_rows(run_rows, cwd)
        metrics["label"] = label
        metrics["is_main"] = False
        result_rows.append(metrics)

    if not result_rows:
        return {
            "has_data": False,
            "rows": [],
            "total_cost": "—",
            "total_cost_usd": 0.0,
            "subagent_count": 0,
        }

    result_rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    total_cost_usd = sum(r["cost_usd"] for r in result_rows)

    cum = 0.0
    for i, row in enumerate(result_rows):
        row["rank"] = i + 1
        row["pct"] = (
            100.0 * row["cost_usd"] / total_cost_usd if total_cost_usd > 0 else 0.0
        )
        cum += row["pct"]
        row["cum_pct"] = cum

    return {
        "has_data": True,
        "rows": result_rows,
        "total_cost": format_cost(total_cost_usd),
        "total_cost_usd": total_cost_usd,
        "subagent_count": sum(1 for r in result_rows if not r["is_main"]),
    }
