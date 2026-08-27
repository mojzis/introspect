"""Tokenscape: where the session's money went, stripe by stripe.

Each piece of content the model saw becomes a *band* — a horizontal
stripe that exists from the turn the content arrived until /compact
discards it. The y-axis is **dollars per turn**, so every column sums
to what that API call actually cost and the area of a stripe is the
total money that piece of content burned across the session.

Token attribution is anchored to API truth, not heuristics:

* ``context_t = input + cache_read + cache_creation`` for turn *t*.
  ``context_t - context_{t-1}`` is the *exact* token count of content
  added between turns.
* The assistant share of that delta is the previous turn's
  ``output_tokens`` (exact — covers redacted thinking too).
* The remainder is split across user-side blocks (prompts, tool
  results, skill loads) proportionally to character count.
* The first turn's residual after visible content is the system
  prompt + tool definitions; the same logic after /compact yields the
  compact summary.

Cost allocation per turn: newly arrived bands pay the cache-write
rate, persisting bands split the remaining input bill (cache reads,
plus any cache rebuilds) proportionally to size. Output tokens are
billed at the output rate on the turn they're generated. Sidechain
(subagent) API calls are bucketed onto the main-chain turn during
which they ran. Column totals therefore tie out to the real bill.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from introspect.pricing import compute_cost_usd, rates_for

from ._helpers import ensure_chart_template, path_basename, safe_json

if TYPE_CHECKING:
    import datetime

    import duckdb

    from introspect.pricing import Rates

logger = logging.getLogger(__name__)

_PER_MILLION = 1_000_000
_CHARS_PER_TOKEN = 4.0
# /compact: total context plummets — history truncated.
_COMPACT_DROP_RATIO = 0.5
_COMPACT_MIN_DROP = 5_000
_TOP_BANDS = 6
# Inline slab captions only when the slab is tall enough to hold them.
_LABEL_MIN_HEIGHT_RATIO = 0.07
_AGENT_ANNOTATION_MIN_SHARE = 0.10

_CATEGORY_LABEL = {
    "overhead": "system prompt & overhead",
    "skill": "skills & commands",
    "prompt": "your prompts",
    "read_edited": "reads of files you edited",
    "read": "reads of other files",
    "tool": "tool output",
    "assistant": "assistant output",
    "agent": "subagents",
    "rest": "everything else",
}
_CATEGORY_FILL = {
    "overhead": "#d8d2c2",  # warm sand — quiet baseline
    "skill": "#c9b458",  # ochre
    "prompt": "#7ca9dd",  # clear blue — the user's own words
    "read_edited": "#e0784a",  # terracotta
    "read": "#f0c49e",  # pale apricot
    "tool": "#72b3a2",  # sea green
    "assistant": "#a99bd8",  # violet
    "agent": "#cc6677",  # rose — subagents live in the red-purple family
    "rest": "#ececef",  # cool gray, hatched — unlike the warm overhead sand
}
_CATEGORY_BORDER = {
    "overhead": "#94896c",
    "skill": "#8a7426",
    "prompt": "#3d6fad",
    "read_edited": "#a64a20",
    "read": "#ab7340",
    "tool": "#327c6b",
    "assistant": "#6a55a8",
    "agent": "#943a4c",
    "rest": "#86868f",
}
# Each Task/Agent run cycles through these so a subagent-heavy session
# doesn't collapse into one flat colour. All sit in the red-purple
# family (agents stay "reddish") but are mutually distinguishable.
_AGENT_RUN_COLORS = (
    ("#cc6677", "#943a4c"),  # rose
    ("#b07aa8", "#7c4a73"),  # plum
    ("#e09a8a", "#a85e4d"),  # salmon
    ("#a64d62", "#6e2f40"),  # wine
    ("#c9a0b8", "#8e607e"),  # mauve
)
_OTHER_AGENTS_COLOR = ("#d9b8bd", "#9c6d75")  # washed-out rose for the fold
# Stripe captions sit on light pastel fills — near-black ink beats the
# mid-tone category borders for readability.
_STRIPE_LABEL_FONT = {"size": 13, "color": "#2d2a26"}


def _agent_run_color(run_index: int) -> tuple[str, str]:
    """(fill, border) for the i-th Task/Agent run — stable across the
    per-turn chart and the stripes chart."""
    return _AGENT_RUN_COLORS[run_index % len(_AGENT_RUN_COLORS)]


# Bottom-to-top stacking order: stable context first, volatile spenders on top.
_CATEGORY_ORDER = (
    "overhead",
    "skill",
    "prompt",
    "read_edited",
    "read",
    "tool",
    "assistant",
    "agent",
)

_SKILL_TEXT_PREFIX = "Base directory for this skill"


def _tool_result_label(tool_name: str, tool_input_raw: str) -> str:
    """One-line label for a tool_result band, e.g. ``Read settings.py``."""
    if not tool_name:
        return "tool result"
    if tool_name == "Read":
        d = safe_json(tool_input_raw)
        return f"Read {path_basename(d.get('file_path'))}"
    if tool_name == "Bash":
        d = safe_json(tool_input_raw)
        cmd = (d.get("command") or "").strip().split()
        head = " ".join(cmd[:2]) if cmd else "(empty)"
        return f"Bash · {head}"
    if tool_name in ("WebFetch", "WebSearch"):
        return tool_name
    if tool_name.startswith("mcp__"):
        return f"mcp · {tool_name[len('mcp__') :]}"
    return tool_name


def _classify_block(  # noqa: PLR0911
    kind: str | None,
    tool_name: str | None,
    tool_input: str | None,
    head: str | None,
    edited_files: set[str],
) -> tuple[str, str] | None:
    """Map a content block to ``(category, label)``.

    Returns ``None`` for assistant-side blocks — their token share comes
    from ``output_tokens`` of the producing turn, not from char counts.
    """
    text = head or ""
    if kind in ("human_prompt", "slash_command", "subagent_prompt"):
        if text.startswith(_SKILL_TEXT_PREFIX):
            # First line names the skill dir — label by its basename.
            first_line = text.split("\n", 1)[0]
            return (
                "skill",
                f"skill {path_basename(first_line.split(':', 1)[-1].strip())}",
            )
        if "<command-message>" in text or text.startswith("<command-name>"):
            return ("skill", "slash command")
        if text.startswith(
            ("<system-reminder", "<task-notification", "<local-command-caveat")
        ):
            # Harness-injected, not something the user typed.
            return ("overhead", "system reminders")
        return ("prompt", "prompt")
    if kind == "tool_result":
        name = tool_name or ""
        if name == "Read":
            file_path = safe_json(tool_input).get("file_path") or ""
            category = "read_edited" if file_path in edited_files else "read"
            return (category, f"Read {path_basename(file_path)}")
        if name in ("Task", "Agent"):
            return ("agent", "subagent result")
        return ("tool", _tool_result_label(name, tool_input or ""))
    return None


@dataclass
class _Band:
    """One piece of context: lives from ``arrival`` to ``end`` (inclusive)."""

    category: str
    label: str
    tokens: float
    arrival: int
    end: int | None = None
    cost: float = 0.0
    per_turn: dict[int, float] = field(default_factory=dict)

    def add_cost(self, turn: int, dollars: float) -> None:
        self.per_turn[turn] = self.per_turn.get(turn, 0.0) + dollars
        self.cost += dollars

    @property
    def final_turn(self) -> int:
        """Last turn this band is alive (``_walk_rows`` always closes bands,
        so ``end`` is only ``None`` mid-walk)."""
        return self.end if self.end is not None else self.arrival


@dataclass
class _Turn:
    """One main-chain assistant API call with its usage numbers."""

    idx: int
    ts: datetime.datetime | None
    model: str | None
    inp: int
    cache_read: int
    cache_creation: int
    cc_5m: int
    cc_1h: int
    out: int
    ctx: int
    event: str | None  # "compact" | "ttl_break" | None


def _fetch_chain_rows(
    db: duckdb.DuckDBPyConnection, session_id: str, *, sidechain: bool
) -> list[tuple]:
    """Content blocks + per-assistant-message usage for one chain, in order.

    ``sidechain=False`` selects the main chain, ``True`` the subagent
    threads. 15 columns: the 14 usage fields ``_walk_rows`` consumes
    plus a trailing ``parent_uuid`` (used to group sidechain rows into
    per-invocation threads; the main-chain path ignores it).
    """
    chain_predicate = "e.is_sidechain" if sidechain else "NOT e.is_sidechain"
    return db.execute(
        f"""
        WITH result_meta AS (
            -- tool_result rows don't expose the result payload in the
            -- enriched view; pull tool_use_id and content size from
            -- raw_messages in one pass.
            SELECT
                rm.uuid AS user_uuid,
                e.block_idx AS block_idx,
                json_extract_string(
                    rm.message, '$.content[' || e.block_idx || '].tool_use_id'
                ) AS result_tool_use_id,
                LENGTH(COALESCE(
                    json_extract_string(
                        rm.message, '$.content[' || e.block_idx || '].content'
                    ),
                    ''
                )) AS result_char_count
            FROM session_messages_enriched e
            JOIN raw_messages rm ON rm.uuid = e.uuid
            WHERE e.session_id = ? AND e.kind = 'tool_result'
        )
        SELECT
            e.uuid,
            e.timestamp,
            e.kind,
            COALESCE(e.tool_name, tc.tool_name) AS tool_name,
            COALESCE(e.tool_input, tc.tool_input) AS tool_input,
            CASE
                WHEN e.kind = 'tool_result' THEN COALESCE(rmeta.result_char_count, 0)
                ELSE LENGTH(COALESCE(e.text, ''))
            END AS char_count,
            LEFT(COALESCE(e.text, ''), 120) AS head,
            amc.input_tokens,
            amc.cache_read_tokens,
            amc.cache_creation_tokens,
            amc.cache_creation_5m,
            amc.cache_creation_1h,
            amc.output_tokens,
            amc.model,
            e.parent_uuid
        FROM session_messages_enriched e
        LEFT JOIN result_meta rmeta
          ON rmeta.user_uuid = e.uuid AND rmeta.block_idx = e.block_idx
        LEFT JOIN tool_calls tc ON tc.tool_use_id = rmeta.result_tool_use_id
        LEFT JOIN assistant_message_costs amc ON amc.uuid = e.uuid
        WHERE e.session_id = ? AND {chain_predicate}
        ORDER BY e.timestamp ASC, e.block_idx ASC
        """,  # noqa: S608 — chain_predicate is a literal, not user input
        [session_id, session_id],
    ).fetchall()


def _fetch_subagent_prompts(
    db: duckdb.DuckDBPyConnection, session_id: str
) -> dict[str, str]:
    """Full prompt text per sidechain root prompt message, keyed by uuid.

    Sidechain roots carry the Task call's ``prompt`` input verbatim —
    the full text (not the 120-char head) is needed for Task linkage.
    """
    rows = db.execute(
        """
        SELECT uuid, text
        FROM session_messages_enriched
        WHERE session_id = ? AND is_sidechain AND kind = 'subagent_prompt'
        """,
        [session_id],
    ).fetchall()
    return {r[0]: r[1] for r in rows if r[0] is not None and r[1] is not None}


def _group_sidechain_rows(rows: list[tuple]) -> list[list[tuple]]:
    """Split sidechain rows into per-invocation threads via parent chains.

    A thread roots wherever the parent leaves the sidechain row set
    (real roots have ``parent_uuid IS NULL``; degenerate data may point
    at a main-chain uuid). Groups are ordered by first timestamp (rows
    arrive timestamp-ordered) and preserve row order within each group.
    """
    parents: dict[str, str | None] = {}
    for row in rows:
        uuid, parent = row[0], row[14]
        if uuid is not None and uuid not in parents:
            parents[uuid] = parent

    roots: dict[str, str] = {}

    def root_of(uuid: str) -> str:
        path: list[str] = []
        seen: set[str] = set()
        current = uuid
        while current not in roots:
            path.append(current)
            seen.add(current)
            parent = parents.get(current)
            if parent is None or parent not in parents or parent in seen:
                break
            current = parent
        root = roots.get(current, current)
        for u in path:
            roots[u] = root
        return root

    groups: dict[str, list[tuple]] = {}
    for row in rows:
        if row[0] is None:
            continue
        groups.setdefault(root_of(row[0]), []).append(row)
    return list(groups.values())


def _agent_label(descr: str | None, stype: str | None) -> str:
    """Label for one Task/Agent invocation — shared by the session
    stripes and the per-run drill-down so the two never desync."""
    return f"agent: {(descr or stype or 'run')[:40]}"


def _fetch_agent_calls(db: duckdb.DuckDBPyConnection, session_id: str) -> list[tuple]:
    """Task/Agent invocations in call order:
    ``(called_at, description, subagent_type, prompt)``."""
    return db.execute(
        """
        SELECT called_at,
               json_extract_string(tool_input, '$.description'),
               json_extract_string(tool_input, '$.subagent_type'),
               json_extract_string(tool_input, '$.prompt')
        FROM tool_calls
        WHERE session_id = ? AND tool_name IN ('Task', 'Agent')
        ORDER BY called_at
        """,
        [session_id],
    ).fetchall()


def _label_and_merge_subagent_groups(
    db: duckdb.DuckDBPyConnection,
    session_id: str,
    groups: list[list[tuple]],
    prompts: dict[str, str],
) -> list[tuple[list[tuple], str]]:
    """``(rows, "agent: <description>")`` per Task/Agent invocation.

    Primary linkage: the group's root prompt text equals a Task/Agent
    call's ``prompt`` input verbatim (each call consumed once). Groups
    without a prompt root — subagent conversations often arrive with a
    ``parent_uuid`` that exists nowhere in the log, so they can't chain
    back to their prompt — inherit the most recent preceding call by
    timestamp (the sweep ``_sidechain_costs_per_turn`` uses, so the
    drill-down agrees with the chart). Remaining groups pair with
    leftover calls in order; anything still unmatched is ``agent: run``.

    Groups resolving to the same call are fragments of one invocation
    (the prompt thread, the conversation, post-break continuations) —
    time-disjoint fragments merge into a single run, in group order
    (groups arrive ordered by first timestamp). A fragment overlapping
    the run in time is a concurrent thread (a forked skill or a nested
    agent), not a continuation — it becomes its own run; see
    ``_chain_disjoint_fragments``.
    """
    calls = _fetch_agent_calls(db, session_id)

    assigned: list[int | None] = [None] * len(groups)
    used = [False] * len(calls)
    for gi, group in enumerate(groups):
        root_text = next((prompts[r[0]] for r in group if r[0] in prompts), None)
        if root_text is None:
            continue
        for ci, (_called_at, _descr, _stype, prompt) in enumerate(calls):
            if not used[ci] and prompt == root_text:
                used[ci] = True
                assigned[gi] = ci
                break

    for gi, group in enumerate(groups):
        if assigned[gi] is not None:
            continue
        first_ts = next((r[1] for r in group if r[1] is not None), None)
        if first_ts is None:
            continue
        for ci, (called_at, _descr, _stype, _prompt) in enumerate(calls):
            if called_at is not None and called_at <= first_ts:
                assigned[gi] = ci

    leftover = (ci for ci in range(len(calls)) if not used[ci])
    by_call: dict[int, list[list[tuple]]] = {}
    unmatched: list[list[tuple]] = []
    for gi, group in enumerate(groups):
        ci = assigned[gi] if assigned[gi] is not None else next(leftover, None)
        if ci is None:
            unmatched.append(group)
        else:
            by_call.setdefault(ci, []).append(group)

    runs = [
        (rows, _agent_label(calls[ci][1], calls[ci][2]))
        for ci, fragments in sorted(by_call.items())
        for rows in _chain_disjoint_fragments(fragments)
    ]
    runs.extend((group, "agent: run") for group in unmatched)
    return runs


def _chain_disjoint_fragments(fragments: list[list[tuple]]) -> list[list[tuple]]:
    """Concatenate one invocation's fragments into time-monotonic chains.

    A dangling ``parent_uuid`` splits a single conversation into
    sequential fragments — those belong end-to-end in one chain. But a
    thread that *overlaps* the chain in time is a concurrent
    conversation (a forked skill load, a nested agent): interleaving it
    would corrupt the context-delta walk and fold the x-axis time ticks
    back on themselves, so it starts a new chain. Each fragment joins
    the first chain it doesn't overlap.
    """
    chains: list[list[tuple]] = []
    chain_ends: list[datetime.datetime | None] = []
    for fragment in fragments:
        stamps = [r[1] for r in fragment if r[1] is not None]
        first = stamps[0] if stamps else None
        last = stamps[-1] if stamps else None
        for i, end in enumerate(chain_ends):
            if first is None or end is None or first >= end:
                chains[i].extend(fragment)
                if last is not None and (end is None or last > end):
                    chain_ends[i] = last
                break
        else:
            chains.append(list(fragment))
            chain_ends.append(last)
    return chains


def _fetch_edited_files(db: duckdb.DuckDBPyConnection, session_id: str) -> set[str]:
    rows = db.execute(
        "SELECT DISTINCT file_path FROM file_writes WHERE session_id = ?",
        [session_id],
    ).fetchall()
    return {r[0] for r in rows if r[0]}


@dataclass
class _AgentRun:
    """One Task/Agent invocation: its sidechain bill and turn span."""

    label: str
    cost: float = 0.0
    first_turn: int | None = None
    last_turn: int | None = None
    per_turn: dict[int, float] = field(default_factory=dict)


def _sidechain_costs_per_turn(
    db: duckdb.DuckDBPyConnection, session_id: str, turns: list[_Turn]
) -> tuple[list[float], list[_AgentRun]]:
    """Bucket each sidechain (subagent) API call's cost onto the main turn
    that was active when it ran.

    Returns ``(per_turn, runs)``: per-turn sidechain dollars, plus one
    ``_AgentRun`` per Task/Agent invocation so each subagent run can
    get its own stripe. Sidechain messages inherit the most recent
    preceding Task/Agent call (same sweep the Cost tab uses); parallel
    agents launched in one turn lump under whichever call came first.
    """
    rows = db.execute(
        """
        SELECT timestamp, model, input_tokens, output_tokens,
               cache_read_tokens, cache_creation_tokens,
               cache_creation_5m, cache_creation_1h
        FROM assistant_message_costs
        WHERE session_id = ? AND is_sidechain
        ORDER BY timestamp
        """,
        [session_id],
    ).fetchall()
    per_turn = [0.0] * len(turns)
    if not rows or not turns:
        return per_turn, []

    timeline = _fetch_agent_calls(db, session_id)
    runs = [
        _AgentRun(label=_agent_label(descr, stype))
        for _called_at, descr, stype, _prompt in timeline
    ]
    fallback_run = _AgentRun(label="agent: run")

    timed = [(t.ts, t.idx) for t in turns if t.ts is not None]
    turn_ts = [ts for ts, _ in timed]
    timeline_idx = 0
    current_run = fallback_run
    for ts, model, inp, out, cr, cc, cc5, cc1 in rows:
        while (
            ts is not None
            and timeline_idx < len(timeline)
            and timeline[timeline_idx][0] is not None
            and timeline[timeline_idx][0] <= ts
        ):
            current_run = runs[timeline_idx]
            timeline_idx += 1
        cost = compute_cost_usd(
            model=model,
            input_tokens=inp or 0,
            output_tokens=out or 0,
            cache_read_tokens=cr or 0,
            cache_creation_5m=(cc5 or 0) if (cc5 or cc1) else (cc or 0),
            cache_creation_1h=cc1 or 0,
        )
        pos = bisect_right(turn_ts, ts) - 1 if ts is not None and turn_ts else 0
        idx = timed[pos][1] if timed and pos >= 0 else 0
        idx = max(0, min(idx, len(turns) - 1))
        per_turn[idx] += cost
        current_run.per_turn[idx] = current_run.per_turn.get(idx, 0.0) + cost
        current_run.cost += cost
        if current_run.first_turn is None:
            current_run.first_turn = idx
        current_run.last_turn = idx

    if fallback_run.cost > 0:
        runs.append(fallback_run)
    return per_turn, [r for r in runs if r.cost > 0]


def _detect_event(
    turns: list[_Turn], ctx: int, uuid: str, ttl_break_uuids: frozenset[str]
) -> str | None:
    """Classify the transition into this turn: /compact, TTL break, or none.

    TTL breaks are not re-derived here. ``ttl_break_uuids`` comes from the
    ``cache_requests`` view — the single detection rule the session-detail
    divider and the cost-overview panel also read — so this event track and
    those surfaces can no longer disagree about what a cache break is.
    """
    if not turns:
        return None
    prev = turns[-1]
    drop = prev.ctx - ctx
    if drop > _COMPACT_MIN_DROP and ctx < prev.ctx * _COMPACT_DROP_RATIO:
        return "compact"
    if uuid in ttl_break_uuids:
        return "ttl_break"
    return None


def _fetch_ttl_break_uuids(
    db: duckdb.DuckDBPyConnection, session_id: str, *, sidechain: bool
) -> frozenset[str]:
    """Request uuids the shared rule flags as cache misses, for one chain."""
    rows = db.execute(
        "SELECT uuid FROM cache_requests"
        " WHERE session_id = ? AND is_sidechain = ? AND cache_miss",
        [session_id, sidechain],
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _walk_rows(  # noqa: PLR0912
    rows: list[tuple],
    edited_files: set[str],
    ttl_break_uuids: frozenset[str] = frozenset(),
) -> tuple[list[_Band], list[_Turn]]:
    """Single pass: build bands (token-attributed content) and turns.

    ``pending`` collects user-side blocks since the last assistant call;
    when the next call's usage arrives, the context delta is split as
    described in the module doc.
    """
    bands: list[_Band] = []
    turns: list[_Turn] = []
    pending: list[tuple[str, str, int]] = []
    pending_out = 0
    seen_uuids: set[str] = set()

    for (
        uuid,
        ts,
        kind,
        tool_name,
        tool_input,
        char_count,
        head,
        inp,
        cache_read,
        cache_creation,
        cc_5m,
        cc_1h,
        out,
        model,
        *_,  # parent_uuid (grouping only) and any future trailing columns
    ) in rows:
        classified = _classify_block(kind, tool_name, tool_input, head, edited_files)
        if classified is not None and (char_count or 0) > 0:
            pending.append((classified[0], classified[1], int(char_count)))

        if inp is None or uuid is None or uuid in seen_uuids:
            continue
        seen_uuids.add(uuid)

        ctx = int(inp) + int(cache_read or 0) + int(cache_creation or 0)
        if model == "<synthetic>" or ctx == 0:
            # Synthetic messages (harness-generated, zero usage) are not
            # API calls — a fake ctx=0 turn would read as a /compact.
            continue

        turn_idx = len(turns)
        event = _detect_event(turns, ctx, str(uuid), ttl_break_uuids)

        if event == "compact":
            for band in bands:
                if band.end is None:
                    band.end = turn_idx - 1
            new_tokens = float(ctx)
        else:
            prev_ctx = turns[-1].ctx if turns else 0
            new_tokens = float(max(0, ctx - prev_ctx))

        if event != "compact" and turns:
            # Previous turn's output entered the context now — exact count.
            assistant_tokens = min(float(pending_out), new_tokens)
            if assistant_tokens >= 1:
                bands.append(
                    _Band("assistant", "assistant output", assistant_tokens, turn_idx)
                )
            new_tokens -= assistant_tokens

        if turn_idx == 0 or event == "compact":
            # Estimate visible blocks at ~4 chars/token; the residual is
            # invisible context — system prompt + tools on turn 0, the
            # summary after /compact.
            estimated = sum(c / _CHARS_PER_TOKEN for _, _, c in pending)
            scale = min(1.0, new_tokens / estimated) if estimated > 0 else 0.0
            for category, label, chars in pending:
                tokens = chars / _CHARS_PER_TOKEN * scale
                if tokens >= 1:
                    bands.append(_Band(category, label, tokens, turn_idx))
            residual = max(0.0, new_tokens - estimated * scale)
            baseline_label = (
                "system prompt + tool definitions"
                if turn_idx == 0
                else "compact summary"
            )
            if residual >= 1:
                bands.append(_Band("overhead", baseline_label, residual, turn_idx))
        else:
            total_chars = sum(c for _, _, c in pending)
            for category, label, chars in pending:
                tokens = new_tokens * chars / total_chars if total_chars else 0.0
                if tokens >= 1:
                    bands.append(_Band(category, label, tokens, turn_idx))

        pending = []
        pending_out = int(out or 0)
        turns.append(
            _Turn(
                idx=turn_idx,
                ts=ts,
                model=model,
                inp=int(inp),
                cache_read=int(cache_read or 0),
                cache_creation=int(cache_creation or 0),
                cc_5m=int(cc_5m or 0),
                cc_1h=int(cc_1h or 0),
                out=int(out or 0),
                ctx=ctx,
                event=event,
            )
        )

    last = len(turns) - 1
    for band in bands:
        if band.end is None:
            band.end = last
    return bands, turns


def _cache_write_cost(turn: _Turn, rates: Rates) -> float:
    """Dollars billed for cache writes on this turn. Older logs may lack
    the 5m/1h breakdown — fall back to the whole total at the 5m rate."""
    if turn.cc_5m or turn.cc_1h:
        return (
            turn.cc_5m * rates.cache_write_5m + turn.cc_1h * rates.cache_write_1h
        ) / _PER_MILLION
    return turn.cache_creation * rates.cache_write_5m / _PER_MILLION


def _allocate_costs(bands: list[_Band], turns: list[_Turn]) -> list[float]:
    """Distribute each turn's exact input bill across its live bands.

    New arrivals pay the cache-write rate for their tokens; persisting
    bands split the remainder (cache reads + rebuilds) proportionally
    to size. Returns per-turn output-generation dollars (assistant).
    """
    generation_costs = [0.0] * len(turns)
    for turn in turns:
        rates = rates_for(turn.model)
        write_cost = _cache_write_cost(turn, rates)
        input_total = (
            turn.inp * rates.input / _PER_MILLION
            + turn.cache_read * rates.cache_read / _PER_MILLION
            + write_cost
        )
        generation_costs[turn.idx] = turn.out * rates.output / _PER_MILLION

        live = [b for b in bands if b.arrival <= turn.idx <= b.final_turn]
        if not live:
            continue
        new = [b for b in live if b.arrival == turn.idx]
        persisting = [b for b in live if b.arrival < turn.idx]
        new_tokens = sum(b.tokens for b in new)
        write_per_token = (
            write_cost / turn.cache_creation if turn.cache_creation else 0.0
        )
        new_cost = min(new_tokens * write_per_token, input_total)
        remainder = input_total - new_cost
        if not persisting:
            # First turn of a segment: everything is new — the whole
            # bill belongs to the arrivals.
            new_cost = input_total
            remainder = 0.0
        for band in new:
            if new_tokens > 0:
                band.add_cost(turn.idx, new_cost * band.tokens / new_tokens)
        persist_tokens = sum(b.tokens for b in persisting)
        for band in persisting:
            if persist_tokens > 0:
                band.add_cost(turn.idx, remainder * band.tokens / persist_tokens)
    return generation_costs


def _pick_named_bands(bands: list[_Band]) -> list[_Band]:
    """The individually-labelled slabs: most expensive content pieces."""
    candidates = [
        b
        for b in bands
        if b.category in ("read", "read_edited", "tool", "agent", "skill")
        and b.cost > 0
    ]
    candidates.sort(key=lambda b: b.cost, reverse=True)
    return candidates[:_TOP_BANDS]


def _time_ticks(turns: list[_Turn]) -> tuple[list[int], list[str]]:
    """~6 evenly spaced x ticks labelled with wall-clock time."""
    n = len(turns)
    if n == 0:
        return [], []
    step = max(1, n // 6)
    vals, texts = [], []
    for i in range(0, n, step):
        ts = turns[i].ts
        vals.append(i + 1)
        texts.append(ts.strftime("%H:%M") if ts is not None else str(i + 1))
    return vals, texts


def _render_figure(  # noqa: PLR0912, PLR0913, PLR0915
    bands: list[_Band],
    turns: list[_Turn],
    generation_costs: list[float],
    agent_costs: list[float],
    agent_runs: list[_AgentRun],
    named: list[_Band],
) -> str:
    """Stacked per-turn cost bars: one trace per named slab + per category."""
    ensure_chart_template()
    n = len(turns)
    xs = list(range(1, n + 1))
    named_ids = {id(b) for b in named}

    # Series assembly, bottom to top. Named slabs sit adjacent to their
    # category aggregate so same-coloured stripes group visually. The
    # band object rides along (None for aggregates) — labels can repeat
    # (the same file Read twice), so identity must not go through them.
    series: list[tuple[str, str, str, str, list[float], _Band | None]] = []

    def _aggregate(category: str) -> list[float]:
        ys = [0.0] * n
        for band in bands:
            if band.category != category or id(band) in named_ids:
                continue
            for t, dollars in band.per_turn.items():
                ys[t] += dollars
        return ys

    def _band_series(band: _Band) -> list[float]:
        ys = [0.0] * n
        for t, dollars in band.per_turn.items():
            ys[t] += dollars
        return ys

    for category in _CATEGORY_ORDER:
        fill, border = _CATEGORY_FILL[category], _CATEGORY_BORDER[category]
        aggregate = _aggregate(category)
        if category == "assistant":
            for t, dollars in enumerate(generation_costs):
                aggregate[t] += dollars
        if any(v > 0 for v in aggregate):
            series.append(
                (_CATEGORY_LABEL[category], category, fill, border, aggregate, None)
            )
        series.extend(
            (band.label, category, fill, border, _band_series(band), band)
            for band in named
            if band.category == category
        )
        if category == "agent":
            # One trace per Task/Agent run, each with its own colour —
            # same assignment order as the stripes chart.
            for run_idx, run in enumerate(agent_runs):
                run_fill, run_border = _agent_run_color(run_idx)
                ys = [0.0] * n
                for t, dollars in run.per_turn.items():
                    ys[t] += dollars
                if any(v > 0 for v in ys):
                    series.append((run.label, category, run_fill, run_border, ys, None))

    fig = go.Figure()
    for label, _category, fill, border, ys, _band in series:
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                name=label,
                marker={
                    "color": fill,
                    "line": {"width": 0.4, "color": border},
                },
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>turn %{{x}}<br>"
                    "$%{y:,.4f} this turn<extra></extra>"
                ),
            )
        )

    annotations: list[dict] = []

    # Stack bottoms per series, for centring inline labels.
    cumulative = [0.0] * n
    series_bottom: list[list[float]] = []
    for _, _, _, _, ys, _band in series:
        series_bottom.append(list(cumulative))
        for i in range(n):
            cumulative[i] += ys[i]
    max_total = max(cumulative) if cumulative and max(cumulative) > 0 else 1.0

    for s_idx, (_label, category, _fill, border, ys, band) in enumerate(series):
        if category == "overhead" and band is None:
            # Label the baseline at the mid-point of its widest stretch.
            active = [i for i in range(n) if ys[i] > 0]
            if not active:
                continue
            mid = active[len(active) // 2]
            total = sum(ys)
            annotations.append(
                {
                    "x": mid + 1,
                    "y": series_bottom[s_idx][mid] + ys[mid] / 2,
                    "xref": "x",
                    "yref": "y",
                    "text": f"system prompt & overhead · ${total:,.2f}",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#3a342a"},
                }
            )
            continue
        if band is None:
            continue
        mid = max(0, min(n - 1, (band.arrival + band.final_turn) // 2))
        if ys[mid] < _LABEL_MIN_HEIGHT_RATIO * max_total:
            continue
        persisted = band.final_turn - band.arrival + 1
        annotations.append(
            {
                "x": mid + 1,
                "y": series_bottom[s_idx][mid] + ys[mid] / 2,
                "xref": "x",
                "yref": "y",
                "text": f"{band.label} · ${band.cost:,.2f} over {persisted} turns",
                "showarrow": False,
                "font": {"size": 11, "color": border},
            }
        )

    total_cost = sum(cumulative)
    agent_total = sum(agent_costs)
    if total_cost > 0 and agent_total / total_cost >= _AGENT_ANNOTATION_MIN_SHARE:
        peak = max(range(n), key=lambda i: agent_costs[i])
        annotations.append(
            {
                "x": peak + 1,
                "y": 1.02,
                "xref": "x",
                "yref": "paper",
                "text": f"subagents · ${agent_total:,.2f} total",
                "showarrow": False,
                "font": {"size": 11, "color": _CATEGORY_BORDER["agent"]},
                "xanchor": "center",
            }
        )

    for turn in turns:
        if turn.event is None:
            continue
        x = turn.idx + 0.5
        if turn.event == "compact":
            fig.add_vline(x=x, line_dash="dash", line_color="#777", line_width=1)
            label = "/compact"
            color = "#555"
        else:
            fig.add_vline(x=x, line_dash="dot", line_color="#c0a000", line_width=1)
            rebuild = _cache_write_cost(turn, rates_for(turn.model))
            label = f"gap · cache rebuilt ${rebuild:,.2f}"
            color = "#a08000"
        annotations.append(
            {
                "x": x,
                "y": 1.07,
                "xref": "x",
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "font": {"size": 10, "color": color},
                "xanchor": "left",
            }
        )

    tick_vals, tick_texts = _time_ticks(turns)
    fig.update_layout(
        barmode="stack",
        bargap=0,
        bargroupgap=0,
        showlegend=False,
        hovermode="closest",
        xaxis={
            "tickvals": tick_vals,
            "ticktext": tick_texts,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "USD per turn", "font": {"size": 12, "color": "#666"}},
            "tickprefix": "$",
            "showgrid": False,
            "zeroline": False,
        },
        margin={"l": 70, "r": 20, "t": 45, "b": 45},
        annotations=annotations,
    )
    return fig.to_json()


# Stripes layout (variant B): only pieces worth at least 1/20 of the
# bill get their own stripe; the rest folds into "everything else".
_STRIPE_MIN_SHARE = 1 / 20

# Per-subagent drill-down: at most this many charts, and only for runs
# that cost at least this much — the rest folds into a one-line note.
_SUBAGENT_CHARTS_MAX = 8
_SUBAGENT_CHART_MIN_COST = 0.10


def _select_subagent_runs(run_ctxs: list[dict]) -> tuple[list[dict], dict]:
    """Top runs worth a chart (≤ 8, ≥ $0.10), rest folded into a note."""
    ordered = sorted(run_ctxs, key=lambda r: r["cost"], reverse=True)
    shown: list[dict] = []
    hidden_count = 0
    hidden_cost = 0.0
    for run in ordered:
        if (
            len(shown) < _SUBAGENT_CHARTS_MAX
            and run["cost"] >= _SUBAGENT_CHART_MIN_COST
        ):
            shown.append(run)
        else:
            hidden_count += 1
            hidden_cost += run["cost"]
    return shown, {"count": hidden_count, "cost": hidden_cost}


def _bill_split(
    turns: list[_Turn], generation_costs: list[float], agent_costs: list[float]
) -> dict[str, float]:
    """Dollars by billing type — answers "how much of this is cache reads?"."""
    read = write = uncached = 0.0
    for turn in turns:
        rates = rates_for(turn.model)
        read += turn.cache_read * rates.cache_read / _PER_MILLION
        write += _cache_write_cost(turn, rates)
        uncached += turn.inp * rates.input / _PER_MILLION
    return {
        "cache_read": read,
        "cache_write": write,
        "uncached_input": uncached,
        "output": sum(generation_costs),
        "subagents": sum(agent_costs),
    }


_ASSISTANT_COHORTS = 4


def _assistant_cohort_stripes(
    bands: list[_Band],
    turns: list[_Turn],
    generation_costs: list[float],
    threshold: float,
) -> list[dict]:
    """Assistant output grows throughout the session — one full-width
    stripe would misleadingly look like it all loaded at turn 1. Split
    it into arrival cohorts (quarters of the session), each starting
    when that slice of output was produced and dragging to the end, so
    the purple stack visibly thickens over time. Cohorts below the
    stripe threshold merge into their neighbour.
    """
    n = len(turns)
    last = n - 1
    bounds = [round(i * n / _ASSISTANT_COHORTS) for i in range(_ASSISTANT_COHORTS + 1)]
    cohorts: list[dict] = []
    for i in range(_ASSISTANT_COHORTS):
        lo, hi = bounds[i], bounds[i + 1] - 1
        if hi < lo:
            continue
        member_arrivals = [
            b.arrival
            for b in bands
            if b.category == "assistant" and lo <= b.arrival <= hi
        ]
        cost = sum(
            b.cost for b in bands if b.category == "assistant" and lo <= b.arrival <= hi
        ) + sum(generation_costs[lo : hi + 1])
        if cost <= 0:
            continue
        cohorts.append(
            {
                "label": "assistant output",
                "category": "assistant",
                "arrival": min(member_arrivals) if member_arrivals else lo,
                "end": last,
                "cost": cost,
            }
        )
    # Merge sub-threshold cohorts into the previous one (or the next,
    # for a small opening cohort) so every stripe stays labelable.
    merged: list[dict] = []
    for cohort in cohorts:
        if merged and (cohort["cost"] < threshold or merged[-1]["cost"] < threshold):
            merged[-1]["cost"] += cohort["cost"]
        else:
            merged.append(cohort)
    return merged


def _build_stripes(
    bands: list[_Band],
    turns: list[_Turn],
    generation_costs: list[float],
    agent_costs: list[float],
    agent_runs: list[_AgentRun],
) -> list[dict]:
    """Sedimentation stripes: one rectangle per expensive piece of content.

    Whatever loads first sits at the bottom and runs to the end (or to
    /compact); later content stacks on top from the turn it arrived.
    Height = total dollars / turns alive, so area = dollars.

    The 1/20 visibility rule is applied per sub-budget: main-chain
    pieces compete against the main-chain total and subagent runs
    against the subagent total — so a subagent-heavy session doesn't
    push every main-chain stripe under the threshold (and vice versa).
    """
    n = len(turns)
    last = n - 1
    subagent_total = sum(agent_costs)
    main_total = sum(b.cost for b in bands) + sum(generation_costs)
    total = (main_total + subagent_total) or 1.0
    main_threshold = (main_total or total) * _STRIPE_MIN_SHARE
    agent_threshold = (subagent_total or total) * _STRIPE_MIN_SHARE

    stripes: list[dict] = []
    accounted = 0.0

    cohort_stripes = _assistant_cohort_stripes(
        bands, turns, generation_costs, main_threshold
    )
    for stripe in cohort_stripes:
        if stripe["cost"] >= main_threshold:
            stripes.append(stripe)
            accounted += stripe["cost"]

    # One stripe per Task/Agent invocation: starts when the run started
    # and drags to the end of the session (the main chain is blocked
    # while agents run, so the raw active span would draw as a needle).
    # Runs below the subagent threshold fold into "other subagents".
    other_agents = 0.0
    other_agents_first: int | None = None
    for run_idx, run in enumerate(agent_runs):
        if run.cost >= agent_threshold and run.cost >= 0.01:  # noqa: PLR2004
            fill, _border = _agent_run_color(run_idx)
            stripes.append(
                {
                    "label": run.label,
                    "category": "agent",
                    "fill": fill,
                    "arrival": run.first_turn or 0,
                    "end": last,
                    "cost": run.cost,
                }
            )
            accounted += run.cost
        else:
            other_agents += run.cost
            first = run.first_turn or 0
            if other_agents_first is None or first < other_agents_first:
                other_agents_first = first
    if other_agents > 0.005:  # noqa: PLR2004
        stripes.append(
            {
                "label": "other subagents",
                "category": "agent",
                "fill": _OTHER_AGENTS_COLOR[0],
                "arrival": other_agents_first or 0,
                "end": last,
                "cost": other_agents,
            }
        )
        accounted += other_agents

    for band in bands:
        if band.category == "assistant" or band.cost < main_threshold:
            continue
        stripes.append(
            {
                "label": band.label or _CATEGORY_LABEL[band.category],
                "category": band.category,
                "arrival": band.arrival,
                "end": band.final_turn,
                "cost": band.cost,
            }
        )
        accounted += band.cost

    rest = total - accounted
    if rest > 0.005:  # noqa: PLR2004 — below half a cent nothing to draw
        stripes.append(
            {
                "label": "everything else",
                "category": "rest",
                "arrival": 0,
                "end": last,
                "cost": rest,
            }
        )

    # Sedimentation order: earliest arrival at the bottom; among
    # same-turn arrivals the biggest goes lowest so the base is stable.
    stripes.sort(key=lambda s: (s["arrival"], -s["cost"]))
    return stripes


def _render_stripes_figure(stripes: list[dict], turns: list[_Turn]) -> str:
    """Variant B: flat labeled rectangles, area = dollars, no y-axis."""
    ensure_chart_template()
    n = len(turns)
    xs = list(range(1, n + 1))

    fig = go.Figure()
    for stripe in stripes:
        width = stripe["end"] - stripe["arrival"] + 1
        height = stripe["cost"] / width
        ys = [
            height if stripe["arrival"] <= t <= stripe["end"] else 0.0 for t in range(n)
        ]
        # No per-bar border — a stripe must read as one solid
        # rectangle, not a comb of per-turn columns.
        marker: dict = {"color": stripe.get("fill", _CATEGORY_FILL[stripe["category"]])}
        if stripe["category"] == "rest":
            # Hatch the catch-all so it can't be mistaken for the
            # system-prompt overhead stripe.
            marker["pattern"] = {
                "shape": "/",
                "fgcolor": "#c8c8cf",
                "fillmode": "overlay",
                "size": 5,
                "solidity": 0.4,
            }
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                name=stripe["label"],
                marker=marker,
                showlegend=False,
                hovertemplate=(
                    f"<b>{stripe['label']}</b><br>"
                    f"${stripe['cost']:,.2f} total · "
                    f"turns {stripe['arrival'] + 1}-{stripe['end'] + 1}"
                    "<extra></extra>"
                ),
            )
        )

    # Stack bottoms for label placement. Stripes sharing the same span
    # (several full-width layers) would all label at the same x —
    # stagger them evenly across the span instead.
    span_groups: dict[tuple[int, int], list[int]] = {}
    for idx, stripe in enumerate(stripes):
        span_groups.setdefault((stripe["arrival"], stripe["end"]), []).append(idx)

    cumulative = [0.0] * n
    annotations: list[dict] = []
    for idx, stripe in enumerate(stripes):
        width = stripe["end"] - stripe["arrival"] + 1
        height = stripe["cost"] / width
        group = span_groups[(stripe["arrival"], stripe["end"])]
        position = (group.index(idx) + 1) / (len(group) + 1)
        label_x = stripe["arrival"] + (width - 1) * position
        anchor = max(stripe["arrival"], min(stripe["end"], round(label_x)))
        annotations.append(
            {
                "x": label_x + 1,
                "y": cumulative[anchor] + height / 2,
                "xref": "x",
                "yref": "y",
                "text": f"{stripe['label']} · ${stripe['cost']:,.2f}",
                "showarrow": False,
                "font": dict(_STRIPE_LABEL_FONT),
            }
        )
        for t in range(stripe["arrival"], stripe["end"] + 1):
            cumulative[t] += height

    for turn in turns:
        if turn.event == "compact":
            fig.add_vline(
                x=turn.idx + 0.5, line_dash="dash", line_color="#777", line_width=1
            )

    tick_vals, tick_texts = _time_ticks(turns)
    fig.update_layout(
        barmode="stack",
        bargap=0,
        bargroupgap=0,
        showlegend=False,
        hovermode="closest",
        xaxis={
            "tickvals": tick_vals,
            "ticktext": tick_texts,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
        },
        margin={"l": 20, "r": 20, "t": 20, "b": 45},
        annotations=annotations,
    )
    return fig.to_json()


def _build_subagent_run_context(
    group_rows: list[tuple],
    label: str,
    edited_files: set[str],
    ttl_break_uuids: frozenset[str] = frozenset(),
) -> dict | None:
    """Stripes data for one subagent invocation (figure rendered later).

    The Plotly figure is deliberately not built here: runs folded into
    the hidden note never need one, so rendering waits until after
    ``_select_subagent_runs``. Returns ``None`` when the thread has no
    priced assistant turns (orphan/degenerate sidechains — e.g. a lone
    prompt with no usage).
    """
    bands, turns = _walk_rows(group_rows, edited_files, ttl_break_uuids)
    if not turns:
        return None
    generation_costs = _allocate_costs(bands, turns)
    cost = sum(b.cost for b in bands) + sum(generation_costs)
    # The 1/20 stripe threshold zooms to this run's own total.
    stripes = _build_stripes(bands, turns, generation_costs, [0.0] * len(turns), [])
    return {
        "label": label,
        "cost": cost,
        "turn_count": len(turns),
        "models": sorted({t.model for t in turns if t.model}),
        "stripes": stripes,
        "turns": turns,
    }


def _assign_subagent_run_colors(shown: list[dict], agent_runs: list[_AgentRun]) -> None:
    """Assign (fill, border) colours to each drill-down chart entry.

    Matches each shown run to the first unconsumed agent_run with the same
    label, assigning it ``_agent_run_color(idx)`` where ``idx`` is the
    position in ``agent_runs``.  Unmatched entries fall back to
    ``_OTHER_AGENTS_COLOR``.

    Note: this assigns colours by label-match order, not by the stripe
    fold threshold used in ``_build_stripes``.  A run folded into 'other
    subagents' in the main chart (below ``_STRIPE_MIN_SHARE``) may still
    receive a distinct colour here if its label appears in ``agent_runs``.
    The drill-down shown/hidden cut-off (``_select_subagent_runs``:
    ≤ 8 runs, ≥ $0.10) differs from the stripe threshold, so the two sets
    can diverge.
    """
    consumed = [False] * len(agent_runs)
    for run in shown:
        fill, border = _OTHER_AGENTS_COLOR
        for idx, agent_run in enumerate(agent_runs):
            if not consumed[idx] and agent_run.label == run["label"]:
                consumed[idx] = True
                fill, border = _agent_run_color(idx)
                break
        run["fill"], run["border"] = fill, border


def _build_subagent_runs(
    db: duckdb.DuckDBPyConnection, session_id: str, edited_files: set[str]
) -> tuple[list[dict], dict]:
    """Per-invocation drill-down charts: fetch → group → label → build → select.

    Degrades to an empty section on any failure — the route-level
    except would otherwise blank the whole tab.
    """
    empty: tuple[list[dict], dict] = [], {"count": 0, "cost": 0.0}
    try:
        sidechain_rows = _fetch_chain_rows(db, session_id, sidechain=True)
        if not sidechain_rows:
            return empty
        groups = _group_sidechain_rows(sidechain_rows)
        prompts = _fetch_subagent_prompts(db, session_id)
        merged = _label_and_merge_subagent_groups(db, session_id, groups, prompts)
        # Subagents run under their own `subagentPromptCacheTtl`; the shared
        # rule scores them off the same view, on the sidechain partition.
        ttl_breaks = _fetch_ttl_break_uuids(db, session_id, sidechain=True)
        run_ctxs = []
        for group, label in merged:
            run_ctx = _build_subagent_run_context(
                group, label, edited_files, ttl_breaks
            )
            if run_ctx is not None:
                run_ctxs.append(run_ctx)
        shown, hidden = _select_subagent_runs(run_ctxs)
        for run in shown:
            run["figure_json"] = _render_stripes_figure(
                run.pop("stripes"), run.pop("turns")
            )
    except Exception:
        logger.exception("Failed to build subagent drill-down for %s", session_id)
        return empty
    return shown, hidden


def build_tokenscape_context(db: duckdb.DuckDBPyConnection, session_id: str) -> dict:
    """Everything the Tokenscape tab needs: figure + the tables under it."""
    no_data = {
        "has_data": False,
        "subagent_runs": [],
        "subagent_runs_hidden": {"count": 0, "cost": 0.0},
    }
    edited_files = _fetch_edited_files(db, session_id)
    rows = _fetch_chain_rows(db, session_id, sidechain=False)
    if not rows:
        return no_data

    ttl_breaks = _fetch_ttl_break_uuids(db, session_id, sidechain=False)
    bands, turns = _walk_rows(rows, edited_files, ttl_breaks)
    if not turns:
        return no_data

    generation_costs = _allocate_costs(bands, turns)
    agent_costs, agent_runs = _sidechain_costs_per_turn(db, session_id, turns)
    named = _pick_named_bands(bands)

    figure_json = _render_figure(
        bands, turns, generation_costs, agent_costs, agent_runs, named
    )
    stripes = _build_stripes(bands, turns, generation_costs, agent_costs, agent_runs)
    figure_json_stripes = _render_stripes_figure(stripes, turns)
    bill_split = _bill_split(turns, generation_costs, agent_costs)
    subagent_runs, subagent_runs_hidden = _build_subagent_runs(
        db, session_id, edited_files
    )
    _assign_subagent_run_colors(subagent_runs, agent_runs)

    category_cost: dict[str, float] = dict.fromkeys(_CATEGORY_ORDER, 0.0)
    for band in bands:
        category_cost[band.category] += band.cost
    category_cost["assistant"] += sum(generation_costs)
    category_cost["agent"] += sum(agent_costs)
    total_cost = sum(category_cost.values())

    category_totals = [
        {
            "label": _CATEGORY_LABEL[category],
            "cost": cost,
            "pct": 100.0 * cost / total_cost if total_cost else 0.0,
        }
        for category, cost in sorted(category_cost.items(), key=lambda kv: -kv[1])
        if cost > 0
    ]
    top_bands = [
        {
            "label": band.label,
            "category": _CATEGORY_LABEL[band.category],
            "cost": band.cost,
            "tokens": int(band.tokens),
            "arrival_turn": band.arrival + 1,
            "persisted": band.final_turn - band.arrival + 1,
            "pct": 100.0 * band.cost / total_cost if total_cost else 0.0,
        }
        for band in named
    ]
    swatches = [
        {"label": _CATEGORY_LABEL[c], "color": _CATEGORY_FILL[c]}
        for c in _CATEGORY_ORDER
        if category_cost[c] > 0
    ]
    return {
        "has_data": True,
        "figure_json": figure_json,
        "figure_json_stripes": figure_json_stripes,
        "bill_split": bill_split,
        # Precomputed: total_cost is 0 for unpriced (unknown-model) sessions,
        # so the template must not divide by it.
        "cache_read_pct": (
            100.0 * bill_split["cache_read"] / total_cost if total_cost else 0.0
        ),
        "turn_count": len(turns),
        "compact_count": sum(1 for t in turns if t.event == "compact"),
        "ttl_break_count": sum(1 for t in turns if t.event == "ttl_break"),
        "total_cost": total_cost,
        "category_totals": category_totals,
        "top_bands": top_bands,
        "swatches": swatches,
        "subagent_runs": subagent_runs,
        "subagent_runs_hidden": subagent_runs_hidden,
    }
