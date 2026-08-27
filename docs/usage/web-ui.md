# Web UI

A FastAPI application with an HTMX + Alpine.js frontend. It serves the session
list and detail views, tool/command statistics, cost analytics, first-prompt
triggers, and search.

```bash
introspy serve
# Runs on http://127.0.0.1:8347 by default
```

Bind a different port or host:

```bash
introspy serve --port 3000 --host 0.0.0.0
```

If the target port is busy, the server falls forward to the next free port.

!!! warning "Binding to a non-loopback host"
    When the server is bound to a non-loopback host (e.g. `--host 0.0.0.0`), the
    local [SQL API](sql-api.md) is automatically disabled. Only bind to a public
    interface on a trusted network.

## Pages

| Path | Page | What it answers |
|---|---|---|
| `/` | Dashboard | Session count, recent sessions, headline token/cost stats. |
| `/sessions` | Sessions | Every session, filterable by model, project, branch, command, provider, and free text; sortable by start, duration, message counts, tool calls, model, project, branch, provider, title, file counts, or cost. |
| `/sessions/{session_id}` | Session detail | One session across five tabs — see [below](#session-detail-tabs). |
| `/search` | Search | Full-text search across all message types (BM25, ILIKE fallback). |
| `/tools` | Tools | Tool-call statistics; filter by name, session, project, or failures only. |
| `/bash` | Bash | Bash invocations grouped by command prefix, with failure counts. |
| `/mcps` | MCPs | MCP tool calls by server and tool, with failures. |
| `/triggers` | Triggers | What each session's *opening prompt* referenced and loaded. |
| `/stats` | Stats | Insights and analytics across the whole corpus. |
| `/cost-overview` | Cost Overview | Where the money goes, portfolio-wide. |
| `/raw` | Raw Data | The underlying JSONL records, filterable by session and record type. |

HTMX fragment routes (`/cost-overview/breakdown`,
`/cost-overview/breakdown/{day}`, `/cost-overview/portfolio`,
`/sessions/{session_id}/cost/bloat`, `/refresh-status`) back the drill-downs on those
pages; they aren't meant to be visited directly.

## Cost Overview

Portfolio-level cost analysis, all sourced from the same `session_stats.cost_usd`
the sessions list shows, so the numbers always agree.

- **Pareto table** — sessions ranked by cost descending, cut at the first row
  that crosses 80% cumulative share. That crossing row is emphasised, and the
  hero stat reads "N sessions = 80% of $X".
- **Binary splits** — three two-row mini-tables showing the cost impact of
  subagent use, huge reads, and skill/slash-command use. Answer: *does this
  behaviour correlate with expensive sessions?*
- **Daily and hourly charts** — stacked bars of cost per day, optionally split
  by model or project. Clicking a bar drills into that day's hours; clicking an
  hour reruns the portfolio panel under that time window.
- **Cache loss** — the premium paid across the window for caches that went
  cold, split into what a longer TTL would recover and what nothing would. See
  [Cache loss](#cache-loss).
- **Prompt-cache TTL** — both TTL policies replayed over the same requests, with
  the margin between them. See [Prompt-cache TTL](#prompt-cache-ttl).

When you filter by day or hour, the cost aggregation and the Pareto rows narrow
to that window, but the subagent and skill classifiers stay all-time session
properties. "Of the cost incurred at hour 14, this much came from sessions that
ever used a subagent" is the intended reading — requiring the `Task` call to
land *inside* hour 14 would be both noisy and unintuitive.

### Spend shapes

Two small graphics per row on the Pareto table and the Subagents tab, describing
exactly the dollars in that row's Cost column:

- **Split bar** — a fixed-width bar divided into cache read (grey), cache write
  (blue), and output (orange). It answers *what kind of tokens did this money
  buy?* A bar that is almost all cache-write means the session kept rebuilding
  context; almost all output means it kept generating.
- **Sparkline** — cost over time within the session. The x-axis is elapsed
  session time and the y-axis is per-message cost; bar **width** scales with
  `sqrt(duration / longest duration in the table)`, so a wide sparkline is a
  long session and a narrow one is a short burst. It answers *was the money
  spent evenly, or in a spike?*

Both are omitted (em-dash) when there is nothing to draw.

## Session detail tabs

| Tab | What it answers |
|---|---|
| Messages | What actually happened, turn by turn — prompts, thinking, tool calls, results, with cache-loss dividers inline. |
| Cost | Where this session's tokens and dollars went, by model and by *bloat* bucket. |
| Tokenscape | Which piece of content burned the money, and for how long. |
| Trajectory | How the session behaved — the tool-call sequence as a glyph strip. |
| Subagents | Which agent spent what (only shown when the session used subagents). |

### Cost tab

A per-model token/cost rollup, plus **bloat** tables: every unit of context the
session paid for, bucketed by what produced it (`file read: db.py`,
`bash: uv run`, `WebFetch`, a write, a conversation turn) and rolled up by
category. The question is *which specific reads and commands are responsible for
this bill?* Selecting a range of messages re-scopes the tables to that range.

### Tokenscape

Each piece of content the model saw becomes a **band** — a horizontal stripe
that exists from the turn it arrived until `/compact` discards it.

- **x-axis**: turns, in order.
- **y-axis**: **dollars per turn**. Every column sums to what that API call
  actually cost, so a stripe's *area* is the total money that piece of content
  burned across the session.

Attribution is anchored to API truth rather than heuristics.
`context_t = input + cache_read + cache_creation`; the difference between
consecutive turns is the exact token count of newly arrived content. The
assistant's share of that delta is the previous turn's `output_tokens` (exact —
it covers redacted thinking too), and the remainder is split across user-side
blocks (prompts, tool results, skill loads) in proportion to character count.
The first turn's residual after visible content is the system prompt plus tool
definitions; the same logic after `/compact` yields the compact summary.

Newly arrived bands pay the cache-write rate; persisting bands split the
remaining input bill proportionally to size; output tokens are billed on the
turn they're generated. Sidechain (subagent) calls are bucketed onto the
main-chain turn during which they ran, so column totals tie out to the real
bill.

The question it answers: *that 40-file read on turn 6 — how much did it cost me
for the rest of the session?*

### Trajectory

The session's tool calls as a left-to-right strip of glyph tiles, one per call,
split into three phases:

- **locate** — everything before the first edit.
- **implement** — from the first edit to the start of the trailing verification
  streak.
- **verify** — from the first verify signal with no edit after it, to the end.

The verify boundary is deliberately the *trailing* streak, so writing a failing
test first (TDD) or a red/green fix loop doesn't collapse the implement phase.

Two granularities: **category** buckets every call (Read, Edit, Write, search,
test, git, pkg, agent, web, mcp, bash) into one glyph; **detail** labels Bash
tiles by their two-word prefix (`git status`, `uv run`) and other tiles by file
basename, so you see *what* ran rather than just the bucket.

Alongside it: total calls, locate-phase length, distinct files touched, maximum
re-reads of a single file, search count, edit count, and which category the
session reached for first. The question it answers: *did this session thrash —
reflexive grep, the same file read six times, a locate phase that never
converged?* That behaviour is invisible in token totals.

### Subagents

One row per agent — the "Main agent" (the non-sidechain part of the session) and
one row per `Task`/`Agent` invocation — sorted cost-descending, with the same
[spend shapes](#spend-shapes) as the Pareto table plus files read and edited.
The question it answers: *was the expensive part of this session me, or a
subagent I fired and forgot?*

## Cache loss

Anthropic's ephemeral prompt cache expires. Walk away mid-session — or fire a
tool that runs for ten minutes — and the next request pays to rebuild context
that would otherwise have been a cheap cache read.

**The rule.** One rule, in one place: the `cache_requests` relation (see
[Cache TTL](../architecture.md#cache-ttl-cache_ttlpy)). A request is a cache
miss when the gap since the previous request outran the TTL that request was
*actually billed at*, and the prefix did not change underneath it. The gap runs
from the end of the previous response to whatever triggered the next one — a
human prompt **or** a tool result. Sidechain turns are scored separately;
subagents have their own `subagentPromptCacheTtl`.

Two things it deliberately does not count:

- **Structural invalidations.** Reading back almost nothing after a sub-5-minute
  gap means the prefix itself changed — a model or effort switch, `/compact`, a
  tool-set change — not that time ran out. Those cost the same under any TTL.
- **Breaks.** A gap longer than an hour is past the longest TTL Claude Code
  offers, so no setting recovers it. Reported separately, never as waste.

**The cost.** The reported figure is the *premium*: the prefix a warm cache
would have read, billed at the write rate instead. It is scoped to that shared
prefix — the tokens the new message itself added would have been written under
any policy.

Events appear inline in the Messages tab as a divider reading
`cache lost · N min gap · ~$X wasted` (or `break · N min gap · no TTL recovers
this`), as "Cache losses" and "Cache breaks" lines in the session header, and as
"Recoverable cache waste" plus "Breaks > 60 min" stats on the Cost Overview
portfolio panel, which follows the same day/hour window as the rest of it.

**Waste is not the whole question.** Zero recoverable waste does not mean
nothing to fix: a 1h TTL charges 2× input on *every* incremental write, so it
can cost money on a session that never pauses at all. The
[TTL panel](#prompt-cache-ttl) beside these stats prices both policies; the CLI
does the same with `introspy cache-ttl`.

**Known limits.**

- The premium is a **lower bound**: it prices the rebuild itself, not the
  secondary cost of later turns reading a cache they should never have had to
  re-create.
- A structural invalidation with a gap between 5 and 60 minutes is
  indistinguishable from a real miss, so it is counted as one.

## Prompt-cache TTL

Sitting beside the cache-loss stats on the Cost Overview portfolio panel: would
a 1h or a 5m `promptCacheTtl` have been cheaper, over the sessions in view?

Both policies are replayed over the same requests. The prefix a request re-sends
is a property of the conversation, not of the setting, so only the read/write
split moves — which makes the comparison a like-for-like one. The panel reports
each policy's total, the margin as a percentage, how many gaps a longer TTL
would rescue, how many are breaks past its reach, and which TTL the sessions
were actually billed at.

**Read the margin, not the sign.** Under about 2% the panel says the two are too
close to call rather than naming a winner — the simulation carries real
modelling error (estimated overlaps on cold requests, gaps measured from log
timestamps), and a 1% edge is inside it.

Subagents are excluded; they carry their own `subagentPromptCacheTtl`. Costs are
list API prices, so on a subscription plan treat the result as a ratio as much
as a dollar figure.

## Triggers

`/triggers` asks one question per session: **did the opening prompt set the
session up properly?**

Columns: file/dir paths named in the first prompt, whether a nested `CLAUDE.md`
or `.claude/rules` file auto-loaded, how many `@`-file expansions the harness
performed, how many skills ran, whether the skill menu was shown, and which
slash commands were used — next to the first 200 characters of the prompt
itself.

Paths are found by a heuristic regex: `@mentions`, rooted paths (`/a/b`,
`./a/b`, `~/a/b`), and any token with a slash plus a file extension. It
deliberately misses bare filenames without a slash (`fix db.py`) so that prose
like "e.g." doesn't match. Skills are counted from `Skill` tool calls, which
covers both skills the user typed as `/name` and skills the model auto-triggered.

The root project `CLAUDE.md` and the global `~/.claude/CLAUDE.md` arrive inline
as a first-message `<system-reminder>` rather than as attachments, so they are
**not** counted in the CLAUDE.md column — that column is about *nested* memory
files loaded on directory entry.

The row to look for: zero referenced paths and zero skills. Those sessions gave
the model no anchor. The same data is available over MCP as the
[`first_prompt_triggers` prompt](mcp.md#prompts).

## Refreshing data

A background loop polls the JSONL files and rebuilds the database automatically
(every 10 minutes by default). You can also trigger a rebuild from the "Refresh
now" button, and scope how much history is loaded with the window picker
(`1` / `7` / `30` days, or the current calendar month).

See [Configuration](../configuration.md) for the environment variables that
control refresh behaviour.

## What the web UI does not do

- It is a **read-only** view of your logs. Nothing you do in the UI changes a
  conversation; the only write is rebuilding the derived DuckDB.
- It has no authentication. Binding it to a non-loopback host exposes every
  conversation on your machine to that network.
- Cost figures are computed from a hardcoded price snapshot and exclude
  request-level modifiers (fast mode, US-pinned inference, OpenAI's
  long-context tier) — see [Architecture](../architecture.md#pricing-pricingpy).
