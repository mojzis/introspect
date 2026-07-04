# First-prompt triggers — analysis plan

Goal: help people see how a session's **opening prompt** set it up — which
files/dirs it referenced, whether that auto-loaded context (CLAUDE.md,
`@`-file expansions, skills), and which skills/commands it triggered — and
eventually share those patterns across a team.

## Why this lives in the MCP, not a skill

The know-how ships through **`query_templates.py`**, the registry that already
fans out three ways (`mcp/_register.py`):

- **cookbook** — every entry renders in the `list_query_templates` MCP tool as
  adaptable reference SQL;
- **`kind="deterministic"`** → auto-registered as a dedicated MCP tool;
- **`kind="exploratory"`** → auto-registered as an MCP prompt that seeds an
  investigation.

A build-time drift check (`_wire_template_adapters`) fails if an entry and its
adapter get out of sync. This is strictly better than a skill for spreading
know-how: it's versioned in the repo, ships with the server, and nobody has to
install or update anything. **Rule of thumb: new session-analysis know-how =
a new registry entry, not a skill.**

## Status

### ✅ Done in this branch — `first_prompt_triggers` exploratory template

Added to `query_templates.py` (+ prompt in `mcp/prompts.py`, wired in
`_register.py`, tested in `tests/test_query_templates.py`). Per session it
returns:

| column | source | meaning |
|---|---|---|
| `first_prompt` | `session_stats.first_prompt` | the opening prompt (first 200 chars) |
| `referenced_paths` | heuristic regex over `first_prompt` | file/dir paths the prompt named |
| `n_referenced_paths` | `len(referenced_paths)` | count, for quick filtering |
| `skills_invoked` | `Skill` tool calls in `session_messages_enriched` | skills that ran (user-typed **or** auto-triggered) |
| `commands` | `message_commands` | distinct slash commands used |

Try it: MCP prompt `first_prompt_triggers`, or via `run_sql` from the cookbook
(`list_query_templates(kind="exploratory")`). The payoff is comparing prompt
*wording* to outcome — e.g. sessions with `n_referenced_paths=0` **and**
`skills_invoked=0` are candidates for a vaguer opening prompt.

**Path regex** (heuristic, deliberately conservative):
`(?:@[\w./~-]+)|(?:[~.]?/[\w.~-]+/[\w./~-]+)|(?:[\w-]+/[\w./~-]*\.[\w]+)`
Catches `@mentions`, rooted paths (`/a/b`, `./a/b`, `~/a/b`), and tokens with a
slash + file extension. Skips bare filenames without a slash (`fix db.py`) to
avoid matching prose like `e.g.`. Widen it if you want bare filenames.

### ⏳ To do locally — the big gap: harness auto-loads

**What's missing:** the *automatic* loading signal the original ask centers on
— CLAUDE.md auto-load, `@`-file expansions, skill listings, hooks — is **not
in any view**. Those live in `type='attachment'` records, and every derived
view (`raw_messages` and everything built on it) filters to
`type IN ('user','assistant')`, dropping attachments entirely.

They *are* still in `raw_data` (unfiltered `SELECT *`). Confirm the shapes on
your real local sessions first:

```sql
-- what attachment subtypes do YOUR local sessions emit?
SELECT json_extract_string(attachment, '$.type') AS atype, count(*)
FROM raw_data WHERE type = 'attachment' GROUP BY 1 ORDER BY 2 DESC;
```

⚠️ The web session I built against emits `skill_listing`,
`mcp_instructions_delta`, `deferred_tools_delta`, `agent_listing_delta`,
`hook_success`, `task_reminder`. **Local CLI sessions likely differ** — in
particular `@`-file expansions and CLAUDE.md auto-load may appear as a
different attachment subtype (or inline in the first user message as a
`<system-reminder>` block). Dump a few real records and pin the actual keys
before building the view:

```sql
SELECT attachment FROM raw_data
WHERE type = 'attachment' AND sessionId = '<a session you know auto-loaded CLAUDE.md>';
```

Also check whether CLAUDE.md / `@`-files arrive as attachments **or** as
`<system-reminder>`-prefixed text inside the first `user` message. `tokenscape`
already classifies skill-load pseudo-messages (user text starting with
`"Base directory for this skill"`) and `<system-reminder>` blocks — see
`api/handlers/tokenscape.py` `_classify_block` / `_SKILL_TEXT_PREFIX` for the
existing precedent to reuse.

#### Proposed view: `session_context_loads`

Add to `db.py` alongside the other derived views (both the materialized-table
and lazy-view paths go through `_create_derived_views` → `_make(...)`):

- One row per auto-loaded context item: `session_id`, `timestamp`,
  `load_kind` (`claude_md` | `file_ref` | `skill_listing` | `hook` | `mcp` |
  `other`), `name` (file path / skill name), `char_len` (from `content`).
- Source: `raw_data WHERE type='attachment'`, `json_extract`-ing the
  `attachment` column — plus, if CLAUDE.md/`@`-files turn out to be inline
  `<system-reminder>` text, a second branch over the first `user` message.
- Register indexes in `_DERIVED_INDEXES`; drop-list the new name at the top of
  `materialize_views` (the `for name in (...)` block).
- Update `schema-notes.md` (document the `attachment` record type — currently
  undocumented) and the MCP `INSTRUCTIONS` in `mcp/server.py`.

Then extend `first_prompt_triggers` to LEFT JOIN a per-session rollup of
`session_context_loads` (e.g. `auto_loaded_claude_md BOOL`,
`n_auto_loaded_files INT`, `auto_loaded_skills`), so a single row shows
prompt → referenced → *actually loaded*. Consider a companion
`kind="deterministic"` template/tool once the columns are stable.

**Tests:** add attachment records to a conftest fixture (extend
`make_*`/`write_jsonl` or add a `make_attachment_message` helper) and assert
the view classifies each `load_kind`. Follow the fixture pattern in
`tests/test_query_templates.py`.

### ⏳ To do locally — web page `/triggers`

Per the "Adding a page" pattern in `CLAUDE.md`:

- `api/handlers/triggers.py` — query `session_stats` + `session_context_loads`
  rollup; reuse `_helpers.py` (`parent`, `conn`, pagination, sort allowlists).
- Route in `api/routes.py`; template `templates/triggers.html` (extends
  `base.html` / `partial.html` via `parent(request)` for HTMX).
- Suggested view: a table of recent sessions (first prompt · referenced paths ·
  auto-loaded files · skills · commands), plus an aggregate — e.g. share of
  sessions whose opening prompt referenced no path and loaded no skill, over
  time. If you chart it, use the `nolegend` skill (server-side `go.Figure` +
  `nolegend.activate()`, embed JSON for `Plotly.newPlot`).
- Tests in `tests/routes/` (use the `_patched_client()` context manager).

### ⏳ Later — team sharing (design deferred)

Not built yet — decide the shape once the analysis columns are proven. Options,
smallest first:

1. **Export command/tool** — `introspect export triggers --format {csv,json,md}`
   (Typer command in `cli.py` + optionally an MCP tool) that dumps the
   per-session trigger table to hand to a team. Lowest lift.
2. **Aggregate report** — a shareable markdown/HTML summary ("this week: N% of
   openers referenced the right project, top skills auto-triggered, …").
3. **Cross-teammate aggregation** — the real feature: combine multiple people's
   logs. Needs a data-collection/privacy story (what leaves a laptop, prompt
   redaction) — treat as its own project.

## Suggested local sequence

1. Run the `attachment` dumps above on your real sessions; pin the true
   `load_kind` sources (attachment subtypes vs. inline `<system-reminder>`).
2. Build `session_context_loads` in `db.py` + tests; update `schema-notes.md`
   and `mcp/server.py` INSTRUCTIONS.
3. Extend `first_prompt_triggers` with the auto-load rollup columns.
4. Build the `/triggers` web page + tests.
5. Revisit team sharing.
6. Run `/python-review` and `uv run poe check` before each push.
