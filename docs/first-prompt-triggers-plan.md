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

### ✅ Done — `session_context_loads` view (harness auto-loads)

The automatic-loading signal now lives in the `session_context_loads` view
(`db.py`), reading `type='attachment'` records back out of `raw_data`. One row
per auto-loaded context item: `session_id`, `timestamp`, `load_kind`, `name`,
`char_len`. Classified subtypes and their sources (confirmed against real local
sessions):

| `load_kind` | attachment `type` | `name` | `char_len` |
|---|---|---|---|
| `claude_md` | `nested_memory` | `displayPath` (nested CLAUDE.md / `.claude/rules/*`) | `len(content)` |
| `file_ref` | `file` | `displayPath` (`@`-file expansion) | `len(content)` |
| `skill_listing` | `skill_listing` | — (skill menu shown) | `len(content)` |
| `mcp` | `mcp_instructions_delta` | `addedNames` | `len(addedBlocks)` |
| `hook` | `hook_success` / `hook_non_blocking_error` | `hookName` | `len(content)` |

Chatter subtypes (`output_style`, `total_tokens_reminder`, `task_reminder`,
`deferred_tools_delta`, `agent_listing_delta`, …) are dropped, not mapped to
`other`. **Known limitation:** the *root* project CLAUDE.md and global
`~/.claude/CLAUDE.md` arrive inline as a first-message `<system-reminder>`, not
as an attachment, so they are not captured — near-universal anyway, so low
signal. `nested_memory` covers the differentiating case (per-directory rules).

`first_prompt_triggers` now LEFT JOINs a per-session rollup:
`auto_loaded_claude_md BOOL`, `n_auto_loaded_files INT`,
`skill_menu_loaded BOOL`. `schema-notes.md` documents the `attachment` record
type; `mcp/server.py` INSTRUCTIONS lists the view. Tests:
`tests/test_db.py::test_session_context_loads` and
`tests/test_query_templates.py::test_first_prompt_triggers_auto_load_rollup`.

Robustness note: `read_json_auto` only emits an `attachment` column when some
record carries one, so `_create_raw_tables` / `_create_views` now guarantee the
column (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / a recreated view) — else
the view fails to bind on attachment-free logs and narrow test fixtures.

<details><summary>Original investigation notes (kept for reference)</summary>

**What was missing:** the *automatic* loading signal the original ask centers on
— CLAUDE.md auto-load, `@`-file expansions, skill listings, hooks — was **not
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

</details>

### ✅ Done — web page `/triggers`

- `api/handlers/triggers.py` — one `_BASE_CTE` over `session_stats` +
  `session_context_loads` rollup + Skill-call count, reusing the shared
  `FIRST_PROMPT_PATH_REGEX` constant (exported from `query_templates.py` so the
  page and the template can't drift). Paginated; project-scoped.
- Route `GET /triggers` in `api/routes.py`; template `templates/triggers.html`
  (extends `base.html` / `partial.html` via `parent(request)`); nav link added.
- Table: recent sessions (title · project · started · paths · CLAUDE.md ·
  @-files · skills · menu · commands). Aggregate stat cards: total sessions,
  **share of "vague openers"** (no path *and* no skill), and count that
  auto-loaded a CLAUDE.md/rule.
- Tests: `tests/routes/test_triggers_page.py` (via `_patched_client`).

No chart yet — the aggregate is three stat cards. A future "vague openers over
time" line chart would use the `nolegend` skill.

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

1. ~~Run the `attachment` dumps; pin the true `load_kind` sources.~~ ✅
2. ~~Build `session_context_loads` in `db.py` + tests; update `schema-notes.md`
   and `mcp/server.py` INSTRUCTIONS.~~ ✅
3. ~~Extend `first_prompt_triggers` with the auto-load rollup columns.~~ ✅
4. ~~Build the `/triggers` web page + tests.~~ ✅
5. **Next:** revisit team sharing (still deferred — see options above).
6. Run `/python-review` and `uv run poe check` before each push.
