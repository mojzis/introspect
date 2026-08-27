---
name: docs-review
context: fork
description: Docs drift review for user-visible changes. Auto-invoke when finishing a task, before marking work complete, or when preparing a PR — after `/python-review`. Checks whether the changes in the diff are reflected in `docs/`, `README.md`, and `CLAUDE.md`, and fixes what is missing or stale.
---

# Docs Review

Judgment gate for prose. Ask of every user-visible change in the diff: does the
owning doc still describe **what this does, what question it answers, and where
it stops being true**? Report 🔴 Must Fix / 🟡 Should Fix / 🟢 Suggestion, then
apply the 🔴s.

**Scope is the diff. Nothing else.** A page that was already wrong before this
diff is not this review's problem.

This is not the drift test. `tests/test_docs_drift.py` is the hard gate for
things that can be enumerated — commands, env vars, relations, MCP tools and
prompts, templates, routes. Do not re-implement it here, and do not hand-check
what it already checks. This skill covers what a test cannot: whether the
sentences around those names are still true.

**Until that test exists**, the enumerable cases are unguarded, so check them
too: any name the diff adds or removes must appear in (or disappear from) the
list that enumerates its kind — `architecture.md`'s views table and module
tree, its env-var table, `usage/mcp.md`'s tool table, `usage/cli.md`'s command
table, and `CLAUDE.md`'s Architecture and Views/tables lists. These lists are
exhaustive by construction; a gap in one is a 🔴.

---

## 1. Get the diff

```bash
git diff main...HEAD          # dirty tree with no branch commits: git diff HEAD
```

If nothing in the diff touches `src/`, `pyproject.toml`, `mkdocs.yml`, or an
agent-config directory (`.claude/`, `.agents/`, `.codex/`), print
**"no user-visible change"** and stop. No findings, no edits.

## 2. Classify each hunk

Map every changed hunk to its owning doc. This table is the review — work it
row by row, not by intuition.

| Change in diff | Owning doc(s) |
|---|---|
| Typer command / option / help text (`cli.py`) | `usage/cli.md` (regenerated section) |
| `os.environ` / `os.getenv` read | `configuration.md` |
| Relation added/removed/columns changed (`db.py` drop-list, `_make(...)`) | `architecture.md` views table; `schema.md` if it changes how raw JSONL is read |
| Pricing rates, cache split, cost formula (`pricing.py`, `sql_fragments.py`) | `architecture.md` pricing section; any page that shows a cost number |
| MCP tool / prompt registered (`mcp/_register.py`, `mcp/prompts.py`) | `usage/mcp.md` |
| Query template (`query_templates.py`) | `usage/mcp.md` cookbook section; `architecture.md` registry section |
| HTML route / handler / template (`api/`, renders Jinja) | `usage/web-ui.md` — the *question it answers and its axes*, not the handler name |
| JSON API route / handler (`api/handlers/query.py`, `sql_query.py`) | `usage/sql-api.md` — request/response shape, row cap, loopback gating |
| Static asset / PWA manifest / `/static` mount | `usage/web-ui.md` if a user can see or install it; otherwise no doc |
| Codex transcoding (`codex.py`) | `schema.md` Codex section; wherever "Claude-only" caveats live |
| Refresh / window / materialize behaviour (`refresh.py`, lifespan) | `configuration.md` refresh section; `usage/web-ui.md` |
| Detection rule / threshold constant (e.g. cache-loss, tokenscape events, trajectory phase split) | the page that describes the visual, plus its stated limits |
| Cross-cutting library module (`sql_query.py`, non-cost rollups in `sql_fragments.py`, `mcp/server.py` `SERVER_INSTRUCTIONS`) | every page that states its limits — grep for the guarantee, not the module name |
| New outbound network call, cache file, or anything written outside the DB | `configuration.md` (the switch + what leaves the machine); `installation.md` |
| Dev-workflow behaviour (`devserve`, per-branch DB, worktrees, pre-commit) | `development.md` — *and* `usage/cli.md` if it is a command flag |
| Skill / hook / `settings.json` under `.claude/`, `.agents/`, `.codex/` | `CLAUDE.md`; `development.md` |
| `pyproject.toml` deps / entry points / description | `installation.md` (incl. version caps a user can hit); mkdocs `site_description` + `llmstxt.markdown_description` if the one-liner changed |
| Removed feature | every page above that mentions it — **grep, don't assume** |

A hunk under `src/` that matches no row is a 🟡: *"unmapped change, decide if
user-visible."* Never drop it silently — an unmapped hunk is either a gap in
this table or a gap in the docs.

## 3. Check each mapped page

Open the owning page. Three questions, in this order:

1. **Is the change mentioned at all?** Missing → 🔴.
2. **Does the prose still describe behaviour and limits correctly?** A renamed
   threshold, a changed default, a caveat that no longer applies, a new caveat
   that does, a scope sentence that has quietly widened ("reads
   `~/.claude/projects/**`" when it now reads two trees). Stale → 🔴.
3. **For a new analytic: does the page say what question it answers?** A page
   that names a feature without saying what you would use it for is not
   documented. Absent → 🟡.

## 4. Cross-page consistency — diff-scoped only

If the diff touched something that appears on more than one page, check every
copy agrees: `README.md` quick-start vs `installation.md`, the config table if
it is still duplicated in `architecture.md` and `configuration.md`, the
"what you can ask" examples in `index.md`. Only for the thing the diff touched.

## 5. Structural checks — cheap, always run

- New page → present in **both** `mkdocs.yml` `nav` **and** `llmstxt.sections`.
- A plan or design doc (`docs/plans/**`, or a root-level `*-plan.md`) → in
  **neither**.
- Reference sections that enumerate code (CLI reference, env-var table, MCP
  tool table) are **hand-written today** — this repo has no `mkdocstrings`,
  `gen-files`, or `mkdocs-typer` generator. Edit them by hand and keep them
  exhaustive. If a generator is ever added, regenerate instead, and confirm
  `.github/workflows/docs.yml` `paths` covers the source that feeds it — today
  that workflow watches `docs/**`, `mkdocs.yml`, `pyproject.toml`, `uv.lock`, and
  its own file, so a `src/` change never rebuilds the site on its own.

## 6. Report, then fix

Report in the `/python-review` format:

- 🔴 **Must Fix** — missing or stale docs for a user-visible change in this diff.
- 🟡 **Should Fix** — unmapped hunk; missing "what question does this answer".
- 🟢 **Suggestion** — including pre-existing staleness, explicitly labelled
  out of scope.

Each finding: **file** + **section** + **one line of what to write**.

Then **apply every 🔴 directly** — small, scoped edits inside the owning page —
and list what was edited. Do not touch pages outside this diff's rows in the
ownership map.

## 7. Exit criteria

```bash
uv run mkdocs build --strict
uv run pytest tests/test_docs_drift.py    # once it exists
```

Both must pass. If the drift test fails on something this diff introduced, that
is a 🔴 no matter how good the prose is.

---

## Non-goals

- **Not a docs rewrite.** Page already stale before this diff → 🟢 "pre-existing,
  out of scope", and move on.
- **Not a style pass.** No reflowing, no hunting for "simply", no tightening
  paragraphs the diff didn't touch.
- **Not a changelog.** Conventional Commits + git-cliff own release notes. This
  skill owns reference docs.

$ARGUMENTS
