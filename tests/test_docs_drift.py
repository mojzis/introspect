"""Guard the published docs against drifting behind the code.

Every check here enumerates something the code *defines* — a CLI command, an
environment variable, a relation, an MCP tool or prompt, a query template, a
route — and asserts the docs mention it. The failure message always names the
missing item and the file expected to mention it, so the fix is obvious.

These are mention checks, not prose reviews: they catch "we shipped a feature
and forgot the docs entirely", which is the failure mode that actually happens.
They cannot tell you a paragraph is out of date.

When one fails, fix the docs. Adding an item to an allowlist is only correct
when the item genuinely isn't user-facing.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "introspect"
DOCS_ROOT = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

# Planning notes live here and are excluded from the mkdocs build. Nothing in
# this directory counts as documenting anything.
PLANS_DIR = DOCS_ROOT / "plans"


def _published_docs() -> dict[str, str]:
    """Map ``docs``-relative path -> text, for every published page."""
    return {
        str(path.relative_to(DOCS_ROOT)): path.read_text()
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if PLANS_DIR not in path.parents
    }


def _doc_text(rel_path: str) -> str:
    """Text of one published doc page."""
    return (DOCS_ROOT / rel_path).read_text()


def _assert_mentioned(items: set[str], rel_path: str, kind: str) -> None:
    """Fail naming every `items` entry absent from the doc at `rel_path`."""
    text = _doc_text(rel_path)
    missing = sorted(item for item in items if item not in text)
    if missing:
        pytest.fail(
            f"docs/{rel_path} does not mention {len(missing)} {kind}: "
            f"{', '.join(missing)}. Document them there (or, if one is not "
            f"user-facing, add it to the allowlist in {Path(__file__).name})."
        )


# --- CLI commands -------------------------------------------------------------


def _typer_command_names() -> set[str]:
    """Every command registered on the Typer app, from the app itself."""
    from typer.main import get_group  # noqa: PLC0415

    from introspect.cli import app  # noqa: PLC0415

    return set(get_group(app).commands)


def test_every_cli_command_is_documented() -> None:
    """Each Typer command appears in the hand-written CLI page."""
    _assert_mentioned(_typer_command_names(), "usage/cli.md", "CLI command(s)")


def _load_cli_doc_generator():
    """Load scripts/gen_cli_docs.py by path.

    `scripts/` is not an importable package, so this goes through importlib
    rather than a plain import (which the type checker can't resolve either).
    """
    spec = importlib.util.spec_from_file_location(
        "gen_cli_docs", REPO_ROOT / "scripts" / "gen_cli_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_is_regenerated() -> None:
    """The generated per-command reference matches the current --help output."""
    generator = _load_cli_doc_generator()
    expected = generator.render()
    actual = generator.OUTPUT_PATH.read_text()
    assert actual == expected, (
        "docs/usage/cli-reference.md is stale — run `uv run poe docs-cli`."
    )


# --- Environment variables ----------------------------------------------------

# Any INTROSPECT_-prefixed name appearing in src/ is treated as an environment
# variable that users can set, and must appear in the configuration page. The
# scan is deliberately over-eager (it also catches names mentioned only in a
# comment) because the failure mode it guards against is an undocumented knob.
#
# Add a name here only when it is an internal identifier that merely shares the
# prefix — and prefer renaming it, so the prefix keeps meaning exactly one
# thing.
NON_ENV_INTROSPECT_NAMES: frozenset[str] = frozenset()

_ENV_NAME_RE = re.compile(r"\bINTROSPECT_[A-Z0-9_]+\b")


def _introspect_env_names() -> set[str]:
    """Every INTROSPECT_* name referenced anywhere under src/."""
    found: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        found.update(_ENV_NAME_RE.findall(path.read_text()))
    return found - NON_ENV_INTROSPECT_NAMES


def test_every_env_var_is_documented() -> None:
    """Each INTROSPECT_* variable read by the code is in the config table."""
    _assert_mentioned(
        _introspect_env_names(), "configuration.md", "environment variable(s)"
    )


def test_env_vars_are_documented_only_in_configuration() -> None:
    """The config table is the single copy — no page re-tabulates it.

    architecture.md used to carry a second copy that drifted. Any page other
    than configuration.md may *reference* a variable, but must not list a
    majority of them, which is what a duplicated table looks like.
    """
    names = _introspect_env_names()
    threshold = len(names) // 2
    for rel_path, text in _published_docs().items():
        if rel_path == "configuration.md":
            continue
        mentioned = {name for name in names if name in text}
        assert len(mentioned) <= threshold, (
            f"docs/{rel_path} mentions {len(mentioned)} of {len(names)} "
            "environment variables — that looks like a second copy of the "
            "configuration table. Keep one copy in configuration.md and link "
            "to it."
        )


# --- Relations ----------------------------------------------------------------


def _drop_list_relations() -> set[str]:
    """Relation names from the drop-list at the top of ``materialize_views``.

    That tuple is the canonical list: every relation the materializer creates
    must be dropped first, so nothing can be added without appearing there.
    Read via ``ast`` rather than importing, so a syntax-level change surfaces
    here rather than as an import error.
    """
    tree = ast.parse((SRC_ROOT / "db.py").read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_views"
    )
    for node in ast.walk(func):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            names = {
                element.value
                for element in node.iter.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if names:
                return names
    pytest.fail(
        "Could not find the relation drop-list in materialize_views() — "
        "if it was restructured, update _drop_list_relations() to match."
    )


def test_every_relation_is_documented() -> None:
    """Each relation materialize_views() creates has a row in the docs."""
    _assert_mentioned(_drop_list_relations(), "architecture.md", "relation(s)")


# --- MCP tools and prompts ----------------------------------------------------


def _registered_mcp_names() -> tuple[set[str], set[str]]:
    """(tool names, prompt names) as actually registered on a FastMCP server.

    Introspected from a live server instance rather than grepped, so tools
    generated from the query-template registry are included exactly as a
    client would see them.
    """
    from introspect.mcp.server import create_mcp_server  # noqa: PLC0415

    server = create_mcp_server()

    async def _collect() -> tuple[set[str], set[str]]:
        tools = await server.list_tools()
        prompts = await server.list_prompts()
        return {t.name for t in tools}, {p.name for p in prompts}

    return asyncio.run(_collect())


def test_every_mcp_tool_is_documented() -> None:
    """Each registered MCP tool, generated ones included, is in the MCP page."""
    tools, _ = _registered_mcp_names()
    _assert_mentioned(tools, "usage/mcp.md", "MCP tool(s)")


def test_every_mcp_prompt_is_documented() -> None:
    """Each registered MCP prompt is in the MCP page."""
    _, prompts = _registered_mcp_names()
    _assert_mentioned(prompts, "usage/mcp.md", "MCP prompt(s)")


# --- Query templates ----------------------------------------------------------


def test_every_query_template_is_documented() -> None:
    """Each entry in the query-template registry is named in the MCP page."""
    from introspect.query_templates import QUERY_TEMPLATES  # noqa: PLC0415

    names = {template.name for template in QUERY_TEMPLATES}
    _assert_mentioned(names, "usage/mcp.md", "query template(s)")


# --- Routes -------------------------------------------------------------------

# HTMX fragment endpoints: they exist to be swapped into a page, not visited.
# The pages that use them are documented; the fragments themselves need no
# entry of their own.
FRAGMENT_ROUTES: frozenset[str] = frozenset(
    {
        "/sessions/{session_id}/cost/bloat",
        "/cost-overview/breakdown",
        "/cost-overview/breakdown/{day}",
        "/cost-overview/portfolio",
        "/refresh",
        "/refresh-status",
    }
)

# Routes documented somewhere other than the web-UI page.
ROUTE_DOC_OVERRIDES: dict[str, str] = {
    "/api/query": "usage/sql-api.md",
    "/api/schema": "usage/sql-api.md",
}

_ROUTE_DECORATOR_RE = re.compile(
    r"@router\.(?:get|post|put|patch|delete)\(\s*\"([^\"]+)\""
)


def _route_paths() -> set[str]:
    """Every path registered in routes.py."""
    return set(
        _ROUTE_DECORATOR_RE.findall((SRC_ROOT / "api" / "routes.py").read_text())
    )


def test_every_route_is_documented() -> None:
    """Each non-fragment route path appears in its documenting page."""
    by_doc: dict[str, set[str]] = {}
    for path in _route_paths() - FRAGMENT_ROUTES:
        rel_path = ROUTE_DOC_OVERRIDES.get(path, "usage/web-ui.md")
        by_doc.setdefault(rel_path, set()).add(path)
    for rel_path, paths in sorted(by_doc.items()):
        _assert_mentioned(paths, rel_path, "route(s)")


def test_fragment_route_allowlist_is_current() -> None:
    """The fragment allowlist names only routes that still exist."""
    stale = sorted(FRAGMENT_ROUTES - _route_paths())
    assert not stale, (
        f"FRAGMENT_ROUTES lists routes that no longer exist in routes.py: "
        f"{', '.join(stale)}. Remove them."
    )


# --- mkdocs nav / llmstxt parity ----------------------------------------------


def _load_mkdocs_config() -> dict:
    """Parse mkdocs.yml, ignoring the `!!python/name:` tags Material uses."""

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
    )
    return yaml.load(MKDOCS_YML.read_text(), Loader=_Loader)


def _nav_pages(nav: object) -> set[str]:
    """Flatten a mkdocs nav tree into the set of page paths it references."""
    if isinstance(nav, str):
        return {nav}
    if isinstance(nav, list):
        return {page for entry in nav for page in _nav_pages(entry)}
    if isinstance(nav, dict):
        return {page for value in nav.values() for page in _nav_pages(value)}
    return set()


def _llmstxt_sections(config: dict) -> dict[str, list[str]]:
    """The llmstxt plugin's `sections` mapping from the plugins list."""
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "llmstxt" in plugin:
            return plugin["llmstxt"].get("sections", {})
    pytest.fail("mkdocs.yml has no llmstxt plugin with a `sections` mapping.")


def test_nav_and_llmstxt_cover_the_same_pages() -> None:
    """Every published page is in both the nav and llms-full.txt, or neither."""
    config = _load_mkdocs_config()
    nav = _nav_pages(config["nav"])
    sections = {page for pages in _llmstxt_sections(config).values() for page in pages}
    only_nav = sorted(nav - sections)
    only_sections = sorted(sections - nav)
    assert not only_nav, (
        f"Pages in mkdocs.yml `nav` but missing from `llmstxt.sections`, so "
        f"they never reach llms-full.txt: {', '.join(only_nav)}."
    )
    assert not only_sections, (
        f"Pages in `llmstxt.sections` but missing from `nav`: "
        f"{', '.join(only_sections)}."
    )


def test_every_published_page_is_in_the_nav() -> None:
    """No published page is orphaned outside the nav."""
    nav = _nav_pages(_load_mkdocs_config()["nav"])
    orphans = sorted(set(_published_docs()) - nav)
    assert not orphans, (
        f"Pages under docs/ that no nav entry references: "
        f"{', '.join(orphans)}. Add them to `nav` and `llmstxt.sections`, or "
        "move them to docs/plans/ if they are planning notes."
    )


def test_no_planning_page_is_published() -> None:
    """Nothing under docs/plans/ reaches the nav, llms-full.txt, or the build."""
    config = _load_mkdocs_config()
    referenced = _nav_pages(config["nav"]) | {
        page for pages in _llmstxt_sections(config).values() for page in pages
    }
    leaked = sorted(page for page in referenced if page.startswith("plans/"))
    assert not leaked, (
        f"Planning pages referenced by mkdocs.yml: {', '.join(leaked)}. "
        "docs/plans/ is excluded from the build and must not appear in `nav` "
        "or `llmstxt.sections`."
    )
    excluded = config.get("exclude_docs", "")
    assert "plans/" in excluded, (
        "mkdocs.yml `exclude_docs` no longer excludes plans/ — planning notes "
        "would be published and swept into llms-full.txt."
    )
