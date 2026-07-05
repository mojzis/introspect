# Installation

Introspect is published on PyPI as
[`introspy`](https://pypi.org/project/introspy/).

## Prerequisites

You need [uv](https://docs.astral.sh/uv/). If you don't have it (it also takes
care of Python for you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Just trying it out?

Run it directly — nothing to install:

```bash
uvx introspy@latest serve
```

Use `introspy@latest` rather than a bare `uvx introspy`: `uvx` reuses its cached
environment indefinitely, so the bare form silently pins you to whatever version
it first downloaded. The `@latest` suffix always fetches the current release.

## Using it regularly?

Install it as a tool, then call it by name:

```bash
uv tool install introspy
introspy serve
```

Installed tools stay pinned until you upgrade them, so pull new releases with:

```bash
uv tool upgrade introspy
```

Both `introspy` and `introspect` are installed as entry points — they are
interchangeable.

Introspect checks PyPI in the background at most once a day and prints a
one-line hint to stderr when a newer release is available. It never
self-updates, sends no data beyond a plain request for the `introspy` release
metadata, and can be turned off with `INTROSPECT_VERSION_CHECK=off` — see
[Configuration](configuration.md#update-check).

## Where your data comes from

Introspect reads Claude Code conversation logs from
`~/.claude/projects/**/*.jsonl` and materializes them into a DuckDB database at
`~/.introspect/introspect.duckdb`. Your logs never leave your machine —
everything runs locally. The one exception is the daily
[update check](configuration.md#update-check): a background request to PyPI for
the latest `introspy` version, carrying no logs and no identifiers, which you
can disable with `INTROSPECT_VERSION_CHECK=off`. See
[Configuration](configuration.md) to point it at a different location or scope
how much history is loaded.
