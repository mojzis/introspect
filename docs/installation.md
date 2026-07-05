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
uvx introspy serve
```

## Using it regularly?

Install it as a tool, then call it by name:

```bash
uv tool install introspy
introspy serve
```

Both `introspy` and `introspect` are installed as entry points — they are
interchangeable.

## Where your data comes from

Introspect reads Claude Code conversation logs from
`~/.claude/projects/**/*.jsonl` and materializes them into a DuckDB database at
`~/.introspect/introspect.duckdb`. Nothing is sent anywhere — everything runs
locally. See [Configuration](configuration.md) to point it at a different
location or scope how much history is loaded.
