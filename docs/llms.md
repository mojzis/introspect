# For LLMs

This site publishes machine-readable versions of the documentation following
the [llms.txt](https://llmstxt.org/) convention, so you can hand the whole
project context to an LLM when you want help using or extending Introspect.

## The files

<div class="grid cards" markdown>

-   :material-file-document-outline: __`llms.txt`__

    A concise, structured index of the documentation with links to each page —
    ideal as a lightweight map for an LLM to navigate from.

    [:octicons-arrow-right-24: /llms.txt](https://mojzis.github.io/introspect/llms.txt)

-   :material-file-document-multiple-outline: __`llms-full.txt`__

    The full documentation concatenated into a single Markdown file. Paste it
    directly into a chat when you want the model to have everything at once.

    [:octicons-arrow-right-24: /llms-full.txt](https://mojzis.github.io/introspect/llms-full.txt)

</div>

Both files are generated automatically at build time by the
[`mkdocs-llmstxt`](https://github.com/pawamoy/mkdocs-llmstxt) plugin, so they
stay in sync with the rest of the docs.

## How to use them

Drop the contents (or the URL) of `llms-full.txt` into your LLM of choice, then
ask questions like:

> Using the Introspect docs, write a `run_sql` MCP call that returns the top 10
> most expensive sessions for the current month.

> Explain how the background refresh loop swaps the DuckDB database without
> interrupting in-flight requests.

> I want to add a new web page to Introspect. Walk me through the
> handler → route → template → test pattern.

For quick reference, the raw contents are also available directly:

- Full dump: <https://mojzis.github.io/introspect/llms-full.txt>
- Index: <https://mojzis.github.io/introspect/llms.txt>
