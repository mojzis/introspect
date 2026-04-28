---
name: nolegend
description: >
  Tufte-style Plotly charts in introspect's FastAPI/Jinja stack. Use this
  whenever you add or modify a chart — bar, line, scatter, sparkline, or
  any go.Figure-driven visualization. Trigger on words like "clean",
  "minimal", "professional", "boardroom-ready", or whenever a new
  visualization is being built or an existing one is unclear. Charts in
  this repo are built server-side with go.Figure + nolegend.activate()
  and bootstrapped client-side via Plotly.newPlot in base.html — there is
  no Plotly Express or marimo here.
---

# nolegend — Plotly visualization rules for introspect

*Remove the legend to become one.*

Every chart should tell a story, not decorate the page. Apply Tufte's
principles, then refine with the helpers from the `nolegend` package.

## Where charts live in this project

- Server side: `go.Figure` built in a handler (e.g.
  `cost_breakdown.py:_build_figure`, `sessions.py:_render_multi_chart`),
  serialised via `fig.to_json()`, embedded into a `<script
  type="application/json">` tag.
- Client side: `base.html`'s `initCostChart` JS finds elements with
  `class="cost-chart"`, parses the JSON, and runs `Plotly.newPlot`. Click
  / select handlers are added via `el.on('plotly_click', ...)` →
  `htmx.ajax(...)` for HTMX fragment swaps.
- Template: `nolegend.activate()` is called lazily on first render
  (`_ensure_template()` pattern — keeps imports side-effect-free).
- Plotly Express is **not** used. Build figures directly with
  `go.Figure` and `add_trace`. Keep Plotly Express advice from upstream
  nolegend out of this project.

## Core principles

1. **Maximize data-ink ratio.** Every pixel encodes data. Remove
   gridlines, borders, backgrounds, decorations. The `tufte` template
   does most of this — don't undo it with overrides.
2. **No chartjunk.** No 3D, no gradient fills, no drop shadows, no
   coloured backgrounds.
3. **Direct-label over legends.** This is the load-bearing rule. With a
   view toggle that flips trace visibility, set
   `showlegend=False` on the layout and put the label on the line itself
   (see "Direct labels on toggled views" below).
4. **Titles convey insight, not description.** When the chart has its
   own caption (most do here, via the surrounding `<h2>`), keep the
   in-figure title empty so the plot area isn't pushed down.
5. **Range frames.** Axes span the data range, not arbitrary round
   numbers. The `tufte` template does this for line/scatter; bars
   default to starting at 0.
6. **Restrained colour.** ≤ 5 colours per chart. Gray (`#b0b0b0`) is
   for context. Keep the most expensive / most important series in the
   highest-contrast slot (e.g. `_INVOCATION_COLOR_PALETTE` puts red at
   index 0 so the runaway subagent stands out).

## Direct labels on toggled views

The session detail Cost tab is the canonical example: one figure with
multiple Scatter traces, view toggle (`Total / Agent / Category /
Invocations`) flips visibility via `Plotly.restyle`. Without a legend
the user can't tell which line is which — so we direct-label every
line trace at its endpoint.

Pattern (from `_render_multi_chart`):

```python
text_labels = [""] * len(ys)
if ys:
    text_labels[-1] = f" {name}"  # leading space avoids hugging the line

fig.add_trace(go.Scatter(
    x=x_axis,
    y=ys,
    mode="lines+text",
    name=name,
    line={"color": color, "width": 1.5},
    text=text_labels,
    textposition="middle right",
    textfont={"color": color, "size": 11},
    cliponaxis=False,
    customdata=...,
    hovertemplate=...,
))
```

Plus, on the layout:

```python
fig.update_layout(
    showlegend=False,
    margin={"l": 60, "r": 110, "t": 20, "b": 60},  # right margin holds labels
)
```

Why per-trace `text` rather than `add_annotation`:

- Annotations are figure-wide and **don't toggle** with trace visibility.
  When the user flips from "Total" to "By agent", annotation labels for
  hidden traces would still float in space.
- Per-trace `text` is part of the trace, so visibility toggling
  (`Plotly.restyle({visible: ...})`) hides the labels too, automatically.
- One label per trace at the last point (rest empty) is enough — the
  reader sees the colour + name pair at the line endpoint.

For stacked bar charts (cost-overview daily/hourly), use
`add_annotation` instead — bars don't have an "endpoint" to attach text
to, so place the label at the peak segment of each top-N group. See
`cost_breakdown.py:_compute_top_group_annotations` for the pattern.

## Hover, click, select interactivity

Charts in this repo are interactive surfaces, not static images. Keep
this in mind:

- `customdata` is your friend. Put per-point identifiers
  (uuid, day, session_id) in `customdata` so the click handler can fire
  the right HTMX request. Example: session cost chart's bucket points
  carry `[first_uuid, last_uuid, msg_count]`.
- `hovertemplate` should reference customdata — it gives the user the
  context they need before they decide to click.
- `dragmode='select'` for any chart that supports box-filtering. Force
  it via `Plotly.relayout` after `newPlot` in case the modebar resets it.
- Line-only traces don't reliably populate `evt.points` on box-select.
  Read `evt.range.x` and look up customdata from `el.data[trace_idx]`.
- Markers always-visible: when a chart has a "marker overlay" trace
  (spike/slope highlights), keep it visible across all view toggles by
  flagging its trace index in the JS `viewMap` logic.

## Colour rules (hard constraints)

- Never use more than 5 colours in a single chart (excluding the
  marker overlay).
- Never use rainbow / jet colorscales — not perceptually uniform.
- Never use red and green together without another differentiator.
- Use `_MAIN_AGENT_COLOR` for "main / total" series so the same colour
  consistently means the same thing across the app's charts.
- For sequential data, `nolegend.SEQUENTIAL` or `viridis`.
- When in doubt: fewer colours. One colour + gray usually suffices.

## Pinned colour map for re-rendered panels

When two panels show overlapping groups (e.g. daily totals and hourly
drill-down by project), build a single canonical colour map from the
all-time aggregate and pass it to both renderers. This way "project
foo" is the same colour in every panel. See
`cost_breakdown.py:_canonical_color_map` for the pattern.

## Chart type selection

Match the chart to the question, not the other way around.

| Question type | Chart |
|---|---|
| How did it change over time? | Line |
| How do groups compare? | Horizontal bar |
| What's the relationship? | Scatter |
| What's the distribution? | Histogram or box |
| What's the composition? | Stacked bar |
| What's the trend per group? | Small multiples (faceted bars/lines) |
| What's the quick trend? | Sparkline |

**Never use:** pie charts, donut charts, 3D charts, radar/spider charts,
gauge charts.

## Typography and titles

- **In-card heading carries the title.** The surrounding `<h2>` in the
  Jinja template is the chart's title — keep `fig.update_layout(title=...)`
  empty so the plot area sits at the top of its card.
- **Axis labels: only when non-obvious.** "Day" on a daily-bucket
  x-axis is fine. "USD" on a cost y-axis is useful. Skip generic
  "Value" / "Count".
- **Font: serif (Georgia).** The `tufte` template handles this. The
  base.html stylesheet pins `#js-plotly-tester` to Georgia too —
  don't remove that rule, it prevents axis-title displacement after
  HTMX swaps.

## Anti-patterns to catch and fix

| Anti-pattern | Fix |
|---|---|
| Legend box with 2+ entries on a toggled chart | Direct labels via per-trace `text` |
| Gridlines visible | `showgrid=False` (template handles it) |
| Default Plotly blue everywhere | Use the project palette constants |
| Title says "Chart of X by Y" | Rewrite the surrounding `<h2>` as the insight |
| Pie chart | Replace with horizontal bar |
| Too many colours (>5) | Aggregate, fold into "Other", or use spotlight |
| Y-axis starts at 0 when data starts at 800 | Trust the tufte template's range frame |
| Annotations on a toggled-trace chart | Use per-trace `text` instead |
| `fig.update_layout(title=...)` set | Drop it — the `<h2>` is the title |
| HTMX swap rebuilds chart from scratch and loses selection state | Use `Plotly.restyle` for visibility flips |

## Checklist before merging a chart change

- [ ] `nolegend.activate()` called (lazy `_ensure_template()` pattern)
- [ ] No legend box (direct labels or single series)
- [ ] No gridlines, no in-figure title
- [ ] ≤ 5 colours, palette pinned across panels if shared
- [ ] Axes have units where non-obvious
- [ ] Right margin ≥ 80–110px when direct-labeling
- [ ] `customdata` carries enough info for click/select handlers
- [ ] `cliponaxis=False` on text-bearing traces so end labels render
- [ ] Test added in `tests/test_routes.py` asserting the figure JSON
  shape (e.g. trace name, customdata presence, no `<polyline>`)
