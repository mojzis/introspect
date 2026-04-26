---
name: nolegend
description: >
  Tufte-style data visualization with Plotly in marimo notebooks.
  Use this skill whenever creating charts, graphs, dashboards, sparklines,
  or any data visualization with Plotly or Plotly Express — even if the user
  doesn't mention Tufte. Also trigger when the user asks for "clean charts",
  "professional charts", "boardroom-ready", "presentation-quality",
  "minimal", or "beautiful" visualizations. Trigger for any marimo notebook
  that involves Plotly plotting, px.line, px.bar, px.scatter, go.Figure,
  or data storytelling.
---

# nolegend — Plotly Visualization Skill

*Remove the legend to become one.*

Create boardroom-ready Plotly charts in marimo notebooks following
Edward Tufte's principles. Every chart should tell a story, not
decorate a slide.

## Core principles

Apply these to EVERY chart you create. No exceptions.

1. **Maximize data-ink ratio.** Every pixel should encode data.
   Remove gridlines, borders, backgrounds, and decorations.
2. **No chartjunk.** No 3D effects, no gradient fills, no drop shadows,
   no unnecessary gridlines, no colored backgrounds.
3. **Direct-label over legends.** Put the label on the data, not in a box.
4. **Titles convey insight, not description.**
   "Revenue peaked in Q3 before seasonal decline" not "Quarterly Revenue".
5. **Range frames.** Axes should span the data range, not arbitrary round numbers.
6. **Small multiples over complex single charts.** Use `facet_col`/`facet_row`.
7. **Restrained color.** 2-3 colors for most charts. Gray is your friend.

## Workflow: px → go refinement

ALWAYS follow this two-phase workflow. Never jump straight to go.Figure
for a new chart.

### Phase 1: Draft with Plotly Express

Start with `px` for speed. Get the data mapping right first.

```python
import plotly.express as px
import nolegend

nolegend.activate()

fig = px.line(
    df, x="date", y="revenue", color="region",
    template="tufte",
    title="Northern region drove 60% of growth in H2",
)
```

Rules for the px draft:
- Always pass `template="tufte"` (or call `nolegend.activate()` once)
- Write the insight-title immediately — it shapes the chart
- Use `facet_col` / `facet_row` for small multiples instead of cramming
  everything into one panel
- Keep the number of traces ≤ 5. If more, aggregate or facet.

### Phase 2: Refine with helpers and go

After the px draft looks structurally right, refine:

```python
# Axes span only the data
nolegend.range_frame(fig)

# Labels on the lines, legend removed
nolegend.direct_label(fig)

# Call out the key insight
nolegend.annotate_point(fig, x="2024-09", y=peak_val,
                     text="Record quarter")

# Fine-tune with update_layout / update_traces
fig.update_layout(
    yaxis_title="Revenue ($M)",
    margin=dict(r=80),  # room for direct labels
)
```

Rules for refinement:
- `nolegend.range_frame(fig)` — use on scatter and line charts, skip on bar charts
- `nolegend.direct_label(fig)` — use when ≤ 5 named traces exist
- `nolegend.annotate_point()` — always annotate the ONE thing the audience
  should remember
- Increase right margin (`margin=dict(r=80)`) when using direct labels
- For bar charts, consider `nolegend.strip_chartjunk(fig)` as a quick pass

## Color rules

Color is the most powerful — and most abused — visual channel.

### Default behavior

The `tufte` template uses the `qualitative` palette: 5 colorblind-safe,
muted colors that don't shout. This is correct for most charts.

### When to change palettes

| Situation | Palette | Code |
|-----------|---------|------|
| 2 series (most common case) | `duo` | `template=nolegend.with_palette("duo")` |
| Grayscale / print | `ink` | `template=nolegend.with_palette("ink")` |
| Financial / business | `earth` | `template=nolegend.with_palette("earth")` |
| Technical / scientific | `slate` | `template=nolegend.with_palette("slate")` |
| Heatmap / ordered data | `sequential` | Use with `color_continuous_scale` |

### The spotlight technique

When ONE series matters and the rest are context, use spotlight:

```python
colors = nolegend.QUALITATIVE.spotlight(index=2)  # highlight 3rd series
fig.update_traces(
    marker_color=[colors[i] for i in range(len(fig.data))]
)
```

This grays out everything except the focal series.

### Color rules (hard constraints)

- **Never use more than 5 colors** in a single chart
- **Never use rainbow / jet colorscales** — they're not perceptually uniform
- **Never use red and green together** without another differentiator
- **Gray (#b0b0b0) is a color** — use it for context, baselines, and
  secondary series
- For sequential data, use `nolegend.SEQUENTIAL` or `viridis`
- When in doubt, use fewer colors. One color + gray usually suffices.

## Chart type selection

Match the chart to the question, not the other way around.

| Question type | Chart | px function |
|--------------|-------|-------------|
| How did it change over time? | Line | `px.line` |
| How do groups compare? | Horizontal bar | `px.bar(..., orientation='h')` |
| What's the relationship? | Scatter | `px.scatter` |
| What's the distribution? | Histogram or box | `px.histogram` / `px.box` |
| What's the composition? | Stacked bar | `px.bar(..., barmode='stack')` |
| How do parts relate to whole? | 100% stacked bar | NOT pie charts |
| What's the trend per group? | Small multiples | `px.line(..., facet_col=...)` |
| What's the quick trend? | Sparkline | `nolegend.sparkline(values)` |

**Never use:** pie charts, donut charts, 3D charts, radar/spider charts,
gauge charts. These all have lower data-ink ratios than alternatives.

## marimo integration

### Basic reactive chart

```python
import marimo as mo
import plotly.express as px
import nolegend

nolegend.activate(dark=mo.app_meta().theme == "dark")

fig = px.scatter(df, x="cost", y="impact", hover_name="project",
                 template="tufte",
                 title="Three projects dominate the efficiency frontier")
nolegend.range_frame(fig)

chart = mo.ui.plotly(fig)
```

### Selection-driven workflow

```python
# Cell 1: chart with selections
chart = mo.ui.plotly(fig)
chart

# Cell 2: downstream — reacts to brush/click
selected = chart.value  # returns filtered DataFrame
mo.md(f"**{len(selected)}** projects selected, "
      f"avg impact: **{selected['impact'].mean():.1f}**")
```

### Dashboard layout

```python
mo.vstack([
    mo.md("## Portfolio Performance"),
    mo.hstack([chart_revenue, chart_margin], widths=[2, 1]),
    mo.hstack([spark_arr, spark_churn, spark_nps]),
])
```

### Sparkline rows

Sparklines work well in marimo `hstack` for KPI dashboards:

```python
def kpi_spark(label, values, current):
    spark = nolegend.sparkline(values, width=120, height=30)
    return mo.vstack([
        mo.md(f"**{label}**"),
        mo.ui.plotly(spark),
        mo.md(f"### {current}"),
    ])

mo.hstack([
    kpi_spark("ARR", arr_trend, "$12.4M"),
    kpi_spark("Churn", churn_trend, "2.1%"),
    kpi_spark("NPS", nps_trend, "67"),
])
```

## Typography and titles

- **Title = insight.** State what the chart shows, not what it is.
  The title is the first thing read; make it the takeaway.
- **Subtitle via annotation.** For methodology or caveats, add a
  subtitle annotation below the title, smaller and grayer.
- **Axis labels: only when non-obvious.** "Year" on a time axis is noise.
  "Revenue ($M)" on the y-axis is useful.
- **Font: serif by default.** The template uses Georgia. Don't override
  with Arial unless matching a corporate deck.

```python
# Adding a subtitle
fig.add_annotation(
    text="Source: Internal BI, Q1-Q4 2025",
    xref="paper", yref="paper",
    x=0, y=1.06,
    showarrow=False,
    font=dict(size=10, color="#999999"),
    xanchor="left",
)
```

## Anti-patterns to catch and fix

When reviewing or generating charts, actively look for and fix these:

| Anti-pattern | Fix |
|-------------|-----|
| Legend box with 2+ entries | `nolegend.direct_label(fig)` |
| Gridlines visible | `showgrid=False` (template handles this) |
| Default Plotly blue | Use tufte template or explicit palette |
| Title says "Chart of X by Y" | Rewrite as insight |
| Pie chart | Replace with horizontal bar |
| Too many colors (>5) | Aggregate categories or use spotlight |
| Y-axis starts at 0 when data starts at 800 | `nolegend.range_frame(fig)` |
| `fig.show()` in marimo | Return `mo.ui.plotly(fig)` instead |
| `plt.show()` anywhere | Wrong library — use Plotly |

## Polars integration

When the user works with Polars DataFrames (common in marimo):

```python
# Plotly Express accepts Polars directly since px 5.16+
fig = px.line(df_polars, x="date", y="value", template="tufte")

# For older versions, convert only the columns you need
fig = px.line(
    df_polars.select(["date", "value"]).to_pandas(),
    x="date", y="value", template="tufte",
)
```

## Checklist before presenting

Run through this before any chart is shown to the user:

- [ ] Title states the insight, not the axes
- [ ] No legend box (direct labels or single series)
- [ ] No gridlines
- [ ] ≤ 5 colors
- [ ] Axes have units where non-obvious
- [ ] Range frame applied (line/scatter)
- [ ] Key data point annotated
- [ ] Right margin sufficient for labels (≥80px if direct-labeling)
- [ ] Works in marimo dark mode (if applicable)
- [ ] Uses `mo.ui.plotly(fig)` not `fig.show()`
