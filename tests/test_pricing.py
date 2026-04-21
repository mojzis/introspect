"""Table-driven tests for ``introspect.pricing``."""

from __future__ import annotations

import pytest

from introspect.pricing import (
    PRICING_CACHE_READ_RATE_SQL,
    PRICING_CACHE_WRITE_1H_RATE_SQL,
    PRICING_CACHE_WRITE_5M_RATE_SQL,
    PRICING_INPUT_RATE_SQL,
    PRICING_OUTPUT_RATE_SQL,
    Rates,
    compute_cost_usd,
    rates_for,
)

# (model_name, expected Rates) — covers every family + a dated suffix variant.
_RATE_CASES = [
    ("claude-opus-4-7", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-6", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-5", Rates(5, 6.25, 10, 0.50, 25)),
    ("claude-opus-4-1", Rates(15, 18.75, 30, 1.50, 75)),
    ("claude-opus-3", Rates(15, 18.75, 30, 1.50, 75)),
    ("claude-sonnet-4-6", Rates(3, 3.75, 6, 0.30, 15)),
    ("claude-sonnet-3-7", Rates(3, 3.75, 6, 0.30, 15)),
    ("claude-haiku-4-5-20251001", Rates(1, 1.25, 2, 0.10, 5)),
    ("claude-haiku-3-5", Rates(0.80, 1, 1.60, 0.08, 4)),
    ("<synthetic>", Rates(0, 0, 0, 0, 0)),
    ("totally-made-up-model", Rates(0, 0, 0, 0, 0)),
    (None, Rates(0, 0, 0, 0, 0)),
    ("", Rates(0, 0, 0, 0, 0)),
]


@pytest.mark.parametrize(("model", "expected"), _RATE_CASES)
def test_rates_for(model, expected):
    """Each model family resolves to its expected rate table."""
    assert rates_for(model) == expected


def test_rates_for_prefix_match_picks_longest():
    """Dated suffix variants resolve via prefix match without overshadowing."""
    # opus-4-1 should match the legacy table, not the current opus-4 one
    assert rates_for("claude-opus-4-1-some-date").input == 15
    # opus-4 alone (legacy) should not silently bind to opus-4-7 etc.
    assert rates_for("claude-opus-4-99").input == 15


def test_compute_cost_usd_zero_for_no_tokens():
    """Empty usage yields zero cost."""
    assert compute_cost_usd(model="claude-opus-4-7") == 0


def test_compute_cost_usd_opus_current():
    """Per-1M math works for opus current generation."""
    # 1M input, 1M output, 1M cache_read, 1M cache_5m, 1M cache_1h
    cost = compute_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_5m=1_000_000,
        cache_creation_1h=1_000_000,
    )
    assert cost == pytest.approx(5 + 25 + 0.50 + 6.25 + 10)


def test_compute_cost_usd_haiku():
    """Per-1M math works for the Haiku 4.5 family with dated suffix."""
    cost = compute_cost_usd(
        model="claude-haiku-4-5-20251001",
        input_tokens=2_000_000,
        output_tokens=500_000,
    )
    # 2 * 1.0 input + 0.5 * 5.0 output
    assert cost == pytest.approx(2 + 2.5)


def test_compute_cost_usd_synthetic_is_zero():
    """Synthetic + unknown models always cost $0."""
    assert (
        compute_cost_usd(
            model="<synthetic>",
            input_tokens=10_000_000,
            output_tokens=10_000_000,
        )
        == 0
    )
    assert (
        compute_cost_usd(
            model="brand-new-model",
            input_tokens=10_000_000,
        )
        == 0
    )


# All five rate dimensions feed SESSION_COST_SUBQUERY; a typo in any one
# would silently mis-bill (cache_read in particular is the dominant cost for
# typical Claude Code sessions).
_SQL_RATE_PAIRS = [
    ("input", PRICING_INPUT_RATE_SQL),
    ("output", PRICING_OUTPUT_RATE_SQL),
    ("cache_read", PRICING_CACHE_READ_RATE_SQL),
    ("cache_write_5m", PRICING_CACHE_WRITE_5M_RATE_SQL),
    ("cache_write_1h", PRICING_CACHE_WRITE_1H_RATE_SQL),
]

_SQL_TEST_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-1",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-haiku-3-5",
    "<synthetic>",
    "unknown-model",
]


@pytest.mark.parametrize("model", _SQL_TEST_MODELS)
@pytest.mark.parametrize(("attr", "sql"), _SQL_RATE_PAIRS)
def test_pricing_sql_matches_python(model, attr, sql):
    """Each SQL CASE rate must agree with the Python ``Rates`` attribute."""
    import duckdb  # noqa: PLC0415

    conn = duckdb.connect(":memory:")
    row = conn.execute(
        f"SELECT {sql} FROM (SELECT ? AS model)",
        [model],
    ).fetchone()
    assert row is not None, model
    # DuckDB returns Decimal for fractional CASE-derived numerics; coerce to
    # float so the equality check is value-based.
    assert float(row[0]) == pytest.approx(getattr(rates_for(model), attr))
